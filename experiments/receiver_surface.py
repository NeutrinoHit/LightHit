#!/usr/bin/env python3
"""Prepare first-entry receiver tables without changing event data manifests.

Examples:
  PYTHONPATH=src python experiments/receiver_surface.py rte --model MODEL.npz \
      --output surface-450.npz --wavelength 450 --frequencies 0 .03 .1 .6 1.2
  PYTHONPATH=src python experiments/receiver_surface.py q1 --model MODEL.npz \
      --output first-entry-q1 --r-min 3 --r-max 300 --radii 97

The model supplies the physical glass radius, water and detector response.
Prepared files remain separate from centre-reference tables. Convergence and
full-event timing must be checked before enabling them in an event manifest.
"""
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import argparse,hashlib,json,platform,sys
import numpy as np


def _hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block:=stream.read(1<<20):digest.update(block)
    return digest.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('kind',choices=('rte','vertex','bank','q1','geometry','q1-time','q1-bands','q1-band-time'))
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--receiver-index',type=int,default=0)
    parser.add_argument('--wavelength',type=float,default=450.)
    parser.add_argument('--band-index',type=int)
    parser.add_argument('--wavelength-nodes',type=int,default=32)
    parser.add_argument('--r-min',type=float,default=3.)
    parser.add_argument('--r-max',type=float,default=300.)
    parser.add_argument('--radii',type=int,default=220)
    parser.add_argument('--radial-grid',type=Path,help='explicit increasing radii in an NPY file')
    parser.add_argument('--jobs',type=int,default=1,help='independent Q1 radial blocks')
    parser.add_argument('--frequency-block',type=int,default=8)
    parser.add_argument('--reuse-bank',type=Path,help='recorded partial bank at lower receiver degree')
    parser.add_argument('--source-degree',type=int,default=32)
    parser.add_argument('--scattering-degree',type=int,default=96)
    parser.add_argument('--k-max',type=float,default=12.)
    parser.add_argument('--acceptance-degree',type=int)
    parser.add_argument('--angular-pairs',type=int,default=96)
    parser.add_argument('--radial-nodes',type=int,default=800)
    parser.add_argument('--frequencies',nargs='+',type=float)
    parser.add_argument('--hit-nodes',type=int,default=33)
    parser.add_argument('--miss-nodes',type=int,default=97)
    parser.add_argument('--time-nodes',type=int,default=257)
    parser.add_argument('--path-nodes',type=int,default=128)
    parser.add_argument('--path-grid-nodes',type=int,default=513)
    parser.add_argument('--fine-atlas',type=Path,help='fine Q1 table for a controlled coarse time level')
    parser.add_argument('--geometry',type=Path,help='unit-sphere geometry atlas for one-band CDF preparation')
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args(argv)
    from lighthit.bgvd import load_model
    from lighthit.model import WavelengthQuadrature
    from lighthit.directional import acceptance_bandwidth,DirectionalCache
    from lighthit.receiver_vertex import ReceiverVertexConfig
    from lighthit.receiver import SphereSurface
    from lighthit.receiver_adjoint import first_entry_cache,SphereTransportConfig
    from lighthit.receiver_first_atlas import FirstEntryAtlas,FirstEntryAtlasConfig
    from lighthit.green import SolverSettings
    from lighthit.cache import CacheGrid
    from lighthit.transport import KernelConfig
    from lighthit.prompt import _spectral_tables
    from lighthit import receiver_adjoint,receiver_first,receiver_first_fold,receiver_first_atlas,receiver_vertex
    model=load_model(args.model);surface=model.detector.receiver_surface
    if not isinstance(surface,SphereSurface):raise ValueError('model must supply spherical glass geometry')
    if not 0<=args.receiver_index<len(model.detector):raise ValueError('receiver index outside model')
    radius=float(surface.radii(len(model.detector))[args.receiver_index]);surface=SphereSurface(radius)
    if not radius<args.r_min<args.r_max or args.radii<2:raise ValueError('require glass radius < r-min < r-max and >=2 radii')
    if args.output.exists() and not args.force:raise FileExistsError(f'{args.output} exists; choose a separate output or --force')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    code={Path(m.__file__).name:_hash(m.__file__) for m in
          (receiver_adjoint,receiver_first,receiver_first_fold,receiver_first_atlas,receiver_vertex)}
    record=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        model_sha256=_hash(args.model),source_sha256=code,
        platform=platform.platform(),python=sys.version.split()[0],numpy=np.__version__,radius_m=radius)
    began=perf_counter();radii=(np.geomspace(args.r_min,args.r_max,args.radii) if args.radial_grid is None
                              else np.load(args.radial_grid,allow_pickle=False))
    if args.radial_grid is not None:record['radial_grid_sha256']=_hash(args.radial_grid)
    if args.kind in ('rte','vertex','bank'):
        degree=acceptance_bandwidth(model.detector.angular_acceptance)[0] if args.acceptance_degree is None else args.acceptance_degree
        wave=args.wavelength
        if args.band_index is not None:
            wavelength=WavelengthQuadrature.gauss_legendre(*model.medium.wavelength_range_nm,args.wavelength_nodes)
            if not 0<=args.band_index<len(wavelength.wavelength_nm):raise ValueError('band index outside quadrature')
            wave=float(wavelength.wavelength_nm[args.band_index])
        omega=np.linspace(0.,1.2,65) if args.frequencies is None else np.asarray(args.frequencies)
        update=lambda i,n:print(json.dumps(dict(frequency=i,total=n,seconds=perf_counter()-began)),flush=True)
        settings=SolverSettings(args.scattering_degree,args.source_degree,args.k_max,.04,10)
        if args.kind=='bank':
            from lighthit.receiver_bank import prepare_receiver_bank
            wavelength=WavelengthQuadrature.gauss_legendre(*model.medium.wavelength_range_nm,args.wavelength_nodes)
            config=ReceiverVertexConfig(angular_pairs=args.angular_pairs,radial_nodes=args.radial_nodes)
            metadata=prepare_receiver_bank([model.medium.band(float(w)) for w in wavelength.wavelength_nm],
                settings,CacheGrid(radii,omega),surface,degree,args.output,config=config,jobs=args.jobs,
                frequency_block=args.frequency_block,progress=update,reuse_directory=args.reuse_bank)
            record.update(metadata=metadata,numerics=asdict(config));sidecar=args.output/'run.json'
            record['seconds']=perf_counter()-began
            sidecar.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            print(json.dumps(dict(finished=True,output=str(args.output),seconds=record['seconds'])),flush=True)
            return 0
        if args.kind=='vertex':
            config=ReceiverVertexConfig(angular_pairs=args.angular_pairs,radial_nodes=args.radial_nodes)
            cache=DirectionalCache.build(model.medium.band(wave),settings,CacheGrid(radii,omega),degree,
                receiver_surface=surface,receiver_config=config,angular_backend='numba',radial_phase='flight',
                progress=update)
        else:
            config=SphereTransportConfig(angular_pairs=args.angular_pairs,radial_nodes=args.radial_nodes)
            cache=first_entry_cache(model.medium.band(wave),settings,
                CacheGrid(radii,omega),surface,degree,config=config,progress=update)
        temporary=args.output.with_suffix('.tmp.npz');cache.save(temporary);temporary.replace(args.output)
        record.update(wavelength_nm=wave,acceptance_degree=degree,numerics=asdict(config),timings=cache.timings_s)
        sidecar=args.output.with_suffix('.json')
    elif args.kind=='q1-band-time':
        from lighthit.receiver_first_spectral import SpectralFirstEntry
        from lighthit.receiver_geometry import ReceiverGeometry
        if args.geometry is None:raise ValueError('--geometry is required for q1-band-time')
        atlas=SpectralFirstEntry.load(args.output)
        atlas.prepare_quadratic(ReceiverGeometry.load(args.geometry),args.output,
            progress=lambda i,n:print(json.dumps(dict(radii=i,total=n,seconds=perf_counter()-began)),flush=True)
                if i%16==0 else None)
        record.update(metadata=atlas.metadata);sidecar=args.output/'deposition-run.json'
    elif args.kind=='q1-time':
        from lighthit.receiver_first_adaptive import AdaptiveFirstEntry
        if args.fine_atlas is None:raise ValueError('--fine-atlas is required for q1-time')
        fine=FirstEntryAtlas.load(args.fine_atlas)
        atlas=AdaptiveFirstEntry.of(fine,stride=4);atlas.save(args.output)
        record.update(metadata=atlas.coarse.metadata);sidecar=args.output/'run.json'
    else:
        kernel=model.kernel(KernelConfig(wavelength_nodes=args.wavelength_nodes))
        tables=_spectral_tables(kernel)
        if args.kind=='q1-bands':
            from lighthit.receiver_first_spectral import SpectralFirstEntry,SpectralFirstEntryConfig
            if args.geometry is None:raise ValueError('--geometry is required for q1-bands')
            config=SpectralFirstEntryConfig(args.time_nodes)
            atlas=SpectralFirstEntry.build_parallel(geometry_directory=args.geometry,radius_m=radius,tables=tables,
                output=args.output,jobs=args.jobs,config=config,
                progress=lambda i,n:print(json.dumps(dict(radii=i,total=n,seconds=perf_counter()-began)),flush=True))
            record.update(numerics=asdict(config),metadata=atlas.metadata);sidecar=args.output/'run.json'
            record['seconds']=perf_counter()-began
            sidecar.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            print(json.dumps(dict(finished=True,output=str(args.output),seconds=record['seconds'])),flush=True)
            return 0
        if args.kind=='geometry':
            from lighthit.receiver_geometry import ReceiverGeometry,ReceiverGeometryConfig
            from lighthit.receiver_first_atlas import _path_extent
            config=ReceiverGeometryConfig(hit_nodes=args.hit_nodes,miss_nodes=args.miss_nodes,
                path_grid_nodes=args.path_grid_nodes,path_nodes=args.path_nodes)
            excess=max(_path_extent(r,radius,model.medium.g,tables,1e-8)-r for r in radii)/radius
            atlas=ReceiverGeometry.build_parallel(radii_unit=radii/radius,g=model.medium.g,max_excess_unit=excess,
                config=config,output=args.output,jobs=args.jobs,overwrite=args.force,
                progress=lambda i,n:print(json.dumps(dict(radii=i,total=n,seconds=perf_counter()-began)),flush=True))
            record.update(numerics=asdict(config));sidecar=args.output/'run.json'
            record['seconds']=perf_counter()-began
            sidecar.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
            print(json.dumps(dict(finished=True,output=str(args.output),seconds=record['seconds'])),flush=True)
            return 0
        config=FirstEntryAtlasConfig(hit_nodes=args.hit_nodes,miss_nodes=args.miss_nodes,
            time_nodes=args.time_nodes,path_nodes=args.path_nodes)
        if args.jobs>1:
            atlas=FirstEntryAtlas.build_parallel(radii_m=radii,surface=surface,g=model.medium.g,
                tables=tables,config=config,output=args.output,jobs=args.jobs,
                overwrite=args.force,
                progress=lambda i,n:print(json.dumps(dict(radii=i,total=n,seconds=perf_counter()-began)),flush=True))
        else:
            atlas=FirstEntryAtlas.build(radii_m=radii,surface=surface,g=model.medium.g,tables=tables,config=config,
                progress=lambda i,n:print(json.dumps(dict(row=i,total=n,seconds=perf_counter()-began)),flush=True)
                     if i%(args.hit_nodes+args.miss_nodes)==0 else None)
            atlas.save(args.output)
        record.update(numerics=asdict(config),metadata=atlas.metadata)
        sidecar=args.output/'run.json'
    record['seconds']=perf_counter()-began
    sidecar.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(finished=True,output=str(args.output),seconds=record['seconds'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
