#!/usr/bin/env python3
"""Build q1 track-response rows from optical properties only; no G4/event input.

The synthetic single OM supplies response functions and area; its position and
orientation are excluded from the cache contract. Every radius is resumable and
checksummed. Charge prefixes stay small/eager; time CDF rows load lazily.
"""
import argparse
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter,process_time
import hashlib,json,os,subprocess,sys,shutil
import numpy as np


def optical_kernel(model_path=None,first_entry_tables=None):
    import lighthit as lh
    if first_entry_tables is not None and model_path is None:
        raise ValueError('first-entry preparation requires a model with actual glass geometry')
    if model_path is not None:
        from lighthit.receiver_vertex import ReceiverVertexConfig
        options=(dict(scattered_receiver='first-entry',first_entry_tables=first_entry_tables,
            scattering_degree=96,k_max_per_m=12.,receiver_vertex=ReceiverVertexConfig(angular_pairs=96,radial_nodes=400))
            if first_entry_tables is not None else {})
        kernel=lh.load_model(model_path).kernel(lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require',**options))
        if first_entry_tables is not None:kernel._receiver_contract()
        kernel.detector=kernel.detector.subset(np.array([0]))
        return kernel
    from bgvd_model import OpticalModule
    from bgvd_model.BaikalWater import BaikalWater
    water=BaikalWater();parameters=np.asarray(OpticalModule.angular_parameters,float)
    medium=lh.SpectralMedium(np.asarray(water.wavelength),np.asarray(water.absorption_inv_length),
        np.asarray(water.scattering_inv_length),np.asarray(water.phase_refraction_index),
        np.asarray(water.group_refraction_index),g=.9,provenance='installed optical provider only')
    def angular(x):
        x=-np.asarray(x,float)
        return np.maximum(sum(parameters[k]*x**k for k in range(len(parameters))),0.)
    def efficiency(w):
        return np.asarray(OpticalModule.efficiency(w),float)*np.asarray(OpticalModule.transmission_gel_glass(w),float)
    efficiency.wavelength_knots_nm=np.unique(np.r_[OpticalModule.eff_wavelengths,OpticalModule.transmission_wavelengths])
    detector=lh.DetectorArray(np.zeros((1,3)),np.array([[0.,0.,1.]]),np.pi*OpticalModule.radius**2,angular,efficiency)
    return lh.TransportKernel(medium,detector,lh.KernelConfig(omega_per_ns=np.array([0.]),cache_policy='require'))


def contract(kernel,cells,time_step_ns,reference_rho,min_radius,max_time_ns,beta=1.):
    from lighthit.experimental.primary_polyline import primary_spectral_quadrature
    from lighthit.prompt import _spectral_tables
    local=copy(kernel);local.wavelength=primary_spectral_quadrature(kernel.medium,kernel.detector,order=2,
        wavelength_range_nm=kernel.config.wavelength_range_nm)
    tables=_spectral_tables(local)
    photon_c=.299792458;speed=beta*photon_c
    roots=1/np.sqrt((beta*kernel.medium.phase_index)**2-1)
    root=float(np.mean(roots))
    scale=.1
    x=root+scale*np.sinh(np.linspace(np.arcsinh((-8.5-root)/scale),np.arcsinh((8.5-root)/scale),cells+1))
    x[0]=-8.5;x[-1]=8.5
    fronts=np.sqrt((kernel.medium.group_index/photon_c)**2-1/speed**2)
    front=float(fronts.min());fine_end=float(fronts.max()+40/reference_rho)
    if kernel.config.scattered_receiver!='centre':
        # The scaled glass front depends on R/rho. Cover its earliest value
        # over the actual radius axis rather than smearing it into the anchor.
        radius=kernel._receiver_contract()['radius_m']
        front-=float(np.max(kernel.medium.group_index))*radius/(photon_c*min_radius)
    dt=time_step_ns/reference_rho
    maximum=max_time_ns/min_radius
    early=np.arange(front-2/reference_rho,fine_end,dt)
    # Fine near the cone/front, geometric excess time in the broad positive tail.
    tail=front+np.geomspace(fine_end-front,maximum-front,240)
    t=np.unique(np.r_[-5.,early,tail,maximum])
    physics=dict(beta=beta,speed_m_per_ns=speed,scattering_order=1,
        acceptance_alpha=np.asarray(kernel.acceptance()['alpha']).tolist(),
        effective_area_m2=float(kernel.detector.effective_area_m2[0]),g=float(kernel.medium.g),
        spectral_nodes=len(local.wavelength.wavelength_nm),spectral_order=2,
        spectral_tables={k:np.asarray(v).tolist() for k,v in tables.items()},
        source_model='unit C0 per metre; C2=-C0/beta²; homogeneous optical medium',
        event_input=False,geometry_input=False,accuracy_certified=False)
    if kernel.config.scattered_receiver!='centre':
        physics['receiver_contract']=kernel._receiver_contract();physics['geometry_input']='glass radius only'
    settings=dict(format_version=1,physics=physics,x_edges=x.tolist(),time_over_rho_edges=t.tolist(),
        interpolation='linear positive normalized CDF; log row amplitude; exact cubic acceptance',
        reference_rho_m=reference_rho,near_front_time_step_at_reference_ns=time_step_ns,
        max_time_at_min_radius_ns=max_time_ns)
    digest=hashlib.sha256(json.dumps(settings,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    from lighthit.experimental.track_first_primitive import track_first_optical_contract
    _,optical_hash=track_first_optical_contract(kernel)
    settings['physics'].update(optical_fingerprint=optical_hash,causal_zero_before_tau0=True)
    return settings,digest


def atomic_json(path,data):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False));os.replace(temporary,path)


def build_row(kernel,radius,x,t,physics,progress=None,workers=1):
    from lighthit.experimental.track_first_primitive import build_track_first_rows,orientation_nodes
    nodes,powers,matrix=orientation_nodes();inverse=np.linalg.inv(matrix)
    rows=build_track_first_rows(kernel,rho_m=radius,orientations=nodes,beta=physics['beta'],
        speed_m_per_ns=physics['speed_m_per_ns'],u_edges_m=radius*x,tau_edges_ns=radius*t,progress=progress,workers=workers)
    result={}
    for key,attribute in [('charge_left','_q_left'),('charge_right','_q_right'),('cdf_left','_cdf_left'),('cdf_right','_cdf_right')]:
        positive=np.stack([getattr(row,attribute) for row in rows])
        result[key]=np.moveaxis(np.tensordot(inverse,positive,axes=(1,0)),0,-1)
    return result,rows[0].metadata


def row_query(row,radius,x,t,powers,look,lo,hi,edges,*,radial_time_origin_ns=0.):
    from lighthit.experimental.track_first_primitive import TrackFirstAtlas
    # Zero radial weight at this stored node isolates u/time/angular accuracy.
    view=SimpleNamespace(radii_m=np.array([radius,radius*(1+1e-9)]),x_edges=x,
        time_over_rho_edges=t,powers=powers,metadata={'causal_zero_before_tau0':True},
        **{key:[value,value] for key,value in row.items()})
    return TrackFirstAtlas.query(view,radius,look,lo,hi,edges,radial_time_origin_ns=radial_time_origin_ns)


def check_geometry(kernel,query,radius,root,*,full=True,beta=1.):
    from lighthit.experimental.track_first_primitive import build_track_first_primitive
    look=np.array([.23,.61,np.sqrt(1-.23**2-.61**2)])
    intervals=[(-1.,4.),(root-.25,root+.2),(root+.6,root+3.)]
    result=[]
    for j,(a,b) in enumerate(intervals if full else intervals[:1]):
        lo,hi=a*radius,b*radius;edges=np.arange(-5.,806.,5.)+.37*j
        direct=build_track_first_primitive(kernel,rho_m=radius,orientation=look,beta=beta,speed_m_per_ns=beta*.299792458,
            u_edges_m=[lo,hi],tau_edges_ns=edges)
        q,bins=query(radius,look,lo,hi,edges)
        qr,br=direct.charge_cells[0],direct.bins_cells[0]
        # A distant interval may have no support in this finite time window.
        # Keep that exact zero check instead of reporting an undefined 0/0.
        if (qr==0 and q!=0) or (not np.any(br) and np.any(bins)):
            raise ValueError('nonzero prediction outside the reference support')
        result.append(dict(rho_m=radius,interval_x=[a,b],orientation=look.tolist(),charge=q,reference_charge=qr,
            charge_relative_error=float(abs(q-qr)/qr) if qr else 0.,
            bins_l1_relative=float(abs(bins-br).sum()/br.sum()) if np.any(br) else 0.,
            reference_cpu_s=direct.metadata['build_cpu_s']))
    return result


def _write_row(path,row,storage):
    """Retain identical NPY members; choose whether ZIP inflation is needed."""
    if storage not in ('compressed','stored'):raise ValueError('unknown row storage')
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        (np.savez if storage=='stored' else np.savez_compressed)(stream,**row)
    os.replace(temporary,path)


def _parallel_rows(args,radii,selected,digest,manifest,records,manifest_path,*,script=None):
    """Build independent radius checkpoints; only the parent joins manifests."""
    from concurrent.futures import ThreadPoolExecutor,as_completed
    script=Path(__file__).resolve() if script is None else Path(script)
    missing=[float(r) for r in selected if float(r) not in records]
    def build(radius):
        index=int(np.flatnonzero(radii==radius)[0])
        directory=args.output/'blocks'/f'radius-{index:04d}';directory.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,str(script),'--output',str(directory),'--radii',*map(str,radii),
            '--build-radii',str(radius),'--beta',str(args.beta),'--cells',str(args.cells),
            '--time-step',str(args.time_step),'--max-time-ns',str(args.max_time_ns),'--workers',str(args.workers),
            '--row-storage',getattr(args,'row_storage','compressed')]
        for flag,value in [('--model',args.model),('--first-entry-tables',args.first_entry_tables)]:
            if value is not None:command.extend([flag,str(value)])
        if args.check_rows:command.append('--check-rows')
        log=directory/'build.log'
        with log.open('ab') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'radius {radius} build failed; see {log}')
        part=json.loads((directory/'manifest.json').read_text())
        if part['optics_grid_hash']!=digest or part['radii_m']!=radii.tolist():
            raise ValueError('parallel row has a different optics/grid contract')
        rows=[r for r in part['rows'] if float(r['rho_m'])==radius]
        if len(rows)!=1:raise ValueError('parallel radius checkpoint is incomplete')
        record=rows[0];name=record['file']
        if Path(name).name!=name:raise ValueError('parallel row path must be local')
        source=directory/name
        if hashlib.sha256(source.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('parallel row checksum differs')
        checks=part.get('row_checks',{}).get(str(radius))
        if args.check_rows and checks is None:raise ValueError('parallel row checks are missing')
        return source,record,checks
    with ThreadPoolExecutor(max_workers=min(args.radius_workers,max(1,len(missing)))) as pool:
        futures=[pool.submit(build,r) for r in missing]
        errors=[]
        for future in as_completed(futures):
            try:source,record,checks=future.result()
            except Exception as error:
                errors.append(error);continue
            target=args.output/record['file']
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=record['sha256']:
                    raise FileExistsError('different parallel row already exists')
            else:
                temporary=target.with_suffix('.join.tmp')
                if temporary.exists():temporary.unlink()
                try:os.link(source,temporary)
                except OSError:shutil.copyfile(source,temporary)
                if hashlib.sha256(temporary.read_bytes()).hexdigest()!=record['sha256']:
                    raise ValueError('joined parallel row checksum differs')
                os.replace(temporary,target)
            radius=float(record['rho_m']);records[radius]=record
            if checks is not None:manifest['row_checks'][str(radius)]=checks
            manifest['rows']=[records[r] for r in sorted(records)];atomic_json(manifest_path,manifest)
            print(json.dumps(dict(radius_completed=record,checks=checks)),flush=True)
        if errors:raise errors[0]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path,help='serialized optics and actual glass radius; positions are not used')
    p.add_argument('--first-entry-tables',type=Path,help='geometry CDF for first glass entry')
    p.add_argument('--workers',type=int,default=1,help='independent source cells; each keeps its complete wavelength sum')
    p.add_argument('--radius-workers',type=int,default=1,help='independent radius processes; each uses --workers source-cell threads')
    p.add_argument('--row-storage',choices=['compressed','stored'],default='compressed',
        help='stored ZIP members avoid repeated time-row inflation; numerical arrays are unchanged')
    p.add_argument('--beta',type=float,default=1.,help='actual source cone and emission-clock speed')
    p.add_argument('--radii',type=float,nargs='+',default=[3,5,8,12,18,25,35,50,70,100,150,220,300])
    p.add_argument('--build-radii',type=float,nargs='+',help='Build only this subset, leaving a resumable incomplete store')
    p.add_argument('--cells',type=int,default=96)
    p.add_argument('--time-step',type=float,default=.1)
    p.add_argument('--max-time-ns',type=float,default=900.)
    p.add_argument('--check-rows',action='store_true')
    p.add_argument('--reuse-rows-from',type=Path)
    p.add_argument('--validate-midpoints',type=float,nargs='*')
    args=p.parse_args()
    if args.workers<1 or args.radius_workers<1:p.error('positive worker counts required')
    import numba
    from lighthit.experimental.track_first_primitive import orientation_nodes,TrackFirstAtlasStore
    if numba.get_num_threads()!=1:p.error('NUMBA_NUM_THREADS=1 required')
    radii=np.asarray(args.radii,float)
    if not np.isfinite(radii).all() or len(radii)<2 or np.any(np.diff(radii)<=0) or radii[0]<3:
        p.error('increasing positive radii >=3m required')
    if args.cells<16 or args.time_step<=0:p.error('positive time step and at least16 source cells required')
    selected=radii if args.build_radii is None else np.asarray(args.build_radii,float)
    if any(not np.any(radii==r) for r in selected):p.error('build radii must occur in full radius axis')
    try:kernel=optical_kernel(args.model,args.first_entry_tables)
    except ValueError as error:p.error(str(error))
    if not 0<args.beta<=1 or np.any(args.beta*kernel.medium.phase_index<=1):p.error('beta must emit throughout the atlas spectrum')
    settings,digest=contract(kernel,args.cells,args.time_step,30.,radii[0],args.max_time_ns,args.beta)
    x=np.asarray(settings['x_edges']);t=np.asarray(settings['time_over_rho_edges'])
    _,powers,matrix=orientation_nodes()
    settings.update(radii_m=radii.tolist(),powers=powers.tolist(),optics_grid_hash=digest,row_storage=args.row_storage,
        orientation_condition_number=float(np.linalg.cond(matrix)))
    args.output.mkdir(parents=True,exist_ok=True);manifest_path=args.output/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if (manifest['optics_grid_hash']!=digest or manifest['radii_m']!=radii.tolist()
                or manifest.get('row_storage','compressed')!=args.row_storage):
            p.error('frozen cache contract differs; choose a new directory')
    else:
        manifest={**settings,'status':'building','rows':[], 'row_checks':{}}
        atomic_json(manifest_path,manifest)
    records={float(v['rho_m']):v for v in manifest['rows']}
    if args.reuse_rows_from:
        import shutil
        old=json.loads((args.reuse_rows_from/'manifest.json').read_text())
        if old['optics_grid_hash']!=digest:p.error('reused rows have a different optics/grid contract')
        if old.get('row_storage','compressed')!=args.row_storage:p.error('reused rows have a different storage layout')
        for record in old['rows']:
            radius=float(record['rho_m'])
            if radius not in records and np.any(radii==radius):
                source=args.reuse_rows_from/record['file']
                if hashlib.sha256(source.read_bytes()).hexdigest()!=record['sha256']:
                    raise ValueError('reused row checksum mismatch')
                target=args.output/record['file']
                if target.exists():
                    if hashlib.sha256(target.read_bytes()).hexdigest()!=record['sha256']:
                        raise FileExistsError('different row already exists at reuse target')
                else:
                    try:os.link(source,target)
                    except OSError:shutil.copyfile(source,target)
                records[radius]={**record,'reused_from':str(args.reuse_rows_from)}
                if str(radius) in old.get('row_checks',{}):manifest['row_checks'][str(radius)]=old['row_checks'][str(radius)]
        manifest['rows']=[records[r] for r in sorted(records)];atomic_json(manifest_path,manifest)
    print(json.dumps(dict(contract_hash=digest,source_cells=len(x)-1,time_edges=len(t),radii=radii.tolist(),
                         event_inputs=False,geometry_inputs=settings['physics']['geometry_input'])),flush=True)
    if args.radius_workers>1:
        _parallel_rows(args,radii,selected,digest,manifest,records,manifest_path)
    root=float(np.mean(1/np.sqrt((args.beta*kernel.medium.phase_index)**2-1)))
    for radius in selected:
        radius=float(radius)
        if radius in records:
            path=args.output/records[radius]['file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=records[radius]['sha256']:
                raise ValueError('existing row checksum mismatch')
            if args.check_rows and str(radius) not in manifest['row_checks']:
                with np.load(path,allow_pickle=False) as data:row={k:data[k].copy() for k in data.files}
                query=lambda r,look,lo,hi,edges:row_query(row,r,x,t,powers,look,lo,hi,edges)
                manifest['row_checks'][str(radius)]=check_geometry(kernel,query,radius,root,beta=args.beta)
                atomic_json(manifest_path,manifest)
            print(json.dumps(dict(reused_radius_m=radius)),flush=True);continue
        start,cpu=perf_counter(),process_time()
        def progress(completed,total):
            if completed%8==0 or completed==total:
                print(json.dumps(dict(rho_m=radius,source_cells_completed=completed,source_cells_total=total,
                    elapsed_seconds=perf_counter()-start)),flush=True)
        row,info=build_row(kernel,radius,x,t,settings['physics'],progress=progress,workers=args.workers)
        filename=f'rho-{radius:010.6f}.npz';path=args.output/filename
        _write_row(path,row,args.row_storage)
        record=dict(rho_m=radius,file=filename,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            build_cpu_s=process_time()-cpu,build_wall_s=perf_counter()-start,
            direct_cpu_s=info['build_cpu_s'],bytes_arrays=sum(v.nbytes for v in row.values()),bytes_file=path.stat().st_size)
        records[radius]=record;manifest['rows']=[records[r] for r in sorted(records)]
        if args.check_rows:
            query=lambda r,look,lo,hi,edges:row_query(row,r,x,t,powers,look,lo,hi,edges)
            manifest['row_checks'][str(radius)]=check_geometry(kernel,query,radius,root,beta=args.beta)
        atomic_json(manifest_path,manifest)
        print(json.dumps(dict(radius_completed=record,checks=manifest['row_checks'].get(str(radius)))),flush=True)
        del row
    if len(records)==len(radii):
        if manifest['status']!='complete':
            qleft=[];qright=[]
            for radius in radii:
                with np.load(args.output/records[float(radius)]['file'],allow_pickle=False) as data:
                    qleft.append(data['charge_left']);qright.append(data['charge_right'])
            path=args.output/'charges.npz'
            with path.with_suffix('.tmp').open('wb') as stream:
                np.savez_compressed(stream,charge_left=np.array(qleft),charge_right=np.array(qright))
            os.replace(path.with_suffix('.tmp'),path)
            manifest['charge_file']=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            manifest['status']='complete';manifest['rows']=[records[float(r)] for r in radii]
            atomic_json(manifest_path,manifest)
        else:
            path=args.output/manifest['charge_file']['file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['charge_file']['sha256']:
                raise ValueError('completed charge cache checksum mismatch')
        if args.validate_midpoints is not None:
            midpoints=args.validate_midpoints or np.sqrt(radii[:-1]*radii[1:]).tolist()
            store=TrackFirstAtlasStore(args.output)
            checks=[]
            for radius in midpoints:
                found=check_geometry(kernel,store.query,float(radius),root,beta=args.beta)
                checks.extend(found);print(json.dumps(dict(held_out_rho_m=radius,checks=found)),flush=True)
            atomic_json(args.output/'midpoint-checks.json',dict(checks=checks,time_rows_loaded=store.time_row_loads,
                max_resident_rows=store.max_resident_rows,resident_bytes=store.resident_bytes))
    print(json.dumps(dict(status=manifest['status'],finished_radii=len(records),total_radii=len(radii))),flush=True)

if __name__=='__main__':main()
