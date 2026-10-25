"""Checkpointed wavelength/frequency preparation of finite receiver kernels.

Only table construction runs in workers. Joined tables retain the ordinary
directional layout used by prepared kernels and spectral folds at runtime.
An incomplete bank has no completion manifest and never replaces input data.
"""
from dataclasses import asdict
from pathlib import Path
import hashlib,json
import numpy as np
from .directional import DirectionalCache
from .cache import CacheGrid
from .receiver import SphereSurface
from .receiver_vertex import ReceiverVertexConfig
from .green import SolverSettings
from .medium import Medium


def _block(options):
    output=Path(options['output']);sidecar=output.with_suffix('.json')
    signature=options['signature']
    if output.exists() and sidecar.exists():
        if json.loads(sidecar.read_text()).get('signature')==signature:
            cache=DirectionalCache.load(output)
            if np.array_equal(cache.grid.omega_per_ns,options['omega']):return str(output)
    reuse=None
    if options.get('reuse'):
        record=options['reuse'];path=Path(record['file']);meta=path.with_suffix('.json')
        if path.is_file() and meta.is_file() and json.loads(meta.read_text()).get('signature')==record['signature']:
            reuse=DirectionalCache.load(path)
    cache=DirectionalCache.build(Medium(**options['medium']),SolverSettings(**options['settings']),
        CacheGrid(np.asarray(options['radii']),np.asarray(options['omega'])),options['acceptance_degree'],
        receiver_surface=SphereSurface(options['radius']),receiver_config=ReceiverVertexConfig(**options['vertex']),
        angular_backend='numba',radial_phase='flight',reuse_receiver_cache=reuse)
    if reuse is not None:cache.timings_s['reused_cache_sha256']=hashlib.sha256(Path(options['reuse']['file']).read_bytes()).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.tmp.npz');cache.save(temporary);temporary.replace(output)
    sidecar.write_text(json.dumps(dict(signature=signature))+'\n')
    return str(output)


def prepare_receiver_bank(media,settings,grid,surface,acceptance_degree,output,*,config=None,
                          jobs=1,frequency_block=8,progress=None,reuse_directory=None):
    """Build bands in bounded frequency blocks; safely resume matching blocks.

    Resume keys include coefficients, radius, all resolutions and the physical
    builder sources. The owner alone joins bands and publishes the manifest.
    """
    from concurrent.futures import ProcessPoolExecutor,as_completed
    from multiprocessing import get_context
    from . import receiver_adjoint,receiver_vertex,directional,receiver_first,_receiver_first_numba
    config=config or ReceiverVertexConfig();output=Path(output)
    if not isinstance(surface,SphereSurface) or surface.radius_m.ndim!=0:
        raise ValueError('one explicit sphere radius required')
    for name,value in [('jobs',jobs),('frequency_block',frequency_block)]:
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<1:
            raise ValueError(f'{name} must be a positive integer')
    media=tuple(media)
    if not media:raise ValueError('at least one wavelength required')
    if len({m.wavelength_nm for m in media})!=len(media):raise ValueError('bank wavelengths must be distinct')
    reusable=None
    if reuse_directory is not None:
        reuse_directory=Path(reuse_directory)
        reusable=json.loads((reuse_directory/'preparation.json').read_text())
        if (reusable.get('schema')!='lighthit/receiver-bank-preparation/1'
                or not 0<=reusable['acceptance_degree']<acceptance_degree):
            raise ValueError('reuse requires a recorded lower receiver degree')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'manifest.json').exists():raise FileExistsError('completed receiver bank already exists')
    digest=hashlib.sha256(Path(__file__).read_bytes())
    for m in (receiver_adjoint,receiver_vertex,directional,receiver_first,_receiver_first_numba):
        digest.update(Path(m.__file__).read_bytes())
    builder=digest.hexdigest();blocks=[]
    for lo in range(0,len(grid.omega_per_ns),frequency_block):
        for band,medium in enumerate(media):
            options=dict(medium=asdict(medium),settings=asdict(settings),radii=grid.radii_m.tolist(),
                omega=grid.omega_per_ns[lo:lo+frequency_block].tolist(),acceptance_degree=int(acceptance_degree),
                radius=float(surface.radius_m),vertex=asdict(config),builder_sha256=builder)
            signature=hashlib.sha256(json.dumps(options,sort_keys=True).encode()).hexdigest()
            if reusable is not None:
                old=dict(options,acceptance_degree=reusable['acceptance_degree'],builder_sha256=reusable['builder_sha256'])
                options['reuse']=dict(signature=hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest(),
                    file=str(reuse_directory/'blocks'/f'band-{band:03d}'/f'omega-{lo:04d}.npz'))
            options.update(signature=signature,output=str(output/'blocks'/f'band-{band:03d}'/f'omega-{lo:04d}.npz'))
            blocks.append(options)
    complete=0
    def finished():
        nonlocal complete
        complete+=1
        if progress is not None:progress(complete,len(blocks))
    if jobs==1:
        for block in blocks:_block(block);finished()
    else:
        with ProcessPoolExecutor(max_workers=jobs,mp_context=get_context('spawn')) as pool:
            futures=[pool.submit(_block,block) for block in blocks]
            for f in as_completed(futures):f.result();finished()
    entries=[]
    for band,medium in enumerate(media):
        parts=[DirectionalCache.load(block['output']) for block in blocks
               if block['medium']==asdict(medium)]
        first=parts[0]
        joined=DirectionalCache(grid,medium,settings,first.table,
            np.concatenate([p.coefficients for p in parts],axis=0),
            timings_s=dict(blocks=len(parts),builder_sha256=builder),angular_backend='numba',
            radial_phase='flight',receiver_contract=first.receiver_contract)
        path=output/f'directional-band-{band:03d}.npz';temporary=path.with_suffix('.tmp.npz')
        joined.save(temporary);temporary.replace(path)
        entries.append(dict(wavelength_nm=medium.wavelength_nm,file=path.name,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    metadata=dict(schema='lighthit/first-entry-receiver-bank/1',bands=entries,builder_sha256=builder,
        receiver_contract=first.receiver_contract,settings=asdict(settings),convergence_verified=False)
    (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata
