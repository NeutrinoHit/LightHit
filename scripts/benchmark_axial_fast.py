"""Compare exact-representation accelerators on LOCAL G4 data; no publishing.

Run from LightHit after applying the additive patch. Existing transport,
source contract, CIC lattice and retained degrees are preserved. The old
apply is compared only on control OMs, not repeated for hours on the array.
"""
from pathlib import Path
import sys, argparse, json, time, platform, resource
import numpy as np

# Same source-tree convention as the repository's scripts.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from lighthit import synthetic_medium, SolverSettings
from lighthit.cache import ResponseCache, CacheGrid
from lighthit.experimental.g4_source import load_event, SourceContract
from lighthit.experimental.axial_source import AxialSource, AxisFrame, axial_response
from lighthit.experimental.event_moments import KernelChannels
from lighthit.experimental.axial_fast import (PreparedAxialKernel,
    compile_axial_source_fast, rigidly_moved_source)
from lighthit.readout import inverse_bins


def array_positions():
    phi=np.arange(7)*2*np.pi/7
    strings=np.zeros((8,3));strings[1:,:2]=60*np.column_stack((np.cos(phi),np.sin(phi)))
    heights=(np.arange(36)-17.5)*15
    centers=np.array([[0.,0.,0.],[110.,0.,0.]])
    vertical=np.column_stack((np.zeros_like(heights),np.zeros_like(heights),heights))
    return (centers[:,None,None,:]+strings[None,:,None,:]+vertical[None,None,:,:]).reshape(-1,3)


def rss_mib():
    raw=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw/(1024 if sys.platform.startswith('linux') else 2**20)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--event',type=int,default=5)
    p.add_argument('--output',default='.build/axial-fast')
    p.add_argument('--cache',default=None)
    p.add_argument('--compiler',choices=['original','cell-gemm'],default='cell-gemm')
    p.add_argument('--compare-compilers',action='store_true',help='Full reference source, expensive and higher RAM')
    p.add_argument('--controls',type=int,default=6)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--frequencies',type=int,default=41);p.add_argument('--omega-max',type=float,default=.5)
    p.add_argument('--degree',type=int,default=32);p.add_argument('--m',type=int,default=4)
    p.add_argument('--cell',type=float,default=.12);p.add_argument('--receiver-block',type=int,default=8)
    a=p.parse_args()
    import numba
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        from contextlib import nullcontext
        threadpool_limits = lambda **kwargs: nullcontext()
    numba.set_num_threads(a.threads)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    report={'event':a.event,'settings':vars(a),'python':sys.version,'platform':platform.platform(),
            'numpy':np.__version__,'numba':numba.__version__,
            'scope':'unchanged source and transport representations; isotropic point receiver; monochromatic transport'}
    with threadpool_limits(limits=1):
        start=time.perf_counter();raw=load_event(a.input,a.event,SourceContract())
        elements=raw.moved(translation=np.array([20.,15.,-20.]))
        report['read_s']=time.perf_counter()-start; report['source_contract']=elements.summary()
        receivers=array_positions();frame=AxisFrame.of(elements)
        distance=np.linalg.norm(receivers-elements.centroid_m,axis=1)
        idx=np.argsort(distance)[np.linspace(0,len(receivers)-1,a.controls).astype(int)]
        omega=np.linspace(0,a.omega_max,a.frequencies);medium=synthetic_medium()
        cache_path=Path(a.cache) if a.cache else out/'transport.npz'
        start=time.perf_counter()
        if cache_path.exists():
            cache=ResponseCache.load(cache_path)
            if not np.array_equal(cache.grid.omega_per_ns,omega) or cache.degree<a.degree:
                raise ValueError('Cached frequencies/degrees do not match the requested benchmark')
            if cache.medium!=medium:raise ValueError('Cached medium differs')
            report['cache_loaded']=True
        else:
            reach=1.5*elements.extent_m
            cache=ResponseCache.build(medium,SolverSettings(24,a.degree,8.,.04,10),
                CacheGrid.geometric(max(3.,.90*(distance.min()-reach)),1.1*(distance.max()+reach),160,omega))
            cache.save(cache_path);report['cache_loaded']=False
        report['cache_load_or_build_s']=time.perf_counter()-start
        factory=AxialSource.of if a.compiler=='original' else compile_axial_source_fast
        start=time.perf_counter()
        source=factory(elements,a.degree,omega,azimuthal_degree=a.m,cell_m=a.cell,
                       element_order=2,frame=frame)
        report['source_compile_s']=time.perf_counter()-start
        report['source_shape']=list(source.channels.shape);report['source_bytes']=source.channels.nbytes
        if a.compare_compilers and a.compiler=='cell-gemm':
            start=time.perf_counter()
            old=AxialSource.of(elements,a.degree,omega,azimuthal_degree=a.m,cell_m=a.cell,
                               element_order=2,frame=frame)
            report['original_source_compile_s']=time.perf_counter()-start
            if not np.array_equal(source.points_m(),old.points_m()):raise AssertionError('Grid changed')
            scale=np.max(np.abs(old.channels));error=0.0
            for b in range(0,len(source.z_m),32):
                error=max(error,float(np.max(np.abs(source.channels[b:b+32]-old.channels[b:b+32]))/scale))
            report['source_max_error_over_max']=error
            if error>1e-10:raise AssertionError('Compiler comparison failed')
            del old
        start=time.perf_counter();prepared=PreparedAxialKernel.from_cache(cache,degree=a.degree)
        report['prepare_spline_s']=time.perf_counter()-start
        start=time.perf_counter()
        fast=prepared.apply(source,receivers,source_omega_per_ns=omega,receiver_block=a.receiver_block)
        report['first_apply_including_possible_JIT_s']=time.perf_counter()-start
        times=[]
        for _ in range(a.repeat):
            start=time.perf_counter()
            fast=prepared.apply(source,receivers,source_omega_per_ns=omega,receiver_block=a.receiver_block)
            times.append(time.perf_counter()-start)
        report['warm_apply_both_orders_s']=times
        # Same stored source, same radial interpolant, no new physical reference implied.
        start=time.perf_counter()
        original=np.stack([axial_response(KernelChannels.of(cache,o,a.degree),source,receivers[idx])
                           for o in range(2)],axis=-1)
        report['original_apply_controls_both_orders_s']=time.perf_counter()-start
        scale=np.maximum(np.max(np.abs(original),axis=(0,2)),1e-300)
        error=float(np.max(np.abs(fast[:,idx]-original)/scale[None,:,None]))
        report['max_error_controls_over_receiver_peak']=error
        if error>1e-9:raise AssertionError('Fast apply comparison failed')
        # Exact relabeling of time: t_relative = t_absolute - origin_d.
        origin=source.reference_ns+distance/medium.speed_m_per_ns
        relative=fast*np.exp(-1j*omega[:,None]*origin[None,:])[:,:,None]
        edges=np.arange(-50.,451.,10.)
        bins=np.stack([inverse_bins(omega,relative[:,:,o],edges) for o in range(2)],axis=-1)
        # Both sides agree algebraically regardless of the discretization, but
        # the physical time profile still needs the existing convergence/MC controls.
        report['time_bin_note']='negative phase used; scattered components only, no clipping; not a new physical validation'
        np.savez_compressed(out/'scattered.npz',omega=omega,spectrum=fast,
            receivers=receivers,time_origin_ns=origin,relative_edges_ns=edges,bins=bins)
        # Do NOT recompile a rigidly moved realization. Rigidly move its already
        # deposited lattice with all three basis axes instead.
        moved=rigidly_moved_source(source,translation=np.array([1.,-.5,.2]),delay_ns=5.)
        start=time.perf_counter()
        prepared.apply(moved,receivers,source_omega_per_ns=omega,receiver_block=a.receiver_block)
        report['new_pose_apply_s']=time.perf_counter()-start
        report['pose_reuses_source_array']=bool(moved.channels is source.channels)
        report['peak_process_MiB']=rss_mib()
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='source_contract'},indent=2))

if __name__=='__main__':main()
