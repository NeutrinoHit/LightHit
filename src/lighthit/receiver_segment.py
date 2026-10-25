"""Continuous segments and true spectral cones on first-entry geometry CDFs.

Used for primary-track table construction and independent segment controls.
All orientations at one position share geometry and wavelength folding.
The event's fast primary path continues to query its prepared track atlas.
"""
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import numpy as np
from .receiver import SphereSurface


@lru_cache(maxsize=16)
def _gauss(order):return np.polynomial.legendre.leggauss(order)


def geometry_for(kernel):
    from .receiver_geometry import ReceiverGeometry
    geometry=getattr(kernel,'_first_entry_geometry',None)
    if geometry is None:
        if kernel.config.first_entry_tables is None:raise FileNotFoundError('first_entry_tables geometry required')
        root=Path(kernel.config.first_entry_tables)
        directory=root/'geometry-primary' if (root/'geometry-primary/manifest.json').exists() else root/'geometry'
        geometry=ReceiverGeometry.load(directory)
        kernel._first_entry_geometry=geometry
    if not np.isclose(geometry.g,kernel.medium.g,rtol=0,atol=1e-12):
        raise ValueError('geometry CDF belongs to another phase function')
    return geometry


def segment_order1(kernel,segments,positions,looks,areas,origins,work,tables,polynomial,edges,config,*,compute_bins=True):
    """Integrate the supplied fixed cones; callers split the actual spectrum.

    Geometry interpolation is linear in all arrival moments and hence in
    the cubic detector acceptance. Path-CDF panel refinement, emission and
    azimuth quadratures have independent convergence controls.
    """
    from ._receiver_segment_numba import segment_pencils
    from ._receiver_geometry_lookup import sum_linear_pencils
    geometry=geometry_for(kernel);R=kernel._receiver_contract()['radius_m']
    edges=np.asarray(edges,float) if compute_bins else np.empty(0)
    n=len(positions);charge=np.zeros(n);bins=np.zeros((n,max(0,len(edges)-1)));nodes=0
    polynomial=np.pad(np.asarray(polynomial,float),(0,4-len(polynomial)))
    work=np.asarray(work,np.int64);began=perf_counter()
    ax,aw=_gauss(int(config.a_gauss));gx,gw=_gauss(int(config.phi_gauss))
    keys=np.column_stack((positions[work],origins[work]))
    unique,inverse=np.unique(keys,axis=0,return_inverse=True)
    for group,key in enumerate(unique):
        indices=work[inverse==group];receiver=np.ascontiguousarray(key[:3]);origin=float(key[3])
        local_looks=np.ascontiguousarray(looks[indices])
        for band in range(len(tables['mut'])):
            ms=float(tables['mus'][band]);mu=float(tables['mut'][band]);speed=float(tables['speed'][band])
            if ms==0:continue
            for i in range(len(segments.length)):
                density=segments.q0[i]*tables['s0w'][band]+segments.q2[i]*tables['s2w'][band]
                if density==0:continue
                ray=segment_pencils(receiver,np.ascontiguousarray(segments.start[i]),np.ascontiguousarray(segments.direction[i]),
                    float(segments.length[i]),float(density),float(segments.cone[i]),float(segments.t0[i]),float(segments.dtda[i]),
                    origin,R,speed,float(config.front_width_ns),float(config.grade_ratio),float(config.grade_min),
                    float(config.panel_max_relative),float(config.phi_w_panel),int(config.phi_uniform),int(config.max_a_breaks),ax,aw,gx,gw,compute_bins)
                q,b=sum_linear_pencils(*ray,local_looks,polynomial,np.asarray(edges,float),R,geometry.g,
                    np.log(geometry.radii_unit*R),geometry.path_u_max,geometry.max_excess_unit,
                    geometry.hit_cdf,geometry.hit_total,geometry.miss_cdf,geometry.miss_total,mu,ms,speed)
                charge[indices]+=areas[indices]*q;bins[indices]+=areas[indices,None]*b;nodes+=len(ray[0])
    return charge,bins,dict(timings=dict(order1_eval_seconds=perf_counter()-began),
        order1_level=np.where(np.isin(np.arange(n),work),2,0).astype(np.int8),
        order1_segments_too_close=0,order1_pencils_total=nodes,order1_batch_omission=np.zeros((n,2)),
        order1_receiver_reference='first-entry',order1_backward_profile='finite sphere geometry CDF',
        order1_quadrature='continuous segment/cone; glass-grazing splits; linear positive path-CDF moments')
