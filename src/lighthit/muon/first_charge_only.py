"""Direct stationary q1 for explicit source segments, without time histograms.

Uses the same physical HG/acceptance and spectral cone as segment prompt
transport. Source subdivision is geometric; emission clocks and output time
windows have no role in this all-time charge. No source grouping, detector
selection, atlas interpolation or new transport approximation is introduced.
"""
from dataclasses import replace
from time import perf_counter,process_time
import numpy as np


def first_scattering_charge_only(kernel,elements,*,config=None,gauss_order=0,panel_width=1.0):
    """Return (charge_pe, metadata) for every receiver in the supplied kernel.

    The caller owns the wavelength quadrature: use primary_spectral_quadrature
    for the corrected composite optical rule. Real beta and its wavelength
    thresholds are retained. Batched receivers avoid repeated setup per OM.
    Numerical longitudinal/path quadrature must still be checked for convergence.
    Positive gauss_order explicitly replaces the stationary path interpolation
    by Gauss integration in asinh((S-r)/sigma), with panel_width and an optical
    attenuation-width guard. The default retains the original path grid.
    """
    from .. import prompt
    from ..sources import SpectralLightElements
    if not isinstance(elements,SpectralLightElements) or elements.cone_model!='spectral':
        raise ValueError('requires SpectralLightElements with spectral cones')
    settings=config or prompt.PromptConfig(element_method='segment',screen=False)
    if (isinstance(gauss_order,bool) or not isinstance(gauss_order,(int,np.integer)) or
            gauss_order<0 or gauss_order==1 or not np.isfinite(panel_width) or panel_width<=0):
        raise ValueError('invalid stationary Gauss order/panel width')
    if settings.element_method!='segment' or settings.screen:
        raise ValueError('stationary exact-segment route requires segment and screen=False')
    tables=prompt._spectral_tables(kernel)
    alpha=np.ascontiguousarray(kernel.acceptance()['alpha'],float)
    coef,ipow,jpow,degree=prompt.acceptance_monomials(alpha)
    detector=kernel.detector
    positions=np.ascontiguousarray(detector.positions_m,float)
    looks=np.ascontiguousarray(-detector.orientations,float)
    areas=np.asarray(detector.effective_area_m2,float)
    work=np.arange(len(positions),dtype=np.int64)
    # Placeholder edges satisfy the shared API; compute_bins=False bypasses
    # their use in stationary grid selection and histogram deposition.
    edges=np.array([0.,1.]);origins=np.zeros(len(positions))
    charge=np.zeros(len(positions));diagnostics=[]
    wall,cpu=perf_counter(),process_time()
    for lam,phase in enumerate(tables['phase_index']):
        if tables['mus'][lam]==0 or (tables['s0w'][lam]==0 and tables['s2w'][lam]==0):continue
        active=elements.beta*phase>1.
        if not np.any(active):continue
        source=elements if np.all(active) else elements.subset(active)
        source=replace(source,reference_phase_index=float(phase),cone_model='frozen')
        segments=prompt._element_segments(source)
        single={key:value[lam:lam+1] for key,value in tables.items()}
        values,bins,meta=prompt._order1('segment',kernel,source,source,segments,
            positions,looks,areas,origins,work,single,coef,ipow,jpow,degree,
            float(kernel.medium.g),edges,settings,np.zeros(len(positions)),compute_bins=False,
            stationary_gauss_order=gauss_order,stationary_panel_width=panel_width)
        if meta.get('order1_segments_too_close',0):
            raise ValueError('segment enters unsupported point-detector near field')
        if bins.shape!=(len(positions),0):
            raise RuntimeError('charge-only kernel unexpectedly allocated time bins')
        charge+=values;diagnostics.append(meta)
    return charge,dict(cpu_seconds=process_time()-cpu,wall_seconds=perf_counter()-wall,
        elements=len(elements),receivers=len(positions),wavelength_nodes=len(tables['wavelength_nm']),
        computed_order=1,computed_bins=False,source_grouping=False,
        stationary_gauss_order=int(gauss_order),stationary_panel_width=float(panel_width),
        emission_clock_used=False,wavelengths=diagnostics)
