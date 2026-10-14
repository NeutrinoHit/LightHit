"""Explicit corrected primary-muon 0+1 component, independent of RTE caches.

Order zero uses original steps and continuous root-split spectral integration.
Order one uses a connected polyline and known-optical-knot composite quadrature.
Secondaries and order >=2 are not represented or silently assigned zero here.
"""
from copy import copy
from dataclasses import dataclass
from time import perf_counter
import numpy as np

from ..ballistic_interval import ballistic_interval
from ..prompt import PromptConfig
from ..muon.primary_polyline import fit_primary_polyline, primary_spectral_quadrature
from ..muon.muon_integral import order1_integrals


@dataclass
class PrimaryPromptResult:
    charge_orders_pe: np.ndarray
    bins_orders_pe: np.ndarray
    time_origin_ns: np.ndarray
    relative_time_edges_ns: np.ndarray
    metadata: dict


def transport_primary_prompt(kernel, source, *, primary_uid=0, time_origin_ns=None,
                             fit_config=None, spectral_order=2, ballistic_config=None,
                             prompt_config=None):
    """Return only primary orders (0,1), on the caller's detectors/time window.

    Composite spectral order two is a measured fast setting, not an automatic
    accuracy guarantee. Repeat with order four/eight for an optical convergence
    control. No error tolerance on geometry is advertised as a detector bound.
    """
    from .._shower_basis import earliest_element_arrival
    from ..sources import SpectralLightElements
    began = perf_counter()
    if not isinstance(source, SpectralLightElements) or source.cone_model != 'spectral':
        raise ValueError('primary prompt requires spectral LightElements')
    mask = source.uid == primary_uid
    if not np.any(mask):
        raise ValueError('primary UID not found')
    primary = source if np.all(mask) else source.subset(mask)
    positions = np.ascontiguousarray(kernel.detector.positions_m)
    if time_origin_ns is None:
        origins = earliest_element_arrival(positions, primary.start_m, primary.direction,
            primary.length_m, primary.start_ns, float(kernel.medium.group_index.min()))
    else:
        origins = np.asarray(time_origin_ns, float)
    edges = kernel.config.relative_time_edges_ns
    zero = ballistic_interval(kernel, primary, config=ballistic_config,
                              time_origin_ns=origins, relative_edges_ns=edges)
    phase = kernel.medium.phase_index
    fit = fit_primary_polyline(primary, config=fit_config, phase_indices=phase)
    local = copy(kernel)
    local.wavelength = primary_spectral_quadrature(kernel.medium, kernel.detector, order=spectral_order,
        wavelength_range_nm=kernel.config.wavelength_range_nm)
    settings = prompt_config or PromptConfig(element_method='segment', screen=False)
    if settings.element_method != 'segment' or settings.screen:
        raise ValueError('primary component requires unscreened segment integration')
    one, bins_one, one_meta = order1_integrals(local, fit.elements, time_origin_ns=origins, config=settings)
    return PrimaryPromptResult(np.column_stack((zero.charge_pe, one)),
        np.stack((zero.bins_pe, bins_one), axis=-1), origins.copy(), np.asarray(edges).copy(),
        dict(computed_orders=[0,1], primary_uid=int(primary_uid), secondaries_computed=False,
             ge2_computed=False, ballistic=zero.metadata, polyline=fit.diagnostics,
             order1=one_meta, spectral_order_per_optical_interval=spectral_order,
             order1_wavelength_nodes=len(local.wavelength.wavelength_nm),
             elapsed_seconds=perf_counter()-began))
