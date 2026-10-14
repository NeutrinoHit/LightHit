"""Multiple scattering (order >=2) of a shower on folded node bands.

The axial shower basis (:mod:`lighthit._shower_basis`) is compiled only in the
node bands of a :class:`lighthit.band_fold.BandFold`, exactly as the per-band
route compiles a band (stencils on the shower axis). Each node source is
scaled by its detector band weight and applied with its slice of the folded
kernels; the other bands enter through the fold. (A layout shared by all bands
through ``geometry_reference``/centroid stencils was tried first: on a
hadronic shower, 0.74 m wide, it gave 5-19% >=2 bin L1 against a finer basis,
against 0.4-1.3% for the axis stencil.) A source collapsed
to a few points predicts the zero-frequency charge of every OM; the exact
basis is evaluated only where the threshold decision needs it.
"""
from dataclasses import replace

import numpy as np

__all__ = ["node_sources", "collapse", "apply_folded"]


def _band_field(elements, wavelength_nm, phase_index):
    """Signed photon field of one band (threshold and yield of that band)."""
    mask = elements.beta * phase_index > 1
    if not np.any(mask):
        return None
    subset = elements if np.all(mask) else elements.subset(mask)
    first = subset.field(0, phase_index=phase_index)
    second = subset.field(2, phase_index=phase_index)
    scale0 = 1.0 / wavelength_nm ** 2
    scale2 = scale0 / phase_index ** 2
    return replace(first, photons=np.ascontiguousarray(
        scale0 * first.photons - scale2 * second.photons))


def node_sources(kernel, elements, basis, fold, omega, frame, geometry=None):
    """Axial-slice sources of the node bands, scaled by ``fold.scale``.

    Each band is compiled as in the per-band route (its own threshold, cone,
    yield and thin-slice centroids; stencils on the axis of ``frame``), so the
    bands' point sets can differ slightly. A node band in which no element
    emits gets zero channels on another band's layout. ``geometry`` is unused
    (kept for call compatibility).
    """
    from .._shower_basis import compile_axial_slices
    wavelength = np.asarray(kernel.wavelength.wavelength_nm, float)
    phase = fold.phase_index
    compiled = {}
    for band in fold.bands:
        field = _band_field(elements, float(wavelength[band]), float(phase[band]))
        if field is None:
            continue
        compiled[band] = compile_axial_slices(
            field, int(kernel.config.source_degree), omega,
            azimuthal_degree=int(kernel.config.azimuthal_degree), cell_m=basis.cell_m,
            axial_order=basis.axial_order, transverse=basis.transverse,
            stencil_m=basis.stencil_m, stencil_threshold_m=basis.stencil_threshold_m,
            time_step_ns=basis.time_step_ns, time_deposit=basis.time_deposit, frame=frame)
    if not compiled:
        raise ValueError("no element emits in any node band")
    template = next(iter(compiled.values()))
    sources = []
    for band in fold.bands:
        item = compiled.get(band)
        if item is None:
            item = replace(template, channels=np.zeros_like(template.channels))
        sources.append(replace(item, channels=item.channels * fold.scale[band]))
    return sources


def collapse(sources, cell_m):
    """Zero-frequency sources reduced to one point per ``cell_m`` cube (prediction).

    Each source's basis points are grouped in cubes of the shower frame; a
    cube keeps the sum of its channels at the centroid of its |l=0| weights.
    (Contiguous axial chunks were tried first; on a hadronic shower with wide
    transverse excursions they mispredicted single OMs by up to a factor 24.)
    """
    reduced = []
    for item in sources:
        weight = np.abs(item.channels[:, 0, 0])
        coordinates = np.column_stack((item.z_m, item.transverse_m))
        keys = np.floor(coordinates / float(cell_m)).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        chunks = [np.flatnonzero(inverse == group) for group in range(int(inverse.max()) + 1)]
        z, transverse = [], []
        for chunk in chunks:
            total = weight[chunk].sum()
            share = (weight[chunk] / total if total > 0
                     else np.full(len(chunk), 1.0 / len(chunk)))
            z.append(np.sum(share * item.z_m[chunk]))
            transverse.append(share @ item.transverse_m[chunk])
        reduced.append(replace(item, z_m=np.asarray(z), transverse_m=np.asarray(transverse),
                               channels=np.ascontiguousarray(np.asarray(
                                   [item.channels[chunk, :, :1].sum(axis=0) for chunk in chunks]))))
    return reduced


def apply_folded(folded, sources, positions, looks, alpha, area, window):
    """Folded order >=2 spectrum (frequency, OM) of node-band sources.

    ``sources`` are already scaled by their band weight; the detector scale
    of every virtual band is the OM effective area.
    """
    from .._shower_basis import apply_multi_spectral
    if not len(positions):
        return np.zeros((len(folded["omega"]), 0), complex)
    out_scale = np.repeat(np.asarray(area, float)[None, :], len(sources), axis=0)
    points = sources[0].points_m()
    if all(np.array_equal(item.points_m(), points) for item in sources[1:]):
        return apply_multi_spectral(folded, sources, positions, looks, alpha, out_scale,
                                    radial_window_m=window)
    total = 0
    for j, item in enumerate(sources):
        band = dict(folded, coefficients=folded["coefficients"][j:j + 1],
                    absorption=folded["absorption"][j:j + 1], speed=folded["speed"][j:j + 1])
        total = total + apply_multi_spectral(band, [item], positions, looks, alpha,
                                             out_scale[j:j + 1], radial_window_m=window)
    return total


def delay_groups(elements, *, prompt_delay_ns, cell_ns, window_end_ns, group_index_max):
    """Split steps by emission delay behind the light front of the source.

    The delay of a step is its start time minus ``t0 + |x - x0| / c`` for the
    earliest step (x0, t0). Prompt steps (delay <= ``prompt_delay_ns``) keep
    the source's own clock. Later steps (neutron captures, decays) are grouped
    in ``cell_ns`` cells, each with its own clock shift (the smallest delay in
    the cell), unless no light of the step can arrive before the end of the
    readout window: arrival minus the OM origin is at least
    ``delay - (n_g,max - 1) |x - x0| / c``, so such steps (``beyond``) only add
    to the all-time charge. Returns ``(prompt_mask, [(mask, shift_ns)], beyond_mask)``.
    """
    c = 0.299792458
    first = int(np.argmin(elements.start_ns))
    distance = np.linalg.norm(elements.start_m - elements.start_m[first], axis=1)
    delay = elements.start_ns - elements.start_ns[first] - distance / c
    prompt = delay <= prompt_delay_ns
    beyond = ~prompt & (delay - (group_index_max - 1.0) * distance / c > window_end_ns)
    within = ~prompt & ~beyond
    groups = []
    if np.any(within):
        cells = np.floor(delay / cell_ns).astype(np.int64)
        for cell in np.unique(cells[within]):
            mask = within & (cells == cell)
            groups.append((mask, float(delay[mask].min())))
    return prompt, groups, beyond
