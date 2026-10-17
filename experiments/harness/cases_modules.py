"""Experimental modules without a script of their own, on small synthetic setups."""
from __future__ import annotations

import time

import numpy as np

from .cases import case
from .record import RESULTS


def _timed(function, repeat=3):
    best, value = np.inf, None
    for _ in range(repeat):
        began = time.perf_counter()
        value = function()
        best = min(best, time.perf_counter() - began)
    return value, best


@case("spline-fast-multipoles", "synthetic", kind="metrics",
      notes="PreparedMultipoles vs ResponseCache.moments_at on 20000 radii; cache of lighthit.synthetic_medium()")
def _(ctx):
    from lighthit import SolverSettings, synthetic_medium
    from lighthit.cache import CacheGrid, ResponseCache
    from lighthit.experimental.spline_fast import PreparedMultipoles
    cache = ctx.cached("cache", lambda: ResponseCache.build(
        synthetic_medium(), SolverSettings(24, 32, 6.0, 0.04, 10),
        CacheGrid.geometric(3.0, 150.0, 120, np.linspace(0.0, 0.5, 41))))
    prepared = ctx.cached("prepared", lambda: PreparedMultipoles.of(cache))
    radii = np.random.default_rng(7).uniform(5.0, 140.0, size=20000)
    reference, t_ref = _timed(lambda: cache.moments_at(radii))
    fast, t_fast = _timed(lambda: prepared.moments_at(radii))
    scale = float(np.abs(reference).max())
    return dict(headline=dict(max_abs_difference_over_max=float(np.abs(fast - reference).max() / scale),
                              speedup=t_ref / t_fast),
                reference_seconds=t_ref, prepared_seconds=t_fast, shape=list(reference.shape))


@case("axial-source-route", "blocked-apply", kind="metrics",
      notes="BlockedAxialKernel (numpy reordered apply, both orders) vs axial_response per order; "
            "synthetic needle of 3000 elements, 24 receivers, degree 32, M=4, 21 frequencies")
def _(ctx):
    from lighthit import Medium, SolverSettings
    from lighthit.cache import CacheGrid, ResponseCache
    from lighthit.experimental.axial_blocked import BlockedAxialKernel
    from lighthit.experimental.axial_source import VACUUM_M_PER_NS, AxialSource, axial_response
    from lighthit.experimental.event_moments import KernelChannels
    from lighthit.experimental.g4_source import LightElements, SourceContract
    medium = Medium(0.04, 0.05, 0.7, 1.35)
    omega = np.linspace(0.0, 0.4, 21)
    cache = ctx.cached("cache", lambda: ResponseCache.build(
        medium, SolverSettings(24, 32, 4.0, 0.05, 8), CacheGrid.geometric(8.0, 80.0, 120, omega)))
    rng = np.random.default_rng(5)
    count = 3000
    start = np.column_stack((rng.normal(scale=0.08, size=count), rng.normal(scale=0.08, size=count),
                             rng.normal(scale=1.2, size=count)))
    direction = rng.normal(scale=0.4, size=(count, 3))
    direction[:, 2] += 1.0
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    length = rng.uniform(0.002, 0.01, count)
    times = (start[:, 2] - start[:, 2].min()) / VACUUM_M_PER_NS + rng.uniform(0, 2.0, count)
    event = LightElements(start, direction, length, rng.uniform(1.0, 4.0, count),
                          np.full(count, 1 / 1.35), times, times + length / 0.22,
                          np.arange(count), np.zeros(count, np.int64), SourceContract(), {"synthetic": True})
    source = AxialSource.of(event, 32, omega, cell_m=0.12, azimuthal_degree=4)
    directions = rng.normal(size=(24, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    receivers = directions * rng.uniform(15.0, 70.0, size=(24, 1))
    blocked = BlockedAxialKernel.of(cache)
    both, t_blocked = _timed(lambda: blocked.apply(source, receivers), repeat=2)
    errors, t_plain = [], 0.0
    for order in (0, 1):
        kernel = KernelChannels.of(cache, order, 32, frequencies=omega)
        value, seconds = _timed(lambda: axial_response(kernel, source, receivers), repeat=2)
        t_plain += seconds
        mine = both[:, :, order]
        errors.append(float(np.abs(mine - value).max() / np.abs(value).max()))
    return dict(headline=dict(max_relative_difference=max(errors), blocked_over_plain_time=t_blocked / t_plain),
                blocked_seconds=t_blocked, plain_seconds_both_orders=t_plain, per_order_difference=errors,
                cells=int(len(source.points_m())))


@case("flash-references", "synthetic-flash", kind="metrics",
      notes="isotropic flash in lighthit.synthetic_medium(): TransportKernel flash path (charge per order, "
            "isotropic OMs, unit acceptance and efficiency) vs stationary P_N modes and the crossing MC; "
            "phase index set to the group index (Medium carries only the group index; a flash has no Cherenkov angle)")
def _(ctx):
    import lighthit as lh
    from lighthit.experimental.crossing_mc import crossing_estimate
    from lighthit.experimental.stationary_modes import StationaryModes, collision_components
    medium = lh.synthetic_medium()
    wl = np.array([400.0, 450.0, 500.0])
    spectral = lh.SpectralMedium(wl, np.full(3, medium.absorption_per_m), np.full(3, medium.scattering_per_m),
                                 np.full(3, medium.group_index), np.full(3, medium.group_index), g=medium.g)
    radii = np.array([20.0, 40.0, 80.0, 128.0])
    axes = np.eye(3)[[0, 1, 2, 0]] * np.array([1, 1, 1, -1])[:, None]
    detector = lh.DetectorArray(axes * radii[:, None], np.tile([0.0, 0.0, -1.0], (4, 1)), 0.05,
                                lambda c: np.ones_like(np.asarray(c, float)),
                                lambda w: np.ones_like(np.asarray(w, float)))
    cache_dir = RESULTS / "work" / "flash-references" / "cache"
    config = lh.KernelConfig(omega_per_ns=np.linspace(0.0, 0.3, 33), radial_range_m=(3.0, 150.0),
                             radial_nodes=120, cache_directory=str(cache_dir), cache_policy="build")
    kernel = ctx.cached("kernel", lambda: lh.TransportKernel(spectral, detector, config))
    photons = 1e10
    flash = lh.IsotropicFlash.monochromatic([0.0, 0.0, 0.0], photons, 450.0)
    ctx.cached("built", lambda: kernel.build(source=flash))
    response = kernel.transport(flash)
    charge = np.asarray(response.charge_components_pe) / (photons * 0.05)
    modes, _ = collision_components(radii, StationaryModes.build(medium, 256))
    sample = crossing_estimate(scattering_per_m=medium.scattering_per_m, g=medium.g,
                               absorptions=[medium.absorption_per_m], radii_m=radii,
                               photons_per_batch=20000, batches=8, max_path_m=2000.0, seed=20261015)
    mc = sample[:, 0, :, :, 0]
    mc_mean, mc_se = mc.mean(axis=0), mc.std(axis=0, ddof=1) / np.sqrt(mc.shape[0])
    mc3 = np.column_stack((mc_mean[:, 0], mc_mean[:, 1], mc_mean[:, 2:].sum(1)))
    se3 = np.column_stack((mc_se[:, 0], mc_se[:, 1], np.sqrt((mc_se[:, 2:] ** 2).sum(1))))
    rel_modes = (charge - modes) / modes
    return dict(headline=dict(max_rel_vs_modes_order0=float(np.abs(rel_modes[:, 0]).max()),
                              max_rel_vs_modes_order1=float(np.abs(rel_modes[:, 1]).max()),
                              max_rel_vs_modes_ge2=float(np.abs(rel_modes[:, 2]).max()),
                              max_pull_vs_mc=float(np.abs((charge - mc3) / np.where(se3 > 0, se3, np.inf)).max())),
                radii_m=radii.tolist(), kernel_fluence=charge.tolist(), modes_fluence=modes.tolist(),
                mc_fluence=mc3.tolist(), mc_standard_error=se3.tolist(),
                note="fluence per photon = charge / (photons x area); isotropic OM, acceptance 1, efficiency 1")
