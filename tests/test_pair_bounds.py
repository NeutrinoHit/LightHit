"""Numerical-cache omission bounds, independent of physical RTE accuracy."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("numba")
from lighthit._shower_basis import (  # noqa: E402
    _radial_norm_bound, _source_norm_bound, apply_multi, apply_multi_spectral)
from lighthit.experimental.axial_source import (  # noqa: E402
    AxisFrame, AxialSource, channel_index)


def _case(seed=71, points=1, bands=2, degree=4, max_m=3, detector_degree=2):
    rng = np.random.default_rng(seed)
    omega = np.array([0., .17, .53])
    log_r = np.log([3., 8., 30.])
    pairs = [(ell, m) for ell in range(detector_degree + 1)
             for m in range(ell + 1)]
    shape = (bands, 2, degree + 1, len(pairs), 4, len(omega))
    coefficients = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    # Signed cubic terms cancel at each interval midpoint. A bound based
    # only on endpoint values or on signed coefficient sums is unsafe here.
    mid = np.diff(log_r)[None, :, None, None, None] / 2
    coefficients[..., 3, :] = -(
        (coefficients[..., 0, :] * mid + coefficients[..., 1, :]) * mid
        + coefficients[..., 2, :]) * mid
    coefficients[..., 3, :] += 0.05 * (1 + 1j)
    bundle = dict(
        degree=degree, acceptance_degree=detector_degree,
        pair_lambda=np.array([p[0] for p in pairs], np.int64),
        pair_mu=np.array([p[1] for p in pairs], np.int64), omega=omega,
        log_r=log_r, coefficients=coefficients.astype(np.complex64),
        absorption=np.linspace(.015, .04, bands),
        speed=np.linspace(.21, .23, bands), phase="flight")
    frame = AxisFrame(np.zeros(3), np.array([0., 0., 1.]),
                      np.array([1., 0., 0.]), np.array([0., 1., 0.]))
    kept, channel_degree = channel_index(degree, max_m)
    sources = []
    for band in range(bands):
        channels = (rng.normal(size=(points, len(kept), len(omega)))
                    + 1j * rng.normal(size=(points, len(kept), len(omega))))
        sources.append(AxialSource(
            frame, np.linspace(-.4, .4, points) if points > 1 else np.zeros(1),
            np.zeros((points, 2)), channels, kept, channel_degree, degree,
            max_m, 2.25 * band, {}))
    alpha = np.array([1.3, -.7, .8])[:detector_degree + 1]
    return rng, bundle, sources, alpha


def _band(bundle, band):
    return dict(bundle, coefficients=bundle["coefficients"][band:band + 1],
                absorption=bundle["absorption"][band:band + 1],
                speed=bundle["speed"][band:band + 1])


def _pair_bounds(bundle, sources, receivers, alpha, scale):
    """Return the implemented triangle bound by (band, point, receiver)."""
    radial = _radial_norm_bound(bundle, bundle["coefficients"], alpha)
    bounds = []
    for band, source in enumerate(sources):
        r = np.linalg.norm(source.points_m()[:, None, :] - receivers[None], axis=2)
        interval = np.clip(np.searchsorted(bundle["log_r"], np.log(r), side="right") - 1,
                           0, len(bundle["log_r"]) - 2)
        bounds.append(_source_norm_bound(source)[:, None] * radial[band, interval]
                      * np.abs(scale[band])[None, :]
                      * np.exp(-bundle["absorption"][band] * r) / (4 * np.pi * r * r))
    return np.array(bounds)


@pytest.mark.parametrize('max_m', [0, 3])
@pytest.mark.parametrize('second', [False, True])
def test_common_flight_factor_matches_independent_spectral_contraction(max_m, second):
    rng, bundle, sources, alpha = _case(points=5, max_m=max_m)
    source0, source2 = sources
    # Both physical fields have the same original emission-clock reference.
    source2 = replace(source2, reference_ns=source0.reference_ns)
    receivers = np.array([[10., 3., -2.], [-7., 12., 6.]])
    looks = rng.normal(size=receivers.shape)
    looks /= np.linalg.norm(looks, axis=1)[:, None]
    scale = np.array([[.2, -.7], [1.1, .8]])
    s0 = np.array([.7, -1.3]); s2 = np.array([.5, -.4]) if second else np.zeros(2)
    combined = [replace(source0, channels=s0[i]*source0.channels-s2[i]*source2.channels)
                for i in range(2)]
    expected = apply_multi_spectral(bundle, combined, receivers, looks, alpha, scale,
                                    channel_dtype=np.complex128)
    actual = apply_multi(bundle, source0, source2, receivers, looks, alpha, s0, s2, scale)
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13*abs(expected).max())


def test_prepared_radial_envelope_reuse_respects_band_frequency_and_acceptance():
    _, bundle, _, alpha = _case()
    coefficients = bundle["coefficients"]
    bundle["_norm_cache"] = {}
    # Caller-owned writable arrays must not reuse a stale numerical bound.
    first = _radial_norm_bound(bundle, coefficients, alpha)
    coefficients *= 2
    np.testing.assert_allclose(_radial_norm_bound(bundle, coefficients, alpha), 2*first)
    assert not bundle["_norm_cache"]
    coefficients.setflags(write=False)
    first = _radial_norm_bound(bundle, coefficients, alpha)
    assert _radial_norm_bound(bundle, coefficients, alpha) is first
    for view, acceptance in [(coefficients[:1], alpha), (coefficients[..., :1], alpha),
                              (coefficients, 2*alpha)]:
        got = _radial_norm_bound(bundle, view, acceptance)
        uncached = {k: v for k, v in bundle.items() if k != "_norm_cache"}
        np.testing.assert_array_equal(got, _radial_norm_bound(uncached, view, acceptance))
    assert len(bundle["_norm_cache"]) == 4


@pytest.mark.parametrize("max_m", [0, 3])
def test_each_signed_complex_pair_is_below_its_envelope(max_m):
    rng, bundle, sources, alpha = _case(max_m=max_m)
    directions = rng.normal(size=(16, 3))
    directions[:3] = [[0., 0., 1.], [0., 0., -1.], [1e-8, 0., 1.]]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    # Exercise both spline intervals, including points near their boundaries.
    radius = np.geomspace(3.000001, 29.99999, len(directions))
    receivers = radius[:, None] * directions
    looks = rng.normal(size=receivers.shape)
    looks /= np.linalg.norm(looks, axis=1)[:, None]
    scale = rng.uniform(.3, 2., size=(len(sources), len(receivers)))
    scale[0, ::2] *= -1
    bounds = _pair_bounds(bundle, sources, receivers, alpha, scale)
    for band, source in enumerate(sources):
        response = apply_multi(_band(bundle, band), source, source, receivers, looks,
                               alpha, np.ones(1), np.zeros(1), scale[band:band + 1])
        assert np.any(np.abs(response) > 1e-8)
        assert np.all(np.max(np.abs(response), axis=0) <= bounds[band, 0])
        # The detector's nonzero signed-m terms contribute materially; this
        # would otherwise merely be an isotropic normalization check.
        isotropic = apply_multi(_band(bundle, band), source, source, receivers, looks,
                                alpha * [1., 0., 0.], np.ones(1), np.zeros(1),
                                scale[band:band + 1])
        assert np.max(np.abs(response - isotropic)) > 1e-6

    # The legacy two-field route subtracts independently weighted fields.
    # Its bound must take absolute weights before combining their norms.
    scale0 = np.array([.7, -1.2])
    scale2 = np.array([1.1, -.4])
    source0, source2 = sources
    first_bound = _pair_bounds(bundle, [source0, source0], receivers, alpha, scale)
    second_bound = _pair_bounds(bundle, [source2, source2], receivers, alpha, scale)
    combined_bound = (np.abs(scale0)[:, None, None] * first_bound
                      + np.abs(scale2)[:, None, None] * second_bound).sum(axis=0)[0]
    combined = apply_multi(bundle, source0, source2, receivers, looks, alpha,
                           scale0, scale2, scale)
    assert np.all(np.max(np.abs(combined), axis=0) <= combined_bound)


def test_isotropic_constant_pair_nearly_saturates_normalization_bound():
    _, bundle, sources, alpha = _case(bands=1, degree=0, max_m=0, detector_degree=0)
    bundle["coefficients"][:] = 0
    bundle["coefficients"][..., 3, :] = 2 - 3j
    source = replace(sources[0], channels=np.full((1, 1, 3), -3 + 4j))
    receivers = np.array([[5., 0., 0.], [17., 0., 0.]])
    scale = np.array([[.4, -1.7]])
    response = apply_multi(bundle, source, source, receivers,
                           np.tile([0., 0., 1.], (2, 1)), alpha,
                           np.ones(1), np.zeros(1), scale)
    bound = _pair_bounds(bundle, [source], receivers, alpha, scale)[0, 0]
    ratio = np.max(np.abs(response), axis=0) / bound
    assert np.all(ratio <= 1.)
    np.testing.assert_allclose(ratio, 1., rtol=2e-10, atol=0.)


def test_point_and_wavelength_budgets_cannot_be_spent_repeatedly():
    """Without either budget divisor the omitted charge exceeds epsilon."""
    _, bundle, sources, _ = _case(points=3, bands=2, degree=0, max_m=0,
                                  detector_degree=0)
    bundle["coefficients"][:] = 0
    bundle["coefficients"][..., 3, :] = 1
    bundle["absorption"][:] = 0
    channels = np.broadcast_to(np.array([1., .8, .4])[:, None, None], (3, 1, 3)).copy()
    sources = [replace(source, z_m=np.zeros(3), channels=channels.astype(complex),
                       reference_ns=0.) for source in sources]
    receivers = np.array([[5., 0., 0.]])
    looks = np.array([[0., 0., 1.]])
    alpha = np.array([np.sqrt(4 * np.pi)])
    scale = np.full((2, 1), 4 * np.pi * 25.)
    tolerance = 3.
    reference = apply_multi_spectral(bundle, sources, receivers, looks, alpha, scale,
                                     channel_dtype=np.complex128)
    fused = apply_multi_spectral(bundle, sources, receivers, looks, alpha, scale,
                                 channel_dtype=np.complex128,
                                 absolute_tolerance_pe=tolerance)
    independent = np.zeros_like(reference)
    for band, source in enumerate(sources):
        independent += apply_multi(_band(bundle, band), source, source,
                                   receivers, looks, alpha, np.ones(1), np.zeros(1),
                                   scale[band:band + 1], absolute_tolerance_pe=tolerance / 2)
    assert reference[0, 0].real == pytest.approx(4.4)
    # Only the .4 point in each band fits the per-point budget. Incorrectly
    # reusing the whole budget for each point or wavelength drops all 4.4 PE.
    assert fused[0, 0].real == pytest.approx(3.6)
    assert independent[0, 0].real == pytest.approx(3.6)
    assert np.max(np.abs(fused - reference)) <= tolerance
    assert np.max(np.abs(independent - reference)) <= tolerance


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_budget_split_across_bands_with_cancellation(dtype):
    rng, bundle, sources, alpha = _case(points=3)
    reference_ns = min(s.reference_ns for s in sources)
    aligned = []
    for source in sources:
        channels = source.channels.copy()
        channels[0] *= 1e-8
        channels[2] = -.85 * channels[1]
        phase = np.exp(1j * bundle["omega"] * (source.reference_ns - reference_ns))
        aligned.append(replace(source, channels=(channels * phase).astype(dtype),
                               reference_ns=reference_ns))
    receivers = np.array([[5., 2., 1.], [12., -3., 4.], [-18., 5., 2.]])
    looks = rng.normal(size=receivers.shape)
    looks /= np.linalg.norm(looks, axis=1)[:, None]
    scale = np.array([[1., -.6, .8], [-.9, 1.2, .5]])
    pair_bounds = _pair_bounds(bundle, aligned, receivers, alpha, scale)
    # Each independently budgeted band must skip the weak point, while
    # retaining both much larger, mutually cancelling source points.
    tolerance = 1.1 * len(aligned) * 3 * pair_bounds[:, 0].max()
    assert np.all(pair_bounds[:, 1:] > tolerance / 3)
    fused_ref = apply_multi_spectral(bundle, aligned, receivers, looks, alpha, scale,
                                     channel_dtype=dtype)
    fused = apply_multi_spectral(bundle, aligned, receivers, looks, alpha, scale,
                                 channel_dtype=dtype, absolute_tolerance_pe=tolerance)
    independent_ref = np.zeros_like(fused_ref)
    independent = np.zeros_like(fused_ref)
    for band, source in enumerate(aligned):
        arguments = (_band(bundle, band), source, source, receivers, looks, alpha,
                     np.ones(1), np.zeros(1), scale[band:band + 1])
        independent_ref += apply_multi(*arguments)
        independent += apply_multi(*arguments,
                                   absolute_tolerance_pe=tolerance / len(aligned))
    assert tolerance < .01 * np.max(np.abs(fused_ref))
    assert np.max(np.abs(fused - fused_ref)) > 0
    assert np.max(np.abs(independent - independent_ref)) > 0
    assert np.max(np.abs(fused - fused_ref)) <= tolerance
    assert np.max(np.abs(independent - independent_ref)) <= tolerance
    np.testing.assert_allclose(fused_ref, independent_ref, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(fused, independent, rtol=2e-12, atol=1e-15)
