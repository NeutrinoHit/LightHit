"""Method 7 on a narrow curved-track fixture, plus its exact ballistic splice.

The fixture deliberately has a nearly single-valued direction at each axial
position. Its convergence test must not be read as a claim that axial bins
converge for a broad electromagnetic shower.
"""
from dataclasses import replace

import numpy as np
import pytest

from lighthit import SolverSettings, synthetic_medium
from lighthit.cache import CacheGrid, ResponseCache
from lighthit.experimental.g4_source import LightElements, SourceContract
from lighthit.experimental.event_moments import KernelChannels, direct_response
from lighthit.experimental.effective_segments import (fit_effective_segments,
                                                       effective_segment_response,
                                                       full_response)


def _curved_shower(n=600, seed=20260920):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    axis = np.array([0.05, 0.0, 1.0])
    axis /= np.linalg.norm(axis)
    bend = np.array([0.6, 0.0, 0.0])
    centre_line = np.outer(t, axis) * 6.0 + np.outer(np.sin(t * 3.0), bend) * 0.3
    jitter = rng.normal(scale=0.01, size=(n, 3))
    start = centre_line + jitter
    tangent = np.gradient(centre_line, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    direction = tangent + rng.normal(scale=0.003, size=tangent.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    length = rng.uniform(1e-4, 0.01, size=n)
    photons = rng.uniform(0.2, 5.0, size=n) * (1.0 - 0.5 * t)
    cone_cosine = 0.76 + 0.01 * np.sin(t * 3.0) + rng.uniform(-0.001, 0.001, size=n)
    assert (cone_cosine > 1.0 / 1.35).all()
    start_ns = t * 20.0 + rng.uniform(0, 0.02, size=n)
    end_ns = start_ns + length / (0.99 * 0.2998)
    contract = SourceContract(phase_index=1.35)
    return LightElements(start, direction, length, photons, cone_cosine,
                         start_ns, end_ns, np.arange(n), np.arange(n), contract, {})


@pytest.fixture(scope="module")
def elements():
    return _curved_shower()


@pytest.fixture(scope="module")
def cache():
    medium = synthetic_medium()
    omega = np.linspace(0.0, 0.2, 5)
    settings = SolverSettings(12, 8, 6.0, 0.04, 10)
    grid = CacheGrid.geometric(5.0, 60.0, 24, omega)
    return ResponseCache.build(medium, settings, grid)


RECEIVERS = np.array([[18.0, 6.0, 3.0], [28.0, -9.0, 8.0], [42.0, 4.0, 5.0]])


def test_fit_reports_no_empty_bins_at_moderate_segment_counts(elements):
    for k in (1, 4, 16):
        effective = fit_effective_segments(elements, k)
        assert len(effective) == k
        assert effective.empty_bins == 0
        assert np.all(np.array([s.length_m for s in effective.segments]) > 0)
        beta = np.array([s.beta for s in effective.segments])
        assert np.all((beta > 0) & (beta <= 1.0))


def test_narrow_track_error_shrinks_as_segments_grow(elements, cache):
    k1 = KernelChannels.of(cache, 0, 8)
    k2 = KernelChannels.of(cache, 1, 8)
    ref1 = direct_response(k1, elements, RECEIVERS)
    ref2 = direct_response(k2, elements, RECEIVERS)
    scale1, scale2 = np.abs(ref1).max(), np.abs(ref2).max()

    errors = []
    for k in (1, 4, 16):
        effective = fit_effective_segments(elements, k)
        approx = effective_segment_response(cache, effective, RECEIVERS, longitudinal_order=16)
        e1 = np.max(np.abs(approx[:, :, 1] - ref1)) / scale1
        e2 = np.max(np.abs(approx[:, :, 2] - ref2)) / scale2
        errors.append(max(e1, e2))

    # Each fourfold increase in segments must clearly shrink the error --
    # not just fail to grow. A regression in the per-segment fit (wrong
    # direction, wrong beta, wrong start time) shows up here as a plateau
    # or a reversal, not as an outright crash. Measured on this fixture the
    # drops are roughly 32x (K=1->4) and 7.5x (K=4->16); the bounds below
    # are deliberately looser than that so the test is robust to small
    # numerical changes while still catching a plateau or reversal.
    assert errors[0] > errors[1] > errors[2]
    assert errors[0] > 20 * errors[1]
    assert errors[1] > 5 * errors[2]
    assert errors[-1] < 5e-3


def test_ballistic_column_is_exact_and_independent_of_segment_count(elements, cache):
    import sys
    from pathlib import Path
    medium = synthetic_medium()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_shower_moments import vectorised_ballistic as reference_ballistic
    exact = reference_ballistic(elements, RECEIVERS, medium, elements.cone_cosine)
    for k in (1, 8):
        effective = fit_effective_segments(elements, k)
        out = full_response(cache, elements, medium, effective, RECEIVERS)
        # Whichever implementation full_response used (numba if available,
        # else the plain numpy formula), it must be the same closed form,
        # not something the segment count can move.
        assert np.allclose(out[0, :, 0], exact, rtol=0, atol=1e-9)


def test_rejects_a_non_positive_segment_count(elements):
    with pytest.raises(ValueError):
        fit_effective_segments(elements, 0)


def test_many_segments_report_empty_bins_instead_of_crashing(elements):
    effective = fit_effective_segments(elements, 5000)
    assert effective.empty_bins > 0
    assert len(effective) <= 5000


def test_beta_one_boundary_survives_weighted_roundoff(elements):
    at_boundary = replace(
        elements,
        cone_cosine=np.full(len(elements), 1.0 / elements.contract.phase_index),
    )
    effective = fit_effective_segments(at_boundary, 256)
    assert np.all(np.array([piece.beta for piece in effective.segments]) <= 1.0)
