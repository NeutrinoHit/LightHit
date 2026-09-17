"""Equivalence and independent controls for the optional recurrence backend."""
from dataclasses import replace
import numpy as np
import pytest

pytest.importorskip("numba")
from lighthit import PointGreenSolver, SolverSettings, synthetic_medium
from lighthit.angular import (
    _free_moments_and_ratios, angular_components, dense_finite_rank_reference,
)


@pytest.mark.parametrize("degree", [0, 1, 16, 160, 450])
@pytest.mark.parametrize("omega", [0.0, -0.13, 0.13, 1.2])
def test_free_backends_agree_across_branches(degree, omega):
    m = synthetic_medium()
    d0 = m.extinction_per_m - 1j * omega / m.speed_m_per_ns
    # Bracket the exact omega=0 branch boundary, and include zero, small k,
    # the root neighbourhood at positive omega, and forward-recursion nodes.
    boundary = m.extinction_per_m / np.sinh(3 / (degree + 3))
    k = np.r_[0.0, np.geomspace(1e-8, 100, 45),
              boundary * np.array([1 - 1e-9, 1, 1 + 1e-9])]
    ref = _free_moments_and_ratios(k, d0, degree, backend="numpy")
    got = _free_moments_and_ratios(k, d0, degree, backend="numba")
    for actual, expected in zip(got, ref):
        assert actual.flags.f_contiguous
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=5e-11, atol=5e-13)


def test_nonzero_ratios_survive_underflow_of_moments():
    b, ratio = _free_moments_and_ratios([0, 1e-7], 0.1 - 0.4j, 450, backend="numba")
    assert b[1, -1] == 0
    assert ratio[1, -1] != 0
    assert np.isfinite(ratio).all()
    np.testing.assert_array_equal(ratio[0], 0)


@pytest.mark.parametrize("omega", [0, 0.1, 0.7])
def test_numba_with_independent_dense_reference(omega):
    m = synthetic_medium()
    k = np.array([0.0, 1e-7, 0.03, 0.2, 1.0])
    parts = angular_components(k, omega, m, 8, 18, backend="numba")
    for i, ki in enumerate(k):
        expected = dense_finite_rank_reference(ki, omega, m, 8, 18, 1500)
        np.testing.assert_allclose(sum(p[i] for p in parts), expected, rtol=3e-10, atol=4e-10)


def test_longer_backward_run_confirms_minimal_ratios():
    # A separate deeper continued-fraction calculation, not a recurrence
    # residual (which would not distinguish the growing solution).
    N = 64
    d0 = 0.092 - 0.6j
    k = np.array([1e-6, 0.02, 0.1, 0.6, 1.0])
    _, got = _free_moments_and_ratios(k, d0, N, backend="numba")
    z = 1j * d0 / k
    root = np.sqrt(z - 1) * np.sqrt(z + 1)
    decay = 1 / (z + root)
    decay = np.where(abs(decay) > 1, 1 / decay, decay)
    eta = -np.log(abs(decay))
    assert np.all(eta * (N + 3) >= 3)
    top = N + 1 + max(64, int(np.ceil(56 / eta.min())))
    ratio = np.zeros(len(k), complex)  # intentionally different initial value
    expected = np.zeros_like(got)
    for ell in range(top, 0, -1):
        ratio = ell / ((2 * ell + 1) * z - (ell + 1) * ratio)
        if ell <= N + 1:
            expected[:, ell - 1] = ratio * np.sqrt((2 * ell + 1) / (2 * ell - 1))
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("direction", [(0., 0., 1.), None])
def test_spectra_readout_and_metadata_match(direction, tmp_path):
    import json
    m = synthetic_medium()
    s = SolverSettings(24, 80, 3.0, 0.05, 8)
    omega = np.linspace(0, 0.5, 25)
    displacement = [[5, 0, 3], [6, 0, -4]]
    ref = PointGreenSolver(m, s).solve(omega, displacement, direction=direction)
    got = PointGreenSolver(m, s, angular_backend="numba").solve(
        omega, displacement, direction=direction)
    np.testing.assert_allclose(got.components, ref.components, rtol=2e-10, atol=2e-15)
    edges = np.arange(-20., 252., 2.)
    for sigma in (0., 3.):
        np.testing.assert_allclose(got.readout(edges, sigma).components,
                                   ref.readout(edges, sigma).components,
                                   rtol=2e-9, atol=2e-15)
    got.save(tmp_path / "spectrum.npz")
    with np.load(tmp_path / "spectrum.npz") as f:
        assert json.loads(str(f["metadata"]))["angular_backend"] == "numba"


@pytest.mark.parametrize("L,J", [(0, 0), (8, 0), (3, 18)])
def test_numba_zero_k_particle_number(L, J):
    m = synthetic_medium()
    omega = 0.1
    b, one, multi = angular_components([0.], omega, m, L, J, backend="numba")
    np.testing.assert_allclose((b + one + multi)[0, 0],
        np.sqrt(2) / (m.absorption_per_m - 1j * omega / m.speed_m_per_ns), rtol=2e-14)
    np.testing.assert_array_equal((b + one + multi)[0, 1:], 0)
