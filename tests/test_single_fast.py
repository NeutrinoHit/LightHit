"""The compiled exact-first-order scalar kernel changes no quadrature math."""
import numpy as np
import pytest

pytest.importorskip("numba")

from lighthit import synthetic_medium
from lighthit.single import (_unit_rule, single_scattering_rate,
                             single_spectrum, single_bins)
from lighthit.single_fast import single_scattering_path_rate


@pytest.mark.parametrize("cosine", [None, -1.0, -0.3, 0.5, 0.9])
def test_scalar_kernel_matches_public_rate(cosine):
    medium = synthetic_medium()
    radius = 20.0
    path = radius + np.geomspace(1e-10, 500.0, 40)
    nodes, weights = _unit_rule()
    cosine_value = np.nan if cosine is None else cosine
    fast = np.array([
        single_scattering_path_rate(
            value, radius, cosine_value, medium.speed_m_per_ns,
            medium.scattering_per_m, medium.extinction_per_m, medium.g,
            nodes, weights)
        for value in path
    ])
    reference = single_scattering_rate(
        path / medium.speed_m_per_ns, radius, cosine, medium)
    np.testing.assert_allclose(fast, reference, rtol=3e-13, atol=0.0)


@pytest.mark.parametrize("cosine", [None, 0.5])
def test_numba_and_numpy_quadratures_agree(cosine):
    medium = synthetic_medium()
    radius = 20.0
    omega = np.linspace(0.0, 0.5, 17)
    reference, reference_error = single_spectrum(
        omega, radius, cosine, medium, backend="numpy")
    fast, fast_error = single_spectrum(
        omega, radius, cosine, medium, backend="numba")
    np.testing.assert_allclose(fast, reference, rtol=2e-12, atol=1e-18)
    np.testing.assert_allclose(fast_error, reference_error, rtol=2e-8, atol=1e-20)

    front = radius / medium.speed_m_per_ns
    edges = np.linspace(front - 1.0, front + 100.0, 31)
    reference_bins = single_bins(edges, radius, cosine, medium, backend="numpy")
    fast_bins = single_bins(edges, radius, cosine, medium, backend="numba")
    np.testing.assert_allclose(fast_bins, reference_bins, rtol=2e-12, atol=1e-18)
