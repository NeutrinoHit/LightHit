"""Analytic path moments at finite emission-window endpoints."""
import numpy as np
import pytest
from scipy.integrate import quad

pytest.importorskip('numba')
from lighthit._surface_ballistic_linear import window_moments


@pytest.mark.parametrize('a,b', [(-1., 1.), (-1., -.999999), (.3, .300001),
    (-.4, .7), (.6, .95), (1-1e-9, 1.), (-1e-9, 1e-9), (-2., 2.), (2., 3.)])
def test_partial_window_moments_against_independent_quadrature(a, b):
    value = window_moments(a, b)
    low, high = max(-1., a), min(1., b)
    if low >= high:
        np.testing.assert_array_equal(value, 0.)
        return
    # Map the integral to a unit interval so even narrow windows are resolved.
    def h(x):
        return low+(high-low)*x
    def square(x):
        return ((1-low)-(high-low)*x)*((1+low)+(high-low)*x)
    functions = [lambda x: np.sqrt(max(0., square(x))),
                 lambda x: h(x)*np.sqrt(max(0., square(x))), square]
    expected = [(high-low)*quad(f, 0., 1., epsabs=1e-13, epsrel=1e-11)[0] for f in functions]
    np.testing.assert_allclose(value, expected, rtol=2e-9, atol=2e-16)


def test_source_window_partition_preserves_all_three_moments():
    whole = np.array(window_moments(-.99, .97))
    cuts = [-.99, -.4, -.02, .7, .97]
    pieces = sum((np.array(window_moments(a, b)) for a, b in zip(cuts[:-1], cuts[1:])), np.zeros(3))
    np.testing.assert_allclose(pieces, whole, rtol=2e-14, atol=2e-16)


def setup(rho=2.):
    import lighthit as lh
    medium = lh.SpectralMedium([350., 450., 610.], [.07, .04, .09], [0., 0., 0.],
                              [1.38, 1.35, 1.32], [1.43, 1.39, 1.36], g=.9)
    centre = rho/np.sqrt(1.35**2-1)+.025
    detector = lh.DetectorArray([[rho, 0., centre], [rho, 0., centre-.08]],
        [[0., 0., -1.], [0., 0., -1.]], .05, lambda x: np.ones_like(x),
        lambda w: np.full_like(w, .3), receiver_surface=lh.SphereSurface(.126))
    kernel = lh.TransportKernel(medium, detector, lh.KernelConfig(
        omega_per_ns=np.array([0.]), acceptance_degree=0))
    def elements(start, lengths):
        start, lengths = np.asarray(start, float), np.asarray(lengths, float)
        n = len(lengths)
        return lh.SpectralLightElements(start, np.tile([0., 0., 1.], (n, 1)), lengths,
            1e8*lengths, -1e8*lengths, np.ones(n), start[:, 2]/.299792458,
            (start[:, 2]+lengths)/.299792458, np.arange(n), np.zeros(n, np.int64), cone_model='spectral')
    return kernel, elements


def test_prepared_spectral_windows_refine_only_requested_receivers():
    from lighthit._surface_ballistic import PreparedBallistic
    from lighthit._surface_ballistic_linear import prepare, response
    from lighthit.ballistic_interval import BallisticIntervalConfig
    k, make = setup()
    s = make([[0., 0., 0.]], [.05])
    p = PreparedBallistic(k, s, BallisticIntervalConfig())
    cached = prepare(k, s, p.tables, p.pairs)
    whole = response(k, s, p.tables, p.pairs, 32, prepared=cached)
    part = response(k, s, p.tables, p.pairs, 32, prepared=cached, receiver_mask=[False, True])
    assert part[0] == 0. and whole[0] > 0.
    assert part[1] == whole[1]


def test_glass_charge_is_invariant_under_source_partition():
    from lighthit.ballistic_interval import first_entry_ballistic
    k, make = setup()
    whole = first_entry_ballistic(k, make([[0., 0., 0.]], [.05])).charge_pe
    split = first_entry_ballistic(k, make([[0., 0., 0.], [0., 0., .013]], [.013, .037])).charge_pe
    np.testing.assert_allclose(split, whole, rtol=2e-6, atol=1e-10)


@pytest.mark.parametrize('rho', [.2, 2., 50.])
def test_linear_control_residual_is_within_its_pair_bound(rho):
    from lighthit._surface_ballistic import PreparedBallistic
    from lighthit._surface_ballistic_linear import response
    from lighthit._surface_ballistic_bounds import bound_pairs
    from lighthit._surface_ballistic_charge import boundary_charge
    from lighthit.ballistic_interval import BallisticIntervalConfig
    k, make = setup(rho)
    s = make([[0., 0., 0.]], [.05])
    p = PreparedBallistic(k, s, BallisticIntervalConfig())
    t, d = p.tables, k.detector
    baseline = response(k, s, t, p.pairs, 64)
    exact, _ = boundary_charge(k, s, 12, 32, 12, tables=t, pairs=p.pairs)
    bound = bound_pairs(*p.pairs, d.positions_m, -d.orientations, d.effective_area_m2,
        p.radii, s.start_m, s.direction, s.length_m, s.coefficient0, s.beta,
        t['panel_edges'], t['phase_nodes'], t['extinction_nodes'],
        p.eta_upper, p.angular_bounds, False, True)
    total = np.bincount(p.pairs[0], weights=bound, minlength=len(d))
    assert np.all(abs(exact-baseline) <= total+1e-10)
