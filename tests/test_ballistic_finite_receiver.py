"""Direct light on the finite OM sphere (``BallisticIntervalConfig.receiver_mode``).

A short step close to an OM lights a thin conical shell. A point receiver at
the OM centre counts the whole shell when it passes through the centre and
nothing when it misses; the photon Monte Carlo counts the rays crossing the
sphere. This was measured on the regenerated 6 GeV muon passing OM 267 at
1.76 m (experiments/ideas/finite-receiver-order0.toml).
"""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('numba')
import lighthit as lh
from lighthit.ballistic_interval import ballistic_interval, BallisticIntervalConfig

POINT = BallisticIntervalConfig(receiver_mode="point")


def setup(position, area=.05, acceptance=lambda x: np.ones_like(x), degree=0):
    medium = lh.SpectralMedium([350., 450., 610.], [.07, .04, .09], [0., 0., 0.],
                               [1.38, 1.35, 1.32], [1.43, 1.39, 1.36], g=.9)

    def efficiency(w):
        return np.interp(w, [350., 410., 415., 610.], [.1, .2, .35, .05])
    efficiency.wavelength_knots_nm = np.array([350., 410., 415., 610.])
    detector = lh.DetectorArray(np.atleast_2d(position), [[0., 0., -1.]], area, acceptance, efficiency,
                                receiver_surface=lh.SphereSurface(np.sqrt(area / np.pi)))
    return lh.TransportKernel(medium, detector, lh.KernelConfig(
        omega_per_ns=np.array([0.]), wavelength_nodes=2, threshold_pe=0., acceptance_degree=degree))


def element(start, length, beta=1., direction=(0., 0., 1.), c0=1e8):
    n = len(np.atleast_1d(length))
    length = np.atleast_1d(np.asarray(length, float))
    start = np.atleast_2d(np.asarray(start, float))
    t0 = start[:, 2] / .299792458
    return lh.SpectralLightElements(start, np.tile(direction, (n, 1)), length, c0 * length,
        -c0 * length / beta ** 2, np.full(n, beta), t0, t0 + length / (beta * .299792458),
        np.arange(n), np.zeros(n, np.int64), cone_model='spectral')


def sphere_reference(kernel, source, radius, ns=48, nphi=24, per_panel=16):
    """Rays from every emission point and wavelength crossing the sphere.

    Extinction is taken to the ray's closest approach to the centre (the
    reference plane of the point receiver); the acceptance is isotropic.
    """
    centre = kernel.detector.positions_m[0]
    knots = np.array([350., 410., 415., 450., 610.])
    gx, gw = np.polynomial.legendre.leggauss(per_panel)
    lam = (.5 * (knots[1:, None] + knots[:-1, None]) + .5 * np.diff(knots)[:, None] * gx).ravel()
    wl = (.5 * np.diff(knots)[:, None] * gw).ravel()
    band = kernel.medium.sample(lam)
    n, ext = band['phase_index'], band['absorption_per_m'] + band['scattering_per_m']
    eff = kernel.detector.spectral_weight(lam)
    sx, sw = np.polynomial.legendre.leggauss(ns)
    px, pw = np.polynomial.legendre.leggauss(nphi)
    total = 0.
    for i in range(len(source)):
        u, L = source.direction[i], source.length_m[i]
        e1 = np.cross(u, [0., 1., 0.] if abs(u[1]) < .9 else [1., 0., 0.]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        cos_t = 1. / (source.beta[i] * n)
        sin_t = np.sqrt(1. - cos_t ** 2)
        photons = source.coefficient0[i] / L * sin_t ** 2 / lam ** 2
        for s, ws in zip(.5 * L * (sx + 1.), .5 * L * sw):
            w = centre - (source.start_m[i] + s * u)
            ww = w @ w
            bperp = np.hypot(w @ e1, w @ e2)
            phi0 = np.arctan2(w @ e2, w @ e1)
            x = (np.sqrt(ww - radius ** 2) - cos_t * (w @ u)) / (sin_t * bperp)
            half = np.arccos(np.clip(x, -1., 1.))
            phi = phi0 + half[:, None] * px[None, :]
            d = (cos_t[:, None, None] * u + sin_t[:, None, None]
                 * (np.cos(phi)[..., None] * e1 + np.sin(phi)[..., None] * e2))
            inner = (np.exp(-ext[:, None] * (d @ w)) * pw).sum(1) * half / (2 * np.pi)
            total += ws * np.sum(wl * photons * eff * inner)
    return total


def test_short_step_near_om_matches_rays_crossing_the_sphere():
    # A 5 cm step whose 450 nm cone passes through the OM centre at 2 m.
    cot = 1. / np.sqrt(1.35 ** 2 - 1.)
    kernel = setup([2., 0., 2. * cot + .025])
    source = element([[0., 0., 0.]], .05)
    radius = np.sqrt(.05 / np.pi)
    finite = ballistic_interval(kernel, source).charge_pe[0]
    point = ballistic_interval(kernel, source, config=POINT).charge_pe[0]
    expected = sphere_reference(kernel, source, radius)
    np.testing.assert_allclose(finite, expected, rtol=5e-3)
    # The point receiver counts the whole shell: twice the sphere here.
    assert point > 2 * finite


def test_om_just_off_a_short_shell_sees_light_the_point_misses():
    cot = 1. / np.sqrt(1.35 ** 2 - 1.)
    kernel = setup([2., 0., 2. * cot - .15])
    source = element([[0., 0., 0.]], .05)
    radius = np.sqrt(.05 / np.pi)
    point = ballistic_interval(kernel, source, config=POINT).charge_pe[0]
    finite = ballistic_interval(kernel, source).charge_pe[0]
    assert point == 0. and finite > 0.
    # Leading order in R/rho = 0.06 at the edge of the window: 2% here.
    np.testing.assert_allclose(finite, sphere_reference(kernel, source, radius), rtol=3e-2)


def test_split_straight_track_is_unchanged_and_long_track_equals_point():
    # Windows of collinear pieces tile: the split does not change the charge.
    kernel = setup([1.5, 0., 4.])
    whole = element([[0., 0., 0.]], 6.)
    pieces = element([[0., 0., 0.], [0., 0., .7], [0., 0., 2.95]], [.7, 2.25, 3.05])
    q = ballistic_interval(kernel, whole).charge_pe
    np.testing.assert_allclose(ballistic_interval(kernel, pieces).charge_pe, q, rtol=1e-9)
    # Far from both ends the window lies inside the track: the point result.
    np.testing.assert_allclose(q, ballistic_interval(kernel, whole, config=POINT).charge_pe, rtol=1e-13)


def test_finite_receiver_bins_sum_to_charge_and_metadata():
    cot = 1. / np.sqrt((.9 * 1.35) ** 2 - 1.)
    kernel = setup([2., 0., 2. * cot + .035])
    source = element([[0., 0., 0.], [0., 0., .04]], [.04, .03], beta=.9)
    edges = np.arange(-5., 40., .5)
    result = ballistic_interval(kernel, source, time_origin_ns=[0.], relative_edges_ns=edges)
    assert result.charge_pe[0] > 0
    np.testing.assert_allclose(result.bins_pe.sum(), result.charge_pe[0], rtol=1e-12)
    assert result.metadata['receiver_mode'] == 'auto'
    assert result.metadata['receiver_surface'] == 'sphere'


def test_far_thin_shell_in_large_source_still_uses_supplied_surface():
    # The centre misses every allowed cone, while the actual sphere intersects
    # the 350 nm edge. Repeating the source must not switch its geometry.
    rho = 80.
    kernel = setup([rho, 0., rho/np.sqrt(1.38**2-1.)-.05])
    count = 9001
    source = element(np.zeros((count, 3)), np.full(count, .01), c0=1e8/count)
    auto = ballistic_interval(kernel, source)
    surface = ballistic_interval(kernel, source, config=BallisticIntervalConfig(receiver_mode='surface'))
    point = ballistic_interval(kernel, source, config=POINT)
    assert point.charge_pe[0] == 0.
    assert auto.charge_pe[0] > 0.
    np.testing.assert_array_equal(auto.charge_pe, surface.charge_pe)


def test_no_surface_is_an_explicit_point_model_without_area_derived_radius():
    kernel = setup([2., 0., 2./np.sqrt(1.35**2-1.)-.15])
    source = element([[0., 0., 0.]], .05)
    point_detector = replace(kernel.detector, receiver_surface=None)
    point_kernel = lh.TransportKernel(kernel.medium, point_detector, kernel.config)
    auto = ballistic_interval(point_kernel, source)
    point = ballistic_interval(point_kernel, source, config=POINT)
    np.testing.assert_array_equal(auto.charge_pe, point.charge_pe)
    with pytest.raises(ValueError, match='receiver_surface geometry'):
        ballistic_interval(point_kernel, source, config=BallisticIntervalConfig(receiver_mode='surface'))



def test_window_reaching_the_cone_axis_near_threshold_bins_equal_charge():
    # A slow step just above threshold with the OM almost on its axis: the
    # sphere's support starts at the Cherenkov threshold, where the arrival
    # time diverges. Binning must not change the charge.
    kernel = setup([.1, 0., 2.])
    source = element([[0., 0., 0.]], .01, beta=1. / 1.37)
    edges = np.arange(-50., 400., 5.)
    plain = ballistic_interval(kernel, source).charge_pe
    binned = ballistic_interval(kernel, source, time_origin_ns=[0.], relative_edges_ns=edges)
    assert plain[0] > 0
    np.testing.assert_allclose(binned.charge_pe, plain, rtol=1e-6)
    np.testing.assert_allclose(binned.bins_pe.sum(), plain[0], rtol=1e-9)
