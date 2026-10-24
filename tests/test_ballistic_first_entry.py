"""First glass entry, degenerate cone geometry and deferred arrival bins."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('numba')
import lighthit as lh
from lighthit.ballistic_interval import first_entry_ballistic as ballistic_interval, _prepare_ballistic_interval


def axial_setup(position=(0., 0., .25)):
    medium = lh.SpectralMedium([350., 610.], [.7, .7], [0., 0.],
                              [1.35, 1.35], [1.4, 1.4], g=.9)
    detector = lh.DetectorArray([position], [[0., 0., -1.]], .05,
        lambda x: np.ones_like(x), lambda w: np.full_like(w, .3),
        receiver_surface=lh.SphereSurface(.2))
    kernel = lh.TransportKernel(medium, detector, lh.KernelConfig(
        omega_per_ns=np.array([0.]), acceptance_degree=0, threshold_pe=0.))
    length = .02
    source = lh.SpectralLightElements(np.zeros((1, 3)), np.array([[0., 0., 1.]]),
        np.array([length]), np.array([1e8*length]), np.array([-1e8*length]), np.ones(1),
        np.zeros(1), np.array([length/.299792458]), np.array([0]), np.array([0]), cone_model='spectral')
    return kernel, source


def test_axial_constant_phase_charge_and_clock_end_at_glass():
    k, s = axial_setup()
    x, w = np.polynomial.legendre.leggauss(96)
    a = (1+x)*s.length_m[0]/2
    along = .25-a
    cosine = 1/1.35
    entry = cosine*along-np.sqrt(.2**2-(1-cosine**2)*along**2)
    # Every azimuth hits the sphere in this geometry. The spectral integral
    # is analytic because n, extinction and efficiency are constant.
    factor = 1e8*(1-cosine**2)*.3*(1/350-1/610)*.05/(np.pi*.2**2)
    expected = factor*np.dot(w, np.exp(-.7*entry))*s.length_m[0]/2
    endpoints = np.array([0., s.length_m[0]])
    r = .25-endpoints
    path = cosine*r-np.sqrt(.2**2-(1-cosine**2)*r**2)
    time = endpoints/.299792458+1.4*path/.299792458
    edges = np.array([-5., time.min()-1e-8, time.max()+1e-8, 5.])
    result = ballistic_interval(k, s, time_origin_ns=[0.], relative_edges_ns=edges)
    np.testing.assert_allclose(result.charge_pe, [expected], rtol=2e-8)
    np.testing.assert_allclose(result.bins_pe[0], [0., expected, 0.], rtol=2e-8, atol=1e-11)
    assert result.metadata['ray_pairs'] == 1
    assert result.metadata['reference_point'] == 'first-entry'


def test_outside_axial_cone_that_misses_has_zero_charge():
    k, s = axial_setup((0., 0., 2.))
    result = ballistic_interval(k, s)
    np.testing.assert_array_equal(result.charge_pe, 0.)


def test_emission_crossing_the_glass_is_rejected():
    k, s = axial_setup((.1, 0., .01))
    with pytest.raises(ValueError, match='intersects the glass'):
        ballistic_interval(k, s)


def test_charge_does_not_request_bins_and_prompt_defers_unselected_oms(monkeypatch):
    from lighthit._surface_ballistic import PreparedBallistic
    k, s = axial_setup()
    k = lh.TransportKernel(k.medium, k.detector, replace(k.config, threshold_pe=1e10))
    def forbidden(*args, **kwargs):
        raise AssertionError('unselected receivers must not request arrival bins')
    monkeypatch.setattr(PreparedBallistic, 'bins', forbidden)
    prepared = _prepare_ballistic_interval(k, s)
    assert prepared.charge().charge_pe[0] > 0
    response = k.transport_prompt(s, lh.PromptConfig(ballistic_method='first-entry', screen=False),
                                  time_origin_ns=[0.])
    assert not response.active.any()
    np.testing.assert_array_equal(response.bins_orders_pe, 0.)


@pytest.mark.parametrize('offset_m', [0., -.12])
def test_dispersive_tilted_receiver_matches_independent_entry_rays(offset_m):
    beta, length, radius = .9, .05, .126
    medium = lh.SpectralMedium([350., 610.], [.07, .04], [0., 0.],
                              [1.38, 1.32], [1.43, 1.36], g=.9)
    look = np.array([.6, .3, np.sqrt(.55)])
    centre = np.array([2., 0., 2/np.sqrt((beta*1.35)**2-1)+length/2+offset_m])
    def accept(x):
        return 1+.3*x+.1*x*x+.02*x*x*x
    detector = lh.DetectorArray([centre], [-look], .05, accept,
        lambda w: np.full_like(w, .3), receiver_surface=lh.SphereSurface(radius))
    k = lh.TransportKernel(medium, detector, lh.KernelConfig(
        omega_per_ns=np.array([0.]), acceptance_degree=3))
    _, original = axial_setup()
    s = replace(original, length_m=np.array([length]), beta=np.array([beta]),
        coefficient0=np.array([1e8*length]), coefficient2=np.array([-1e8*length/beta**2]),
        end_ns=np.array([length/(beta*.299792458)]))
    gx, gw = np.polynomial.legendre.leggauss(64)
    lam = 480+130*gx
    sampled = medium.sample(lam)
    cosine = 1/(beta*sampled['phase_index'])
    sine = np.sqrt((1-cosine)*(1+cosine))
    # An independent wavelength/source/azimuth integral, with a sine map
    # resolving grazing entry rays at the cap endpoints.
    total = 0.
    phi_node = np.sin(np.pi*gx/2)
    phi_weight = np.pi/2*np.cos(np.pi*gx/2)*gw
    for a, weight_a in zip((1+gx)*length/2, gw*length/2):
        w = centre-np.array([0., 0., a])
        outside = w@w-radius**2
        half = np.arccos(np.clip((np.sqrt(outside)-cosine*w[2])/(sine*w[0]), -1, 1))
        phi = half[:, None]*phi_node
        dx, dy = sine[:, None]*np.cos(phi), sine[:, None]*np.sin(phi)
        dot = dx*w[0]+cosine[:, None]*w[2]
        entry = outside/(dot+np.sqrt(np.maximum(0., dot**2-outside)))
        angular = (accept(dx*look[0]+dy*look[1]+cosine[:, None]*look[2])
                   *np.exp(-sampled['absorption_per_m'][:, None]*entry))
        angular = (angular*phi_weight).sum(1)*half/(2*np.pi)
        total += weight_a*np.dot(130*gw, 1e8*sine**2/lam**2*.3*angular)
    expected = total*.05/(np.pi*radius**2)
    result = ballistic_interval(k, s, time_origin_ns=[0.], relative_edges_ns=np.arange(-5., 40., .25))
    np.testing.assert_allclose(result.charge_pe, [expected], rtol=2e-6, atol=1e-10)
    np.testing.assert_allclose(result.bins_pe.sum(), result.charge_pe[0], rtol=2e-12)
    # Charge-only uses the new path-moment control, while bins retain their
    # disk control. Both must recover the same independent boundary rays.
    charge_only = ballistic_interval(k, s)
    np.testing.assert_allclose(charge_only.charge_pe, [expected], rtol=2e-6, atol=1e-10)
