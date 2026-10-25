"""Independent normalization controls for the collision-sampling reference."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

pytest.importorskip('numba')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from harness.ge2_reference import cap_score, collision_batch


@pytest.mark.parametrize('g', [0., .9])
@pytest.mark.parametrize('relative_distance', [1.01, 1.1, 2., 5., 20., 200.])
def test_cap_mass_matches_analytic_hg_probability(g, relative_distance):
    radius = .216
    low = math.sqrt(1-1/relative_distance**2)
    if g == 0:
        expected = (1-low)/2
    else:
        expected = ((1-g*g)/(2*g)
                    *(1/(1-g)-1/math.sqrt((1-g)**2+2*g*(1-low))))
    empty = np.empty(0)
    actual = cap_score(np.array([0., 0., relative_distance*radius]),
        np.array([0., 0., 1.]), np.array([0., 0., 1.]), radius, 0., g,
        np.array([1., 0., 0., 0.]), empty, empty, empty, empty)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-13)


@pytest.mark.parametrize('importance', [0, 4])
@pytest.mark.parametrize('fixed_wavelength_nm', [0., 450.])
def test_first_collision_and_all_order_normalization(importance, fixed_wavelength_nm):
    n, phase_index = 160_000, 1.35
    cosine = 1/phase_index
    mu_a, mu_s, g, r = .05, .02, .9, 30.
    mu_t, albedo = mu_a+mu_s, mu_s/(mu_a+mu_s)
    max_sin2 = np.array([1-cosine*cosine])
    start = np.zeros((1, 3))
    axis = np.array([[0., 0., 1.]])
    receivers = np.array([[0., 0., r]])
    empty = np.empty(0)
    total, square, _, normal, normal2 = collision_batch(n, 92571,
        start, axis, np.zeros(1), np.ones(1), np.ones(1),
        np.array([350., 600.]), np.full(2, phase_index), np.full(2, mu_t),
        np.full(2, mu_s), np.array([[1., 0., 0., 0., 0., 0.]]), 1., max_sin2, g,
        receivers, axis, np.array([.1]), np.ones(1), np.array([1., 0., 0., 0.]),
        empty, empty, empty, empty, np.ones(1), importance, fixed_wavelength_nm)
    # Integral of every collision density over all space. This checks the
    # forced collisions, each importance ratio and roulette compensation.
    expected = np.array([albedo, albedo**2/(1-albedo)])
    mean = normal/n
    error = np.sqrt(np.maximum(0., normal2-normal*normal/n)/(n-1)/n)
    rounding = n*np.finfo(float).eps*abs(expected)
    assert np.all(abs(mean-expected) <= 5*error+rounding)

    # For a point on the axis every cone azimuth has the same first-order
    # response. Its independent one-dimensional flight integral is nonsingular.
    direction = np.array([math.sqrt(1-cosine*cosine), 0., cosine])
    def integrand(flight):
        towards = receivers[0]-flight*direction
        distance = np.linalg.norm(towards)
        angle = np.dot(direction, towards)/distance
        hg = (1-g*g)/(4*np.pi*(1+g*g-2*g*angle)**1.5)
        return mu_s*np.exp(-mu_t*(flight+distance))*hg/distance**2
    reference = quad(integrand, 0., np.inf, epsabs=1e-13, epsrel=1e-10)[0]
    value = total[0, 0]/n
    sigma = math.sqrt(max(0., square[0, 0]-total[0, 0]**2/n)/(n-1)/n)
    assert abs(value-reference) <= 5*sigma


def _monochromatic_inputs():
    from types import SimpleNamespace
    from lighthit import SpectralLightElements
    from lighthit.receiver import SphereSurface
    def sample(wavelength):
        return {key:np.full(np.shape(wavelength),value) for key,value in
                [('phase_index',1.35),('group_index',1.36),
                 ('absorption_per_m',.05),('scattering_per_m',.02)]}
    detector=SimpleNamespace(positions_m=np.array([[0.,0.,30.]]),
        orientations=np.array([[0.,0.,-1.]]),effective_area_m2=np.ones(1),
        receiver_surface=SphereSurface(.216),
        spectral_weight=lambda wavelength:np.full(np.shape(wavelength),.2))
    class Detector(SimpleNamespace):
        def __len__(self):return len(self.positions_m)
    detector=Detector(**vars(detector))
    detector.spectral_efficiency=detector.spectral_weight
    medium=SimpleNamespace(g=.9,sample=sample,wavelength_nm=np.array([400.,500.]),
                           wavelength_range_nm=(400.,500.))
    kernel=SimpleNamespace(detector=detector,medium=medium,
                           config=SimpleNamespace(wavelength_range_nm=None),
                           acceptance=lambda:{'alpha':np.array([4*np.pi])})
    source=SpectralLightElements(np.zeros((1,3)),np.array([[0.,0.,1.]]),
        np.array([.01]),np.ones(1),-np.ones(1),np.ones(1),np.zeros(1),
        np.array([.01/.299792458]),np.zeros(1,int),np.zeros(1,int))
    return kernel,source


def test_monochromatic_reference_has_differential_spectral_units():
    from harness.ge2_reference import estimate
    kernel,source=_monochromatic_inputs()
    a=estimate(kernel,source,[0],samples=4000,seed=75342,wavelength_nm=400.)
    b=estimate(kernel,source,[0],samples=4000,seed=75342,wavelength_nm=500.)
    # Identical optics, cone and random stream: only the physical
    # Frank--Tamm wavelength density changes, without a spectral width.
    np.testing.assert_allclose(b['charge_pe'],np.asarray(a['charge_pe'])*(400/500)**2,rtol=2e-15)
    np.testing.assert_allclose(b['mc_sigma_pe'],np.asarray(a['mc_sigma_pe'])*(400/500)**2,rtol=2e-15)
    expected=.2*(1-1/1.35**2)/400**2
    assert a['normalization_yield']==pytest.approx(expected,rel=2e-15)
    assert a['charge_unit']=='pe/nm' and a['spectral_sampling']=='monochromatic'


def test_continuous_reference_retains_integrated_frank_tamm_normalization():
    from harness.ge2_reference import estimate
    kernel,source=_monochromatic_inputs()
    result=estimate(kernel,source,[0],samples=4000,seed=918234)
    expected=.2*(1-1/1.35**2)*(1/400-1/500)
    assert result['normalization_yield']==pytest.approx(expected,rel=2e-14)
    assert result['spectral_sampling']=='continuous'
    assert 'charge_unit' not in result


@pytest.mark.parametrize('wavelength', [True,0.,-1.,np.nan,np.inf,[450.]])
def test_monochromatic_reference_rejects_invalid_wavelength(wavelength):
    from harness.ge2_reference import estimate
    kernel,source=_monochromatic_inputs()
    with pytest.raises(ValueError,match='wavelength'):
        estimate(kernel,source,[0],samples=2,wavelength_nm=wavelength)
