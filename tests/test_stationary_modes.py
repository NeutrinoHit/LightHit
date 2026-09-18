import numpy as np
import pytest
from lighthit import synthetic_medium, Medium, SolverSettings
from lighthit.cache import ResponseCache,CacheGrid
from lighthit.single import single_spectrum
from lighthit.experimental.stationary_modes import StationaryModes, single_scalar_current

@pytest.mark.parametrize('radius',[12.,40.,128.])
def test_independent_first_order_scalar(radius):
    m=synthetic_medium();a=single_scalar_current(radius,m)
    b=single_spectrum(np.array([0.]),radius,None,m)[0][0].real
    assert a[0]==pytest.approx(b,rel=2e-8)
    assert 0<=a[1]<=a[0]

def test_positive_eigensum_converges_and_has_finite_leading_mode():
    m=synthetic_medium();a=StationaryModes.build(m,128);b=StationaryModes.build(m,256)
    qa,fa=a.scalar_current([11,40,128,400,800]);qb,fb=b.scalar_current([11,40,128,400,800])
    np.testing.assert_allclose(qa,qb,rtol=2e-8)
    np.testing.assert_allclose(fa,fb,rtol=2e-8)
    assert np.all(qa>0) and np.all(fa<qa) and a.isolated_leading_mode
    assert a.leading_attenuation_per_m==pytest.approx(.0780239173344,rel=2e-11)
    assert a.leading_mean_cosine==pytest.approx(.897160798784,rel=2e-11)

def test_radial_balance():
    m=synthetic_medium();a=StationaryModes.build(m,128)
    r=np.array([20.,80.,150.]);dr=1e-3
    q,f=a.scalar_current(r)
    _,fp=a.scalar_current(r+dr);_,fm=a.scalar_current(r-dr)
    divergence=((r+dr)**2*fp-(r-dr)**2*fm)/(2*dr*r*r)
    np.testing.assert_allclose(divergence,-m.absorption_per_m*q,rtol=1e-8)

def test_agrees_with_original_fourier_route():
    m=synthetic_medium();r=np.array([20.,40.,80.])
    c=ResponseCache.build(m,SolverSettings(96,1,6,.025,10),CacheGrid(r,np.array([0.])))
    b=np.exp(-m.extinction_per_m*r)/(4*np.pi*r*r)
    q=b+c.moments[0,:,0,:].real.sum(1)
    f=b+c.moments[0,:,1,:].real.sum(1)/3
    qe,fe=StationaryModes.build(m,128).scalar_current(r)
    np.testing.assert_allclose(q,qe,rtol=2e-5)
    np.testing.assert_allclose(f,fe,rtol=2e-5)

def test_no_scattering_limit():
    m=Medium(.04,0.,.7,1.35);r=np.array([20.,40.,80.])
    q,f=StationaryModes.build(m,512).scalar_current(r)
    exact=np.exp(-m.absorption_per_m*r)/(4*np.pi*r*r)
    np.testing.assert_allclose(q,exact,rtol=2e-7)
    np.testing.assert_allclose(f,exact,rtol=2e-7)
