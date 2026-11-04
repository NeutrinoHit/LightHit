"""The stopping-tail compiler preserves source weight and emission clocks."""
import numpy as np
import pytest
from scipy.special import eval_legendre
from lighthit import SpectralLightElements
from lighthit.experimental.ge2_beta_response import subdivide_tail_elements, cone_beta_taylor, cone_beta_quadratic


def test_subdivision_preserves_spectral_weight_and_clock():
    source=SpectralLightElements(np.array([[1.,2.,3.],[8.,2.,3.]]),
        np.array([[1.,0.,0.],[0.,1.,0.]]),np.array([7.,.2]),
        np.array([14.,.4]),np.array([-15.,-.8]),np.array([.97,.8]),
        np.array([10.,40.]),np.array([38.,41.]),np.arange(2),np.array([0,0]),cone_model='spectral')
    parts=subdivide_tail_elements(source,.125)
    assert parts.length_m.max()<=.125
    np.testing.assert_allclose(parts.coefficient0.sum(),source.coefficient0.sum())
    np.testing.assert_allclose(parts.coefficient2.sum(),source.coefficient2.sum())
    np.testing.assert_allclose(parts.start_m[55]+parts.length_m[55]*parts.direction[55],[8.,2.,3.])
    np.testing.assert_allclose(parts.end_ns[55],38.)
    np.testing.assert_allclose(parts.end_ns[-1],41.)
    np.testing.assert_allclose((parts.end_ns[:56]-parts.start_ns[:56])/parts.length_m[:56],4.)
    assert np.all(parts.beta[:56]==.97) and np.all(parts.beta[56:]==.8)
    with pytest.raises(ValueError):subdivide_tail_elements(source,0)


def test_cone_response_taylor_remainder_is_cubic():
    wave=np.array([370.,550.]);n=np.array([1.36,1.34]);ell=np.arange(33)
    coefficients=cone_beta_taylor(wave,n,32)
    errors=[]
    for delta in [1e-4,5e-5]:
        x=(1+delta)/n
        exact=(1-x[:,None]**2)*eval_legendre(ell[None,:],x[:,None])
        exact*=np.sqrt((2*ell+1)/(4*np.pi))[None,:]/wave[:,None]**2
        approx=sum(coefficients[k]*delta**k for k in range(3))
        errors.append(np.max(abs(exact-approx)))
    assert 7.8<errors[0]/errors[1]<8.2


def test_quadratic_cone_keeps_two_modes_and_exact_beta_one():
    wave=np.array([370.,480.,550.]);n=np.array([1.36,1.35,1.34]);ell=np.arange(33)
    coefficients=cone_beta_quadratic(wave,n,32,.99)
    taylor=cone_beta_taylor(wave,n,32)
    assert coefficients.shape==taylor.shape
    np.testing.assert_array_equal(coefficients[0],taylor[0])
    maximum=1/.99-1
    def exact(delta):
        x=(1+delta)/n[:,None]
        return ((1-x*x)*eval_legendre(ell[None,:],x)
            *np.sqrt((2*ell+1)/(4*np.pi))[None,:]/wave[:,None]**2)
    for delta in [maximum/2,maximum]:
        np.testing.assert_allclose(sum(coefficients[k]*delta**k for k in range(3)),exact(delta),rtol=1e-12,atol=1e-21)
    # Independent interior cones test the approximation between its anchors.
    points=np.linspace(0,maximum,41)
    new_error=max(np.max(abs(sum(coefficients[k]*d**k for k in range(3))-exact(d))) for d in points)
    old_error=max(np.max(abs(sum(taylor[k]*d**k for k in range(3))-exact(d))) for d in points)
    assert new_error<old_error/4
    np.testing.assert_array_equal(cone_beta_quadratic(wave,n,32,1.),taylor)


@pytest.mark.parametrize('beta_min',[0.,1.01,.5,np.nan])
def test_quadratic_cone_rejects_unsupported_intervals(beta_min):
    with pytest.raises(ValueError,match='above-threshold'):
        cone_beta_quadratic([480.],[1.35],32,beta_min)
