"""Finite radial importance quadrature preserves photons and old moments."""
import numpy as np
from scipy.special import erfcx
from lighthit._receiver_radial_rule import radial_rule
from lighthit._prompt_numba import _prepare_gauss_inverse,_GAUSS_TAU,_GAUSS_W


def test_finite_radial_rule_preserves_mass_and_second_moment():
    x,w=_prepare_gauss_inverse(_GAUSS_TAU,_GAUSS_W)
    for u,v,a,b,cap in [(0.,0.,.03**2,.03**2,.008),(.02,.01,.03**2,.01**2,.008),
                        (0.,0.,.03**2,0.,.008),(.4,.2,.03**2,.02**2,.008),(0.,0.,0.,0.,.01)]:
        p0,t0,p1,t1=radial_rule(u,v,a,b,cap,x,w)
        assert p0>=0 and p1>=0 and 0<=t0<=np.pi and 0<=t1<=np.pi
        np.testing.assert_allclose(p0+p1,1.,rtol=2e-12,atol=1e-14)
        np.testing.assert_allclose(p0*t0*t0+p1*t1*t1,u*u+v*v+a+b,rtol=2e-10,atol=1e-14)


def test_isotropic_importance_moment_matches_analytic_rayleigh_mean():
    x,w=_prepare_gauss_inverse(_GAUSS_TAU,_GAUSS_W);sigma=.03;cap=.008
    p0,t0,p1,t1=radial_rule(0.,0.,sigma*sigma,sigma*sigma,cap,x,w)
    actual=p0/np.hypot(t0,cap)+p1/np.hypot(t1,cap)
    expected=np.sqrt(np.pi/2)/sigma*erfcx(cap/(np.sqrt(2)*sigma))
    np.testing.assert_allclose(actual,expected,rtol=2e-8)
