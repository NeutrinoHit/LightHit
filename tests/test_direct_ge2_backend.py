"""Contraction identity and direct moving-source quadrature invariants."""
import numpy as np
import pytest
pytest.importorskip('numba')
from lighthit.experimental.track_ge2_primitive import cone_point_evaluator


@pytest.mark.parametrize('flight',[False,True])
def test_fused_matches_generic_for_spectral_cones_and_signed_modes(flight):
    rng=np.random.default_rng(17)
    degree=4;out=3
    pairs=[(ell,m) for ell in range(out+1) for m in range(ell+1)]
    pl=np.array([p[0] for p in pairs]);pm=np.array([p[1] for p in pairs])
    omega=np.array([0.,.13,.77]);waves=np.array([380.,480.,580.])
    coefficient=rng.normal(size=(3,3,degree+1,len(pl),4,3))+1j*rng.normal(size=(3,3,degree+1,len(pl),4,3))
    bundle=dict(degree=degree,acceptance_degree=out,pair_lambda=pl,pair_mu=pm,
        log_r=np.log([3.,10.,30.,100.]),coefficients=coefficient,absorption=np.array([.02,.03,.04]),
        speed=np.array([.21,.22,.23]),omega=omega,phase='flight' if flight else 'absolute')
    kwargs=dict(impact_m=5.,look=np.array([[.6,0.,.8],[-.3,.4,np.sqrt(.75)]]),
        alpha=np.array([1.,.2,-.1,.02]),wavelength_nm=waves,wavelength_weight_nm=np.array([10.,20.,30.]),
        phase_index=np.array([1.3,1.35,1.4]),efficiency=np.array([.1,.2,.3]),beta=.75,effective_area_m2=.04)
    points=np.array([-21.,-5.,0.,8.,70.,150.])
    # First case includes a below-threshold wavelength. Second case is signed.
    for harmonics in [None,rng.normal(size=(3,degree+1))]:
        a=cone_point_evaluator(bundle,**kwargs,source_harmonics=harmonics,backend='generic')(points)
        b=cone_point_evaluator(bundle,**kwargs,source_harmonics=harmonics,backend='fused-looks')(points)
        np.testing.assert_allclose(a,b,rtol=3e-13,atol=2e-15)
        np.testing.assert_array_equal(b[:,-1],0.)


@pytest.mark.parametrize('quadrature',['fixed12','fixed24','adaptive'])
def test_direct_preserves_density_clock_and_partition(monkeypatch,quadrature):
    from types import SimpleNamespace
    from lighthit.experimental.universal_primary import DirectGe2
    import lighthit.experimental.track_ge2_primitive as primitive
    omega=np.array([0.,.13,.77]);clock=3.7
    def evaluator(bundle,**kw):
        return lambda u:np.exp(1j*u[:,None]*clock*omega)
    monkeypatch.setattr(primitive,'cone_point_evaluator',evaluator)
    kernel=SimpleNamespace(wavelength=SimpleNamespace(wavelength_nm=np.array([480.]),weight_nm=np.ones(1)),
        medium=SimpleNamespace(sample=lambda w:dict(absorption_per_m=np.array([.03]),phase_index=np.array([1.35]))),
        detector=SimpleNamespace(spectral_weight=lambda w:np.ones(1),effective_area_m2=np.ones(1)),
        acceptance=lambda:dict(alpha=np.ones(1)))
    direct=DirectGe2(kernel,'unused',quadrature=quadrature)
    direct.bundle=dict(omega=omega,coefficients=np.ones((1,1,1,1,4,3)),absorption=np.array([.03]))
    args=(5.,np.array([0.,0.,1.]),.99,clock)
    for length in [.01,.07,.2,.5,.9,3.7]:
        result=direct(*args,-.2,length-.2,2.5,7.,True)
        np.testing.assert_allclose(result,2.5*length*np.exp(7j*omega),rtol=3e-15,atol=3e-15)
        split=direct(*args,-.2,length/2-.2,2.5,7.,True)+direct(*args,length/2-.2,length-.2,2.5,7.,True)
        np.testing.assert_allclose(result,split,rtol=3e-15,atol=3e-15)
