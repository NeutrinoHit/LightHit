"""Physical invariants of the finite moving-source Fourier primitive."""
import numpy as np
import pytest
from lighthit.experimental.track_ge2_primitive import SegmentPrimitive


def test_moving_phase_and_charge_analytic():
    omega=np.array([0.,.1,.7]); clock=3.4
    # Constant point response: independent closed-form finite moving source.
    evaluator=lambda u:np.ones((len(u),len(omega)),complex)
    row=SegmentPrimitive.build(evaluator,np.linspace(-10,10,101),omega,order=8,clock_ns_per_m=clock)
    a,b=7.31,-3.27;t=19.5
    expected=np.empty(3,complex);expected[0]=a-b
    expected[1:]=(np.exp(-1j*omega[1:]*clock*a)-np.exp(-1j*omega[1:]*clock*b))/(-1j*omega[1:]*clock)
    expected*=np.exp(1j*omega*t)
    np.testing.assert_allclose(row.segment(a,b,closest_time_ns=t),expected,rtol=3e-12,atol=2e-12)


def test_off_grid_partition_invariance_and_time_translation():
    omega=np.array([0.,.2,.9])
    evaluator=lambda u:np.exp(-u[:,None]**2/4)*np.exp(1j*omega[None,:]*np.sqrt(3+u[:,None]**2))
    row=SegmentPrimitive.build(evaluator,np.linspace(-8,8,65),omega,order=8)
    a,m,b=6.777,.123,-5.888
    direct=row.segment(a,b)
    np.testing.assert_allclose(row.segment(a,m)+row.segment(m,b),direct,atol=1e-15,rtol=1e-14)
    np.testing.assert_allclose(row.segment(a,b,closest_time_ns=7),direct*np.exp(7j*omega),atol=1e-15)
    np.testing.assert_allclose(row.segment(a,a),0,atol=0)


def test_radial_domain_is_explicitly_truncated():
    omega=np.array([0.,.1]);evaluator=lambda u:np.ones((len(u),2))
    row=SegmentPrimitive.build(evaluator,[-1,0,1],omega,order=8)
    np.testing.assert_allclose(row.segment(10,-10),row.segment(1,-1))
    np.testing.assert_allclose(row.segment(10,2),0,atol=0)
    with pytest.raises(ValueError):row.segment(-1,1)


def test_signed_response_is_preserved():
    omega=np.array([0.,.1]);evaluator=lambda u:-np.ones((len(u),2))
    row=SegmentPrimitive.build(evaluator,[-1,0,1],omega,order=8)
    assert row.segment(.5,-.5)[0].real == pytest.approx(-1.)


def test_orientation_basis_and_impact_reuse_at_heldout_geometry():
    from lighthit.experimental.track_ge2_primitive import TrackGe2MiniAtlas,look_polynomials
    omega=np.array([0.,.3]);clock=3.5
    coefficients=np.arange(1,11)/10
    def factory(rho,looks):
        angular=look_polynomials(looks)@coefficients
        return lambda u:angular[:,None,None]*(1+.01*rho*rho)*np.exp(1j*u[None,:,None]*clock*omega)
    atlas=TrackGe2MiniAtlas.build(factory,[25,30,35],np.linspace(-10,10,21),omega,clock_ns_per_m=clock)
    look=np.array([.3,-.4,np.sqrt(.75)])
    rho=27.5;a,b=7.3,-5.8
    expected=(a-b)*(1+.01*rho*rho)*(look_polynomials(look)@coefficients)
    np.testing.assert_allclose(atlas.segment(rho,look,a,b),expected,rtol=1e-13)
    reflected=look.copy();reflected[1]*=-1
    np.testing.assert_allclose(atlas.segment(rho,look,a,b),atlas.segment(rho,reflected,a,b),atol=0)
    with pytest.raises(ValueError):atlas.segment(24,look,a,b)


def test_saved_atlas_reuses_arbitrary_orientation(tmp_path):
    from lighthit.experimental.track_ge2_primitive import TrackGe2MiniAtlas,look_polynomials
    omega=np.array([0.,.2]);clock=3.3
    def factory(rho,looks):
        return lambda u:(1+.1*rho)*np.broadcast_to((look_polynomials(looks)@np.arange(10))[:,None,None],(len(looks),len(u),len(omega)))
    original=TrackGe2MiniAtlas.build(factory,[25,27.5,30,32.5,35],np.linspace(-5,5,11),omega,clock_ns_per_m=clock)
    original.metadata={'beta':1.,'wavelength_nm':[480.],'radial_range_m':[3.,300.]}
    original.save(tmp_path/'atlas.npz')
    loaded=TrackGe2MiniAtlas.load(tmp_path/'atlas.npz')
    look=np.array([.6,0,.8])
    np.testing.assert_array_equal(original.segment(29.1,look,3.7,-2.2,alignment='scaled'),loaded.segment(29.1,look,3.7,-2.2,alignment='scaled'))
    assert loaded.metadata==original.metadata
