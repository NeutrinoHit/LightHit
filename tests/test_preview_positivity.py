import numpy as np
from lighthit.muon._preview_positivity import observer_bernstein_map,positive_on_disk,certified_moments


def test_circle_boundary_is_certified_without_requiring_positive_square():
    _,transform=observer_bernstein_map([.3,.5,.2,-.05])
    polynomial=np.array([1.01,0.,0.,-1.,0.,-1.,0.,0.,0.,0.])
    assert positive_on_disk(transform@polynomial)


def test_small_negative_pocket_between_sampled_looks_is_rejected():
    _,transform=observer_bernstein_map([.3,.5,.2,-.05]);x,z=.317,.423
    polynomial=np.array([x*x+z*z-1e-10,-2*x,-2*z,1.,0.,1.,0.,0.,0.,0.])
    assert not positive_on_disk(transform@polynomial)


def test_physical_tilted_photon_moments_have_positive_response_for_every_look():
    transform,_=observer_bernstein_map([.3082,.54192,.19831,-.04912])
    rng=np.random.default_rng(5);directions=rng.normal(size=(40,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    x,z=directions[:,0],directions[:,2]
    moments=np.column_stack((np.ones(40),x,z,x*x,x*z,z*z,x**3,x*x*z,x*z*z,z**3))
    assert certified_moments(moments,transform).all()
