"""Positive collision-prefix identities, including first-entry killing."""
import numpy as np
from numba import njit
from lighthit.muon._surface_preview_mc import positive_multiple_cell


@njit
def isotropic_tail(dx,dy,dz,sx,sy,sz,band,arguments,out):
    out[:]=0.;out[0]=1.;out[3]=1/3;out[5]=1/3


def test_unbounded_neumann_sum_of_constant_isotropic_field():
    mean,se=positive_multiple_cell(50000,14,(0.,0.),(3.2,3.2),0.,.9,
        np.array([1.,1.]),np.array([.2,.1]),np.array([1.,.3]),np.array([1.,.8]),isotropic_tail,())
    expected=.2/(1-.2)+.3*.1/(1-.1)
    assert abs(mean[0]-expected)<5*se[0]
    np.testing.assert_allclose(mean[[3,5]],mean[0]/3,rtol=2*50000*np.finfo(float).eps)


def test_first_entry_kills_prefix_before_second_collision():
    R=.216;distance=3.2;mu=1.;albedo=.2
    mean,se=positive_multiple_cell(50000,37,(0.,0.),(distance,distance),R,.9,
        np.array([mu]),np.array([mu*albedo]),np.ones(1),np.ones(1),isotropic_tail,(),fixed_prefix=1)
    expected=albedo*(-np.expm1(-mu*(distance-R)))
    assert abs(mean[0]-expected)<5*se[0]
