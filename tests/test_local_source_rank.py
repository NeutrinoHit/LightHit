"""Source compression preserves stationary light and known modal sources."""
import numpy as np
from lighthit.experimental.local_source_rank import charge_preserving_ranks


def test_separable_emission_pulse_needs_one_mode():
    rng=np.random.default_rng(2);angles=rng.normal(size=(3,17));w=np.linspace(0,1.2,33)
    phase=np.exp(1j*np.array([2.,20.,200.])[:,None]*w)*np.exp(-w*w)
    source=angles[:,:,None]*phase[:,None,:]
    _,candidate,meta=next(charge_preserving_ranks(source,[1]))
    np.testing.assert_allclose(candidate,source,rtol=1e-13,atol=1e-13)
    assert meta['stationary_bitwise_equal']


def test_joint_modes_reconstruct_without_losing_charge():
    rng=np.random.default_rng(7);angular=rng.normal(size=(2,13,3));w=np.linspace(0,1.2,17)
    pulses=np.stack([np.exp(1j*3*w),np.exp(1j*7*w)-1,np.exp(-w*w)-1])
    source=angular@pulses
    variants=list(charge_preserving_ranks(source,[1,2,3]))
    errors=[]
    for _,candidate,meta in variants:
        np.testing.assert_array_equal(candidate[:,:,0],source[:,:,0])
        errors.append(meta['source_relative_frobenius'])
    assert errors[2]<1e-13
    assert errors[0]>=errors[1]>=errors[2]


def test_zero_stationary_coefficients_are_supported():
    rng=np.random.default_rng(9);source=rng.normal(size=(2,5,7))+1j*rng.normal(size=(2,5,7))
    source[:,:,0]=0.
    _,candidate,_=next(charge_preserving_ranks(source,[6]))
    np.testing.assert_allclose(candidate,source,rtol=1e-13,atol=1e-13)
