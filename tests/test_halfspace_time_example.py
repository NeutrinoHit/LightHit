"""The executable time example retains a quantitative bandwidth diagnostic."""
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr

from experiments.halfspace_time_profile import temporal_profile


def test_gaussian_band_bound_covers_missing_tail_of_a_delta_pulse():
    # A unit delta has H(omega)=1. Its Gaussian-broadened bin integral is
    # known exactly; no transport discretization enters this check.
    omega = np.linspace(0.,3.,6001)
    spectrum = np.zeros((len(omega),3),complex)
    spectrum[:,2] = 1.
    args = SimpleNamespace(time_start_ns=-.5,time_stop_ns=.5,
                           bin_width_ns=.1,timing_sigma_ns=1.)
    edges,components,prompt,total,diagnostic = temporal_profile(omega,spectrum,0.,args)
    exact = np.diff(ndtr(edges))
    error = np.abs(total-exact)
    assert error.sum() <= diagnostic['gaussian_omitted_band_l1_bound']
    assert error.max() <= diagnostic['gaussian_omitted_band_max_bin_bound']
    np.testing.assert_array_equal(total,components[:,2])
    assert prompt.sum() == 0
