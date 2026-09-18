import numpy as np
import pytest
pytest.importorskip('numba')
from lighthit.experimental.shell_mc import shell_estimate, ratio_with_standard_error

def test_positive_weighted_paths_and_absorption_covariance():
    da=1e-5
    b=shell_estimate(absorptions=[.07-da,.07,.07+da],radii_m=[20.],photons_per_batch=300,batches=4)
    assert np.all(b[...,0]>=0)
    assert np.all(np.abs(b[...,1])<=b[...,0]+1e-30)
    mean=b.mean(0)
    ratio,_=ratio_with_standard_error(b)
    derivative=(ratio[2]-ratio[0])/(2*da)
    q=mean[1,...,0]
    covariance=mean[1,...,4]/q-(mean[1,...,1]/q)*(mean[1,...,3]/q)
    np.testing.assert_allclose(derivative,-covariance,rtol=2e-5,atol=1e-6)
    assert np.all(mean[0,...,0]>=mean[2,...,0])
