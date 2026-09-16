import numpy as np
import pytest
from scipy.special import ndtr
from lighthit.readout import inverse_bins


def test_frequency_inversion_gaussian_delta():
    omega=np.linspace(0,5,2001); t0=12.;sigma=2.
    edges=np.arange(-4,30,1.)
    spectrum=np.exp(1j*omega*t0)[:,None]
    actual=inverse_bins(omega,spectrum,edges,sigma)[0]
    expected=ndtr((edges[1:]-t0)/sigma)-ndtr((edges[:-1]-t0)/sigma)
    np.testing.assert_allclose(actual,expected,atol=5e-14)


@pytest.mark.parametrize("omega,edges,sigma", [([0],[0,1],0),([.1,.2],[0,1],0),([0,.1],[1,0],0),([0,.1],[0,1],-1)])
def test_invalid_readout(omega,edges,sigma):
    with pytest.raises(ValueError):inverse_bins(omega,np.ones((len(omega),1)),edges,sigma)
