import numpy as np
import pytest
from scipy.special import ndtr
from lighthit.readout import inverse_bins
from lighthit.transport import TransportResponse


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


def test_response_rebins_only_fourier_derived_orders_at_smaller_cutoff():
    omega = np.array([0.0, 0.5, 1.0])
    edges = np.array([-1.0, 0.0, 1.0])
    origin = np.array([3.0])
    spectrum = np.zeros((3, 1, 3), complex)
    spectrum[:, 0, 2] = np.exp(1j * omega * origin[0])
    components = np.zeros((1, 2, 3))
    components[0, :, 1] = [0.25, 0.75]
    components[0, :, 2] = inverse_bins(
        omega, spectrum[:, :, 2] * np.exp(-1j * omega[:, None] * origin), edges)[0]
    response = TransportResponse(
        None, omega, edges, spectrum, components, np.zeros((1, 3)), origin,
        np.ones(1, bool), "test", {"fourier_inverted_orders": [2]})

    got = response.components_at_frequency_cutoff(0.5)
    expected_tail = inverse_bins(
        omega[:2], spectrum[:2, :, 2] * np.exp(-1j * omega[:2, None] * origin), edges)[0]
    np.testing.assert_array_equal(got[0, :, 1], components[0, :, 1])
    np.testing.assert_allclose(got[0, :, 2], expected_tail)


@pytest.mark.parametrize("cutoff", [0.0, np.nan, 0.1])
def test_response_rebin_rejects_invalid_frequency_cutoff(cutoff):
    response = TransportResponse(
        None, np.array([0.0, 0.5]), np.array([0.0, 1.0]),
        np.zeros((2, 1, 3), complex), np.zeros((1, 1, 3)),
        np.zeros((1, 3)), np.zeros(1), np.ones(1, bool), "test", {})
    with pytest.raises(ValueError):
        response.components_at_frequency_cutoff(cutoff)
