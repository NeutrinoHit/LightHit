"""A stationary query must use its compact prepared tensor, not a dense slice."""
from types import SimpleNamespace
import numpy as np
from lighthit.shower_basis import Ge2KernelStore


def test_prepared_kernel_frequency_selection_prefers_compact_bundle():
    kernel=SimpleNamespace(wavelength=SimpleNamespace(wavelength_nm=np.array([400.,500.])),
        config=SimpleNamespace(source_degree=3),_directional_cache_key=lambda w:str(w))
    store=Ge2KernelStore();indices=np.arange(3)
    coefficients=(np.arange(2*2*4*3*4*3).reshape(2,2,4,3,4,3)+1j).astype(np.complex64)
    coefficients.setflags(write=False)
    common=dict(degree=3,acceptance_degree=1,pair_lambda=np.array([0,1,1]),pair_mu=np.array([0,0,1]),
        omega=np.array([0.,.3,.6]),log_r=np.log([3.,10.,30.]),absorption=np.array([.03,.04]),
        speed=np.array([.22,.225]),phase='flight')
    full=dict(common,coefficients=coefficients)
    compact=coefficients[...,:1].copy();compact.setflags(write=False)
    zero=dict(common,coefficients=compact,omega=np.array([0.]))
    store._bundles[store._key(kernel,indices)]=full
    store._bundles[store._key(kernel,[0])]=zero
    prepared=store.kernel(kernel,400.,indices,frequencies=slice(0,1))
    assert np.shares_memory(prepared.coefficients,compact)
    assert not np.shares_memory(prepared.coefficients,coefficients)
    np.testing.assert_array_equal(prepared.coefficients[...,0,:],coefficients[0,...,:1])
    np.testing.assert_array_equal(prepared.omega_per_ns,[0.])
    del store._bundles[store._key(kernel,[0])]
    prepared=store.kernel(kernel,400.,indices,frequencies=slice(0,1))
    assert np.shares_memory(prepared.coefficients,coefficients)
    assert len(store._bundles)==1
