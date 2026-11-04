"""Positive response integration conserves modes and is fitted without events."""
import numpy as np
import pytest
from lighthit.model import WavelengthQuadrature
from lighthit.experimental.spectral_response_basis import fit_response_cubature,evaluate_cubature


def test_positive_cubature_integrates_low_rank_family_and_held_out_features():
    reference=WavelengthQuadrature.gauss_legendre(350.,610.,30)
    x=(reference.wavelength_nm-480.)/130.
    # Different training and withheld mixtures of the same response functions.
    rng=np.random.default_rng(5);functions=np.array([np.ones(30),x,x*x,np.exp(.3*x)])
    train=rng.normal(size=(20,4))@functions
    model=fit_response_cubature(reference,train,4)
    assert len(model.selected_indices)<=5
    assert np.all(model.quadrature.weight_nm>0)
    np.testing.assert_allclose(model.quadrature.weight_nm.sum(),reference.weight_nm.sum(),rtol=1e-12)
    held_out=rng.normal(size=(11,4))@functions
    np.testing.assert_allclose(evaluate_cubature(model,held_out),held_out@reference.weight_nm,rtol=1e-10,atol=1e-9)


def test_invalid_shape_rank_and_nan_rejected():
    reference=WavelengthQuadrature.gauss_legendre(350.,610.,9)
    for values,rank in [(np.ones((4,8)),2),(np.full((4,9),np.nan),2),(np.ones((4,9)),True),(np.ones((4,9)),4)]:
        with pytest.raises(ValueError):fit_response_cubature(reference,values,rank)


def test_optical_contract_is_independent_of_geometry_but_rejects_changed_water():
    from dataclasses import replace
    import lighthit as lh
    from lighthit.experimental.track_first_primitive import track_first_optical_contract
    medium=lh.SpectralMedium([350.,610.],[.05,.04],[.015,.014],[1.36,1.34],[1.38,1.36])
    def efficiency(w):return np.ones_like(w)*.2
    efficiency.wavelength_knots_nm=np.array([350.,610.])
    detector=lh.DetectorArray([[10.,0.,25.]],[[0.,0.,-1.]],.05,lambda x:np.ones_like(x),efficiency)
    kernel=lh.TransportKernel(medium,detector,lh.KernelConfig(omega_per_ns=np.array([0.])))
    _,digest=track_first_optical_contract(kernel)
    reference=WavelengthQuadrature.gauss_legendre(350.,610.,9)
    x=reference.wavelength_nm
    model=fit_response_cubature(reference,np.stack([np.ones(9)+factor*x for factor in [.1,.2,.3,.4]]),2,
        metadata={'optics_fingerprint':digest})
    changed=replace(detector,positions_m=np.array([[90.,20.,10.]]),orientations=np.array([[0.,1.,0.]]))
    model.validate_optics(lh.TransportKernel(medium,changed,kernel.config))
    other=replace(medium,absorption_per_m=medium.absorption_per_m*1.1)
    with pytest.raises(ValueError,match='optical contract'):
        model.validate_optics(lh.TransportKernel(other,detector,kernel.config))


def test_legacy_cubature_accepts_only_verified_portable_contract():
    import copy
    import lighthit as lh
    from lighthit.muon.track_first_primitive import track_first_optical_contract
    from lighthit._portable import fingerprint
    from test_model_file import model as fixture_model
    kernel=fixture_model().kernel(lh.KernelConfig(omega_per_ns=np.array([0.])))
    actual,_=track_first_optical_contract(kernel);recorded=copy.deepcopy(actual)
    recorded['spectral_tables']['mut'][0]*=1+1e-12
    reference=WavelengthQuadrature.gauss_legendre(400.,500.,9);x=reference.wavelength_nm
    cube=fit_response_cubature(reference,np.stack([np.ones(9)+a*x for a in [.1,.2,.3,.4]]),2,
        metadata=dict(optics_fingerprint=fingerprint(recorded,allow_nan=False)))
    with pytest.raises(ValueError,match='optical contract'):cube.validate_optics(kernel)
    cube.validate_optics(kernel,recorded_contract=recorded)
    corrupted=copy.deepcopy(recorded);corrupted['spectral_tables']['mut'][0]*=1.01
    with pytest.raises(ValueError,match='optical contract'):cube.validate_optics(kernel,recorded_contract=corrupted)
    other=copy.deepcopy(cube.metadata);other['optical_contract']=recorded
    from dataclasses import replace
    portable=replace(cube,metadata=other);portable.validate_optics(kernel)
    changed=replace(kernel.medium,absorption_per_m=kernel.medium.absorption_per_m*1.01)
    with pytest.raises(ValueError,match='optical contract'):
        portable.validate_optics(lh.TransportKernel(changed,kernel.detector,kernel.config))
