"""Operator-lane isolation and original-ballistic contracts of coarse preview."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("numba")
import lighthit as lh
from lighthit.experimental.charge_preview import (
    ChargePreviewConfig, _operator_lanes, charge_preview)
from lighthit._shower_basis import compile_axial_slices, apply_multi_spectral


def bundle():
    coeff=np.zeros((1,2,3,1,4,3),np.complex64)
    coeff[:,:,0,0,3,:]=.001
    coeff[:,:,2,0,3,:]=.0001
    return dict(degree=2,acceptance_degree=0,phase='flight',
                pair_lambda=np.array([0],np.int64),pair_mu=np.array([0],np.int64),
                omega=np.array([0.,.1,.2]),log_r=np.log([1.,20.,200.]),
                coefficients=coeff,absorption=np.array([.02]),speed=np.array([.22]))


def test_operator_lanes_equal_two_independent_applications_and_keep_sign():
    second=bundle();first=dict(second,coefficients=-2.5*second['coefficients'])
    lanes=_operator_lanes(first,second,2)
    source=lh.CherenkovTrack([0.,0.,-2.],[0.,0.,1.],4.,cone_model='spectral').to_elements(step_m=.1)
    field=source.field(0)
    # An enormous source clock must not create any charge-only time lattice.
    field=replace(field,start_ns=field.start_ns+1e9,end_ns=field.end_ns+1e9)
    compiled=compile_axial_slices(field,2,[0.],azimuthal_degree=0,cell_m=.5)
    assert compiled.summary['time_rows']==compiled.summary['slices']
    receivers=np.array([[5.,0.,0.],[12.,2.,0.]])
    looks=np.tile([0.,0.,1.],(2,1));alpha=np.array([4*np.pi]);scale=np.ones((1,2))
    repeated=replace(compiled,channels=np.repeat(compiled.channels,2,axis=2))
    actual=apply_multi_spectral(lanes,[repeated],receivers,looks,alpha,scale)
    expected=[]
    for operator in [first,second]:
        zero=dict(operator,omega=np.array([0.]),coefficients=operator['coefficients'][...,:1])
        expected.append(apply_multi_spectral(zero,[compiled],receivers,looks,alpha,scale)[0])
    np.testing.assert_allclose(actual,np.asarray(expected),rtol=1e-13,atol=1e-10)
    assert np.all(actual[0].real<0)
    assert np.all(actual[1].real>0)


def test_operator_lanes_reject_incompatible_radial_geometry():
    second=bundle();first=dict(second,log_r=second['log_r']+.01)
    with pytest.raises(ValueError,match='log_r'):
        _operator_lanes(first,second,2)


def test_preview_compiles_only_zero_and_does_not_run_prompt(monkeypatch,tmp_path):
    import lighthit.experimental.charge_preview as module
    import lighthit._shower_basis as engine
    medium=lh.SpectralMedium([350.,610.],[.05,.05],[.02,.02],[1.35,1.35],[1.36,1.36])
    detector=lh.DetectorArray([[1.5,0.,0.],[12.,2.,0.]],[[0.,0.,1.]]*2,.05,
                              lambda x:np.ones_like(x),lambda w:.25*np.ones_like(w))
    kernel=lh.TransportKernel(medium,detector,lh.KernelConfig(wavelength_nodes=1,
                          radial_range_m=(1.,200.),cache_directory=tmp_path,threshold_pe=0.))
    source=lh.CherenkovTrack([0.,0.,-2.],[0.,0.,1.],4.,cone_model='spectral').to_elements(step_m=.1)
    reference=kernel.transport_prompt(source,lh.PromptConfig(element_method='segment',screen=False))
    assert np.any(reference.charge_orders_pe[:,0]>0)
    operator=bundle();lanes=_operator_lanes(dict(operator,coefficients=2*operator['coefficients']),operator,2)
    monkeypatch.setattr(module,'load_charge_operators',lambda *a,**kw:(lanes,{}))
    def forbidden(*a,**kw):
        raise AssertionError('preview must not invoke transport_prompt')
    monkeypatch.setattr(kernel,'transport_prompt',forbidden)
    original_compile=engine.compile_axial_slices;seen=[]
    def charge_only(*a,**kw):
        np.testing.assert_array_equal(a[2],[0.]);seen.append(True)
        assert kw['azimuthal_degree']==1
        return original_compile(*a,**kw)
    monkeypatch.setattr(engine,'compile_axial_slices',charge_only)
    result=charge_preview(kernel,source,ge2_store=None,first_pack_directory=tmp_path,
                           config=ChargePreviewConfig(degree=2,azimuthal_degree=1,cell_m=.5))
    assert seen
    np.testing.assert_allclose(result.charge_components_pe[:,0],reference.charge_orders_pe[:,0],rtol=1e-14)
    assert result.charge_components_pe.shape==(2,3)
    assert result.metadata['source_compilation_frequency_grid']==[0.]
    assert not result.metadata['preview_is_a_bound']
    blocked=charge_preview(kernel,source,ge2_store=None,first_pack_directory=tmp_path,
                 config=ChargePreviewConfig(degree=2,azimuthal_degree=1,cell_m=.5,use_detector_tree=True,source_block_size=4))
    np.testing.assert_allclose(blocked.charge_components_pe,result.charge_components_pe,rtol=2e-13,atol=1e-12)
    for name in result.flags:
        np.testing.assert_array_equal(blocked.flags[name],result.flags[name])


def test_preview_config_is_frozen_and_validates_geometry():
    from dataclasses import FrozenInstanceError
    options=ChargePreviewConfig()
    with pytest.raises(FrozenInstanceError):
        options.cell_m=0.
    with pytest.raises(ValueError):
        ChargePreviewConfig(cell_m=0.)
    with pytest.raises(ValueError):
        ChargePreviewConfig(degree=2,azimuthal_degree=3)
