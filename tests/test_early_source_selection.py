"""Geometric sparsity must preserve every requested source coefficient."""
from dataclasses import replace
import numpy as np
import pytest
import lighthit as lh
from lighthit._shower_basis import compile_axial_slices
from lighthit.experimental.axial_source import AxisFrame


@pytest.mark.parametrize('method',['gemm','loop'])
@pytest.mark.parametrize('time_rule',['linear','cubic'])
@pytest.mark.parametrize('transverse,centre',[('collapse','axis'),('stencil','axis'),('stencil','centroid')])
def test_sparse_compilation_matches_complete_source(method,time_rule,transverse,centre):
    source=lh.SyntheticShower.gaussian(lh.SourcePose([0.,0.,0.],[.1,.2,.97],-10.),
        charged_track_length_m=10.,elements=100,seed=4).elements
    f0,f2=source.field(0),source.field(2)
    kwargs=dict(degree=8,omega_per_ns=np.linspace(0,1.2,17),azimuthal_degree=3,
        axial_order=3,cell_m=.3,transverse=transverse,stencil_center=centre,
        time_step_ns=.25,time_deposit=time_rule,frame=AxisFrame.of(f0),
        deposit_method=method,min_group=2)
    full=compile_axial_slices([f0,f2],**kwargs)
    points=full[0].points_m()
    selected=points[:,2]>np.quantile(points[:,2],.7)
    # Select a few isolated stencil points, too; retaining a cell may require
    # all its moments even when its central point is not requested.
    selected[::13]=True
    sparse=compile_axial_slices([f0,f2],point_selection=selected,**kwargs)
    late=compile_axial_slices([f0,f2],point_selection=selected,prune_selected=False,**kwargs)
    for complete,actual,control in zip(full,sparse,late):
        np.testing.assert_array_equal(actual.points_m(),complete.points_m()[selected])
        np.testing.assert_allclose(actual.channels,complete.channels[selected],rtol=1e-11,atol=1e-11)
        np.testing.assert_array_equal(control.channels,complete.channels[selected])
        assert actual.reference_ns==complete.reference_ns
        assert actual.summary['time_rows']<complete.summary['time_rows']
        assert sum(actual.summary['angular_elements_per_node'])<=sum(complete.summary['angular_elements_per_node'])


def test_empty_selection_allocates_no_time_or_angular_deposit():
    f=lh.SyntheticShower.gaussian(lh.SourcePose([0.,0.,0.],[0.,0.,1.],0.),
        charged_track_length_m=2.,elements=20,seed=2).elements.field(0)
    options=dict(degree=4,omega_per_ns=np.linspace(0,1,5),transverse='stencil')
    full=compile_axial_slices(f,**options)
    empty=compile_axial_slices(f,point_selection=np.zeros(len(full.z_m),bool),**options)
    assert empty.channels.shape==(0,full.channels.shape[1],5)
    assert empty.summary['time_rows']==0
    assert not any(empty.summary['angular_elements_per_node'])


def test_zero_pass_and_fixed_geometry_keep_the_same_sparse_layout():
    f=lh.SyntheticShower.gaussian(lh.SourcePose([0.,0.,0.],[.1,.2,.97],0.),
        charged_track_length_m=2.,elements=50,seed=2).elements.field(0)
    frame=AxisFrame.of(f)
    options=dict(degree=6,azimuthal_degree=2,cell_m=.3,frame=frame,
        stencil_center='centroid',geometry_reference=f,time_deposit='cubic',time_step_ns=.5)
    zero=compile_axial_slices(f,omega_per_ns=[0.],**options)
    mask=zero.z_m>np.median(zero.z_m)
    spectral=compile_axial_slices(f,omega_per_ns=np.linspace(0,1,5),point_selection=mask,**options)
    np.testing.assert_array_equal(spectral.points_m(),zero.points_m()[mask])
    np.testing.assert_allclose(spectral.channels[:,:,0],zero.channels[mask,:,0],rtol=1e-11,atol=1e-11)


def test_fixed_geometry_makes_disjoint_source_responses_additive():
    f=lh.SyntheticShower.gaussian(lh.SourcePose([0.,0.,0.],[.1,.2,.97],0.),
        charged_track_length_m=5.,elements=100,seed=5).elements.field(0)
    options=dict(degree=6,azimuthal_degree=2,omega_per_ns=np.linspace(0,1.2,17),
        frame=AxisFrame.of(f),geometry_reference=f,stencil_center='centroid',cell_m=1.,
        time_step_ns=.5,time_deposit='cubic',axial_clock_ns_per_m=1/.299792458)
    full=compile_axial_slices(f,**options);selected=full.z_m>np.median(full.z_m)
    fields=[]
    for mask in [np.arange(len(f))<len(f)//2,np.arange(len(f))>=len(f)//2]:
        part=replace(f,**{name:getattr(f,name)[mask] for name in
            ['start_m','direction','length_m','photons','cone_cosine','start_ns','end_ns','row_index','uid']})
        fields.append(compile_axial_slices(part,point_selection=selected,**options))
    np.testing.assert_array_equal(fields[0].points_m(),fields[1].points_m())
    np.testing.assert_allclose(fields[0].channels+fields[1].channels,full.channels[selected],rtol=1e-10,atol=1e-10)
