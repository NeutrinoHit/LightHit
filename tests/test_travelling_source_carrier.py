"""A travelling source should interpolate the slowly varying adjoint."""
from dataclasses import replace
import numpy as np
import lighthit as lh
from lighthit._shower_basis import compile_axial_slices
from lighthit.experimental.axial_source import AxisFrame


def source():
    f=lh.SyntheticShower.gaussian(lh.SourcePose([0.,0.,0.],[.1,.2,.97],0.),
        charged_track_length_m=10.,elements=60,seed=4).elements.field(0)
    frame=AxisFrame.of(f);slope=1/.299792458
    start=frame.rotate(f.start_m-frame.centre_m)[:,2]
    end=frame.rotate(f.start_m+f.length_m[:,None]*f.direction-frame.centre_m)[:,2]
    return replace(f,start_ns=10.+slope*start,end_ns=10.+slope*end),frame,slope


def test_travelling_plane_adjoint_is_exact_on_coarse_cubic_cells():
    f,frame,slope=source();omega=np.linspace(0,1.2,17)
    options=dict(degree=0,azimuthal_degree=0,omega_per_ns=omega,cell_m=2.,axial_order=3,
        transverse='collapse',element_order=2,frame=frame,time_deposit='cubic',time_step_ns=.5)
    result=compile_axial_slices(f,axial_clock_ns_per_m=slope,**options)
    def amplitude(z):return 1.+.05*z+.001*z*z+1e-5*z**3
    adjoint=amplitude(result.z_m)[:,None]*np.exp(-1j*slope*result.z_m[:,None]*omega)
    actual=(result.channels[:,0]*adjoint).sum(0)*np.exp(1j*omega*result.reference_ns)
    exact=np.zeros(len(omega),complex)
    x,w=np.polynomial.legendre.leggauss(2)
    for fraction,weight in zip((x+1)/2,w/2):
        z,_,_,t=frame.coordinates(f,fraction)
        exact+=((f.photons*weight*amplitude(z)/np.sqrt(4*np.pi))[:,None]*
            np.exp(1j*(t-slope*z)[:,None]*omega)).sum(0)
    np.testing.assert_allclose(actual,exact,rtol=1e-11,atol=1e-11)
    original=compile_axial_slices(f,**options)
    uncorrected=(original.channels[:,0]*adjoint).sum(0)*np.exp(1j*omega*original.reference_ns)
    assert np.max(abs(uncorrected-exact))/np.max(abs(exact))>.01


def test_travelling_carrier_preserves_stationary_source_and_sparse_selection():
    f,frame,slope=source();options=dict(degree=6,azimuthal_degree=2,cell_m=1.,frame=frame,
        time_deposit='cubic',time_step_ns=.5,transverse='stencil')
    zero=compile_axial_slices(f,omega_per_ns=[0.],**options)
    full=compile_axial_slices(f,omega_per_ns=np.linspace(0,1.2,17),axial_clock_ns_per_m=slope,**options)
    np.testing.assert_array_equal(full.points_m(),zero.points_m())
    np.testing.assert_allclose(full.channels[:,:,0],zero.channels[:,:,0],rtol=1e-10,atol=1e-10)
    selected=full.z_m>np.median(full.z_m)
    sparse=compile_axial_slices(f,omega_per_ns=np.linspace(0,1.2,17),axial_clock_ns_per_m=slope,
        point_selection=selected,**options)
    np.testing.assert_allclose(sparse.channels,full.channels[selected],rtol=1e-11,atol=1e-11)
