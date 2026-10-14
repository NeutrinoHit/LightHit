"""Fourth-order source-clock approximation against exact Fourier phases."""
from dataclasses import replace
import numpy as np
import pytest
import lighthit as lh
from lighthit._shower_basis import compile_axial_slices
from lighthit.experimental.axial_fast import compile_axial_source_fast
from lighthit.experimental.axial_source import AxisFrame


def field():
    s=lh.SyntheticShower.gaussian(lh.SourcePose([12.,3.,0.],[.3,.2,.93],-123.4),
        charged_track_length_m=20.,elements=100,seed=3).elements.field(0)
    # Include delayed elements; clocks need not follow the shower front.
    return replace(s,start_ns=s.start_ns+np.arange(len(s))*.273,
        end_ns=s.end_ns+np.arange(len(s))*.273)


@pytest.mark.parametrize('step',[.25,.5,1.])
def test_cubic_phase_error_and_charge_conservation(step):
    f=field();frame=AxisFrame.of(f);omega=np.linspace(0,1.2,13)
    exact=compile_axial_source_fast(f,6,omega,azimuthal_degree=2,cell_m=.3,
        element_order=2,frame=frame,axial_only=True)
    approx=compile_axial_slices(f,6,omega,azimuthal_degree=2,cell_m=.3,axial_order=1,
        transverse='collapse',element_order=2,time_step_ns=step,time_deposit='cubic',frame=frame)
    np.testing.assert_allclose(approx.z_m,exact.z_m,rtol=0,atol=1e-12)
    a=approx.channels*np.exp(1j*omega*approx.reference_ns)[None,None,:]
    b=exact.channels*np.exp(1j*omega*exact.reference_ns)[None,None,:]
    # The positive l=0 coefficient has no harmonic cancellation: the sum of
    # per-point interpolation bounds is an actual absolute-error bound.
    mass=b[:,0,0].real.sum()
    error=abs(a[:,0]-b[:,0]).sum(0)
    assert np.all(error<=3*(omega*step)**4/128*mass+1e-11*mass)
    np.testing.assert_allclose(a[:,:,0],b[:,:,0],rtol=1e-10,atol=1e-11*mass)


def test_cubic_clock_translation_and_zero_pass():
    f=field();frame=AxisFrame.of(f);omega=np.linspace(0,1.2,13)
    kwargs=dict(azimuthal_degree=2,cell_m=.3,axial_order=3,transverse='stencil',
        frame=frame,time_step_ns=.5,time_deposit='cubic')
    a=compile_axial_slices(f,6,omega,**kwargs)
    shifted=replace(f,start_ns=f.start_ns+1000.,end_ns=f.end_ns+1000.)
    b=compile_axial_slices(shifted,6,omega,**kwargs)
    np.testing.assert_allclose(a.points_m(),b.points_m(),rtol=0,atol=1e-12)
    np.testing.assert_allclose(a.channels,b.channels,rtol=1e-9,atol=1e-10)
    assert b.reference_ns-a.reference_ns==pytest.approx(1000.)
    zero=compile_axial_slices(f,6,[0.],**kwargs)
    np.testing.assert_allclose(zero.channels[:,:,0],a.channels[:,:,0],rtol=1e-10,atol=1e-10)


def test_basis_accepts_explicit_cubic_and_rejects_unknown_rule():
    assert lh.ShowerBasis(time_step_ns=.5,time_deposit='cubic').time_deposit=='cubic'
    with pytest.raises(ValueError,match='time_deposit'):lh.ShowerBasis(time_deposit='unsupported')
