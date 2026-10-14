"""Retarded-time inversion, exact polynomial phases and mapping lifetime."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.integrate import quad
from lighthit.experimental.filon_ge2 import retarded_inverse,wavelength_plane,backward_ge2_filon_many


@pytest.mark.parametrize('clock',[3.33564095198152,1/.22,6.])
def test_backward_inverse_including_unit_slowness_ratio(clock):
    for rho in [3.01,40.,299.]:
        for speed in [.205,.22]:
            u=np.array([-500.,-50.,-1.,-1e-5,0.])
            tau=np.hypot(rho,u)/speed-clock*u
            np.testing.assert_allclose(retarded_inverse(tau,rho,speed,clock),u,rtol=2e-13,atol=2e-11)


def test_one_plane_mapping_is_exact_readonly_and_closed(tmp_path):
    data=(np.arange(48).reshape(2,3,8)+1j).astype(np.complex64)
    path=tmp_path/'coefficients.npy';np.save(path,data)
    full=np.load(path,mmap_mode='r')
    with wavelength_plane(full,1) as plane:
        np.testing.assert_array_equal(plane,data[1:2]);mapping=plane._mmap
        assert not plane.flags.writeable
    assert mapping.closed
    np.testing.assert_array_equal(full,data)
    # A noncontiguous view cannot be remapped as a contiguous plane.
    with wavelength_plane(full[...,::2],0) as plane:
        np.testing.assert_array_equal(plane,data[0:1,:,::2])
    assert not full._mmap.closed


@pytest.mark.parametrize('mixed',[False,True])
def test_filon_exact_for_polynomial_amplitude_with_real_spectral_flight_clocks(monkeypatch,mixed):
    import lighthit.experimental.track_ge2_primitive as primitive
    omega=np.array([0.,.1,.4]);speed=np.array([.21,.225]);clock=3.4;rho=4.2
    waves=np.array([400.,500.]);absorption=np.array([.03,.04])
    kernel=SimpleNamespace(wavelength=SimpleNamespace(wavelength_nm=waves,weight_nm=np.ones(2)),
        medium=SimpleNamespace(sample=lambda w:dict(absorption_per_m=absorption,phase_index=np.array([1.35,1.34]))),
        detector=SimpleNamespace(spectral_weight=lambda w:np.ones_like(w),effective_area_m2=np.array([.05])),
        acceptance=lambda:dict(alpha=np.array([4*np.pi])))
    bundle=dict(phase='flight',degree=2,acceptance_degree=1,omega=omega,speed=speed,absorption=absorption,
        log_r=np.log([3.,10.,50.]),coefficients=np.zeros((2,2,3,3,4,3),np.complex64))
    def amplitude(local,**kwargs):
        velocity=local['speed'][0]
        def evaluate(u):
            radius=np.hypot(rho,u);tau=radius/velocity-clock*u
            jacobian_inverse=clock-u/(velocity*radius)
            value=(1+.03*tau+.0001*tau*tau)*jacobian_inverse
            return np.broadcast_to(value[:,None],(len(u),len(omega))).astype(complex)
        return evaluate
    monkeypatch.setattr(primitive,'cone_point_evaluator',amplitude)
    lo,hi,density,closest=(-1.,0.,4.,7.) if mixed else (-10.,-2.,4.,7.)
    query=(rho,np.array([0.,0.,1.]),.99,clock,lo,hi,density,closest)
    result,meta=backward_ge2_filon_many(kernel,bundle,[query],order=4,allow_gauss=mixed)
    reference=np.zeros(3,complex)
    for v in speed:
        low=np.hypot(rho,hi)/v-clock*hi;high=np.hypot(rho,lo)/v-clock*lo
        for j,w in enumerate(omega):
            f=lambda t:(1+.03*t+.0001*t*t)*np.exp(1j*w*t)
            reference[j]+=quad(lambda t:f(t).real,low,high,epsabs=1e-10)[0]+1j*quad(lambda t:f(t).imag,low,high,epsabs=1e-10)[0]
    reference*=density*np.exp(1j*closest*omega)
    np.testing.assert_allclose(result[0],reference,rtol=2e-12,atol=2e-10)
    assert meta['clock_actual'] and meta['beta_actual']
    assert meta['gauss_queries']==int(mixed)
