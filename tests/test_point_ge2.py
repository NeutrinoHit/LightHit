"""Separable real angular sources agree with ordinary complex-field transport."""
import numpy as np
import pytest
from lighthit.experimental.point_ge2 import apply_point_spectral
from lighthit.experimental.axial_source import AxisFrame,AxialSource
from lighthit.experimental.axial_fast import _layout
from lighthit._shower_basis import apply_multi_spectral


@pytest.mark.parametrize('degree,m,detector_degree',[(4,0,2),(5,2,3)])
@pytest.mark.parametrize('flight',[False,True])
def test_point_factorisation_matches_complex_source(degree,m,detector_degree,flight):
    rng=np.random.default_rng(5);nlam=2;omega=np.linspace(0,1.2,9);logr=np.linspace(np.log(3),np.log(300),6)
    pairs=[(l,p) for l in range(detector_degree+1) for p in range(l+1)]
    co=rng.normal(size=(nlam,5,degree+1,len(pairs),4,len(omega)))+1j*rng.normal(size=(nlam,5,degree+1,len(pairs),4,len(omega)))
    bundle=dict(degree=degree,acceptance_degree=detector_degree,omega=omega,log_r=logr,
        pair_lambda=np.array([x[0] for x in pairs]),pair_mu=np.array([x[1] for x in pairs]),
        coefficients=co,absorption=np.array([.04,.06]),speed=np.array([.22,.215]),phase='flight' if flight else 'absolute')
    rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(rotation)<0:rotation[:,0]*=-1
    frame=AxisFrame(np.array([3.,5.,7.]),rotation[:,2],rotation[:,0],rotation[:,1])
    points=np.array([[1.,2.,3.],[2.,4.,6.],[-3.,4.,10.],[2.,1.,2000.]])
    receivers=np.array([[40.,1.,20.],[-20.,30.,-40.]])
    looks=np.tile([0.,0.,1.],(2,1));times=np.array([1.,30.,5000.,8000.])
    starts,aa,bb,kept,channel_degree=_layout(degree,m)
    angular=rng.normal(size=(nlam,len(points),len(kept)));angular[1,1]=0
    scale=rng.uniform(.1,2,size=(nlam,2));alpha=rng.uniform(.2,1,size=detector_degree+1)
    local=frame.rotate(points-frame.centre_m);reference=times.min();sources=[]
    for values in angular:
        channels=values[:,:,None]*np.exp(1j*(times-reference)[None,:]*omega[:,None]).T[:,None,:]
        sources.append(AxialSource(frame,local[:,2],local[:,:2],channels,kept,channel_degree,degree,m,reference,{}))
    expected=apply_multi_spectral(bundle,sources,receivers,looks,alpha,scale,channel_dtype=np.complex128,radial_window_m=(3.,300.))
    actual=apply_point_spectral(bundle,frame,points,angular,times,receivers,looks,alpha,scale,azimuthal_degree=m)
    np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12*abs(expected).max())
    # Changing the global source clock may only multiply the Fourier spectrum
    # by its exact phase. A 1 ms offset exceeds the cached delay period.
    shifted=apply_point_spectral(bundle,frame,points,angular,times+1e6,receivers,looks,alpha,scale,azimuthal_degree=m)
    np.testing.assert_allclose(shifted,actual*np.exp(1j*omega*1e6)[:,None],
                               rtol=3e-9,atol=3e-9*abs(actual).max())
    with pytest.raises(ValueError,match='must be real'):
        apply_point_spectral(bundle,frame,points,angular+0j,times,receivers,looks,alpha,scale,azimuthal_degree=m)
    with pytest.raises(ValueError,match='three coordinates'):
        apply_point_spectral(bundle,frame,points,angular,times,receivers[:,:2],looks[:,:2],alpha,scale,azimuthal_degree=m)
    with pytest.raises(ValueError,match='one-dimensional'):
        apply_point_spectral(bundle,frame,points,angular,times[:,None],receivers,looks,alpha,scale,azimuthal_degree=m)
