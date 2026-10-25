"""First-entry exterior transport and compatibility with fast multipoles."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from lighthit.cache import CacheGrid
from lighthit.directional import DirectionalCache, directional_response
from lighthit.green import SolverSettings
from lighthit.medium import synthetic_medium
from lighthit.receiver import SphereSurface
from lighthit.receiver_adjoint import (SphereTransportConfig, first_entry_cache,
    _operators, _ballistic_moments)
from lighthit.receiver_first import first_entry_profile


@pytest.mark.parametrize('lam',[0,1,2,3])
def test_free_ray_moments_obey_streaming_and_orientation_rotation(lam):
    modes,streaming,angular=_operators(lam,8)
    r=np.array([1.,3.,30.]);step=r*1e-5;d=.12-.07j
    h=_ballistic_moments(r,.216,d,lam,modes)
    derivative=(_ballistic_moments(r+step,.216,d,lam,modes)
                -_ballistic_moments(r-step,.216,d,lam,modes))/(2*step[:,None])
    # The free streaming equation follows from actual cap rays. Low modes
    # do not meet the finite Galerkin boundary, so they must satisfy it
    # independently of the closure used for the highest modes.
    residual=derivative@streaming.T+h@angular.T/r[:,None]-d*h@(streaming+np.eye(len(modes))).T
    low=modes[:,1]<8
    assert np.max(abs(residual[:,low])/abs(h).max(axis=1)[:,None])<2e-8


def _isotropic_first_entry_q1(medium,r,R):
    x,w=np.polynomial.legendre.leggauss(16)
    gx,gw=np.polynomial.legendre.leggauss(6)
    cap=np.arcsin(R/r)
    cuts=np.unique([0.,cap,2*cap,.3,1.,np.pi]);result=0.
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        for theta,weight in zip((lo+hi)/2+(hi-lo)/2*x,(hi-lo)/2*w):
            profile=first_entry_profile(SphereSurface(R),distance_m=r,
                direction_cosine=np.cos(theta),g=medium.g,
                max_path_m=r-np.log(1e-12)/medium.extinction_per_m,
                path_nodes=96,path_order=6,scatter_order=16)
            curve=PchipInterpolator(profile.path_m,profile.cumulative_m_inv[:,0])
            half=np.diff(profile.path_m)/2
            path=(profile.path_m[1:]+profile.path_m[:-1])[:,None]/2+half[:,None]*gx
            value=np.sum(half[:,None]*gw*np.exp(-medium.extinction_per_m*path)*curve.derivative()(path))
            result+=weight*np.sin(theta)/2*value
    return result*medium.scattering_per_m


def test_stationary_isotropic_q1_matches_independent_first_flight_profiles():
    medium=replace(synthetic_medium(),g=0.)
    grid=CacheGrid(np.array([3.,30.]),np.array([0.]))
    cache=first_entry_cache(medium,SolverSettings(8,8,4.,.1,6),grid,SphereSurface(.216),0,
        config=SphereTransportConfig(angular_pairs=24,radial_nodes=500))
    expected=np.array([_isotropic_first_entry_q1(medium,r,.216) for r in grid.radii_m])
    # Unit isotropic source and unit isotropic acceptance contract t_000
    # directly: the two Y00 factors cancel alpha0=4*pi.
    np.testing.assert_allclose(cache.coefficients[0,:,0,0,0].real,expected,rtol=.006)
    assert np.all(cache.coefficients[0,:,0,0,1].real>0)


def test_zero_scattering_and_persistence_keep_the_receiver_contract(tmp_path):
    cache=first_entry_cache(replace(synthetic_medium(),scattering_per_m=0.),
        SolverSettings(4,4,2.,.1,4),CacheGrid(np.array([2.,5.]),np.array([0.,.1])),
        SphereSurface([.2,.4]),2,receiver=1,
        config=SphereTransportConfig(angular_pairs=4,radial_nodes=32))
    assert not np.any(cache.coefficients)
    path=tmp_path/'sphere.npz';cache.save(path);loaded=DirectionalCache.load(path)
    assert loaded.receiver_contract==cache.receiver_contract
    assert loaded.receiver_contract['radius_m']==.4
    assert loaded.receiver_contract['schema']=='lighthit/first-entry-sphere/1'
    assert loaded.receiver_contract['convergence_verified'] is False


def test_new_multipoles_use_the_existing_fused_apply_without_a_geometry_loop():
    pytest.importorskip('numba')
    from lighthit._directional_numba import PreparedDirectionalKernel
    from lighthit.experimental.axial_source import AxialSource,AxisFrame,channel_index
    from lighthit.experimental.event_moments import real_spherical_harmonics
    degree=4;omega=np.array([0.,.03]);medium=replace(synthetic_medium(),g=0.)
    cache=first_entry_cache(medium,SolverSettings(4,degree,2.,.1,4),
        CacheGrid(np.array([3.,5.,8.]),omega),SphereSurface(.216),2,
        config=SphereTransportConfig(angular_pairs=8,radial_nodes=80))
    frame=AxisFrame(np.zeros(3),np.array([0.,0.,1.]),np.array([1.,0.,0.]),np.array([0.,1.,0.]))
    kept,channel_degree=channel_index(degree,degree)
    u=np.array([[.6,0.,.8]])
    channels=np.broadcast_to(real_spherical_harmonics(degree,u)[:,:,None],(1,len(kept),len(omega))).astype(complex)
    source=AxialSource(frame,np.zeros(1),np.zeros((1,2)),channels,kept,channel_degree,degree,degree,0.,{})
    positions=np.array([[0.,0.,5.],[4.,0.,3.]])
    looks=np.array([[0.,0.,1.],[.6,.8,0.]])
    alpha=np.array([4*np.pi,.3,.1])
    reference=directional_response(cache,source,positions,looks,alpha,source_omega_per_ns=omega)
    fast=PreparedDirectionalKernel.from_cache(cache).apply(source,positions,looks,alpha,source_omega_per_ns=omega)
    np.testing.assert_allclose(fast,reference,rtol=1e-11,atol=1e-15)
    assert fast.shape==(len(omega),len(positions),2)


@pytest.mark.parametrize('options',[dict(angular_pairs=True),dict(angular_pairs=1),
    dict(radial_nodes=8),dict(ballistic_quadrature=2),dict(exterior_relative_bound=0.),
    dict(exterior_relative_bound=np.nan)])
def test_invalid_resolutions_are_rejected(options):
    with pytest.raises(ValueError):SphereTransportConfig(**options)


def test_source_inside_glass_and_insufficient_output_degree_are_rejected():
    args=(synthetic_medium(),SolverSettings(4,8,2.,.1,4),
          CacheGrid(np.array([.1,3.]),np.array([0.])),SphereSurface(.216),0)
    with pytest.raises(ValueError,match='outside'):first_entry_cache(*args)
    args=args[:2]+(CacheGrid(np.array([1.,3.]),np.array([0.])),)+args[3:]
    with pytest.raises(ValueError,match='source degree'):
        first_entry_cache(*args,config=SphereTransportConfig(angular_pairs=4))


def test_centre_cache_directory_scan_cannot_adopt_surface_multipoles():
    from test_ballistic_finite_receiver import setup
    kernel=setup([0.,0.,4.])
    kernel.config=replace(kernel.config,source_degree=4,scattering_degree=4,
        radial_range_m=(3.,5.),radial_nodes=4)
    grid=CacheGrid.geometric(3.,5.,4,kernel.config.omega_per_ns)
    wavelength=float(kernel.wavelength.wavelength_nm[0])
    cache=first_entry_cache(kernel.medium.band(wavelength),kernel._solver_settings(),grid,
        kernel.detector.receiver_surface,0,
        config=SphereTransportConfig(angular_pairs=4,radial_nodes=32))
    metadata=cache._metadata()
    assert not kernel._table_matches('directional',wavelength,metadata,grid.radii_m,grid.omega_per_ns)
    metadata=dict(metadata,receiver_contract={})
    assert kernel._table_matches('directional',wavelength,metadata,grid.radii_m,grid.omega_per_ns)


@pytest.mark.parametrize('frequency',[0.,.1])
def test_bounded_collision_series_matches_full_exterior_resummation(frequency):
    from lighthit.receiver_adjoint import _solve
    args=dict(pairs=8,points=100,high=10.,omega=frequency,tail=1e-9)
    direct=_solve(synthetic_medium(),.216,0,**args,collision_solver='direct')[2]
    diagnostics={}
    series=_solve(synthetic_medium(),.216,0,**args,collision_solver='series',
        collision_relative_bound=1e-10,diagnostics=diagnostics)[2]
    np.testing.assert_allclose(series,direct,rtol=1e-9,atol=1e-14)
    assert diagnostics['first_omitted_collision']>2
