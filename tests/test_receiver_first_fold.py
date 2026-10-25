"""Finite-glass clocks, spectral units, tilted acceptance and omitted tails."""
import numpy as np
import pytest
from lighthit.receiver import SphereSurface
from lighthit.receiver_first import first_entry_profile,MOMENT_POWERS
from lighthit.receiver_first_fold import fold_first_entry,acceptance_moment_weights


def test_cubic_tilted_acceptance_includes_even_transverse_moments():
    x,w=np.polynomial.legendre.leggauss(20)
    phi=(np.arange(40)+.5)*2*np.pi/40
    dx=np.sqrt(1-x[:,None]**2)*np.cos(phi)
    dy=np.sqrt(1-x[:,None]**2)*np.sin(phi)
    dz=np.broadcast_to(x[:,None],dx.shape)
    weight=w[:,None]*np.exp(.2*dx-.3*dz)/len(phi)
    moments=np.array([np.sum(weight*dx**i*dz**j) for i,j in MOMENT_POWERS])
    look=np.array([.3,.4,np.sqrt(.75)]);poly=np.array([1.,.2,.1,-.05])
    actual=moments@acceptance_moment_weights(look,poly)
    expected=np.sum(weight*np.polynomial.polynomial.polyval(look[0]*dx+look[1]*dy+look[2]*dz,poly))
    assert actual==pytest.approx(expected,rel=2e-14)


@pytest.fixture(scope='module')
def profile():
    return first_entry_profile(SphereSurface(1.),distance_m=6.,direction_cosine=np.cos(.7),
        g=.5,max_path_m=80.,path_nodes=128,path_order=8,scatter_order=16)


def test_spectral_clock_has_glass_front_and_preserves_all_time_charge(profile):
    times=np.array([0.,4.99/.25,5.5/.25,6./.25,80./.2,500.])
    args=dict(extinction_per_m=[.12,.15],scattering_per_m=[.03,.05],
        speed_m_per_ns=[.25,.2],spectral_weights=[[.4,.2],[.7,.3]],time_edges_ns=times)
    folded=fold_first_entry(profile,**args)
    np.testing.assert_array_equal(folded.cumulative[:2],0.)
    assert folded.cumulative[2,0,0]>0  # Actual entry, before the centre clock.
    np.testing.assert_allclose(folded.cumulative[-1],folded.charge,rtol=1e-14,atol=1e-20)
    assert np.all(np.diff(folded.cumulative[:,:,0],axis=0)>=-1e-18)
    refined=fold_first_entry(profile,**args,quadrature_order=16)
    np.testing.assert_allclose(folded.charge,refined.charge,rtol=1e-10,atol=1e-15)
    # Wavelength contributions use their own speeds and extinction once.
    separate=[]
    for band in range(2):
        part={key:([value[band]] if key!='time_edges_ns' else value) for key,value in args.items()}
        separate.append(fold_first_entry(profile,**part))
    np.testing.assert_allclose(folded.cumulative,sum(p.cumulative for p in separate),rtol=1e-14,atol=1e-20)


def test_finite_extent_reports_a_bound_instead_of_declaring_zero_tail():
    rows=[first_entry_profile(SphereSurface(.5),distance_m=3.,direction_cosine=.3,
        g=0.,max_path_m=extent,path_nodes=128) for extent in (5.,30.)]
    options=dict(extinction_per_m=[.5],scattering_per_m=[.1],speed_m_per_ns=[.22],
        spectral_weights=[[1.,.5]])
    short,long=[fold_first_entry(p,**options) for p in rows]
    assert np.all(short.tail_bound>0)
    assert np.all(long.tail_bound<short.tail_bound)
    assert np.all(abs(long.charge-short.charge)<=short.tail_bound[:,None])


def test_hitting_pencil_has_no_first_scattering_after_the_tangent_path():
    p=first_entry_profile(SphereSurface(1.),distance_m=6.,direction_cosine=1.,g=.5,
        max_path_m=10.,path_nodes=128)
    f=fold_first_entry(p,extinction_per_m=[.12],scattering_per_m=[.03],
        speed_m_per_ns=[.25],spectral_weights=[[1.,1.]])
    np.testing.assert_array_equal(f.tail_bound,0.)


@pytest.mark.parametrize('look,poly',[(np.zeros(3),[1.]),([0.,0.,1.],[1.]*5),
    ([0.,0.,1.],[np.nan])])
def test_invalid_acceptance_contract_is_rejected(look,poly):
    with pytest.raises(ValueError):acceptance_moment_weights(look,poly)


def test_scaled_grazing_mesh_keeps_strict_physical_paths_and_cumulative_mass():
    options=dict(distance_m=.2188797502843352,direction_cosine=.16168004567225175,g=.9,
                 max_path_m=.035388488022702384,path_order=8,scatter_order=16)
    profiles=[first_entry_profile(SphereSurface(.216),**options,path_nodes=n) for n in (256,512)]
    for profile in profiles:
        assert np.all(np.diff(profile.path_m)>0)
        assert np.isfinite(profile.cumulative_m_inv).all()
        f=fold_first_entry(profile,extinction_per_m=[.1],scattering_per_m=[.02],
            speed_m_per_ns=[.22],spectral_weights=[[1.,.5]])
        assert f.charge[0,0]>0
    np.testing.assert_allclose(profiles[0].cumulative_m_inv[-1],profiles[1].cumulative_m_inv[-1],rtol=1e-4,atol=1e-7)
