"""Guards for the connected primary q1 approximation, independent of G4 files."""
from dataclasses import replace
import numpy as np
import pytest
from lighthit import SpectralLightElements
from lighthit.experimental.primary_polyline import PrimaryPolylineConfig, fit_primary_polyline


def trajectory(turns=None, beta=None, count=64):
    turns=np.zeros(count) if turns is None else np.asarray(turns)
    beta=np.ones(count) if beta is None else np.asarray(beta)
    directions=np.column_stack((np.sin(turns),np.zeros(count),np.cos(turns)))
    start=np.vstack((np.zeros(3),np.cumsum(directions,axis=0)[:-1]))
    times=np.r_[0.,np.cumsum(1./(.299792458*beta))]
    return SpectralLightElements(start,directions,np.ones(count),np.ones(count),-1./beta**2,
        beta,times[:-1],times[1:],np.arange(count),np.zeros(count),cone_model='spectral')


def test_straight_track_is_one_chord_and_preserves_clock_and_spectrum():
    source=trajectory(); fit=fit_primary_polyline(source)
    assert len(fit.elements)==1
    np.testing.assert_array_equal(fit.ranges,[[0,len(source)]])
    assert fit.elements.start_ns[0]==source.start_ns[0]
    assert fit.elements.end_ns[0]==source.end_ns[-1]
    np.testing.assert_allclose(fit.elements.photon_density_per_nm([350,480,610],[1.35,1.34,1.33]),
                               source.photon_density_per_nm([350,480,610],[1.35,1.34,1.33]).sum(axis=0,keepdims=True))


def test_curved_chords_share_endpoints_and_cover_each_original_step_once():
    source=trajectory(np.linspace(0,.05,64)); fit=fit_primary_polyline(source)
    assert 1<len(fit.elements)<len(source)
    np.testing.assert_array_equal(fit.ranges[1:,0],fit.ranges[:-1,1])
    assert fit.ranges[0,0]==0 and fit.ranges[-1,1]==len(source)
    np.testing.assert_allclose(fit.elements.start_m[1:],fit.elements.start_m[:-1]+
        fit.elements.length_m[:-1,None]*fit.elements.direction[:-1],atol=1e-12)
    np.testing.assert_array_equal(fit.elements.start_ns[1:],fit.elements.end_ns[:-1])
    for j,(lo,hi) in enumerate(fit.ranges):
        assert fit.elements.coefficient0[j]==source.coefficient0[lo:hi].sum()
        assert fit.elements.coefficient2[j]==source.coefficient2[lo:hi].sum()


@pytest.mark.parametrize('gap_kind',['space','time'])
def test_dark_gap_is_preserved_even_below_geometry_tolerance(gap_kind):
    source=trajectory(); boundary=32
    if gap_kind=='space':
        pos=source.start_m.copy(); pos[boundary:,0]+=1e-5
        source=replace(source,start_m=pos)
    else:
        t0=source.start_ns.copy();t1=source.end_ns.copy()
        t0[boundary:]+=1e-5;t1[boundary:]+=1e-5
        source=replace(source,start_ns=t0,end_ns=t1)
    fit=fit_primary_polyline(source)
    np.testing.assert_array_equal(fit.ranges,[[0,boundary],[boundary,64]])
    assert fit.diagnostics['disconnected_runs']==2


def test_stopping_tail_is_refined_locally_without_energy_switch():
    beta=np.r_[np.ones(48),np.linspace(.999,.76,16)]
    source=trajectory(beta=beta)
    fit=fit_primary_polyline(source,phase_indices=[1.33,1.35])
    early=fit.ranges[fit.ranges[:,1]<=48]
    late=fit.ranges[fit.ranges[:,0]>=48]
    assert len(early)<4 and len(late)==16
    for lo,hi in fit.ranges:
        assert hi-lo==1 or np.ptp(beta[lo:hi])<=PrimaryPolylineConfig().beta_tolerance


def test_threshold_crossing_forces_refinement_even_with_loose_beta_limit():
    source=trajectory(beta=np.r_[np.full(32,.74),np.full(32,.76)])
    cfg=PrimaryPolylineConfig(beta_tolerance=.1,time_tolerance_ns=100.)
    fit=fit_primary_polyline(source,config=cfg,phase_indices=[1.34])
    assert all(not (lo<32<hi) for lo,hi in fit.ranges)


def test_inconsistent_spectral_beta_fields_are_not_silently_merged():
    source=replace(trajectory(),coefficient2=np.full(64,-2.))
    fit=fit_primary_polyline(source)
    assert len(fit.elements)==len(source)
    np.testing.assert_array_equal(fit.elements.beta,source.beta)


def test_round_trip_loop_does_not_create_zero_length_chord():
    source=trajectory(np.r_[np.zeros(32),np.full(32,np.pi)])
    fit=fit_primary_polyline(source)
    assert len(fit.elements)==2
    assert np.all(fit.elements.length_m>0)


@pytest.mark.parametrize('value',[0,-1,np.nan,np.inf])
def test_invalid_tolerances_are_rejected(value):
    with pytest.raises(ValueError):
        PrimaryPolylineConfig(position_tolerance_m=value)


def test_multiple_uids_and_bad_chronology_are_rejected():
    source=trajectory()
    with pytest.raises(ValueError,match='UID'):
        fit_primary_polyline(replace(source,uid=np.arange(64)))
    with pytest.raises(ValueError,match='times'):
        fit_primary_polyline(replace(source,end_ns=source.start_ns-1))


def test_polyline_route_rejects_point_or_automatic_quadrature():
    from types import SimpleNamespace
    from lighthit.experimental.primary_polyline import order1_primary_polyline
    with pytest.raises(ValueError,match='segment integration'):
        order1_primary_polyline(None,trajectory(),time_origin_ns=[],
            prompt_config=SimpleNamespace(element_method='grouped'))


def test_composite_spectrum_integrates_known_piecewise_quadratic_exactly():
    from types import SimpleNamespace
    from lighthit.experimental.primary_polyline import primary_spectral_quadrature
    medium=SimpleNamespace(wavelength_range_nm=(350.,610.),wavelength_nm=np.array([350.,400.,610.]))
    def efficiency(w):
        return np.maximum(w-450.,0.)**2
    efficiency.wavelength_knots_nm=np.array([300.,450.,650.])
    detector=SimpleNamespace(spectral_efficiency=efficiency)
    rule=primary_spectral_quadrature(medium,detector,order=2)
    np.testing.assert_allclose(rule.weight_nm.sum(),260.)
    np.testing.assert_allclose(rule.weight_nm@efficiency(rule.wavelength_nm),160.**3/3,rtol=1e-14)
    assert np.all(rule.weight_nm>0)
    assert np.all((rule.wavelength_nm>350)&(rule.wavelength_nm<610))


def test_composite_spectrum_requires_known_callable_structure():
    from types import SimpleNamespace
    from lighthit.experimental.primary_polyline import primary_spectral_quadrature
    medium=SimpleNamespace(wavelength_range_nm=(350.,610.),wavelength_nm=np.array([350.,610.]))
    detector=SimpleNamespace(spectral_efficiency=lambda x:np.ones_like(x))
    with pytest.raises(ValueError,match='known efficiency'):
        primary_spectral_quadrature(medium,detector)
    rule=primary_spectral_quadrature(medium,detector,efficiency_knots_nm=[])
    assert len(rule.weight_nm)==2
