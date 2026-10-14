"""Local algorithm controls; real G4 checks live in probe_muon_integral.py."""
from dataclasses import replace

import numpy as np
import pytest

import lighthit as lh
from lighthit.experimental.muon_integral import fit_track_integrals, order1_integrals


def primary():
    return lh.CherenkovTrack([0., 0., -20.], [0., 0., 1.], 40.,
                             cone_model="spectral").to_elements(step_m=.1)


def test_fit_preserves_two_spectral_fields_and_contiguous_step_ranges():
    source = primary()
    fitted = fit_track_integrals(source)
    assert len(fitted.elements) < len(source) / 10
    np.testing.assert_allclose(fitted.elements.coefficient0.sum(), source.coefficient0.sum(), rtol=1e-15)
    np.testing.assert_allclose(fitted.elements.coefficient2.sum(), source.coefficient2.sum(), rtol=1e-15)
    assert fitted.ranges[0, 0] == 0
    assert fitted.ranges[-1, 1] == len(source)
    np.testing.assert_array_equal(fitted.ranges[:-1, 1], fitted.ranges[1:, 0])
    assert fitted.maximum_position_error_m < 1e-12
    assert fitted.maximum_time_error_ns < 1e-12
    assert fitted.elements.cone_model == "spectral"


def test_fit_does_not_smear_a_large_yield_jump():
    source = primary()
    c0, c2 = source.coefficient0.copy(), source.coefficient2.copy()
    c0[len(source)//3] *= 10
    c2[len(source)//3] *= 10
    fitted = fit_track_integrals(replace(source, coefficient0=c0, coefficient2=c2))
    assert any(lo == len(source)//3 and hi == lo+1 for lo, hi in fitted.ranges)


def test_fit_requires_one_ordered_primary():
    source = primary()
    with pytest.raises(ValueError, match="one primary"):
        fit_track_integrals(replace(source, uid=np.arange(len(source))))
    with pytest.raises(ValueError, match="chronological"):
        fit_track_integrals(replace(source, start_ns=source.start_ns[::-1]))


def test_fit_preserves_wavelength_threshold_on_each_original_step():
    source = primary().subset(np.arange(len(primary())) < 2)
    phase = 1.35
    beta = np.array([1/phase-1e-5, 1/phase+1e-5])
    source = replace(source, beta=beta, coefficient2=-source.coefficient0/beta**2)
    assert len(fit_track_integrals(source).elements) == 1
    fitted = fit_track_integrals(source, phase_indices=[phase, 1.34])
    assert len(fitted.elements) == 2
    np.testing.assert_array_equal(fitted.elements.beta, beta)
    # The unmerged emitting step must survive, even if effective beta of the
    # combined piece would have put the entire piece below this threshold.
    wavelength = np.array([480.])
    before = source.photon_density_per_nm(wavelength, np.array([phase]))
    after = fitted.elements.photon_density_per_nm(wavelength, np.array([phase]))
    np.testing.assert_array_equal(before, after)
    assert after[1, 0] > 0


def test_fit_checks_every_internal_endpoint_not_only_midpoints():
    angle = 4e-4
    direction = np.column_stack((np.array([1., -1., -1., 1.]) * np.sin(angle),
                                  np.zeros(4), np.full(4, np.cos(angle))))
    length = np.full(4, .25)
    ends = np.cumsum(length[:, None] * direction, axis=0)
    starts = np.vstack((np.zeros((1, 3)), ends[:-1]))
    times = np.arange(5) * .25 / .299792458
    source = lh.SpectralLightElements(
        starts, direction, length, length, -length, np.ones(4),
        times[:-1], times[1:], np.arange(4), np.zeros(4, int), cone_model="spectral")
    tolerance = 7.5e-5
    fitted = fit_track_integrals(source, max_length_m=1., position_tolerance_m=tolerance)
    # One piece would have midpoint errors 5e-5 but endpoint errors 1e-4.
    assert len(fitted.elements) == 2
    largest = 0.
    for piece, (lo, hi) in enumerate(fitted.ranges):
        delta = np.vstack((starts[lo:hi], ends[lo:hi])) - fitted.elements.start_m[piece]
        axis = fitted.elements.direction[piece]
        coordinate = np.clip(delta @ axis, 0., fitted.elements.length_m[piece])
        largest = max(largest, np.linalg.norm(delta - coordinate[:, None] * axis, axis=1).max())
    assert largest <= tolerance
    assert fitted.maximum_position_error_m == pytest.approx(largest, abs=1e-14)


def test_fit_checks_internal_endpoint_times():
    source = primary().subset(np.arange(len(primary())) < 4)
    times = np.array([0., 1., 4., 7., 8.])
    source = replace(source, start_ns=times[:-1], end_ns=times[1:])
    fitted = fit_track_integrals(source, time_tolerance_ns=.75)
    # The single-line midpoint residual is only .5 ns, but the interior
    # endpoints have 1 ns error and must force a split.
    assert len(fitted.elements) > 1
    for piece, (lo, hi) in enumerate(fitted.ranges):
        start = fitted.elements.start_m[piece]
        axis = fitted.elements.direction[piece]
        slope = (fitted.elements.end_ns[piece] - fitted.elements.start_ns[piece]) / fitted.elements.length_m[piece]
        for positions, actual_times in (
            (source.start_m[lo:hi], source.start_ns[lo:hi]),
            (source.start_m[lo:hi] + source.length_m[lo:hi, None] * source.direction[lo:hi], source.end_ns[lo:hi])):
            predicted = fitted.elements.start_ns[piece] + ((positions - start) @ axis) * slope
            assert np.max(np.abs(predicted - actual_times)) <= .75
    assert fitted.maximum_time_error_ns <= .75


@pytest.mark.parametrize("gap", ["position", "time"])
def test_fit_does_not_emit_across_missing_dark_intervals(gap):
    source = primary().subset(np.arange(len(primary())) < 2)
    starts, t0, t1 = source.start_m.copy(), source.start_ns.copy(), source.end_ns.copy()
    if gap == "position":
        starts[1, 2] += .05
        t0[1] += .05 / .299792458
        t1[1] += .05 / .299792458
    else:
        # Smaller than the fit's time tolerance, but this is an actual dark
        # interval, not an allowed interpolation error on a continuous track.
        t0[1] += .01
        t1[1] += .01
    source = replace(source, start_m=starts, start_ns=t0, end_ns=t1)
    fitted = fit_track_integrals(source)
    assert len(fitted.elements) == 2
    for name in ("start_m", "length_m", "start_ns", "end_ns", "coefficient0", "coefficient2"):
        np.testing.assert_array_equal(getattr(fitted.elements, name), getattr(source, name))


def test_fit_allows_roundoff_at_contiguous_step_joins():
    source = primary().subset(np.arange(len(primary())) < 2)
    starts, t0, t1 = source.start_m.copy(), source.start_ns.copy(), source.end_ns.copy()
    starts[1, 2] = np.nextafter(starts[1, 2], np.inf)
    t0[1] = np.nextafter(t0[1], np.inf)
    t1[1] = np.nextafter(t1[1], np.inf)
    fitted = fit_track_integrals(replace(source, start_m=starts, start_ns=t0, end_ns=t1))
    assert len(fitted.elements) == 1


def test_integral_fit_converges_to_original_segment_quadrature(tmp_path):
    pytest.importorskip("numba")
    medium = lh.SpectralMedium([350., 610.], [.08, .05], [.03, .02],
                                [1.35, 1.335], [1.40, 1.35], g=.9)
    det = lh.DetectorArray([[5., 0., 0.], [20., 5., 0.], [0., 0., 30.]],
                           [[0., 0., 1.]]*3, .05,
                           lambda x: np.ones_like(x), lambda w: .25*np.ones_like(w))
    config = lh.KernelConfig(wavelength_nodes=2, threshold_pe=0.,
                relative_time_edges_ns=np.arange(-20., 300., 5.),
                cache_directory=tmp_path)
    kernel = lh.TransportKernel(medium, det, config)
    source = primary()
    origins = lh.CherenkovTrack([0.,0.,-20.], [0.,0.,1.], 40.).earliest_arrival_ns(
        det.positions_m, 1.35)
    ref_q, ref_b, _ = order1_integrals(kernel, source, time_origin_ns=origins)
    fit_q, fit_b, _ = order1_integrals(kernel, fit_track_integrals(source).elements,
                                       time_origin_ns=origins)
    np.testing.assert_allclose(fit_q, ref_q, rtol=1e-3)
    assert np.max(np.abs(fit_b-ref_b).sum(1) / np.abs(ref_b).sum(1)) < .005
    # Uses the medium's spectral cone and retains finite-track axial response.
    assert np.all(fit_q > 0)
