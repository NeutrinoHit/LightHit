"""Order >=2 of source points closer than the radial minimum (near-field clamp)."""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

pytest.importorskip("numba")
import lighthit as lh
from lighthit import near_field
from lighthit.muon.track_ge2_primitive import cone_point_evaluator


def _bundle(bands=1, omega=(0., .3, .6), seed=3):
    rng = np.random.default_rng(seed)
    shape = (bands, 2, 3, 3, 4, len(omega))
    coefficients = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    return dict(degree=2, acceptance_degree=1, omega=np.asarray(omega),
                pair_lambda=np.array([0, 1, 1]), pair_mu=np.array([0, 0, 1]),
                log_r=np.log([3., 20., 300.]), coefficients=coefficients,
                absorption=np.full(bands, .03), speed=np.full(bands, .22), phase="flight")


def _evaluator(bundle, impact, look, near):
    return cone_point_evaluator(
        bundle, impact_m=impact, look=look, alpha=np.array([4 * np.pi, 1.]),
        wavelength_nm=np.array([480.]), wavelength_weight_nm=np.ones(1),
        phase_index=np.array([1.35]), efficiency=np.ones(1), effective_area_m2=.05,
        backend="fused-looks", near_field=near)


def test_clamp_takes_the_amplitude_at_the_radial_minimum_along_the_same_direction():
    bundle = _bundle()
    look = np.array([.3, .4, np.sqrt(1 - .25)])
    impact, u = .6, np.array([-1.2, .8])
    radius = np.hypot(impact, u)
    near = _evaluator(bundle, impact, look, True)(u)
    # The same directions at 3 m: scale the points along their rays.
    at = 3 * (1 + 1e-9)          # just inside the tables
    outer = np.array([_evaluator(bundle, impact * at / r, look, False)(np.array([x * at / r]))[0]
                      for x, r in zip(u, radius)])
    # Only the flight phase keeps the actual distance.
    phase = np.exp(1j * bundle["omega"][None, :] * (radius[:, None] - at) / .22)
    np.testing.assert_allclose(near, outer * phase, rtol=1e-7)


def test_without_near_field_points_inside_the_radial_minimum_raise():
    with pytest.raises(ValueError, match="radial minimum"):
        _evaluator(_bundle(), .6, np.array([0., 0., 1.]), False)(np.array([-1.]))


def test_near_portions_are_the_parts_of_elements_inside_the_radius():
    source = lh.SpectralLightElements(
        np.array([[0., 0., -10.], [0., 0., 0.], [50., 0., 0.]]), np.tile([0., 0., 1.], (3, 1)),
        np.array([20., 1., 1.]), np.ones(3), -np.ones(3), np.ones(3), np.zeros(3), np.ones(3),
        np.arange(3), np.zeros(3, int), cone_model="spectral")
    portions = near_field.near_portions(np.array([[1., 0., 0.], [0., 0., 100.]]), source, 3.)
    assert list(portions) == [0]
    (first, s0, s1), (second, t0, t1) = sorted(portions[0])
    half = np.sqrt(9 - 1)
    assert first == 0 and (s0, s1) == pytest.approx((10 - half, 10 + half))
    assert second == 1 and (t0, t1) == pytest.approx((0., 1.))


def _fake_response(n=3, bins=4, omega=(0., .3, .6)):
    from lighthit.transport import HybridTransportResponse
    omega = np.asarray(omega)
    return HybridTransportResponse(
        None, omega, np.linspace(0., 16., bins + 1), np.zeros((len(omega), n), complex),
        np.zeros((n, bins, 3)), np.ones((n, 3)), np.array([5., 6., 7.]),
        np.array([True, False, True]), "test", {"fine_charge_mask": [True, True, False]})


def test_added_near_field_is_recorded_warned_and_kept_to_computed_oms(monkeypatch):
    omega = np.array([0., .3, .6])
    spectrum = np.zeros((3, 3), complex)
    spectrum[:, :] = np.exp(1j * omega * 10.)[:, None] * np.array([.2, .3, .4])
    fake = near_field.NearFieldGe2(np.array([0, 1, 2]), np.array([.2, .3, .4]), spectrum,
                                   dict(radius_m=3., charge_pe=[.2, .3, .4]))
    monkeypatch.setattr(near_field, "near_field_ge2", lambda *a, **k: fake)
    kernel = SimpleNamespace(config=SimpleNamespace(readout_omega_per_ns=None),
                             detector=SimpleNamespace(identifiers={"uid": [11, 12, 13]}))
    response = _fake_response()
    with pytest.warns(RuntimeWarning, match=r"uid \[11, 12, 13\]"):
        near_field.add_near_field(response, kernel, None, dict(omega=omega), label="muon")
    # OM 2 has only a coarse charge (fine_charge_mask): nothing is added there.
    np.testing.assert_allclose(response.charge_components_pe[:, 2], [1.2, 1.3, 1.])
    np.testing.assert_allclose(response.charge_components_pe[:, :2], 1.)
    # Bins only for the active OM 0; the stored spectrum is on the absolute clock.
    assert np.any(response.components_pe[0, :, 2]) and not np.any(response.components_pe[1:, :, 2])
    np.testing.assert_allclose(response.spectrum_ge2_pe[:, 1],
                               .3 * np.exp(1j * omega * (10. + 6.)))
    record, = response.metadata["near_field"]
    assert record["label"] == "muon" and record["charge_added_oms"] == [0, 1]
    assert record["bins_added_oms"] == [0]


def test_no_near_point_leaves_the_response_unchanged(monkeypatch):
    fake = near_field.NearFieldGe2(np.zeros(0, int), np.zeros(3), np.zeros((3, 3), complex), {})
    monkeypatch.setattr(near_field, "near_field_ge2", lambda *a, **k: fake)
    response = _fake_response()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        near_field.add_near_field(response, None, None, dict(omega=np.array([0., .3, .6])))
    assert "near_field" not in response.metadata
    np.testing.assert_array_equal(response.charge_components_pe, 1.)


def _kernel_and_bundle():
    medium = lh.SpectralMedium([350., 610.], [.05, .05], [.03, .03], [1.35, 1.35], [1.4, 1.4])
    def efficiency(w):
        return np.ones_like(w) * .2
    detector = lh.DetectorArray([[1., 0., 0.]], [[0., 0., -1.]], .05,
                                lambda x: np.ones_like(x), efficiency)
    kernel = lh.TransportKernel(medium, detector, lh.KernelConfig(omega_per_ns=np.linspace(0, .6, 3)))
    bands = len(kernel.wavelength.wavelength_nm)
    bundle = _bundle(bands)
    bundle["absorption"] = kernel.medium.sample(kernel.wavelength.wavelength_nm)["absorption_per_m"]
    return kernel, bundle


def _track(length, start_z=-2., start_ns=0.):
    return lh.SpectralLightElements(
        np.array([[0., 0., start_z]]), np.array([[0., 0., 1.]]), np.array([length]),
        np.array([1e4 * length]), np.array([-1e4 * length]), np.ones(1),
        np.array([start_ns]), np.array([start_ns + length / .299792458]),
        np.zeros(1, int), np.zeros(1, int), cone_model="spectral")


def test_near_field_is_additive_over_element_splits_and_follows_the_clock():
    kernel, bundle = _kernel_and_bundle()
    whole = near_field.near_field_ge2(kernel, _track(4.), bundle, time_origin_ns=np.zeros(1))
    assert whole.oms.tolist() == [0] and whole.charge_pe[0] != 0
    # The second half starts 2 m later on the same clock.
    second = near_field.near_field_ge2(kernel, _track(2., 0., 2 / .299792458), bundle,
                                       time_origin_ns=np.zeros(1)).spectrum_pe
    first = near_field.near_field_ge2(kernel, _track(2., -2.), bundle,
                                      time_origin_ns=np.zeros(1)).spectrum_pe
    np.testing.assert_allclose(first + second, whole.spectrum_pe, rtol=1e-9)
    shifted = near_field.near_field_ge2(kernel, _track(4.), bundle, time_origin_ns=np.array([7.]))
    np.testing.assert_allclose(shifted.spectrum_pe,
                               whole.spectrum_pe * np.exp(-1j * bundle["omega"][:, None] * 7.),
                               rtol=1e-12)
