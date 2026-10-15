"""Axial-slice shower basis for order >=2 (``transport_hybrid(ge2_basis=...)``).

Lightweight synthetic checks; measured G4/BGVD accuracy is in docs/accuracy.md.
"""
from dataclasses import replace

import numpy as np
import pytest

import lighthit as lh

pytest.importorskip("numba")
from lighthit._shower_basis import compile_axial_slices  # noqa: E402
from lighthit.experimental.axial_fast import compile_axial_source_fast  # noqa: E402
from lighthit.experimental.axial_source import AxisFrame  # noqa: E402

from test_prompt import config, detector, spectral_medium  # noqa: E402


def shower(elements=400, length=20.0, seed=3):
    return lh.SyntheticShower.gaussian(
        lh.SourcePose([12., 3., 0.], [0.3, 0.2, 0.93], 0.),
        charged_track_length_m=length, elements=elements, seed=seed)


def harmonic_00(source, omega_index=0):
    return source.channels[:, 0, omega_index].real


@pytest.mark.parametrize("transverse", ["collapse", "stencil"])
@pytest.mark.parametrize("axial_order", [1, 3])
def test_moments_of_the_basis_equal_those_of_the_chords(transverse, axial_order):
    field = shower().elements.field(0)
    frame = AxisFrame.of(field)
    source = compile_axial_slices(field, 4, [0.0], azimuthal_degree=2, cell_m=0.3,
                                  axial_order=axial_order, transverse=transverse,
                                  element_order=2, frame=frame)
    points = frame.rotate(source.points_m() - frame.centre_m)
    weight = harmonic_00(source) * np.sqrt(4 * np.pi)
    nodes, node_weights = np.polynomial.legendre.leggauss(2)
    exact = {name: 0.0 for name in ("0", "z", "zz", "zzz", "x", "y", "xx", "yy")}
    for fraction, node_weight in zip((nodes + 1) / 2, node_weights / 2):
        z, offset, _, _ = frame.coordinates(field, fraction)
        x, y = offset @ frame.first, offset @ frame.second
        q = field.photons * node_weight
        for name, value in (("0", 1), ("z", z), ("zz", z * z), ("zzz", z ** 3),
                            ("x", x), ("y", y), ("xx", x * x), ("yy", y * y)):
            exact[name] += float(np.sum(q * value))
    got = {"0": weight.sum(), "z": weight @ points[:, 2], "zz": weight @ points[:, 2] ** 2,
           "zzz": weight @ points[:, 2] ** 3, "x": weight @ points[:, 0],
           "y": weight @ points[:, 1], "xx": weight @ points[:, 0] ** 2,
           "yy": weight @ points[:, 1] ** 2}
    scale = {"0": exact["0"], "z": 1e-3 * exact["0"]}
    assert got["0"] == pytest.approx(exact["0"], rel=1e-12)
    assert got["z"] == pytest.approx(exact["z"], rel=1e-10, abs=1e-10 * scale["0"])
    if axial_order == 3:
        for name in ("zz", "zzz"):
            assert got[name] == pytest.approx(exact[name], rel=1e-10, abs=1e-10 * scale["0"])
    if transverse == "stencil":
        for name in ("x", "y", "xx", "yy"):
            assert got[name] == pytest.approx(exact[name], rel=1e-10, abs=1e-12 * scale["0"])
    else:
        assert np.all(np.abs(points[:, :2]) < 1e-12)


def test_time_lattice_error_is_within_its_bound():
    """Linear-collapse basis = cloud-in-cell axial source up to exp(i w t)."""
    field = shower().elements.field(0)
    frame = AxisFrame.of(field)
    omega = np.linspace(0.0, 1.2, 13)
    reference = compile_axial_source_fast(field, 6, omega, azimuthal_degree=2, cell_m=0.3,
                                          element_order=2, frame=frame, axial_only=True)
    for step in (0.05, 0.02):
        basis = compile_axial_slices(field, 6, omega, azimuthal_degree=2, cell_m=0.3,
                                     axial_order=1, transverse="collapse",
                                     element_order=2, time_step_ns=step, frame=frame)
        np.testing.assert_allclose(basis.z_m, reference.z_m, rtol=0, atol=1e-12)
        total = np.abs(reference.channels).sum(axis=(0, 1))
        error = np.abs(basis.channels - reference.channels).sum(axis=(0, 1))
        assert error[0] <= 1e-12 * total[0]
        assert np.all(error <= (omega * step) ** 2 / 8 * total + 1e-12 * total[0])


@pytest.mark.parametrize("transverse", ["collapse", "stencil"])
def test_charge_compilation_does_not_need_a_time_lattice(transverse):
    field = shower().elements.field(0)
    # A long time span previously allocated a mostly empty slice x time map.
    field = replace(field, start_ns=field.start_ns * 1000,
                    end_ns=field.end_ns * 1000)
    options = dict(azimuthal_degree=2, cell_m=.3, transverse=transverse,
                   stencil_threshold_m=.03)
    zero = compile_axial_slices(field, 6, [0.], **options)
    full = compile_axial_slices(field, 6, [0., .03], **options)
    np.testing.assert_allclose(zero.points_m(), full.points_m(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(zero.channels[:, :, 0], full.channels[:, :, 0],
                               rtol=2e-12, atol=1e-10)
    assert zero.summary["time_rows"] == zero.summary["slices"]
    assert full.summary["time_rows"] > zero.summary["time_rows"]


@pytest.mark.parametrize("transverse", ["collapse", "stencil"])
def test_selected_point_compilation_is_an_exact_slice(transverse):
    field = shower(elements=80).elements.field(0)
    options = dict(azimuthal_degree=2, cell_m=.3, transverse=transverse)
    full = compile_axial_slices(field, 6, [0., .1, .3], **options)
    selected = np.arange(len(full.z_m)) % 3 == 1
    part = compile_axial_slices(field, 6, [0., .1, .3], point_selection=selected, **options)
    np.testing.assert_array_equal(part.points_m(), full.points_m()[selected])
    np.testing.assert_array_equal(part.channels, full.channels[selected])
    rounded = compile_axial_slices(field, 6, [0., .1, .3], point_selection=selected,
                                    channel_dtype=np.complex64, **options)
    np.testing.assert_array_equal(rounded.channels, part.channels.astype(np.complex64))
    empty = compile_axial_slices(field, 6, [0., .1], point_selection=np.zeros(len(selected), bool), **options)
    assert empty.channels.shape == (0, full.channels.shape[1], 2)


def test_pair_screen_has_an_absolute_spectral_error_bound(small_kernel):
    from lighthit._shower_basis import apply_multi, apply_multi_spectral
    k = small_kernel
    bundle = k.ge2_kernel_store().bundle(k, np.arange(len(k.config.omega_per_ns)))
    field = shower(elements=50).elements.field(0)
    source = compile_axial_slices(field, bundle["degree"], bundle["omega"],
                                  azimuthal_degree=2, cell_m=.5)
    receivers = source.frame.centre_m + np.array([[8., 1., 3.], [35., 4., 2.]])
    looks = np.tile([0., 0., 1.], (2, 1))
    alpha = k.acceptance()["alpha"]
    scale = np.ones((1, 2))
    args = (bundle, source, source, receivers, looks, alpha,
            np.ones(1), np.zeros(1), scale)
    radial = (3., k.config.radial_range_m[1])
    ref = apply_multi(*args, radial_window_m=radial)
    tolerance = float(np.max(abs(ref))) * 10
    got = apply_multi(*args, radial_window_m=radial, absolute_tolerance_pe=tolerance)
    assert np.any(got != ref), "test must exercise an actually omitted contribution"
    assert np.max(abs(got-ref)) <= tolerance
    # Include complex source coefficients and rounded fused source blocks.
    fused_ref = apply_multi_spectral(bundle, [source], receivers, looks, alpha, scale,
                                     radial_window_m=radial)
    fused = apply_multi_spectral(bundle, [source], receivers, looks, alpha, scale,
                                 radial_window_m=radial, absolute_tolerance_pe=tolerance)
    assert np.any(fused != fused_ref)
    assert np.max(abs(fused-fused_ref)) <= tolerance
    exact = apply_multi(*args, radial_window_m=radial, absolute_tolerance_pe=0.)
    np.testing.assert_array_equal(exact, ref)


@pytest.fixture(scope="module")
def small_kernel(tmp_path_factory):
    from lighthit.model import WavelengthQuadrature
    kernel = lh.TransportKernel(
        spectral_medium(), detector(count=10),
        config(tmp_path_factory.mktemp("cache"), omega_per_ns=np.linspace(0., 1.2, 17),
               threshold_pe=0.0, azimuthal_degree=2))
    kernel.wavelength = WavelengthQuadrature(kernel.wavelength.wavelength_nm[:1].copy(),
                                             kernel.wavelength.weight_nm[:1].copy())
    return kernel


def test_basis_route_keeps_prompt_orders_and_converges_to_the_hybrid(small_kernel, tmp_path):
    source = shower()
    reference = small_kernel.transport_hybrid(source, frequency_nodes=17)
    errors = []
    for cell in (0.5, 0.25):
        basis = small_kernel.transport_hybrid(
            source, frequency_nodes=17, ge2_basis=lh.ShowerBasis(cell_m=cell))
        assert isinstance(basis, lh.HybridTransportResponse)
        np.testing.assert_array_equal(basis.charge_components_pe[:, :2],
                                      reference.charge_components_pe[:, :2])
        np.testing.assert_array_equal(basis.components_pe[:, :, :2],
                                      reference.components_pe[:, :, :2])
        np.testing.assert_array_equal(basis.time_origin_ns, reference.time_origin_ns)
        np.testing.assert_array_equal(basis.active, reference.active)
        q, q_ref = basis.charge_components_pe[:, 2], reference.charge_components_pe[:, 2]
        assert np.all(np.abs(q - q_ref) <= 5e-3 * np.abs(q_ref) + 1e-12)
        errors.append(np.abs(basis.components_pe[:, :, 2]
                             - reference.components_pe[:, :, 2]).sum()
                      / np.abs(reference.components_pe[:, :, 2]).sum())
        assert basis.metadata["ge2_source"] == "axial-slice shower basis"
        assert basis.select(">=2").charge_pe.shape == q.shape
    assert errors[-1] < 0.05
    # the prepared order >=2 kernels round-trip through a memory-mapped pack
    store = small_kernel.ge2_kernel_store()
    assert len(store) > 0
    store.save(tmp_path / "pack")
    loaded = lh.TransportKernel(small_kernel.medium, small_kernel.detector, small_kernel.config)
    loaded.wavelength = small_kernel.wavelength
    loaded.ge2_kernel_store(tmp_path / "pack")
    dense = loaded.ge2_kernel_store().bundle(loaded, np.arange(17))
    sparse = loaded.ge2_kernel_store().bundle(loaded, np.arange(0, 17, 2))
    assert np.shares_memory(sparse["coefficients"], dense["coefficients"])
    assert loaded.ge2_kernel_store().nbytes == dense["coefficients"].nbytes
    np.testing.assert_array_equal(sparse["omega"], dense["omega"][::2])
    again = loaded.transport_hybrid(source, frequency_nodes=17,
                                    ge2_basis=lh.ShowerBasis(cell_m=0.25))
    np.testing.assert_allclose(again.components_pe, basis.components_pe, rtol=0, atol=0)
    cubic = loaded.transport_hybrid(source, frequency_nodes=17,
        ge2_basis=lh.ShowerBasis(cell_m=.25, time_deposit="cubic", time_step_ns=.5))
    # Exercise the public transport route, including its absolute-time carrier.
    np.testing.assert_array_equal(cubic.components_pe[:, :, :2], again.components_pe[:, :, :2])
    np.testing.assert_allclose(cubic.charge_components_pe, again.charge_components_pe,
                               rtol=1e-12, atol=1e-15)
    delta = abs(cubic.components_pe[:, :, 2]-again.components_pe[:, :, 2]).sum()
    assert delta <= 1e-3 * abs(again.components_pe[:, :, 2]).sum()


def test_basis_arguments_are_checked():
    with pytest.raises(ValueError):
        lh.ShowerBasis(cell_m=0)
    with pytest.raises(ValueError):
        lh.ShowerBasis(axial_order=2)
    with pytest.raises(ValueError):
        lh.ShowerBasis(transverse="ring")
    with pytest.raises(ValueError):
        lh.ShowerBasis(track_uid=True)
    with pytest.raises(ValueError):
        lh.ShowerBasis(track_transverse="ring")


def test_explicit_track_component_keeps_prompt_and_sums_the_rte(small_kernel):
    track = lh.CherenkovTrack.centered([12., 3., 0.], [0.3, 0.2, 0.93], 20.)
    elements = track.to_elements(step_m=0.5)
    uid = np.zeros(len(elements), np.int64)
    uid[len(uid) // 2:] = 1
    source = lh.G4Shower(replace(elements, uid=uid), "unit-test", 0)
    baseline = small_kernel.transport_hybrid(
        source, frequency_nodes=17, ge2_basis=lh.ShowerBasis())
    split = small_kernel.transport_hybrid(
        source, frequency_nodes=17, ge2_basis=lh.ShowerBasis(track_uid=0))
    np.testing.assert_array_equal(split.charge_components_pe[:, :2],
                                  baseline.charge_components_pe[:, :2])
    np.testing.assert_array_equal(split.components_pe[:, :, :2],
                                  baseline.components_pe[:, :, :2])
    np.testing.assert_array_equal(split.active, baseline.active)
    assert len(split.metadata["ge2_basis_summary"]["components"]) == 2
    np.testing.assert_allclose(split.charge_components_pe[:, 2],
                               baseline.charge_components_pe[:, 2], rtol=0.03,
                               atol=1e-9)
    with pytest.raises(ValueError, match="absent"):
        small_kernel.transport_hybrid(
            source, frequency_nodes=17, ge2_basis=lh.ShowerBasis(track_uid=42))


@pytest.mark.parametrize("source_m", [0, 2])
def test_wavelength_merged_apply_equals_the_per_wavelength_kernel(tmp_path, source_m):
    from lighthit._shower_basis import apply_multi
    kernel = lh.TransportKernel(
        spectral_medium(), detector(count=6),
        config(tmp_path, omega_per_ns=np.linspace(0., 1.2, 9), azimuthal_degree=source_m))
    field0 = shower().elements.field(0)
    field2 = shower().elements.field(2)
    omega = kernel.config.omega_per_ns
    s0, s2 = compile_axial_slices([field0, field2], 6, omega, azimuthal_degree=source_m,
                                  cell_m=0.5)
    indices = np.arange(len(omega))
    store = lh.Ge2KernelStore()
    bundle = store.bundle(kernel, indices)
    positions = kernel.detector.positions_m
    looks = -kernel.detector.orientations
    alpha = kernel.acceptance()["alpha"]
    wavelengths = kernel.wavelength.wavelength_nm
    phase = kernel.medium.sample(wavelengths)["phase_index"]
    scale0, scale2 = 1 / wavelengths ** 2, 1 / (wavelengths ** 2 * phase ** 2)
    out_scale = np.outer(kernel.wavelength.weight_nm, np.linspace(0.5, 1.5, len(positions)))
    merged = apply_multi(bundle, s0, s2, positions, looks, alpha, scale0, scale2, out_scale)
    separate = sum(
        out_scale[lam][None, :] * store.kernel(kernel, w, indices).apply(
            s0, positions, looks, alpha, source_omega_per_ns=omega, second_source=s2,
            first_scale=scale0[lam], second_scale=scale2[lam], ge2_only=True)[:, :, 1]
        for lam, w in enumerate(wavelengths))
    np.testing.assert_allclose(merged, separate, rtol=1e-12,
                               atol=1e-12 * np.abs(separate).max())
    first_only = apply_multi(bundle, s0, s2, positions, looks, alpha,
                             scale0, np.zeros_like(scale2), out_scale)
    first_reference = sum(
        out_scale[lam][None, :] * store.kernel(kernel, w, indices).apply(
            s0, positions, looks, alpha, source_omega_per_ns=omega,
            second_source=s2, first_scale=scale0[lam], second_scale=0., ge2_only=True)[:, :, 1]
        for lam, w in enumerate(wavelengths))
    np.testing.assert_allclose(first_only, first_reference, rtol=1e-12,
                               atol=1e-12 * np.abs(first_reference).max())


def test_radial_window_matches_explicitly_retained_source_points(small_kernel):
    from lighthit._shower_basis import apply_multi
    source = shower(elements=120, length=20.)
    field0, field2 = source.elements.field(0), source.elements.field(2)
    omega = small_kernel.config.omega_per_ns
    first, second = compile_axial_slices(
        [field0, field2], small_kernel.config.source_degree, omega,
        azimuthal_degree=small_kernel.config.azimuthal_degree, cell_m=0.5)
    receiver = small_kernel.detector.positions_m[:1]
    look = -small_kernel.detector.orientations[:1]
    radii = np.linalg.norm(first.points_m() - receiver[0], axis=1)
    low = small_kernel.config.radial_range_m[0]
    high = float(np.median(radii))
    assert low < high < radii.max()
    keep = (radii >= low) & (radii <= high)
    assert keep.any() and (~keep).any()
    sample = small_kernel.medium.sample(small_kernel.wavelength.wavelength_nm)
    wavelengths = small_kernel.wavelength.wavelength_nm
    scale0 = 1 / wavelengths ** 2
    scale2 = 1 / (wavelengths ** 2 * sample["phase_index"] ** 2)
    output_scale = np.ones((len(wavelengths), 1))
    bundle = small_kernel.ge2_kernel_store().bundle(
        small_kernel, np.arange(len(omega)))
    arguments = (receiver, look, small_kernel.acceptance()["alpha"],
                 scale0, scale2, output_scale)
    windowed = apply_multi(bundle, first, second, *arguments,
                           radial_window_m=(low, high))
    def retained(item):
        return replace(item, z_m=item.z_m[keep],
                       transverse_m=item.transverse_m[keep],
                       channels=item.channels[keep])
    reference = apply_multi(bundle, retained(first), retained(second), *arguments)
    np.testing.assert_allclose(windowed, reference, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="radial window"):
        apply_multi(bundle, first, second, *arguments,
                    radial_window_m=(low, small_kernel.config.radial_range_m[1] + 1))


def test_local_time_origin_uses_nearest_segment_instead_of_event_start():
    from lighthit._shower_basis import earliest_element_arrival
    receiver = np.array([[0., 10., 0.]])
    starts = np.array([[1000., 0., 0.], [0., 0., 0.]])
    direction = np.array([[1., 0., 0.], [1., 0., 0.]])
    length = np.array([1., 20.])
    start_ns = np.array([0., 100.])
    origin = earliest_element_arrival(receiver, starts, direction, length,
                                      start_ns, 1.4)
    np.testing.assert_allclose(origin, 100. + 10. * 1.4 / 0.299792458)
    assert origin[0] < 0. + np.linalg.norm(receiver[0] - starts[0]) * 1.4 / 0.299792458


def test_radial_tail_screen_refuses_unchecked_truncation(small_kernel):
    with pytest.raises(ValueError, match="estimated light beyond"):
        small_kernel.transport_hybrid(
            shower(elements=80), frequency_nodes=17,
            ge2_basis=lh.ShowerBasis(radial_high_m=30.,
                                     radial_tail_tolerance_pe=0.))


def test_spectral_cone_basis_matches_two_band_hybrid(tmp_path):
    kernel = lh.TransportKernel(
        spectral_medium(), detector(count=4),
        config(tmp_path, omega_per_ns=np.linspace(0., 1.2, 9),
               scattering_degree=4, source_degree=4, azimuthal_degree=2,
               radial_nodes=16, threshold_pe=0.0))
    source = replace(shower(elements=80, length=10.).elements,
                     cone_model="spectral")
    reference = kernel.transport_hybrid(source, frequency_nodes=9)
    basis = kernel.transport_hybrid(
        source, frequency_nodes=9, ge2_basis=lh.ShowerBasis(cell_m=0.25))
    np.testing.assert_array_equal(basis.charge_components_pe[:, :2],
                                  reference.charge_components_pe[:, :2])
    np.testing.assert_array_equal(basis.components_pe[:, :, :2],
                                  reference.components_pe[:, :, :2])
    np.testing.assert_array_equal(basis.active, reference.active)
    q = reference.charge_components_pe[:, 2]
    assert np.all(np.abs(basis.charge_components_pe[:, 2] - q)
                  <= 0.01 * np.abs(q) + 1e-10)
    assert (np.abs(basis.components_pe[:, :, 2]
                   - reference.components_pe[:, :, 2]).sum()
            <= 0.05 * np.abs(reference.components_pe[:, :, 2]).sum())
    assert basis.metadata["cone_model"] == "spectral"
    fused = kernel.transport_hybrid(
        source, frequency_nodes=9,
        ge2_basis=lh.ShowerBasis(cell_m=0.25, spectral_fusion=True))
    np.testing.assert_array_equal(fused.charge_components_pe[:, :2],
                                  basis.charge_components_pe[:, :2])
    np.testing.assert_allclose(fused.charge_components_pe[:, 2],
                               basis.charge_components_pe[:, 2], rtol=2e-5, atol=1e-9)
    np.testing.assert_allclose(fused.components_pe[:, :, 2],
                               basis.components_pe[:, :, 2], rtol=2e-5, atol=1e-9)


def test_spectral_fusion_keeps_wavelength_dependent_point_exceptions(tmp_path, monkeypatch):
    import lighthit._shower_basis as sb
    original = sb.compile_axial_slices
    def shifted(field, *args, **kwargs):
        compiled = original(field, *args, **kwargs)
        z = compiled.z_m.copy()
        z[0] += 1e-10 * float(field.photons.sum())
        return replace(compiled, z_m=z)
    monkeypatch.setattr(sb, "compile_axial_slices", shifted)
    kernel = lh.TransportKernel(
        spectral_medium(), detector(count=2),
        config(tmp_path, omega_per_ns=np.linspace(0., 1.2, 9),
               scattering_degree=4, source_degree=4, azimuthal_degree=2,
               radial_nodes=16, threshold_pe=0.0))
    source = replace(shower(elements=60).elements, cone_model="spectral")
    basis = lh.ShowerBasis(stencil_threshold_m=0.)
    independent = kernel.transport_hybrid(source, frequency_nodes=9, ge2_basis=basis)
    fused = kernel.transport_hybrid(source, frequency_nodes=9,
                                    ge2_basis=replace(basis, spectral_fusion=True))
    assert fused.metadata["spectral_geometry_shared_points"] > 0
    assert fused.metadata["spectral_geometry_separate_points"] > 0
    np.testing.assert_allclose(fused.spectrum_ge2_pe, independent.spectrum_ge2_pe,
                               rtol=2e-5, atol=1e-9)


def test_fused_spectral_bands_match_individual_bands(tmp_path):
    from lighthit._shower_basis import apply_multi, apply_multi_spectral
    kernel = lh.TransportKernel(
        spectral_medium(), detector(count=3),
        config(tmp_path, omega_per_ns=np.linspace(0., 1.2, 9),
               scattering_degree=4, source_degree=4, azimuthal_degree=2,
               radial_nodes=16))
    elements = replace(lh.CherenkovTrack.centered(
        [12., 3., 0.], [0.3, 0.2, 0.93], 12.).to_elements(step_m=0.5),
        cone_model="spectral")
    wavelengths = kernel.wavelength.wavelength_nm
    sample = kernel.medium.sample(wavelengths)
    frame = AxisFrame.of(elements.field(0, phase_index=max(sample["phase_index"])))
    omega = kernel.config.omega_per_ns
    sources = []
    for wavelength, phase in zip(wavelengths, sample["phase_index"]):
        first = elements.field(0, phase_index=phase)
        second = elements.field(2, phase_index=phase)
        first_scale = 1 / wavelength ** 2
        combined = replace(first, photons=first_scale * first.photons
                           - first_scale / phase ** 2 * second.photons)
        sources.append(compile_axial_slices(
            combined, 4, omega, azimuthal_degree=2, cell_m=0.5, frame=frame))
    assert np.array_equal(sources[0].points_m(), sources[1].points_m())
    bundle = kernel.ge2_kernel_store().bundle(kernel, np.arange(len(omega)))
    positions = kernel.detector.positions_m
    looks = -kernel.detector.orientations
    alpha = kernel.acceptance()["alpha"]
    out_scale = np.outer(kernel.wavelength.weight_nm,
                         kernel.detector.effective_area_m2)
    fused = apply_multi_spectral(bundle, sources, positions, looks, alpha, out_scale)
    separate = np.zeros_like(fused)
    for lam, item in enumerate(sources):
        band = dict(bundle,
                    coefficients=bundle["coefficients"][lam:lam + 1],
                    absorption=bundle["absorption"][lam:lam + 1],
                    speed=bundle["speed"][lam:lam + 1])
        separate += apply_multi(band, item, item, positions, looks, alpha,
                                np.ones(1), np.zeros(1), out_scale[lam:lam + 1])
    np.testing.assert_allclose(fused, separate, rtol=2e-6,
                               atol=1e-11 * np.max(np.abs(separate)))


@pytest.mark.parametrize("cone_model", ["frozen", "spectral"])
def test_a_rigid_motion_of_the_whole_scene_leaves_the_basis_result_unchanged(
        tmp_path, cone_model):
    """Arbitrary position and orientation: the basis lives in the event's frame.

    Only the azimuthal orientation of the transverse stencil (fixed by the
    frame convention, not by the event) may differ, which is a
    basis-truncation effect far below the tolerance.
    """
    from lighthit.model import WavelengthQuadrature
    from scipy.spatial.transform import Rotation
    rotation = Rotation.from_rotvec([0.4, -1.1, 0.7]).as_matrix()
    shift = np.array([30.0, -12.0, 5.0])
    base = detector(count=10)
    moved = lh.DetectorArray(base.positions_m @ rotation.T + shift,
                             base.orientations @ rotation.T, base.effective_area_m2,
                             base.angular_acceptance, base.spectral_efficiency)
    results = []
    original = replace(shower().elements, cone_model=cone_model)
    for array, source in ((base, original), (moved, original.moved(rotation, shift))):
        kernel = lh.TransportKernel(
            spectral_medium(), array,
            config(tmp_path, omega_per_ns=np.linspace(0., 1.2, 17), threshold_pe=0.0,
                   azimuthal_degree=2))
        kernel.wavelength = WavelengthQuadrature(kernel.wavelength.wavelength_nm[:1].copy(),
                                                 kernel.wavelength.weight_nm[:1].copy())
        # exact per-element prompt rule: the grouped prompt compression bins
        # directions on a world-frame pixel grid (0.13 % change here)
        results.append(kernel.transport_hybrid(
            source, frequency_nodes=17, ge2_basis=lh.ShowerBasis(),
            prompt_config=lh.PromptConfig(element_method="segment")))
    a, b = results
    np.testing.assert_allclose(b.charge_components_pe[:, :2], a.charge_components_pe[:, :2],
                               rtol=1e-9, atol=1e-15)
    q = a.charge_components_pe[:, 2]
    assert np.all(np.abs(b.charge_components_pe[:, 2] - q) <= 1e-3 * np.abs(q) + 1e-15)
    assert (np.abs(b.components_pe[:, :, 2] - a.components_pe[:, :, 2]).sum()
            <= 1e-2 * np.abs(a.components_pe[:, :, 2]).sum())
