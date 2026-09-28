"""Axial-slice shower basis for order >=2 (``transport_hybrid(ge2_basis=...)``).

Lightweight synthetic checks.  The real G4/BGVD validation is
``scripts/shower_basis/`` and ``docs/research/shower-basis.md``.
"""
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
    again = loaded.transport_hybrid(source, frequency_nodes=17,
                                    ge2_basis=lh.ShowerBasis(cell_m=0.25))
    np.testing.assert_allclose(again.components_pe, basis.components_pe, rtol=0, atol=0)


def test_basis_arguments_are_checked():
    with pytest.raises(ValueError):
        lh.ShowerBasis(cell_m=0)
    with pytest.raises(ValueError):
        lh.ShowerBasis(axial_order=2)
    with pytest.raises(ValueError):
        lh.ShowerBasis(transverse="ring")


def test_wavelength_merged_apply_equals_the_per_wavelength_kernel(tmp_path):
    from lighthit._shower_basis import apply_multi
    kernel = lh.TransportKernel(
        spectral_medium(), detector(count=6),
        config(tmp_path, omega_per_ns=np.linspace(0., 1.2, 9), azimuthal_degree=2))
    field0 = shower().elements.field(0)
    field2 = shower().elements.field(2)
    omega = kernel.config.omega_per_ns
    s0, s2 = compile_axial_slices([field0, field2], 6, omega, azimuthal_degree=2,
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


def test_a_rigid_motion_of_the_whole_scene_leaves_the_basis_result_unchanged(tmp_path):
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
    original = shower().elements
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
