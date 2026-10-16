"""Prompt transport: scattering orders 0 and exactly 1 without the RTE cache.

Lightweight synthetic checks only; measured G4/BGVD accuracy is in
docs/accuracy.md.
"""
from dataclasses import replace

import numpy as np
import pytest

import lighthit as lh
from lighthit.medium import Medium
from lighthit.prompt import (OrderNotComputedError, PromptConfig,
                             PromptTransportResponse, acceptance_monomials,
                             acceptance_polynomial)
from lighthit.prompt_reference import (pencil_geometry, pencil_order1_bins,
                                       pencil_order1_monte_carlo)
from lighthit.single import single_bins, single_spectrum

pytest.importorskip("numba")
from lighthit import _prompt_numba as kernels  # noqa: E402

BGVD_LIKE_ALPHA = np.array([4.70363, 2.14654, 0.33227, -0.03527])
ISOTROPIC_ALPHA = np.array([4 * np.pi])


def cubic_acceptance(x):
    x = np.asarray(x, float)
    return 0.3082 + 0.54192 * x + 0.19831 * x ** 2 - 0.04912 * x ** 3


def spectral_medium(scattering=(0.03, 0.02)):
    return lh.SpectralMedium([350., 610.], [0.08, 0.05], list(scattering),
                             [1.35, 1.335], [1.40, 1.35], g=0.9,
                             provenance="prompt test medium")


def detector(count=12, seed=0, acceptance=cubic_acceptance):
    rng = np.random.default_rng(seed)
    positions = rng.uniform(-40, 40, size=(count, 3))
    positions[:, 0] += 12.0
    return lh.DetectorArray(positions, rng.normal(size=(count, 3)), 0.05,
                            acceptance,
                            lambda w: 0.25 * np.ones_like(np.asarray(w, float)))


def config(tmp_path, **changes):
    base = dict(wavelength_nodes=2, omega_per_ns=np.linspace(0., 1.2, 81),
                relative_time_edges_ns=np.arange(-20., 300., 5.),
                scattering_degree=6, source_degree=6, azimuthal_degree=2,
                cell_m=0.5, k_max_per_m=2., k_panel_per_m=0.1, k_order=6,
                radial_range_m=(1., 200.), radial_nodes=24, threshold_pe=0.0,
                cache_directory=tmp_path)
    base.update(changes)
    return lh.KernelConfig(**base)


def track():
    return lh.CherenkovTrack.centered([12., 3., 0.], [0.3, 0.2, 0.93], 40., beta=1.0)


def shower():
    return lh.SyntheticShower.gaussian(
        lh.SourcePose([12., 3., 0.], [0.3, 0.2, 0.93], 0.),
        charged_track_length_m=20., elements=48, seed=3)


def test_long_element_source_uses_local_arrival_origin(tmp_path):
    from lighthit._shower_basis import earliest_element_arrival
    source = lh.SpectralLightElements(
        start_m=np.array([[1000., 0., 0.], [0., 0., 0.]]),
        direction=np.array([[1., 0., 0.], [1., 0., 0.]]),
        length_m=np.array([1., 20.]), coefficient0=np.array([1e7, 2e8]),
        coefficient2=np.array([-1e7, -2e8]), beta=np.ones(2),
        start_ns=np.array([0., 100.]), end_ns=np.array([3.34, 166.72]),
        row_index=np.arange(2), uid=np.arange(2), cone_model="spectral")
    det = lh.DetectorArray([[15., 10., 0.]], [[0., 0., 1.]], .05,
                           lambda c: np.ones_like(c), lambda w: np.ones_like(w) * .25)
    kernel = lh.TransportKernel(spectral_medium(scattering=(0., 0.)), det, config(tmp_path))
    group = float(np.min(kernel.medium.sample(kernel.wavelength.wavelength_nm)["group_index"]))
    origin = earliest_element_arrival(det.positions_m, source.start_m, source.direction,
                                      source.length_m, source.start_ns, group)
    auto = kernel.transport_prompt(source)
    explicit = kernel.transport_prompt(source, time_origin_ns=origin)
    np.testing.assert_array_equal(auto.time_origin_ns, origin)
    np.testing.assert_array_equal(auto.bins_orders_pe, explicit.bins_orders_pe)
    assert auto.bins_orders_pe[:, :, 0].sum() > 0
    assert auto.metadata["time_origin"] == "earliest local segment and fastest group flight"


def test_single_scattering_on_extension_of_a_finite_track_is_not_dropped(tmp_path):
    source = lh.CherenkovTrack([0., 0., 0.], [0., 0., 1.], 10., cone_model="spectral")
    positions = np.array([[0., 0., 30.], [1e-4, 0., 30.],
                          [0., 0., -20.], [1e-4, 0., -20.]])
    det = lh.DetectorArray(positions, np.tile([0., 0., 1.], (4, 1)), .05,
                           lambda c: np.ones_like(c), lambda w: np.ones_like(w) * .25)
    kernel = lh.TransportKernel(spectral_medium(), det, config(tmp_path))
    response = kernel.transport_prompt(source)
    assert np.all(response.charge_orders_pe[:, 1] > 0)
    np.testing.assert_allclose(response.charge_orders_pe[[0, 2], 1],
                               response.charge_orders_pe[[1, 3], 1], rtol=1e-7)
    np.testing.assert_allclose(response.bins_orders_pe[[0, 2], :, 1],
                               response.bins_orders_pe[[1, 3], :, 1], rtol=1e-5, atol=1e-12)


def run_pencil(medium, alpha, edges, receiver, look, position, direction,
               emission_time=0.0, step=0.1):
    coef, ipow, jpow, degree = acceptance_monomials(alpha)
    apoly = acceptance_polynomial(coef, ipow, jpow, degree)
    direction = np.asarray(direction, float) / np.linalg.norm(direction)
    look = np.asarray(look, float) / np.linalg.norm(look)   # kernels take unit vectors
    mus = np.array([medium.scattering_per_m])
    mut = np.array([medium.extinction_per_m])
    speed = np.array([medium.speed_m_per_ns])
    edges = np.asarray(edges, float)
    kappa = -mut * speed
    eedge = np.exp(np.outer(kappa, edges))
    y_g = (1 - abs(medium.g)) / (1 + abs(medium.g))
    charge, bins = kernels.pencils_order1(
        np.atleast_2d(receiver).astype(float), np.atleast_2d(look).astype(float),
        np.ones(1), np.zeros(1), np.atleast_2d(position).astype(float),
        np.atleast_2d(direction).astype(float), np.ones(1), np.zeros(1),
        np.array([emission_time]), apoly, medium.g, y_g, step, 2.5 * step, 25.0,
        mus, mut, speed, np.ones(1), np.zeros(1), edges, eedge,
        kernels.interval_table(edges, kappa), 4, 4096)
    return charge[0], bins[0]


# ------------------------------------------------------------ order-1 core

@pytest.mark.parametrize("emission_time", [100_000.0, -100_000.0])
def test_out_of_window_batch_keeps_integrated_charge(emission_time):
    edges = np.array([-20., 0., 20.])
    mus = np.array([0.03])
    mut = np.array([0.08])
    speed = np.array([0.22])
    kappa = -mut * speed

    def deposit(t_emit):
        pre = np.zeros((1, 3))
        acc = np.zeros((1, len(edges) - 1, 4))
        charge = np.zeros(1)
        kernels._batch_deposit(
            3, np.array([10., 11., 12.]), np.ones(3), np.zeros(3),
            10., t_emit, 25., mus, mut, speed, np.ones(1), np.zeros(1),
            edges, np.exp(np.outer(kappa, edges)), pre, acc, charge, 4,
            np.empty(3), np.empty(3), 0., 0., 0., 0.)
        return charge[0], pre, acc

    reference, _, _ = deposit(0.)
    charge, pre, acc = deposit(emission_time)
    assert reference > 0
    assert charge == pytest.approx(reference, rel=1e-12)
    assert not np.any(pre)
    assert not np.any(acc)


def test_spectral_cone_sums_independent_wavelength_contributions(tmp_path):
    from lighthit.model import WavelengthQuadrature

    medium = spectral_medium()
    kernel = lh.TransportKernel(medium, detector(count=3), config(tmp_path))
    elements = shower().elements
    keep = np.arange(len(elements)) == 0
    elements = replace(elements.subset(keep), cone_model="spectral")
    settings = PromptConfig(element_method="segment")
    nodes = kernel.wavelength.wavelength_nm.copy()
    weights = kernel.wavelength.weight_nm.copy()
    together = kernel.transport_prompt(elements, settings)
    assert together.metadata["cherenkov_cone"] == "wavelength-dependent medium phase index"
    summed = np.zeros_like(together.charge_orders_pe)
    for node, weight in zip(nodes, weights):
        kernel.wavelength = WavelengthQuadrature(np.array([node]),
                                                np.array([weight]))
        summed += kernel.transport_prompt(elements, settings).charge_orders_pe
    np.testing.assert_allclose(together.charge_orders_pe, summed,
                               rtol=1e-6, atol=1e-14)
    assert together.charge_orders_pe[:, 1].sum() > 0


def test_full_spectral_cone_matches_frozen_at_one_wavelength(tmp_path, monkeypatch):
    from lighthit.model import WavelengthQuadrature
    from lighthit.experimental import axial_fast

    kernel = lh.TransportKernel(spectral_medium(), detector(count=1),
                                config(tmp_path))
    node = float(kernel.wavelength.wavelength_nm[0])
    weight = float(kernel.wavelength.weight_nm[0])
    phase = float(kernel.medium.sample(np.array([node]))["phase_index"][0])
    kernel.wavelength = WavelengthQuadrature(np.array([node]),
                                            np.array([weight]))
    elements = shower().elements
    one = elements.subset(np.arange(len(elements)) == 0)
    frozen = kernel.transport(replace(one, reference_phase_index=phase))
    frequencies_compiled = []
    original_compile = axial_fast.compile_axial_source_fast

    def record_compile(source, degree, frequencies, **kwargs):
        frequencies_compiled.append(len(frequencies))
        return original_compile(source, degree, frequencies, **kwargs)

    monkeypatch.setattr(axial_fast, "compile_axial_source_fast", record_compile)
    spectral = kernel.transport(replace(one, cone_model="spectral"))
    assert frequencies_compiled == [1, len(kernel.config.omega_per_ns)]
    assert spectral.metadata["compiled_source_memory_mib"] == pytest.approx(
        spectral.metadata["compiled_source_cells"]
        * spectral.metadata["compiled_source_channels"]
        * len(kernel.config.omega_per_ns) * 16 / 2 ** 20)
    np.testing.assert_allclose(spectral.charge_components_pe,
                               frozen.charge_components_pe,
                               rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(spectral.components_pe,
                               frozen.components_pe, rtol=1e-10, atol=1e-14)


def test_full_spectral_cone_sums_wavelength_charges(tmp_path):
    from lighthit.model import WavelengthQuadrature

    kernel = lh.TransportKernel(spectral_medium(), detector(count=1),
                                config(tmp_path))
    nodes = kernel.wavelength.wavelength_nm.copy()
    weights = kernel.wavelength.weight_nm.copy()
    elements = shower().elements
    one = elements.subset(np.arange(len(elements)) == 0)
    spectral = kernel.transport(replace(one, cone_model="spectral"))
    summed = np.zeros_like(spectral.charge_components_pe)
    for node, weight in zip(nodes, weights):
        phase = float(kernel.medium.sample(np.array([node]))["phase_index"][0])
        kernel.wavelength = WavelengthQuadrature(np.array([node]),
                                                np.array([weight]))
        summed += kernel.transport(replace(one,
                                           reference_phase_index=phase)).charge_components_pe
    np.testing.assert_allclose(spectral.charge_components_pe, summed,
                               rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("cone_model", ["frozen", "spectral"])
def test_hybrid_uses_prompt_low_orders_and_rte_ge2(tmp_path, cone_model):
    from lighthit.model import WavelengthQuadrature

    omega = np.linspace(0., 1.2, 17)
    kernel = lh.TransportKernel(
        spectral_medium(), detector(count=2),
        config(tmp_path, omega_per_ns=omega, threshold_pe=0.0))
    kernel.wavelength = WavelengthQuadrature(
        kernel.wavelength.wavelength_nm[:1].copy(),
        kernel.wavelength.weight_nm[:1].copy())
    one = shower().elements.subset(np.arange(48) == 0)
    one = replace(one, cone_model=cone_model)

    full = kernel.transport(one)
    prompt = kernel.transport_prompt(one, compute_bins_for_all=True)
    hybrid = kernel.transport_hybrid(one, frequency_nodes=len(omega))
    assert isinstance(hybrid, lh.HybridTransportResponse)
    np.testing.assert_allclose(hybrid.charge_components_pe[:, :2],
                               prompt.charge_orders_pe, rtol=0, atol=0)
    np.testing.assert_allclose(hybrid.components_pe[:, :, :2],
                               prompt.bins_orders_pe, rtol=0, atol=0)
    np.testing.assert_allclose(hybrid.charge_components_pe[:, 2],
                               full.charge_components_pe[:, 2], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hybrid.components_pe[:, :, 2],
                               full.components_pe[:, :, 2], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hybrid.spectrum_ge2_pe,
                               full.spectrum_pe[:, :, 2], rtol=1e-12, atol=1e-14)
    assert hybrid.metadata["fourier_inverted_orders"] == [2]

    sparse = kernel.transport_hybrid(one, frequency_nodes=5)
    indices = np.rint(np.linspace(0, len(omega) - 1, 5)).astype(int)
    np.testing.assert_allclose(sparse.spectrum_ge2_pe,
                               full.spectrum_pe[indices, :, 2], rtol=1e-12,
                               atol=1e-14)
    compact_kernel = lh.TransportKernel(
        spectral_medium(), detector(count=2),
        config(tmp_path / "compact", omega_per_ns=omega[indices],
               readout_omega_per_ns=omega, threshold_pe=0.0))
    compact_kernel.wavelength = kernel.wavelength
    compact = compact_kernel.transport_hybrid(one, frequency_nodes=5)
    np.testing.assert_allclose(compact.spectrum_ge2_pe,
                               sparse.spectrum_ge2_pe, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(compact.components_pe,
                               sparse.components_pe, rtol=1e-12, atol=1e-14)
    saved = tmp_path / "hybrid.npz"
    sparse.save(saved)
    with np.load(saved, allow_pickle=False) as archive:
        assert "spectrum_ge2_pe" in archive
        assert "spectrum_pe" not in archive
    lh.viewer_payload(sparse)


def test_spectral_prompt_global_screen_preserves_active_oms(tmp_path):
    kernel = lh.TransportKernel(
        spectral_medium(), detector(count=12),
        config(tmp_path, threshold_pe=0.001))
    source = replace(shower().elements, cone_model="spectral")
    settings = PromptConfig(element_method="grouped")
    exact = kernel.transport_prompt(source, replace(settings, screen=False))
    screened = kernel.transport_prompt(
        source, settings, selection_extra_charge=np.zeros(len(kernel.detector)))
    assert screened.metadata["order1_global_screen"]["coarse_only_modules"] > 0
    np.testing.assert_array_equal(screened.active, exact.active)
    np.testing.assert_allclose(screened.charge_orders_pe[exact.active],
                               exact.charge_orders_pe[exact.active], rtol=1e-12,
                               atol=1e-14)
    np.testing.assert_allclose(screened.bins_orders_pe[exact.active],
                               exact.bins_orders_pe[exact.active], rtol=1e-12,
                               atol=1e-14)


def test_frozen_screen_widening_includes_other_scattering_orders(tmp_path, monkeypatch):
    """A >=2 contribution must participate in both screening decisions."""
    from lighthit import prompt

    kernel = lh.TransportKernel(spectral_medium(), detector(count=2),
                                config(tmp_path, threshold_pe=1.0))
    source = shower().elements
    pencils = {key: np.zeros((1, 3)) for key in
               ("position", "direction", "major")}
    pencils.update({key: np.zeros(1) for key in
                    ("w0", "w2", "time", "var_major", "var_minor")})
    pencils["pos_cov"] = np.zeros((1, 6))
    pencils["ring_nodes_deposited"] = np.array(1)
    monkeypatch.setattr(prompt, "compress", lambda *args: pencils)
    monkeypatch.setattr(prompt, "_min_distance_to_segments",
                        lambda positions, segments: np.ones(len(positions)))

    def ballistic(field, positions, *args, **kwargs):
        return np.zeros(len(positions)), np.zeros((len(positions),
                                                   len(kernel.config.relative_time_edges_ns) - 1))

    monkeypatch.setattr(prompt, "_ballistic_backend", lambda: ballistic)
    calls = []

    def grouped(*args):
        targets = args[4]
        compute_bins = args[-3]
        calls.append((targets.tolist(), compute_bins))
        # OM0 reveals a tenfold underestimate. OM1 was initially below
        # threshold, but the widened estimate plus its >=2 light selects it.
        q = np.array([1.0, 0.3]) if compute_bins else np.array([0.1, 0.02])
        bins = np.zeros((2, len(kernel.config.relative_time_edges_ns) - 1))
        bins[:, 0] = q
        return q, bins, np.ones(2, dtype=int), np.ones(2, dtype=int)

    monkeypatch.setattr(kernels, "grouped_pencils_order1", grouped)
    response = kernel.transport_prompt(
        source, PromptConfig(element_method="grouped"),
        selection_extra_charge=np.array([1.0, 0.8]))
    assert calls == [([0, 1], False), ([0], True), ([1], True)]
    np.testing.assert_array_equal(response.active, [True, True])
    np.testing.assert_array_equal(response.charge_orders_pe[:, 1], [1.0, 0.3])
    assert response.metadata["order1_screen"]["widened_modules"] == 1
    assert response.metadata["order1_level"] == [2, 2]


def _grouped_test_arguments(emission_time=0.):
    """Small two-wavelength kernel input shared by grouped numerical checks."""
    coef, ipow, jpow, degree = acceptance_monomials(BGVD_LIKE_ALPHA)
    edges = np.arange(-20., 300., 5.)
    mut = np.array([0.08, 0.05])
    speed = np.array([0.22, 0.23])
    kappa = -mut * speed
    arguments = (
        np.array([[10., 3., 7.], [20., 5., 2.]]),
        np.array([[0., 0., 1.], [0., 0., 1.]]),
        np.array([0.05, 0.05]), np.zeros(2), np.arange(2, dtype=np.int64),
        np.array([[0., 0., 0.], [0., 0., 1.]]),
        np.array([[0., 0., 1.], [0., 0., 1.]]),
        np.array([1., 2.]), np.array([-0.5, -1.]),
        np.full(2, emission_time), np.array([[1., 0., 0.], [1., 0., 0.]]),
        np.full(2, 1e-5), np.full(2, 1e-5), np.zeros((2, 6)),
        kernels._GAUSS_TAU, kernels._GAUSS_W, coef, ipow, jpow, degree,
        0.9, 0.1 / 1.9, 0.1, 0.25, 25.,
        np.array([0.03, 0.02]), mut, speed, np.ones(2), np.full(2, 0.5),
        edges, np.exp(np.outer(kappa, edges)), kernels.interval_table(edges, kappa),
        4, 2048, 6., 2e-3, 0.08, 0.5, 0.02, float(speed.max()), 1)
    return arguments


@pytest.mark.parametrize("emission_time", [0.0, 100_000.0, -100_000.0])
def test_grouped_charge_only_preserves_quadrature(emission_time):
    """Coarse screening may skip bins, but must retain exactly the same charge."""
    arguments = _grouped_test_arguments(emission_time)
    full = kernels.grouped_pencils_order1(*arguments)
    charge_only = kernels.grouped_pencils_order1(*arguments, False)
    assert np.all(full[0] > 0)
    np.testing.assert_array_equal(charge_only[0], full[0])
    np.testing.assert_array_equal(charge_only[2], full[2])
    np.testing.assert_array_equal(charge_only[3], full[3])
    assert charge_only[1].shape == (2, 0)


@pytest.mark.parametrize("compute_bins", [False, True])
def test_batch_charge_screen_keeps_near_light_and_omits_weak_far_light(compute_bins):
    arguments = list(_grouped_test_arguments())
    arguments[5] = np.array([[0., 0., 0.], [0., 0., -400.]])
    # Keep the weak far pulse inside the readout, so the bin comparison
    # exercises omission of actual histogram mass, not an empty late tail.
    arguments[9] = np.array([0., -1750.])
    full = kernels.grouped_pencils_order1(*arguments, compute_bins, 0., np.zeros((2, 2)))
    omission = np.zeros((2, 2))
    tolerance = 1e-10
    screened = kernels.grouped_pencils_order1(
        *arguments, compute_bins, tolerance, omission)
    assert np.all(omission[:, 0] == 1)
    assert np.all(omission[:, 1] > 0)
    assert np.all(omission[:, 1] <= tolerance)
    assert np.all(full[0] > 100 * tolerance)
    np.testing.assert_array_equal(screened[2], full[2])
    np.testing.assert_array_equal(screened[3], full[3] - 1)
    assert np.all(np.abs(screened[0] - full[0]) <= tolerance)
    if compute_bins:
        assert np.any(screened[1] != full[1])
        assert np.all(np.abs(screened[1] - full[1]).sum(axis=1) <= tolerance)
    # An explicit zero budget must leave both charge and bins bitwise intact.
    default = kernels.grouped_pencils_order1(*arguments, compute_bins)
    np.testing.assert_array_equal(default[0], full[0])
    np.testing.assert_array_equal(default[1], full[1])


@pytest.mark.parametrize("g", [-.9, 0., .9])
def test_batch_charge_estimate_bounds_continuous_grouped_integral(g):
    from scipy.integrate import quad

    coef = np.array([1., -.7, .3])
    ipow = np.array([0, 1, 0])
    jpow = np.array([0, 0, 2])
    moments0 = np.array([2., -.4, .8])
    moments2 = np.array([-1.2, .3, -.5])
    theta, radius, area = .35, 12., .05
    row = np.zeros((1, 12))
    row[0, :3], row[0, 3:6] = moments0, moments2
    row[0, 6:8] = [1., theta]
    mus, mut = np.array([.03, .04]), np.array([.08, .09])
    s0w, s2w = np.array([1., .8]), np.array([.5, .4])
    bound = kernels._grouped_batch_charge_estimate(
        row, np.array([0]), np.array([0]), 0, 1, 3, coef, radius, area,
        g, mus, mut, s0w, s2w)

    def absolute_density(chi):
        path = radius * (np.cos(theta) + np.sin(theta) * np.tan(chi / 2))
        phase = (1 - g * g) / (4 * np.pi * (1 + g * g - 2 * g * np.cos(chi)) ** 1.5)
        angular = coef * np.cos(chi) ** ipow * np.sin(chi) ** jpow
        value = mus * np.exp(-mut * path) * phase * (
            s0w * np.dot(angular, moments0) + s2w * np.dot(angular, moments2))
        return area * np.abs(value).sum()

    integral, error = quad(absolute_density, theta, np.pi, epsabs=1e-12)
    assert integral > 0
    assert integral + error <= bound


def test_spectral_batch_budget_is_divided_once_and_reported(tmp_path, monkeypatch):
    from lighthit import prompt

    kernel = lh.TransportKernel(spectral_medium(), detector(count=2), config(tmp_path))
    budgets = []

    def fake_order1(mode, kernel, source, elements, segments, positions, looks,
                    areas, origins, work, tables, coef, ipow, jpow, degree, g,
                    edges, settings, charge, **kwargs):
        budgets.append(settings.batch_charge_tolerance_pe)
        omission = np.column_stack((np.ones(2),
                                    np.full(2, settings.batch_charge_tolerance_pe / 2)))
        return np.ones(2), np.zeros((2, len(edges) - 1)), {
            "timings": {}, "order1_batch_omission": omission}

    def ballistic(field, positions, *args, **kwargs):
        return np.zeros(len(positions)), np.zeros((len(positions),
                                                   len(kernel.config.relative_time_edges_ns) - 1))

    monkeypatch.setattr(prompt, "_order1", fake_order1)
    monkeypatch.setattr(prompt, "_ballistic_backend", lambda: ballistic)
    monkeypatch.setattr(prompt, "_min_distance_to_segments",
                        lambda positions, segments: np.ones(len(positions)))
    tolerance = 1e-4
    result = kernel.transport_prompt(
        replace(shower().elements, cone_model="spectral"),
        PromptConfig(element_method="grouped", screen=False,
                     batch_charge_tolerance_pe=tolerance))
    assert len(budgets) == len(kernel.wavelength.wavelength_nm)
    np.testing.assert_allclose(budgets, tolerance / len(budgets))
    meta = result.metadata["order1_batch_screen"]
    assert result.metadata["order1_backward_profile"].startswith("grouped-finite-backward-v1:")
    np.testing.assert_allclose(meta["omitted_charge_estimate_pe_per_om"], tolerance / 2)
    np.testing.assert_array_equal(meta["skipped_batches_per_om"], len(budgets))


def test_batch_screen_retries_ambiguous_threshold_without_omission(tmp_path, monkeypatch):
    from lighthit import prompt

    kernel = lh.TransportKernel(spectral_medium(), detector(count=2),
                                config(tmp_path, threshold_pe=1.))
    tolerance = 1e-4
    budgets = []

    def fake_order1(*args, **kwargs):
        settings = args[-2]
        budgets.append(settings.batch_charge_tolerance_pe)
        value = 1. - tolerance / 2 if settings.batch_charge_tolerance_pe else 1. + tolerance / 2
        return np.full(2, value), np.zeros((2, len(args[-3]) - 1)), {"timings": {}}

    def ballistic(field, positions, *args, **kwargs):
        return np.zeros(len(positions)), np.zeros((len(positions),
                                                   len(kernel.config.relative_time_edges_ns) - 1))

    monkeypatch.setattr(prompt, "_order1", fake_order1)
    monkeypatch.setattr(prompt, "_ballistic_backend", lambda: ballistic)
    monkeypatch.setattr(prompt, "_min_distance_to_segments",
                        lambda positions, segments: np.ones(len(positions)))
    result = kernel.transport_prompt(
        shower().elements, PromptConfig(element_method="grouped", screen=False,
                                        batch_charge_tolerance_pe=tolerance))
    assert budgets == [tolerance, 0.]
    assert np.all(result.active)
    assert result.metadata["order1_batch_screen_fallback"] == "near-threshold OM"


def _backward_grouped_arguments(delta, sigma=0., g=.9, alpha=ISOTROPIC_ALPHA):
    args = list(_grouped_test_arguments())
    delta = np.atleast_1d(delta)
    count = len(delta)
    args[0] = 10 * np.column_stack((np.sin(delta), np.zeros(count), -np.cos(delta)))
    args[1] = np.tile([.6, 0., .8], (count, 1))
    args[2], args[3], args[4] = np.ones(count), np.zeros(count), np.arange(count, dtype=np.int64)
    args[5], args[6] = np.zeros((1, 3)), np.array([[0., 0., 1.]])
    args[7], args[8], args[9] = np.ones(1), np.zeros(1), np.zeros(1)
    args[10] = np.array([[1., 0., 0.]])
    args[11], args[12], args[13] = np.full(1, sigma**2), np.full(1, sigma**2), np.zeros((1, 6))
    args[16:20] = acceptance_monomials(alpha)
    args[20], args[21] = g, (1 - abs(g)) / (1 + abs(g))
    args[25:30] = [np.array([.03]), np.array([.08]), np.array([.299792458 / 1.37]),
                   np.ones(1), np.zeros(1)]
    args[30] = np.arange(0., 251., 10.)
    kappa = -args[26] * args[27]
    args[31], args[32] = np.exp(np.outer(kappa, args[30])), kernels.interval_table(args[30], kappa)
    return args


def _ray_path_reference(receiver, look, g, alpha, edges):
    """Independent first-scatter-distance integral, regular behind the source."""
    from scipy.integrate import quad

    receiver, look = np.asarray(receiver), np.asarray(look)
    radius = np.linalg.norm(receiver)
    cosine = receiver[2] / radius
    speed = .299792458 / 1.37
    acceptance = alpha * (2 * np.arange(len(alpha)) + 1) / (4 * np.pi)

    def density(length):
        toward = receiver - np.array([0., 0., length])
        rho = np.linalg.norm(toward)
        scatter_cos = toward[2] / rho
        phase = (1 - g*g) / (4*np.pi * (1 + g*g - 2*g*scatter_cos)**1.5)
        response = np.polynomial.legendre.legval(toward @ look / rho, acceptance)
        return .03 * np.exp(-.08 * (length + rho)) * phase * response / rho**2

    def distance(path):
        return (path*path - radius*radius) / (2 * (path - radius*cosine))

    total = quad(density, 0., np.inf, epsabs=1e-15, epsrel=1e-11)[0]
    bins = np.zeros(len(edges) - 1)
    for j, (lo, hi) in enumerate(zip(speed*edges[:-1], speed*edges[1:])):
        if hi > radius:
            bins[j] = quad(density, distance(max(lo, radius)), distance(hi),
                           epsabs=1e-15, epsrel=1e-11)[0]
    return total, bins


@pytest.mark.parametrize("g", [-.9, 0., .9])
@pytest.mark.parametrize("alpha", [ISOTROPIC_ALPHA, BGVD_LIKE_ALPHA])
def test_backward_grouped_charge_and_bins_match_ray_path_reference(g, alpha):
    delta = np.array([0., 1e-5, .01, .3, np.pi/2 - 1e-6, np.pi/2 + 1e-6])
    args = _backward_grouped_arguments(delta, g=g, alpha=alpha)
    charge, bins, _, _ = kernels.grouped_pencils_order1(
        *args, True, 0., np.zeros((len(delta), 2)))
    for index in range(len(delta)):
        expected, expected_bins = _ray_path_reference(
            args[0][index], args[1][index], g, alpha, args[30])
        assert expected > 0
        assert abs(charge[index] - expected) < 3e-4 * expected
        assert np.abs(bins[index] - expected_bins).sum() < 5e-4 * expected


@pytest.mark.parametrize("sigma", [.001, .01])
def test_backward_gaussian_pencil_has_smooth_covariance_error(sigma):
    """No sigma²/delta² pole: retained smooth-profile error is O(sigma²)."""
    from numpy.polynomial.hermite import hermgauss
    from numpy.polynomial.laguerre import laggauss

    delta = np.array([0., 1e-5, 1e-4, .001, .01])
    args = _backward_grouped_arguments(delta, sigma=sigma)
    charge = kernels.grouped_pencils_order1(
        *args, False, 0., np.zeros((len(delta), 2)))[0]
    node, weight = hermgauss(13)
    x, y = np.meshgrid(node, node)
    directions = np.column_stack((np.sqrt(2)*sigma*x.ravel(),
                                   np.sqrt(2)*sigma*y.ravel(), np.ones(x.size)))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    angular_weights = (weight[:, None] * weight[None, :] / np.pi).ravel()
    node, path_weights = laggauss(80)
    scatter = directions[:, None, :] * (node / .08)[None, :, None]
    for index, receiver in enumerate(args[0]):
        toward = receiver[None, None, :] - scatter
        rho = np.linalg.norm(toward, axis=2)
        cosine = np.einsum("nli,ni->nl", toward, directions) / rho
        phase = (1 - .9**2) / (4*np.pi * (1 + .9**2 - 1.8*cosine)**1.5)
        expected = angular_weights @ (.03/.08 * (np.exp(-.08*rho)*phase/rho**2 @ path_weights))
        assert abs(charge[index] / expected - 1) < 2*sigma**2 + 1e-5


def test_backward_bound_and_batch_normalizations_are_separate():
    # Force two angular groups in the same (tau,r) bin, one in each
    # hemisphere. A shared batch would mix w/r and w/(r*sin(theta)) means.
    args = _backward_grouped_arguments([0.])
    args[0] = np.array([[10., 0., 0.]])
    directions = np.array([[.6, 0., .8], [-.6, 0., .8]])
    args[5] = np.zeros((2, 3))
    args[6] = directions
    args[7], args[8], args[9] = np.ones(2), np.zeros(2), np.zeros(2)
    args[10] = np.array([[0., 1., 0.], [0., 1., 0.]])
    args[11], args[12], args[13] = np.zeros(2), np.zeros(2), np.zeros((2, 6))
    combined = kernels.grouped_pencils_order1(*args, True, 0., np.zeros((1, 2)))
    separate = []
    for i in range(2):
        one = list(args)
        for j in range(5, 14):
            one[j] = args[j][i:i+1].copy()
        separate.append(kernels.grouped_pencils_order1(*one, True, 0., np.zeros((1, 2))))
    assert combined[3][0] == 2
    np.testing.assert_allclose(combined[0], separate[0][0] + separate[1][0], rtol=1e-12)
    np.testing.assert_allclose(combined[1], separate[0][1] + separate[1][1], rtol=1e-10, atol=1e-15)
    # The backward analytic omission estimate is finite and nonzero at pi.
    backward = _backward_grouped_arguments([0.])
    omission = np.zeros((1, 2))
    reference = kernels.grouped_pencils_order1(*backward, False, 0., np.zeros((1, 2)))
    skipped = kernels.grouped_pencils_order1(*backward, False, 1., omission)
    assert omission[0, 0] == 1
    assert 0 < reference[0][0] <= omission[0, 1] <= 1
    assert skipped[0][0] == 0


def test_reference_formula_is_the_single_py_directed_limit():
    medium = Medium(0.05, 0.03, 0.9, 1.37, 450.)
    edges = np.arange(-10., 200., 5.)
    for radius, cosine in ((20., 0.3), (15., 0.98), (30., -0.5)):
        theta = np.arccos(cosine)
        charge, bins = pencil_order1_bins(
            radius, theta, 0.0, 0.0, medium, ISOTROPIC_ALPHA, edges,
            time_origin_ns=radius / medium.speed_m_per_ns)
        reference = single_bins(edges + radius / medium.speed_m_per_ns, radius,
                                cosine, medium, backend="numpy")
        spectrum, _ = single_spectrum(np.array([0.]), radius, cosine, medium,
                                      backend="numpy")
        assert charge == pytest.approx(spectrum[0].real, rel=1e-9)
        np.testing.assert_allclose(bins, reference, rtol=1e-8, atol=1e-12 * charge)


def test_reference_directional_acceptance_against_monte_carlo():
    medium = Medium(0.05, 0.03, 0.9, 1.37, 450.)
    receiver, direction, look = [14., -6., 9.], [0.2, 0.3, 0.93], [0.1, -0.8, 0.59]
    r, theta, c1, c3, _ = pencil_geometry([0, 0, 0], direction, receiver, look)
    charge, _ = pencil_order1_bins(r, theta, c1, c3, medium, BGVD_LIKE_ALPHA,
                                   np.array([0., 1.]))
    mean, error = pencil_order1_monte_carlo([0, 0, 0], direction, receiver, look,
                                            medium, BGVD_LIKE_ALPHA,
                                            samples=400_000, seed=4)
    assert abs(charge - mean) < 4 * error


@pytest.mark.parametrize("cosine", [0.3, 0.95, 0.9999, -0.8])
def test_fast_pencil_matches_single_py_in_the_scalar_limit(cosine):
    medium = Medium(0.05, 0.03, 0.9, 1.37, 450.)
    radius = 25.0
    edges = np.arange(-10., 400., 5.)
    theta = np.arccos(cosine)
    direction = [np.sin(theta), 0.0, np.cos(theta)]
    charge, bins = run_pencil(medium, ISOTROPIC_ALPHA, edges, [0., 0., radius],
                              [0., 0., 1.], [0., 0., 0.], direction)
    reference = single_bins(edges, radius, cosine, medium, backend="numpy")
    total, _ = single_spectrum(np.array([0.]), radius, cosine, medium,
                               backend="numpy")
    assert charge == pytest.approx(total[0].real, rel=2e-4)
    assert np.abs(bins - reference).sum() < 5e-4 * total[0].real


def test_fast_pencil_converges_to_the_adaptive_reference():
    medium = Medium(0.05, 0.03, 0.9, 1.37, 450.)
    edges = np.arange(-10., 400., 5.)
    receiver, direction, look = [30., -10., 25.], [0.7, -0.2, 0.68], [-0.2, 0.9, 0.39]
    r, theta, c1, c3, _ = pencil_geometry([0, 0, 0], direction, receiver, look)
    reference_charge, reference_bins = pencil_order1_bins(
        r, theta, c1, c3, medium, BGVD_LIKE_ALPHA, edges, emission_time_ns=2.0)
    errors = []
    for step in (0.2, 0.1, 0.05):
        charge, bins = run_pencil(medium, BGVD_LIKE_ALPHA, edges, receiver, look,
                                  [0, 0, 0], direction, emission_time=2.0, step=step)
        errors.append(np.abs(bins - reference_bins).sum() / reference_charge)
        assert charge == pytest.approx(reference_charge, rel=5 * errors[-1] + 1e-6)
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-4


def test_gaussian_inverse_mean_against_closed_forms():
    from scipy.integrate import quad
    from scipy.special import i0e
    tau, weight = kernels._GAUSS_TAU, kernels._GAUSS_W
    for sigma, distance in ((1.0, 0.0), (0.3, 0.2), (0.5, 2.0), (1.0, 7.0)):
        # isotropic: sqrt(pi/2)/sigma exp(-x) I0(x), x = d^2 / (4 sigma^2)
        x = distance ** 2 / (4 * sigma ** 2)
        exact = np.sqrt(np.pi / 2) / sigma * i0e(x)
        value = kernels.gauss_inverse_mean(distance, 0.0, sigma ** 2, sigma ** 2,
                                           tau, weight)
        assert value == pytest.approx(exact, rel=1e-10)
    for u, v, var in ((0.5, 0.2, 1.0), (0.0, 0.3, 0.2), (2.0, 0.05, 0.5)):
        # a line: 1D Gaussian of variance var along u, zero width
        def integrand(y):
            return (np.exp(-y * y / (2 * var)) / np.sqrt(2 * np.pi * var)
                    / np.hypot(y - u, v))
        exact = quad(integrand, -np.inf, np.inf, points=None, limit=400)[0]
        value = kernels.gauss_inverse_mean(u, v, var, 1e-30, tau, weight)
        assert value == pytest.approx(exact, rel=2e-4)   # 48-node rule, line limit


# ------------------------------------------------------- public semantics

@pytest.mark.parametrize("field,value", [
    ("max_a_breaks", 2), ("max_a_breaks", 3.5),
    ("max_path_nodes", 2), ("max_path_nodes", 3.5),
    ("screen_pixel_face", 0), ("screen_pixel_face", 1.5),
    ("screen_ring_nodes", 0), ("screen_ring_nodes", 1.5),
    ("screen_cell_m", 0), ("screen_cell_m", np.inf),
    ("screen_margin", 0.9), ("screen_margin", np.nan),
    ("batch_charge_tolerance_pe", -1), ("batch_charge_tolerance_pe", np.inf),
    ("batch_charge_tolerance_pe", np.nan),
    ("min_distance_m", -1), ("min_distance_m", np.inf),
    ("max_distance_m", 0), ("max_distance_m", np.inf),
])
def test_prompt_config_rejects_invalid_controls(field, value):
    with pytest.raises(ValueError, match=field):
        PromptConfig(**{field: value})


def test_prompt_requires_numba_before_preparing_source(tmp_path, monkeypatch):
    from lighthit import prompt

    kernel = lh.TransportKernel(spectral_medium(), detector(), config(tmp_path))
    monkeypatch.setattr(prompt, "find_spec", lambda name: None)
    with pytest.raises(ImportError, match=r"transport_prompt.*lighthit\[accelerate\]"):
        kernel.transport_prompt(track())


def test_zero_scattering_leaves_only_order_zero(tmp_path):
    kernel = lh.TransportKernel(spectral_medium((0., 0.)), detector(),
                                config(tmp_path))
    for source in (track(), shower()):
        response = kernel.transport_prompt(source)
        assert np.all(response.charge_orders_pe[:, 1] == 0)
        assert np.all(response.bins_orders_pe[:, :, 1] == 0)
        if isinstance(source, lh.CherenkovTrack):
            assert response.charge_orders_pe[:, 0].sum() > 0
        np.testing.assert_array_equal(response.select("0+1").charge_pe,
                                      response.charge_orders_pe[:, 0])


def test_ballistic_is_identical_to_the_full_path(tmp_path):
    kernel = lh.TransportKernel(spectral_medium(), detector(), config(tmp_path))
    for source in (track(), shower()):
        prompt = kernel.transport_prompt(source)
        full = kernel.transport(source)
        both = prompt.active & full.active
        assert both.sum() >= 3
        np.testing.assert_allclose(prompt.charge_orders_pe[both, 0],
                                   full.charge_components_pe[both, 0],
                                   rtol=1e-12, atol=1e-300)
        np.testing.assert_allclose(prompt.bins_orders_pe[both, :, 0],
                                   full.components_pe[both, :, 0],
                                   rtol=1e-12, atol=1e-18)
        np.testing.assert_array_equal(prompt.time_origin_ns, full.time_origin_ns)


def test_prompt_response_semantics_and_round_trip(tmp_path):
    kernel = lh.TransportKernel(spectral_medium(), detector(), config(tmp_path))
    response = kernel.transport_prompt(track())
    assert isinstance(response, PromptTransportResponse)
    assert response.computed_orders == (0, 1)
    assert response.metadata["order_ge2"] == "not computed"
    assert response.metadata["computed_orders"] == [0, 1]
    for choice in (2, "2", ">=2", "all"):
        with pytest.raises(OrderNotComputedError):
            response.select(choice)
    prompt = response.select("0+1")
    np.testing.assert_allclose(prompt.charge_pe, response.charge_orders_pe.sum(axis=1))
    np.testing.assert_allclose(prompt.bins_pe, response.bins_pe)
    window = response.window_charge_orders_pe
    outside = response.outside_window_pe()
    np.testing.assert_allclose(window + outside,
                               np.where(response.active[:, None],
                                        response.charge_orders_pe, 0.0),
                               rtol=1e-12, atol=1e-15)
    assert response.charge_orders_pe[:, 1].sum() > 0
    path = response.save(tmp_path / "prompt.npz")
    loaded = PromptTransportResponse.load(path, kernel.detector)
    np.testing.assert_array_equal(loaded.charge_orders_pe, response.charge_orders_pe)
    np.testing.assert_array_equal(loaded.bins_orders_pe, response.bins_orders_pe)
    assert loaded.computed_orders == (0, 1)
    with pytest.raises(OrderNotComputedError):
        loaded.select(">=2")


def test_viewer_distinguishes_not_computed_from_zero(tmp_path):
    from lighthit.viewer import validate_event_viewer, viewer_payload
    kernel = lh.TransportKernel(spectral_medium(), detector(), config(tmp_path))
    response = kernel.transport_prompt(track())
    payload = viewer_payload(response, source=track())
    event = payload["events"][0]
    assert event["computed_orders"] == [0, 1]
    assert np.all(np.asarray(event["components"])[:, :, 2] == 0)
    event["components"][0][0][2] = 1.0     # a placeholder may never hold data
    with pytest.raises(ValueError, match="not computed"):
        validate_event_viewer(payload)


def test_full_transport_and_its_select_are_unchanged(tmp_path):
    kernel = lh.TransportKernel(spectral_medium(), detector(), config(tmp_path))
    full = kernel.transport(track())
    assert full.charge_components_pe.shape[1] == 3
    selected = full.select("0+1")
    np.testing.assert_allclose(selected.charge_pe,
                               full.charge_components_pe[:, :2].sum(axis=1))


def test_prompt_never_touches_the_rte_cache(tmp_path, monkeypatch):
    from lighthit import cache, directional, transport, _spectral_fold

    def forbidden(*args, **kwargs):
        raise AssertionError("transport_prompt must not use the RTE machinery")

    monkeypatch.setattr(cache.ResponseCache, "build", forbidden)
    monkeypatch.setattr(cache.ResponseCache, "load", forbidden)
    monkeypatch.setattr(directional.DirectionalCache, "build", forbidden)
    monkeypatch.setattr(directional.DirectionalCache, "load", forbidden)
    monkeypatch.setattr(transport.TransportKernel, "_cache_for", forbidden)
    monkeypatch.setattr(transport.TransportKernel, "_directional_cache_for", forbidden)
    monkeypatch.setattr(transport.TransportKernel, "transport", forbidden)
    monkeypatch.setattr(transport.TransportKernel, "build", forbidden)
    monkeypatch.setattr(_spectral_fold, "optional_folded_cache", forbidden)
    monkeypatch.setattr(_spectral_fold, "load_folded_cache", forbidden)
    empty = tmp_path / "no-cache"
    kernel = lh.TransportKernel(spectral_medium(), detector(),
                                config(empty, cache_policy="require"))
    for source in (track(), shower()):
        response = kernel.transport_prompt(source)
        assert response.charge_orders_pe[:, 1].sum() > 0
    assert not empty.exists() or not any(empty.iterdir())


# ----------------------------------------------------- source algorithms

def test_track_quadrature_converges(tmp_path):
    kernel = lh.TransportKernel(spectral_medium(), detector(), config(tmp_path))
    base = PromptConfig()
    fine = replace(base, a_gauss=6, front_width_ns=0.6, phi_uniform=32,
                   path_step=0.05, path_step_tail=0.1, grade_min=1e-6)
    reference = kernel.transport_prompt(track(), fine)
    charge = reference.charge_orders_pe[:, 1]
    scale = np.maximum(charge, 0.01)       # includes the 0.01 p.e. threshold
    errors = []
    for settings in (replace(base, a_gauss=2, front_width_ns=10.),
                     base,
                     replace(base, a_gauss=4, front_width_ns=1.25)):
        response = kernel.transport_prompt(track(), settings)
        errors.append(np.max(np.abs(response.bins_orders_pe[:, :, 1]
                                    - reference.bins_orders_pe[:, :, 1]).sum(axis=1)
                             / scale))
    assert errors[0] > errors[2]
    assert errors[1] < 5e-3 and errors[2] < 3e-3


def test_element_algorithms_agree_on_a_small_shower(tmp_path):
    kernel = lh.TransportKernel(spectral_medium(), detector(), config(tmp_path))
    source = shower()
    segment = kernel.transport_prompt(
        source, replace(PromptConfig(), element_method="segment"))
    ring = kernel.transport_prompt(
        source, replace(PromptConfig(), element_method="ring", ring_a_step_m=0.02))
    grouped = kernel.transport_prompt(
        source, replace(PromptConfig(), element_method="grouped", pixel_face=64,
                        ring_nodes=192))
    assert segment.metadata["prompt_algorithm"] == "segment"
    charge = segment.charge_orders_pe[:, 1]
    scale = np.maximum(charge, 0.01)
    assert np.max(np.abs(ring.charge_orders_pe[:, 1] - charge) / scale) < 3e-3
    assert np.max(np.abs(grouped.charge_orders_pe[:, 1] - charge) / scale) < 5e-2
    np.testing.assert_array_equal(ring.charge_orders_pe[:, 0],
                                  segment.charge_orders_pe[:, 0])


PENCIL_FIELDS = ("position", "direction", "w0", "w2", "time", "major",
                 "var_major", "var_minor", "pos_cov")


def test_compression_does_not_depend_on_its_cell_ranges(monkeypatch):
    from lighthit import prompt
    elements = lh.SyntheticShower.gaussian(
        lh.SourcePose([12., 3., 0.], [0.3, 0.2, 0.93], 0.),
        charged_track_length_m=20., elements=400, seed=5).elements
    segments = prompt._element_segments(elements)
    settings = PromptConfig(pixel_face=16, ring_nodes=48)
    whole = prompt.compress(elements, segments, settings)
    # one cell per range at most: many ranges, run in parallel
    monkeypatch.setattr(prompt, "_COMPRESS_RANGE_NODES", settings.ring_nodes)
    split = prompt.compress(elements, segments, settings)
    for name in PENCIL_FIELDS:
        np.testing.assert_array_equal(split[name], whole[name])
    # a single table filled in element order gives the same pencils
    start = np.ascontiguousarray(elements.start_m, float)
    direction = np.ascontiguousarray(elements.direction, float)
    length = np.asarray(elements.length_m, float)
    ends = start + length[:, None] * direction
    origin = np.minimum(start.min(0), ends.min(0)) - settings.cell_m
    high = np.maximum(start.max(0), ends.max(0)) + settings.cell_m
    dims = np.ceil((high - origin) / settings.cell_m).astype(np.int64) + 1
    arrays = (start, direction, length, np.asarray(elements.coefficient0, float),
              np.asarray(elements.coefficient2, float), np.asarray(segments.cone, float),
              np.asarray(elements.start_ns, float), np.asarray(elements.end_ns, float))
    a_step = 0.5 * settings.cell_m
    _, element, sub = kernels.subsegment_cells(start, direction, length, arrays[6], arrays[7],
                                               settings.cell_m, a_step, origin, dims)
    keys, rows = kernels.compress_subsegments(
        *arrays, settings.cell_m, settings.pixel_face, settings.ring_nodes, a_step, origin,
        dims, element, sub, 1 << 16)
    single = kernels.finish_pencils(keys, rows, np.argsort(keys))
    for name, value in zip(PENCIL_FIELDS, single):
        np.testing.assert_array_equal(whole[name], value)
    assert np.sum(whole["w0"]) == pytest.approx(np.sum(elements.coefficient0), rel=1e-12)


def test_grouped_blocks_keep_their_scratch_within_the_budget(monkeypatch):
    from lighthit import prompt
    monkeypatch.setattr(prompt, "_thread_count", lambda: 8)
    assert prompt._grouped_blocks(100, 1000) == prompt._blocks(100)
    pencils = 3_000_000
    blocks = prompt._grouped_blocks(100, pencils)
    assert 1 <= blocks < 8
    assert blocks * prompt._GROUPED_BYTES_PER_PENCIL * pencils <= prompt._GROUPED_SCRATCH_BYTES
    assert prompt._grouped_blocks(100, 10 ** 9) == 1


def graded_nodes(low, high, count, power=3):
    """Gauss-Legendre in t with x = low + (high-low) t^power: graded at low."""
    t, w = np.polynomial.legendre.leggauss(count)
    t = 0.5 * (t + 1)
    w = 0.5 * w
    return low + (high - low) * t ** power, (high - low) * power * t ** (power - 1) * w


def test_isotropic_point_limit_matches_single_py(tmp_path):
    """Cones whose axes are uniform on the sphere form an isotropic flash.

    Only the angle gamma between an axis and the module direction matters for
    an isotropic module, with density d(cos gamma)/2.  The cone passes through
    the module direction at gamma = theta_C, where the azimuth-integrated
    order 1 has a logarithmic singularity, so cos(gamma) is integrated with
    Gauss-Legendre rules graded towards cos(theta_C) from both sides.
    """
    cone = 1.0 / 1.35
    upper_x, upper_w = graded_nodes(cone, 1.0, 40)
    lower_x, lower_w = graded_nodes(cone, -1.0, 60)
    cosine = np.r_[upper_x, lower_x]
    weight = np.r_[upper_w, -lower_w] / 2          # d(cos gamma)/2, positive
    sine = np.sqrt(1 - cosine ** 2)
    axes = np.column_stack((cosine, sine, np.zeros_like(cosine)))   # module on +x
    count = len(axes)
    length = np.full(count, 1e-5)
    elements = lh.SpectralLightElements(
        -0.5 * length[:, None] * axes, axes, length, weight, np.zeros(count),
        np.ones(count), np.zeros(count), np.zeros(count), np.arange(count),
        np.arange(count), 1.35)
    medium = lh.SpectralMedium([449., 451.], [0.05, 0.05], [0.03, 0.03],
                               [1.34, 1.34], [1.37, 1.37], g=0.9)
    radius = np.array([12., 25., 40.])
    positions = np.column_stack((radius, np.zeros(3), np.zeros(3)))
    isotropic = lh.DetectorArray(positions, [[1., 0., 0.]] * 3, 1.0,
                                 lambda x: np.ones_like(np.asarray(x, float)),
                                 lambda w: np.ones_like(np.asarray(w, float)))
    kernel = lh.TransportKernel(medium, isotropic, config(
        tmp_path, wavelength_nodes=1, wavelength_range_nm=(449.5, 450.5),
        relative_time_edges_ns=np.arange(-5., 400., 5.)))
    photons = kernel.wavelength.weight_nm[0] / kernel.wavelength.wavelength_nm[0] ** 2
    band = medium.band(float(kernel.wavelength.wavelength_nm[0]))
    exact = [single_spectrum(np.array([0.]), r, None, band, backend="numpy")[0][0].real
             for r in radius]
    for method in ("ring", "segment"):
        response = kernel.transport_prompt(
            elements, replace(PromptConfig(), element_method=method))
        for index, r in enumerate(radius):
            assert response.charge_orders_pe[index, 1] / photons == pytest.approx(
                exact[index], rel=2e-3)
    edges = kernel.config.relative_time_edges_ns + response.time_origin_ns[1]
    reference = single_bins(edges, radius[1], None, band, backend="numpy")
    assert (np.abs(response.bins_orders_pe[1, :, 1] / photons - reference).sum()
            < 5e-3 * exact[1])


@pytest.mark.parametrize("make_source", [track, shower])
def test_backward_monte_carlo_agrees(tmp_path, make_source):
    """Independent adjoint next-event estimator (experimental) vs prompt."""
    from lighthit.experimental.prompt_backward import backward_order1
    kernel = lh.TransportKernel(spectral_medium(), detector(), config(tmp_path))
    source = make_source()
    prompt = kernel.transport_prompt(source, PromptConfig(element_method="segment"))
    q = prompt.charge_orders_pe[:, 1]
    top = np.argsort(-q)[:3]
    mc = backward_order1(kernel, source, top, prompt.time_origin_ns[top],
                         samples=400_000, batches=16, seed=7)
    assert np.all(np.abs(mc["charge"] - q[top]) < 5 * mc["charge_se"] + 2e-3 * q[top])
    window = prompt.bins_orders_pe[top, :, 1].sum(axis=1)
    assert np.all(np.abs(mc["bins"].sum(axis=1) - window)
                  < 5 * mc["charge_se"] + 2e-3 * q[top])
