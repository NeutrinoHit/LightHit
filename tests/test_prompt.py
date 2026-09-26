"""Prompt transport: scattering orders 0 and exactly 1 without the RTE cache.

Lightweight synthetic checks only; real G4/BGVD validation is in
``scripts/prompt_benchmark.py`` (not part of the test suite).
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
