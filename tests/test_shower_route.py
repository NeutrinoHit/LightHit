"""Shower route (lighthit.shower) and band-folded order >=2 kernels.

Synthetic inputs only; measured G4/BGVD accuracy is in docs/accuracy.md.
"""
from dataclasses import replace

import numpy as np
import pytest

import lighthit as lh

pytest.importorskip("numba")
from lighthit.band_fold import BandFold, fold_bundle, load_folded, save_folded  # noqa: E402
from lighthit.shower import ShowerConfig, transport_shower  # noqa: E402
from lighthit.shower.ge2 import apply_folded, node_sources  # noqa: E402

from test_prompt import config, detector, spectral_medium  # noqa: E402


def shower(elements=80, length=10.0, seed=3):
    source = lh.SyntheticShower.gaussian(
        lh.SourcePose([12., 3., 0.], [0.3, 0.2, 0.93], 0.),
        charged_track_length_m=length, elements=elements, seed=seed)
    return replace(source.elements, cone_model="spectral")


@pytest.fixture(scope="module")
def kernel4(tmp_path_factory):
    return lh.TransportKernel(
        spectral_medium(), detector(count=5),
        config(tmp_path_factory.mktemp("cache"), wavelength_nodes=4,
               omega_per_ns=np.linspace(0., 1.2, 9), scattering_degree=4, source_degree=4,
               azimuthal_degree=2, radial_nodes=16, threshold_pe=0.0))


def test_band_fold_interpolation_keeps_node_bands_exact(kernel4):
    fold = BandFold.of(kernel4, rank=2)
    nodes = list(fold.bands)
    for ell in range(fold.interpolation.shape[0]):
        np.testing.assert_allclose(fold.interpolation[ell][nodes], np.eye(2), atol=1e-9)
    assert fold.residual.max() <= 1.0 + 1e-12
    # a full-rank fold interpolates nothing
    full = BandFold.of(kernel4, rank=4)
    assert full.residual.max() < 1e-8


def test_band_fold_grows_bands_greedily_beyond_the_exhaustive_limit(kernel4, monkeypatch):
    import lighthit.band_fold as band_fold
    best = BandFold.of(kernel4, rank=1)
    monkeypatch.setattr(band_fold, "EXHAUSTIVE_SUBSETS", 1)
    assert BandFold.of(kernel4, rank=1).bands == best.bands   # the first step is exhaustive
    greedy = BandFold.of(kernel4, rank=3)
    assert list(greedy.bands) == sorted(set(greedy.bands)) and len(greedy.bands) == 3
    assert set(best.bands) <= set(greedy.bands)
    assert BandFold.of(kernel4, rank=4).bands == (0, 1, 2, 3)


def test_prepared_fold_is_reused_only_for_its_own_settings(kernel4, tmp_path):
    import json
    from lighthit.shower import ShowerTransport
    from lighthit.shower.data import ShowerData
    every = np.arange(len(kernel4.config.omega_per_ns))
    store = kernel4.ge2_kernel_store()
    bundle = store.bundle(kernel4, every)
    store.save(tmp_path / "pack")
    optimal = BandFold.of(kernel4, rank=2).bands
    bands = (0, 3) if optimal != (0, 3) else (1, 2)
    save_folded(fold_bundle(bundle, BandFold.of(kernel4, rank=2, bands=bands)), tmp_path / "fold")
    data = ShowerData.of(kernel4.config.cache_directory, tmp_path / "pack", tmp_path / "fold")

    def tables(rank, frequencies=len(every)):
        route = object.__new__(ShowerTransport)
        route.data, route.kernel, route.threads, route._tables, route._compiled = (
            data, kernel4, None, None, {})
        route.config = ShowerConfig(frequency_nodes=frequencies, band_rank=rank)
        return route.tables()

    assert tables(2)["fold"].bands == bands                         # prepared: its bands
    assert tables(3)["fold"].bands == BandFold.of(kernel4, rank=3).bands   # other rank: in memory
    fewer = tables(2, frequencies=5)                                 # other frequencies: in memory
    assert fewer["fold"].bands == optimal and len(fewer["folded"]["omega"]) == 5
    meta = json.loads((tmp_path / "fold" / "folded.json").read_text())
    meta["band_fold"]["worst_residual_design"] *= 1.5
    (tmp_path / "fold" / "folded.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="another medium or detector"):
        tables(2)


def _per_band(kernel, elements, sources_by_band, positions, looks, area):
    from lighthit._shower_basis import apply_multi
    bundle = kernel.ge2_kernel_store().bundle(kernel, np.arange(len(kernel.config.omega_per_ns)))
    alpha = kernel.acceptance()["alpha"]
    total = 0
    for lam, item in enumerate(sources_by_band):
        band = dict(bundle, coefficients=bundle["coefficients"][lam:lam + 1],
                    absorption=bundle["absorption"][lam:lam + 1], speed=bundle["speed"][lam:lam + 1])
        total = total + apply_multi(band, item, item, positions, looks, alpha, np.ones(1),
                                    np.zeros(1), (item.summary["band_scale"] * area)[None, :])
    return total


def test_full_rank_fold_converges_to_the_per_band_apply(kernel4, tmp_path):
    """Full rank interpolates nothing: only the cubic re-fit of the folded
    radial kernels remains, converging at fourth order with subdivision
    (this test grid has 16 radial nodes; the production cache has 220)."""
    from lighthit.axial_source import AxisFrame
    elements = shower()
    fold = BandFold.of(kernel4, rank=4)
    omega = kernel4.config.omega_per_ns
    phase_max = float(np.max(fold.phase_index))
    geometry = elements.field(0, phase_index=phase_max)
    frame = AxisFrame.of(geometry)
    basis = lh.ShowerBasis(cell_m=0.5)
    nodes = node_sources(kernel4, elements, basis, fold, omega, frame, geometry)
    unscaled = []
    for band, item in zip(fold.bands, nodes):
        item = replace(item, channels=item.channels / fold.scale[band])
        item.summary["band_scale"] = fold.scale[band]
        unscaled.append(item)
    order = np.argsort(fold.bands)
    positions = kernel4.detector.positions_m
    looks = -kernel4.detector.orientations
    area = kernel4.detector.effective_area_m2
    reference = _per_band(kernel4, elements, [unscaled[i] for i in order], positions, looks, area)
    bundle = kernel4.ge2_kernel_store().bundle(kernel4, np.arange(len(omega)))
    scale = np.max(np.abs(reference))
    errors = []
    for subdivide in (1, 2, 4):
        folded = fold_bundle(bundle, fold, subdivide=subdivide)
        result = apply_folded(folded, nodes, positions, looks, kernel4.acceptance()["alpha"],
                              area, kernel4.config.radial_range_m)
        errors.append(np.max(np.abs(result - reference)) / scale)
    assert errors[1] < errors[0] / 8 and errors[2] < errors[1] / 8
    assert errors[2] < 2e-4
    save_folded(folded, tmp_path / "fold")
    again = load_folded(tmp_path / "fold")
    np.testing.assert_array_equal(np.asarray(again["coefficients"]), folded["coefficients"])
    assert again["band_fold"]["bands"] == list(fold.bands)


def test_shower_route_matches_the_per_band_basis_route(kernel4):
    elements = shower()
    prompt = lh.PromptConfig(screen=False)
    reference = kernel4.transport_hybrid(elements, frequency_nodes=9,
                                         ge2_basis=lh.ShowerBasis(cell_m=0.5),
                                         prompt_config=prompt)
    route = transport_shower(kernel4, elements, config=ShowerConfig(
        basis=lh.ShowerBasis(cell_m=0.5), prompt=prompt, frequency_nodes=9, band_rank=4,
        fold_subdivide=8))
    np.testing.assert_array_equal(route.charge_components_pe[:, :2],
                                  reference.charge_components_pe[:, :2])
    np.testing.assert_array_equal(route.active, reference.active)
    q = reference.charge_components_pe[:, 2]
    assert np.all(np.abs(route.charge_components_pe[:, 2] - q) <= 0.02 * np.abs(q) + 1e-12)
    bins = reference.components_pe[:, :, 2]
    assert (np.abs(route.components_pe[:, :, 2] - bins).sum() <= 0.05 * np.abs(bins).sum())
    assert route.metadata["route"] == "lighthit.shower"
    assert route.metadata["ge2_predictor"]["fallback_to_exact"] is False


def test_predicted_charge_never_activates_an_om(kernel4, monkeypatch):
    import lighthit.shower.ge2 as ge2
    original = ge2.collapse

    def tiny(sources, pieces):
        return [replace(item, channels=item.channels * 1e-6) for item in original(sources, pieces)]
    monkeypatch.setattr(ge2, "collapse", tiny)
    kernel = lh.TransportKernel(kernel4.medium, kernel4.detector,
                                replace(kernel4.config, threshold_pe=1e-6))
    elements = shower()
    route = transport_shower(kernel, elements, config=ShowerConfig(
        basis=lh.ShowerBasis(cell_m=0.5), prompt=lh.PromptConfig(), frequency_nodes=9,
        band_rank=2))
    predictor = route.metadata["ge2_predictor"]
    # a prediction 1e6 too small fails its margin: every reachable OM is exact
    assert predictor["fallback_to_exact"] is True
    levels = np.asarray(route.metadata["ge2_charge_level"])
    assert np.all(levels[levels > 0] == 2)


def test_fused_prompt_bands_keep_direct_light_and_stay_close(kernel4):
    elements = shower(elements=200)
    plain = kernel4.transport_prompt(elements, lh.PromptConfig(screen=False, element_method="grouped"))
    fused = kernel4.transport_prompt(elements, lh.PromptConfig(screen=False, element_method="grouped",
                                                               fuse_bands=True))
    np.testing.assert_array_equal(fused.charge_orders_pe[:, 0], plain.charge_orders_pe[:, 0])
    q1, f1 = plain.charge_orders_pe[:, 1], fused.charge_orders_pe[:, 1]
    assert np.abs(f1.sum() - q1.sum()) <= 0.05 * q1.sum()
    assert fused.metadata["order1_fused_bands"]["pixel_face"] == 64
    assert plain.metadata["order1_fused_bands"] is None


def test_shower_config_rejects_track_and_pair_options():
    with pytest.raises(ValueError):
        ShowerConfig(basis=lh.ShowerBasis(track_uid=0))
    with pytest.raises(ValueError):
        ShowerConfig(band_rank=0)


def test_delayed_copy_is_the_same_light_on_a_shifted_clock(kernel4):
    """A copy of the source emitted 250 ns later (50 bins of 5 ns) adds the
    same >=2 waveform shifted by exactly 50 bins, read on its own clock."""
    first = shower()
    late = replace(first, start_ns=first.start_ns + 250.0, end_ns=first.end_ns + 250.0,
                   row_index=first.row_index + len(first))
    names = ("start_m", "direction", "length_m", "coefficient0", "coefficient2", "beta",
             "start_ns", "end_ns", "row_index", "uid")
    both = replace(first, **{name: np.concatenate([getattr(first, name), getattr(late, name)])
                             for name in names})
    config = ShowerConfig(basis=lh.ShowerBasis(cell_m=0.5), prompt=lh.PromptConfig(screen=False),
                          frequency_nodes=9, band_rank=2, delayed_cell_ns=100.0)
    alone = transport_shower(kernel4, first, config=config)
    pair = transport_shower(kernel4, both, config=config, time_origin_ns=alone.time_origin_ns)
    assert pair.metadata["delayed_light"]["groups_in_window"] == 1
    np.testing.assert_allclose(pair.charge_components_pe[:, 2], 2 * alone.charge_components_pe[:, 2],
                               rtol=1e-6, atol=1e-15)
    # The copy's absolute spectrum is A's times exp(i w 250): demodulated by
    # origin + 250 it is A's, read on the edges seen 250 ns later.
    from lighthit.readout import inverse_bins
    relative = alone.spectrum_ge2_pe * np.exp(-1j * alone.omega_per_ns[:, None]
                                              * alone.time_origin_ns[None, :])
    copy = inverse_bins(alone.omega_per_ns, relative, kernel4.config.relative_time_edges_ns - 250.0)
    expected = alone.components_pe[:, :, 2] + copy
    scale = np.abs(expected).max()
    # Away from the window start the copy is the original shifted by 50 bins.
    np.testing.assert_allclose(copy[:, 50:], alone.components_pe[:, :-50, 2], rtol=1e-9,
                               atol=1e-12 * scale)
    assert np.max(np.abs(pair.components_pe[:, :, 2] - expected)) <= 1e-6 * scale


def test_build_prepares_a_saveable_ge2_pack(tmp_path):
    """build() fills the >=2 pack, so the saved pack is complete and reusable."""
    from lighthit.shower_basis import Ge2KernelStore

    def kernel():
        return lh.TransportKernel(
            spectral_medium(), detector(count=5),
            config(tmp_path / "cache", wavelength_nodes=4, omega_per_ns=np.linspace(0., 1.2, 9),
                   scattering_degree=4, source_degree=4, azimuthal_degree=2,
                   radial_nodes=16, threshold_pe=0.0))

    source = shower()
    built = kernel().build(source=source)
    assert len(built.ge2_kernel_store()) == 1
    built.ge2_kernel_store().save(tmp_path / "pack")
    loaded = Ge2KernelStore.load(tmp_path / "pack")
    assert len(loaded) == 1
    settings = ShowerConfig(frequency_nodes=9)
    first = transport_shower(built, source, config=settings)
    again = transport_shower(kernel(), source, config=settings, store=loaded)
    np.testing.assert_allclose(again.bins_pe, first.bins_pe, rtol=1e-12, atol=1e-15)
    flash = lh.IsotropicFlash.monochromatic([12., 3., 0.], 1e6, 450.)
    assert len(kernel().build(source=flash).ge2_kernel_store()) == 0   # one line: no pack


def test_shower_transport_object_and_detail_threshold(tmp_path):
    """ShowerTransport on prepared tables; detail_threshold_pe keeps active at threshold_pe."""
    from lighthit.shower import ShowerData, ShowerTransport
    from lighthit.shower.api import FREQUENCIES_PER_NS, READOUT_PER_NS

    def kernel(**changes):
        return lh.TransportKernel(
            spectral_medium(), detector(count=5),
            config(tmp_path / "cache", wavelength_nodes=4, omega_per_ns=FREQUENCIES_PER_NS,
                   readout_omega_per_ns=READOUT_PER_NS, scattering_degree=4, source_degree=4,
                   azimuthal_degree=2, radial_nodes=16, **changes))

    source = shower()
    built = kernel(threshold_pe=0.0).build(source=source)
    built.ge2_kernel_store().save(tmp_path / "pack")
    data = ShowerData.of(tmp_path / "cache", tmp_path / "pack")
    settings = ShowerConfig(band_rank=2)
    direct = transport_shower(built, source, config=settings)
    route = ShowerTransport(kernel(threshold_pe=0.0), data, config=settings)
    np.testing.assert_allclose(route(source).bins_pe, direct.bins_pe, rtol=1e-12, atol=1e-15)
    charge = np.asarray(direct.charge_pe)
    threshold = float(np.quantile(charge[charge > 0], 0.6))
    plain = ShowerTransport(kernel(threshold_pe=threshold), data, config=settings)(source)
    lowered = ShowerTransport(kernel(threshold_pe=threshold, detail_threshold_pe=0.0),
                              data, config=settings)(source)
    np.testing.assert_array_equal(lowered.active, np.asarray(lowered.charge_pe) >= threshold)
    assert lowered.metadata["detail_threshold_pe"] == 0.0
    levels = lambda r: np.asarray(r.metadata["ge2_charge_level"])
    assert np.count_nonzero(levels(lowered) == 2) >= np.count_nonzero(levels(plain) == 2)
