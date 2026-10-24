"""A positive first-entry bound permits omission at the requested tolerance."""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('numba')
from test_ballistic_linear_control import setup
from lighthit._surface_ballistic import PreparedBallistic
from lighthit._surface_ballistic_bounds import full_charge_bounds
from lighthit._surface_ballistic_charge import boundary_charge
from lighthit.ballistic_interval import BallisticIntervalConfig
from lighthit._ballistic_interval_numba import interval_candidates, _phase_order, _supported_panels


@pytest.mark.parametrize('rho', [.13, .2, 2., 50., 500.])
@pytest.mark.parametrize('length', [.005, .5])
def test_full_bound_dominates_independent_positive_entry_quadrature(rho, length):
    kernel, make = setup(rho)
    source = make([[0., 0., 0.]], [length])
    prepared = PreparedBallistic(kernel, source, BallisticIntervalConfig())
    t, d = prepared.tables, kernel.detector
    pairs = interval_candidates(d.positions_m, prepared.radii, np.inf, source.start_m,
        source.direction, source.length_m, source.coefficient0, source.beta, t['phase_nodes'])
    upper = full_charge_bounds(*pairs, d.positions_m, d.effective_area_m2,
        prepared.radii, source.start_m, source.direction, source.length_m,
        source.coefficient0, source.beta, t['panel_edges'], t['phase_nodes'],
        t['extinction_nodes'], prepared.eta_upper, prepared.angular_bounds[0])
    exact, _ = boundary_charge(kernel, source, 12, 32, 12, tables=t, pairs=pairs)
    total = np.bincount(pairs[0], weights=upper, minlength=len(d))
    assert np.all(exact <= total*(1+1e-12)+1e-20)
    assert np.all(total >= 0.)


@pytest.mark.parametrize('phases', [[1.4,1.38,1.38,1.35,1.32],
    [1.32,1.35,1.38,1.38,1.4], [1.35,1.35,1.35], [1.32,1.38,1.35,1.4]])
def test_supported_panel_range_keeps_all_overlapping_spectral_intervals(phases):
    phases=np.asarray(phases)
    _,_,monotonic,increasing,ordered=_phase_order(phases)
    for low,high in [(1.1,1.5),(1.35,1.38),(1.349999,1.350001),(1.38,1.39),(1.4,np.inf)]:
        first,last=_supported_panels(low,high,monotonic,increasing,ordered)
        for panel,(left,right) in enumerate(zip(phases[:-1],phases[1:])):
            if max(left,right)>=low and min(left,right)<high:
                assert first<=panel<last


def test_whole_om_can_be_omitted_only_within_its_absolute_error_budget():
    kernel, make = setup(2.)
    source = make([[0., 0., 0.]], [.05])
    source = replace(source, coefficient0=source.coefficient0*1e-9,
                     coefficient2=source.coefficient2*1e-9)
    config = BallisticIntervalConfig(absolute_tolerance_pe=1e-6)
    prepared = PreparedBallistic(kernel, source, config)
    assert prepared.candidate_count > 0 and len(prepared.pairs[0]) == 0
    assert np.all(prepared.omitted_full > 0.)
    assert np.all(prepared.omitted_full <= config.absolute_tolerance_pe/8)
    reference, _ = boundary_charge(kernel, source, 12, 32, 12, tables=prepared.tables)
    charge = prepared.charge()
    binned = prepared.bins([0, 1], [0., 0.], np.linspace(-10., 30., 41))
    np.testing.assert_array_equal(charge.charge_pe, 0.)
    np.testing.assert_array_equal(binned.bins_pe, 0.)
    assert np.all(reference <= prepared.omitted_full)


def test_relative_tolerance_does_not_remove_bright_pairs_before_charge_is_known():
    kernel, make = setup(2.)
    source = make([[0., 0., 0.]], [.05])
    prepared = PreparedBallistic(kernel, source,
        BallisticIntervalConfig(relative_tolerance=.1, absolute_tolerance_pe=1e-15))
    assert len(prepared.pairs[0]) == prepared.candidate_count
    assert not prepared.omitted_full.any()


def test_cached_charge_reaches_one_bin_without_changing_entry_response():
    kernel, make = setup(2.)
    source = make([[0., 0., 0.]], [.05])
    config = BallisticIntervalConfig()
    prepared = PreparedBallistic(kernel, source, config)
    charge = prepared.charge()
    edges = np.array([-100., 100.])
    cached = prepared.bins([0, 1], [0., 0.], edges)
    direct = PreparedBallistic(kernel, source, config).bins([0, 1], [0., 0.], edges)
    assert cached.metadata['reused_charge_pairs'] == 2
    assert cached.metadata['integrated_pairs'] == 0
    tolerance = config.absolute_tolerance_pe+config.relative_tolerance*abs(direct.charge_pe)
    assert np.all(abs(cached.bins_pe-direct.bins_pe).sum(1) < tolerance)
    np.testing.assert_allclose(cached.bins_pe[:, 0], charge.charge_pe,
                               rtol=8*np.finfo(float).eps, atol=0.)


def test_charge_cache_does_not_assign_a_crossing_pair_to_a_single_bin():
    kernel, make = setup(2.)
    source = make([[0., 0., 0.]], [.05])
    prepared = PreparedBallistic(kernel, source, BallisticIntervalConfig())
    prepared.charge()
    edges = np.linspace(-1., 30., 1241)
    cached = prepared.bins([0, 1], [0., 0.], edges)
    assert cached.metadata['reused_charge_pairs'] == 0
    assert cached.metadata['integrated_pairs'] == 2


def test_partial_panel_reuse_respects_bins_and_receiver_reordering():
    kernel, make = setup(2.)
    source = make([[0., 0., 0.]], [.05])
    config = BallisticIntervalConfig()
    prepared = PreparedBallistic(kernel, source, config)
    prepared.charge()
    edges = np.arange(0., 31., 1.)
    cached = prepared.bins([1, 0], [0., 0.], edges)
    direct = PreparedBallistic(kernel, source, config).bins([1, 0], [0., 0.], edges)
    assert cached.metadata['reused_charge_panels'] > 0
    assert cached.metadata['integrated_pairs'] > 0
    tolerance = config.absolute_tolerance_pe+config.relative_tolerance*abs(direct.charge_pe)
    assert np.all(abs(cached.bins_pe-direct.bins_pe).sum(1) < tolerance)


def test_insufficient_memory_disables_panel_cache_without_changing_physics(monkeypatch):
    from lighthit import _memory
    kernel, make = setup(2.)
    source = make([[0., 0., 0.]], [.05])
    config = BallisticIntervalConfig()
    monkeypatch.setattr(_memory, 'available_bytes', lambda: 0)
    prepared = PreparedBallistic(kernel, source, config)
    assert not prepared.cache_panels
    charge = prepared.charge()
    assert prepared.panel_charge is None
    binned = prepared.bins([0, 1], [0., 0.], [-100., 100.])
    assert binned.metadata['reused_charge_panels'] == 0
    tolerance = config.absolute_tolerance_pe+config.relative_tolerance*abs(charge.charge_pe)
    assert np.all(abs(binned.bins_pe[:, 0]-charge.charge_pe) < tolerance)
