"""Certified packet-receiver omission for grouped secondary Q1."""
from types import SimpleNamespace

import numpy as np
import pytest

from lighthit import prompt
from lighthit.experimental.packet_skip import (bound_constants, packet_first_bounds,
                                               seed_packets, select_skips)
from lighthit.experimental.secondary_first import evaluate_secondary_first
from lighthit.experimental.track_first_bounds import combined_first_scattering_segment_bounds
from lighthit.sources import SpectralLightElements
from test_secondary_first import case as secondary_case
from test_track_first_bounds import case as bound_case


def test_numba_bound_reproduces_combined_python_bound_for_one_band():
    kernel, _ = bound_case()
    rng = np.random.default_rng(4)
    n = 7
    direction = rng.normal(size=(n, 3)); direction /= np.linalg.norm(direction, axis=1)[:, None]
    beta = np.array([1., .999, .9, .8, 1., .75, .95])
    c0 = rng.uniform(1., 3., n)
    source = SpectralLightElements(rng.uniform(-5., 5., (n, 3)), direction,
        rng.uniform(.05, 4., n), c0, -c0/beta**2, beta, np.zeros(n), np.ones(n),
        np.arange(n), np.arange(n), cone_model='spectral')
    receivers = np.array([[30., 0., 0.], [0., 80., 10.], [-150., 20., -40.], [3., 2., 250.]])
    tables = prompt._spectral_tables(kernel)
    fast = packet_first_bounds(source, receivers, np.ones(len(receivers)), tables,
                               bound_constants(kernel), azimuth_nodes=0)
    refined = packet_first_bounds(source, receivers, np.ones(len(receivers)), tables,
                                  bound_constants(kernel), azimuth_nodes=16)
    assert np.all(refined <= fast)
    assert fast.shape == (1, len(receivers))
    for j, receiver in enumerate(receivers):
        slow, _ = combined_first_scattering_segment_bounds(kernel, source, receiver)
        np.testing.assert_allclose(fast[0, j], slow.sum(), rtol=1e-10)
        assert fast[0, j] >= slow.sum()


def test_selection_respects_certificate_seeds_and_unsupported_pairs():
    rng = np.random.default_rng(1)
    bounds = rng.lognormal(-8., 4., (40, 6))
    bounds[3, 2] = np.inf
    bounds[5, :] = 0.
    seed = seed_packets(bounds, .5)
    assert seed[3, 2] and not seed[5].any()
    assert np.all(seed[np.argmax(np.where(np.isfinite(bounds), bounds, 0.), axis=0), np.arange(6)])
    charge = rng.uniform(1e-3, 1., 6)
    skip = select_skips(bounds, rng.uniform(1., 100., 40), seed, charge, 1e-3)
    assert not (skip & seed).any() and not skip[3, 2] and skip[5].all()
    omitted = np.where(skip, bounds, 0.).sum(0)
    assert np.all(omitted <= 1e-3*charge)
    assert skip.sum() > 6


@pytest.mark.parametrize('epsilon', [1e-3, .5])
def test_skipping_route_is_certified_against_full_secondary_q1(tmp_path, epsilon):
    kernel, source, packets = secondary_case(tmp_path)
    # A weak packet 250 m below both near modules, emitting away from them.
    far = SpectralLightElements(np.array([[11., 4., -250.]]), np.array([[0., 0., -1.]]),
        np.array([.15]), np.array([1e6]), np.array([-1e6]), np.array([1.]),
        np.array([5.]), np.array([5.5]), np.array([6]), np.array([4]), cone_model='spectral')
    source = SpectralLightElements(**{name: np.concatenate([getattr(source, name), getattr(far, name)])
        for name in ('start_m', 'direction', 'length_m', 'coefficient0', 'coefficient2', 'beta',
                     'start_ns', 'end_ns', 'row_index', 'uid')}, cone_model='spectral')
    from lighthit.experimental.muon_packets import SourcePacket
    packets = packets + [SourcePacket('general', np.array([6]), 4)]
    ids = np.array([1, 0, 2]); origins = np.array([.7, -.3, 0.])
    import lighthit as lh
    settings = lh.PromptConfig(element_method='grouped', screen=False, batch_charge_tolerance_pe=1e-6)
    q0, b0, _ = evaluate_secondary_first(kernel, source, packets, ids, origins, config=settings)
    q, b, meta = evaluate_secondary_first(kernel, source, packets, ids, origins, config=settings,
                                          skip_epsilon=epsilon, skip_seed_fraction=.3)
    skip = meta['packet_skip']
    assert skip['epsilon'] == epsilon and len(skip['bands']) == 9
    for band in skip['bands']:
        assert band['max_omitted_bound_over_seed_charge'] <= epsilon*(1+1e-9)
    # Omitted grouped Q1 is nonnegative mass below epsilon times the computed charge.
    assert np.all(q <= q0*(1+1e-12)+1e-30)
    assert np.all(q0-q <= epsilon*q*(1+1e-9)+1e-30)
    # Bins cover a finite window: their omitted L1 is bounded by the all-time
    # computed charge, not by the computed in-window bin sum.
    l1 = np.abs(b-b0).sum(1)
    assert np.all(l1 <= epsilon*q*(1+1e-9)+1e-30)
    if epsilon == .5:
        assert sum(band['skipped_pairs'] for band in skip['bands']) > 0


def test_packet_bounds_exceed_each_packet_q1_on_spectral_kernel(tmp_path):
    kernel, source, packets = secondary_case(tmp_path)
    import lighthit as lh
    settings = lh.PromptConfig(element_method='grouped', screen=False, batch_charge_tolerance_pe=0.)
    ids = np.array([1, 0]); origins = np.zeros(2)
    tables = prompt._spectral_tables(kernel)
    common = source.beta*tables['phase_index'].max() > 1.
    positions = kernel.detector.positions_m[ids]; areas = kernel.detector.effective_area_m2[ids]
    for packet in packets[1:]:
        q, _, _ = evaluate_secondary_first(kernel, source, [packet], ids, origins, config=settings)
        selected = np.sort(packet.indices[common[packet.indices]])
        bound = packet_first_bounds(source.subset(np.isin(np.arange(len(source)), selected)),
                                    positions, areas, tables, bound_constants(kernel)).sum(0)
        assert np.all(q <= bound)


@pytest.mark.parametrize('receiver', [[40., 0., 20.], [180., 0., -50.], [20., 0., 100.],
                                      [0., 60., 52.], [-30., 25., 5.], [3., 1., 300.]])
def test_azimuth_bound_exceeds_independent_single_scattering_integral(receiver):
    # Independent quadrature over segment position, Cherenkov azimuth and the
    # first flight; HG, exact attenuation and constant unit acceptance.
    from dataclasses import replace
    kernel, source = bound_case()
    source = replace(source, start_m=np.zeros((1, 3)), length_m=np.array([.2]),
                     coefficient0=np.array([.2]), coefficient2=np.array([-.2]))
    tables = prompt._spectral_tables(kernel)
    mus, mut, g = tables['mus'][0], tables['mut'][0], .9
    photons = .2*tables['s0w'][0]-.2*tables['s2w'][0]
    cone = 1/1.34; sincone = np.sqrt(1-cone**2)
    px, pw = np.polynomial.legendre.leggauss(256); phi = np.pi*(px+1); pw = pw/2
    ax, aw = np.polynomial.legendre.leggauss(8)
    y, v = np.polynomial.legendre.leggauss(400); t = (y+1)/2
    ell = 40*t/(1-t)**2; lw = v/2*40*(1+t)/(1-t)**3
    directions = np.column_stack((sincone*np.cos(phi), sincone*np.sin(phi), np.full(len(phi), cone)))
    value = 0.
    for a, w in zip((ax+1)*.1, aw*.1/.2):
        R = np.array(receiver)-np.array([0., 0., a])
        last = R-directions[:, None, :]*ell[None, :, None]
        d = np.linalg.norm(last, axis=2)
        chi = np.sum(directions[:, None, :]*last, axis=2)/d
        p = (1-g*g)/(4*np.pi*(1+g*g-2*g*chi)**1.5)
        value += w*np.sum(pw[:, None]*lw[None, :]*mus*np.exp(-mut*(ell[None, :]+d))*p/d**2)
    value *= photons
    bound = packet_first_bounds(source, np.array([receiver]), np.ones(1), tables,
                                bound_constants(kernel), azimuth_nodes=16)[0, 0]
    old = packet_first_bounds(source, np.array([receiver]), np.ones(1), tables,
                              bound_constants(kernel), azimuth_nodes=0)[0, 0]
    assert bound >= value and bound <= old


def test_joint_config_skip_options_are_validated():
    from lighthit.experimental.joint_muon import JointMuonConfig
    JointMuonConfig(first_backend='grouped', first_skip_epsilon=1e-3,
                    ge2_backend='joint', ge2_skip_epsilon=1e-3, ge2_skip_safety=3.)
    assert JointMuonConfig().first_backend == 'grouped' and JointMuonConfig().ge2_blas_precision == 'float32'
    for options in [dict(first_backend='atlas', first_skip_epsilon=1e-3, first_fuse_bands=False),
                    dict(first_backend='grouped', first_skip_epsilon=1.),
                    dict(ge2_backend='basis', ge2_skip_epsilon=1e-3),
                    dict(ge2_backend='joint', ge2_skip_epsilon=1e-3, ge2_skip_safety=.5),
                    dict(ge2_backend='joint', ge2_skip_epsilon=-1e-3)]:
        with pytest.raises(ValueError):
            JointMuonConfig(**options)


@pytest.mark.parametrize('epsilon', [0., 1e-3])
def test_band_fusion_route_runs_and_reports_units(tmp_path, epsilon):
    import lighthit as lh
    kernel, source, packets = secondary_case(tmp_path)
    ids = np.array([1, 0, 2]); origins = np.array([.7, -.3, 0.])
    settings = lh.PromptConfig(element_method='grouped', screen=False, batch_charge_tolerance_pe=1e-6)
    q0, b0, _ = evaluate_secondary_first(kernel, source, packets, ids, origins, config=settings)
    q, b, meta = evaluate_secondary_first(kernel, source, packets, ids, origins, config=settings,
                                          fuse_bands=True, skip_epsilon=epsilon)
    assert meta['fused_bands']['bands'] == list(range(9))
    assert np.all(np.isfinite(q)) and np.all(np.isfinite(b))
    # Fusion is an approximation; it must stay close on this small case.
    near = q0 > 1e-3*q0.max()
    np.testing.assert_allclose(q[near], q0[near], rtol=.1)
    if epsilon:
        assert all(row['max_omitted_bound_over_seed_charge'] <= epsilon*(1+1e-9)
                   for row in meta['packet_skip']['bands'])


def test_ge2_batch_options_are_validated():
    from lighthit.experimental.joint_muon import JointMuonConfig
    JointMuonConfig(ge2_backend='joint', ge2_operator='blas', ge2_batch_points=2000, ge2_blas_precision='float32')
    for options in [dict(ge2_backend='joint', ge2_operator='scalar', ge2_batch_points=100),
                    dict(ge2_backend='joint', ge2_operator='blas', ge2_batch_points=-1),
                    dict(ge2_backend='joint', ge2_operator='blas', ge2_blas_precision='half')]:
        with pytest.raises(ValueError):
            JointMuonConfig(**options)


def test_multi_source_blas_equals_sum_of_single_calls():
    # Synthetic pack: random coefficients on a log-radius grid; two frames.
    from lighthit.experimental.point_ge2_blas import apply_point_spectral_blas, apply_point_spectral_blas_multi
    rng = np.random.default_rng(7)
    degree, nlam, nfreq = 3, 2, 5
    pair_lambda = np.array([0, 1, 1, 2, 2, 2, 3, 3, 3, 3]); pair_mu = np.array([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    log_r = np.linspace(np.log(1.), np.log(400.), 12)
    coefficients = (rng.normal(size=(nlam, 11, degree+1, 10, 4, nfreq))
                    + 1j*rng.normal(size=(nlam, 11, degree+1, 10, 4, nfreq))).astype(np.complex64)
    bundle = dict(degree=degree, acceptance_degree=3, pair_lambda=pair_lambda, pair_mu=pair_mu,
                  omega=np.linspace(0, 1., nfreq), log_r=log_r, phase='flight',
                  absorption=np.array([.02, .03]), speed=np.array([.22, .21]), coefficients=coefficients)
    width = int(np.sum([2*min(l, 2)+1 for l in range(degree+1)]))
    receivers = rng.uniform(-60, 60, (5, 3)); looks = rng.normal(size=(5, 3)); looks /= np.linalg.norm(looks, axis=1)[:, None]
    alpha = np.array([1., .5, .2, .05]); scale = rng.uniform(.5, 1., (nlam, 5))
    items = []
    for shift in (np.zeros(3), np.array([10., -5., 3.])):
        frame = SimpleNamespace(centre_m=shift, rotate=lambda v: np.asarray(v))
        items.append((frame, rng.normal(size=(6, 3))*4+shift, rng.normal(size=(nlam, 6, width)),
                      rng.uniform(0., 50., 6), np.array([0, 2, 3, 4])))
    total = np.zeros((nfreq, 5), complex)
    for frame, pts, ang, clocks, targets in items:
        total[:, targets] += apply_point_spectral_blas(bundle, frame, pts, ang, clocks, receivers[targets], looks[targets],
                                                       alpha, scale[:, targets], azimuthal_degree=2, radial_window_m=(3., 300.))
    for precision, tol in (('float64', 1e-12), ('float32', 1e-5)):
        multi = apply_point_spectral_blas_multi(bundle, items, receivers, looks, alpha, scale, azimuthal_degree=2,
                                                radial_window_m=(3., 300.), precision=precision)
        assert np.abs(multi-total).max() <= tol*np.abs(total).max()
