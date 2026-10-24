"""Coarse angular integration, emission conservation and charge invariances."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import i0e, i1e

pytest.importorskip('numba')
from lighthit import _prompt_numba as kernels
from lighthit import _prompt_screen as screen
from lighthit import prompt
from test_prompt import (_grouped_test_arguments, shower, spectral_medium,
                         detector, config)
import lighthit as lh


@pytest.mark.parametrize('sigma,mu', [(.05, 0.), (.05, .03), (.08, .05), (.1, .1)])
def test_radial_rule_matches_inverse_weighted_moments(sigma, mu):
    tangent, weights = kernels._prepare_gauss_inverse(kernels._GAUSS_TAU, kernels._GAUSS_W)
    w0, t0, w1, t1 = screen.inverse_radius_rule(mu, 0., sigma**2, sigma**2,
                                               tangent, weights)
    x = mu**2 / (4*sigma**2)
    inverse = np.sqrt(np.pi/2)/sigma*i0e(x)
    radial = sigma*np.sqrt(np.pi/2)*((1+2*x)*i0e(x)+2*x*i1e(x))
    assert 0 < t0 < t1 and min(w0, w1) > 0
    np.testing.assert_allclose([w0*t0**j+w1*t1**j for j in range(4)],
                              [inverse, 1., radial, mu**2+2*sigma**2], rtol=2e-10)


@pytest.mark.parametrize('sigma,mu', [(.05, 0.), (.08, .05), (.1, .1)])
def test_two_nodes_recover_hg_curvature_lost_at_the_mean(sigma, mu):
    # Independent angular-path integral, averaged over an isotropic Gaussian.
    x, w = np.polynomial.legendre.leggauss(160)
    def regular(theta):
        chi = theta+(np.pi-theta)*(x+1)/2
        g = .9
        hg = (1-g*g)/(4*np.pi*(1+g*g-2*g*np.cos(chi))**1.5)
        path = np.cos(theta)+np.sin(theta)*np.tan(chi/2)
        return np.sum(w*hg*np.exp(-3.5*path))*(np.pi-theta)/2
    def measure(t):
        return np.exp(-(t-mu)**2/(2*sigma**2))*i0e(mu*t/sigma**2)/sigma**2
    exact = quad(lambda t: measure(t)*regular(t), 0, mu+10*sigma, epsabs=1e-10)[0]
    tangent, weights = kernels._prepare_gauss_inverse(kernels._GAUSS_TAU, kernels._GAUSS_W)
    w0, t0, w1, t1 = screen.inverse_radius_rule(mu, 0., sigma**2, sigma**2,
                                               tangent, weights)
    one = (w0+w1)*regular(1/(w0+w1))
    two = w0*regular(t0)+w1*regular(t1)
    assert one/exact < .96
    assert abs(two/exact-1) < .015


@pytest.mark.parametrize('factor', [1, 2, 4])
def test_nested_pencils_conserve_emission_at_every_parent(factor):
    elements = shower().elements
    cfg = prompt.PromptConfig(screen_refine_factor=factor)
    tree = screen.compress_screen(elements, prompt._element_segments(elements), cfg)
    roots = np.flatnonzero(tree['hierarchy_parent'] == -2)
    for root in roots:
        child = slice(root+1, tree['next_node'][root])
        np.testing.assert_allclose(tree['w0'][child].sum(), tree['w0'][root], rtol=1e-12)
        np.testing.assert_allclose(tree['w2'][child].sum(), tree['w2'][root], rtol=1e-12)
    np.testing.assert_allclose(tree['w0'][roots].sum(), elements.coefficient0.sum(), rtol=1e-12)


def test_zero_emission_builds_an_empty_hierarchy():
    elements = shower().elements
    elements = replace(elements, coefficient0=np.zeros(len(elements)), coefficient2=np.zeros(len(elements)))
    tree = screen.compress_screen(elements, prompt._element_segments(elements), prompt.PromptConfig())
    assert not len(tree['w0']) and not len(tree['next_node'])


def test_charge_screen_ignores_emission_clocks_and_preserves_thread_results(monkeypatch):
    args = list(_grouped_test_arguments())
    args[41] = 2
    options = dict(hierarchy_parent=np.full(2, -1, np.int64), next_node=np.array([1, 2]))
    monkeypatch.setattr(prompt, '_thread_count', lambda: 1)
    reference = screen.screen_order1(*args, False, **options)
    for shift in [-1e6, 1e6]:
        moved = list(args)
        moved[9] = args[9]+shift
        moved[3] = args[3]-shift
        monkeypatch.setattr(prompt, '_thread_count', lambda: 2)
        actual = screen.screen_order1(*moved, False, **options)
        np.testing.assert_array_equal(actual[0], reference[0])
        np.testing.assert_array_equal(actual[2], reference[2])


@pytest.mark.parametrize('fine_charge', [1., 1.5])
def test_fine_om_never_rescales_an_unselected_om(tmp_path, monkeypatch, fine_charge):
    kernel = lh.TransportKernel(spectral_medium(), detector(2), config(tmp_path, threshold_pe=1.))
    source = shower().elements
    pencils = {key: np.zeros((1, 3)) for key in ['position', 'direction', 'major']}
    pencils.update({key: np.zeros(1) for key in ['w0', 'w2', 'time', 'var_major', 'var_minor']})
    pencils['pos_cov'] = np.zeros((1, 6))
    pencils['ring_nodes_deposited'] = np.array(1)
    monkeypatch.setattr(prompt, 'compress', lambda *a: pencils)
    tree = dict(pencils, hierarchy_parent=np.array([-1]), next_node=np.array([1]))
    monkeypatch.setattr(screen, 'compress_screen', lambda *a: tree)
    monkeypatch.setattr(prompt, '_min_distance_to_segments', lambda *a: np.ones(2))
    def ballistic(field, positions, *args, **kwargs):
        return np.zeros(len(positions)), np.zeros((len(positions), len(kernel.config.relative_time_edges_ns)-1))
    monkeypatch.setattr(prompt, '_ballistic_backend', lambda: ballistic)
    def grouped(*args, **kw):
        q = np.array([fine_charge, 0.]) if args[42] else np.array([2., 1e-6])
        return q, np.zeros((2, len(kernel.config.relative_time_edges_ns)-1)), np.ones(2), np.ones(2)
    monkeypatch.setattr(kernels, 'grouped_pencils_order1', grouped)
    monkeypatch.setattr(screen, 'screen_order1', grouped)
    result = kernel.transport_prompt(source, prompt.PromptConfig(element_method='grouped'))
    assert result.charge_orders_pe[1, 1] == 1e-6
    assert 'coarse_calibration' not in result.metadata['order1_screen']


def test_hierarchy_is_independent_of_compression_ranges(monkeypatch):
    elements = shower().elements
    segments = prompt._element_segments(elements)
    cfg = prompt.PromptConfig()
    whole = screen.compress_screen(elements, segments, cfg)
    monkeypatch.setattr(prompt, '_COMPRESS_RANGE_NODES', 24)
    split = screen.compress_screen(elements, segments, cfg)
    for name in whole:
        np.testing.assert_array_equal(split[name], whole[name])
