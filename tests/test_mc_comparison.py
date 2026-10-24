"""Statistical errors of weighted photon references in the idea registry."""
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.harness.cases_quadrature import _by_order


@pytest.mark.parametrize("record_orders", [False, True])
def test_saved_mc_hits_preserve_particle_identity_and_order(tmp_path, record_orders):
    from experiments.photon_mc import _save_hits

    class Hits(SimpleNamespace):
        def __len__(self):
            return len(self.uid)

    hits = Hits(uid=[101, 102, 101], track_uid=[7, 9, 7],
                t_ns=[1., 2., 3.], phe=[.2, .4, .6])
    path = tmp_path / 'hits.npz'
    _save_hits(path, hits, {}, [np.array([0, 1]), np.array([2])] if record_orders else None)
    with np.load(path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['track_uid'], [7, 9, 7])
        assert saved['track_uid'].dtype == np.int64
        np.testing.assert_array_equal(saved['uid'], hits.uid)
        np.testing.assert_array_equal(saved['phe'], hits.phe)
        if record_orders:
            np.testing.assert_array_equal(saved['order'], [0, 1, 2])
        else:
            assert 'order' not in saved
    with pytest.raises(ValueError, match='one entry per hit'):
        _save_hits(path, hits, {}, [np.array([0, 1])])


@pytest.mark.parametrize("numerator,ratio,ratio_sigma", [(10., .5, .05), (40., 2., .2)])
def test_ratio_errors_follow_the_weighted_mc_denominator(numerator, ratio, ratio_sigma):
    # 100 equal-weight photons give MC = 20 +/- 2 pe. The deterministic
    # numerator 10 pe then gives LH/MC = 0.5 +/- 0.05, not +/- 0.1.
    kernel = SimpleNamespace(detector=SimpleNamespace(identifiers={"uid": [101]}))
    response = SimpleNamespace(charge_components_pe=np.array([[numerator, 0., 0.]]))
    hits = [(np.full(100, 101), np.full(100, .2), np.zeros(100, int))]
    result = _by_order(kernel, response, hits, np.array([20.]), [0., 50.])
    row, headline = result["by_distance"][0], result["headline"]
    assert row["total_ratio"] == pytest.approx(ratio)
    assert row["mc_relative_sigma"] == pytest.approx(.1)
    assert row["total_sigma"] == pytest.approx(ratio_sigma)
    assert row["ratio_sigma"] == [pytest.approx(ratio_sigma), None, None]
    assert headline["total_ratio_sigma"] == pytest.approx(ratio_sigma)
    assert headline["order0_sigma"] == pytest.approx(ratio_sigma)
    assert headline["order0_mc_relative_sigma"] == pytest.approx(.1)
    assert headline["order1_ratio"] is None
    assert headline["order1_sigma"] is None


def test_no_mc_hits_does_not_report_a_ratio_or_nan():
    kernel = SimpleNamespace(detector=SimpleNamespace(identifiers={"uid": [101]}))
    response = SimpleNamespace(charge_components_pe=np.array([[1., 0., 0.]]))
    with np.errstate(divide="raise", invalid="raise"):
        result = _by_order(kernel, response, [], np.array([20.]), [0., 50.])
    assert result["by_distance"] == []
    assert result["headline"]["total_ratio"] is None
    assert result["headline"]["total_ratio_sigma"] is None
