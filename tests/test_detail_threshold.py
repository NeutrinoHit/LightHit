"""KernelConfig.detail_threshold_pe: detail computed below the activity threshold."""
from types import SimpleNamespace

import numpy as np
import pytest

import lighthit as lh
from lighthit.transport import detail_threshold


def test_detail_threshold_lowers_selection_and_reports_active_at_the_threshold():
    kernel = SimpleNamespace(config=lh.KernelConfig(threshold_pe=0.01, detail_threshold_pe=1e-3))
    response = SimpleNamespace(charge_pe=np.array([0.02, 0.005, 0.0]),
                               active=np.array([True, True, False]), metadata={})
    with detail_threshold(kernel) as outer:
        assert kernel.config.threshold_pe == 1e-3 and kernel.config.detail_threshold_pe is None
        with detail_threshold(kernel) as inner:       # a source route inside an event route
            assert not inner.lowered
        assert inner.report(response) is response and response.active.tolist() == [True, True, False]
        outer.report(response, extra=np.array([0.0, 0.006, 0.0]))
    assert kernel.config.threshold_pe == 0.01 and kernel.config.detail_threshold_pe == 1e-3
    assert response.active.tolist() == [True, True, False]   # 0.005 + 0.006 of other sources
    outer.report(response)
    assert response.active.tolist() == [True, False, False]
    assert response.metadata["threshold_pe"] == 0.01 and response.metadata["detail_threshold_pe"] == 1e-3


def test_detail_threshold_is_inactive_without_a_lower_value():
    for detail in (None, 0.01, 0.1):
        kernel = SimpleNamespace(config=lh.KernelConfig(threshold_pe=0.01, detail_threshold_pe=detail))
        with detail_threshold(kernel) as block:
            assert not block.lowered and kernel.config.threshold_pe == 0.01
    assert not detail_threshold(None).lowered
    with pytest.raises(ValueError, match="detail_threshold_pe"):
        lh.KernelConfig(detail_threshold_pe=-1.0)
