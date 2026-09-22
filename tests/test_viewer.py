import json
from types import SimpleNamespace

import numpy as np
import pytest

from lighthit import DetectorArray, IsotropicFlash
from lighthit.viewer import (SCHEMA, load_event_result, save_event_result,
                             merge_event_viewers, validate_event_viewer, viewer_payload,
                             write_event_viewer)


def payload():
    positions = [[0.0, 0.0, -1.0], [1.0, 0.0, 1.0]]
    components = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3) / 100
    components[0, 0, 2] *= -1  # signed Fourier bins are a supported diagnostic
    return {
        "schema": SCHEMA,
        "package_version": "test",
        "units": {"position": "m", "time": "ns", "signal": "m^-2"},
        "detector": {"label": "test", "positions_m": positions,
                     "cluster_id": [0, 0], "string_id": [0, 1],
                     "module_id": [0, 0]},
        "medium": {"provenance": "synthetic"},
        "readout": {"relative_time_edges_ns": [-1.0, 0.0, 1.0, 2.0]},
        "events": [{
            "event_id": "event", "label": "test event",
            "sources": [{"type": "axis", "position_m": [0.0, 0.0, 0.0],
                         "direction": [0.0, 0.0, 1.0], "extent_m": 2.0}],
            "time_origin_ns": [10.0, 11.0],
            "active": [True, False],
            "components": components.tolist(),
            "charge_components": components.sum(axis=1).tolist(),
            "diagnostics": {"elapsed_seconds": 0.1},
        }],
    }


def test_signed_payload_round_trip(tmp_path):
    value = payload()
    validate_event_viewer(value)
    path = save_event_result(value, tmp_path / "events.json")
    assert load_event_result(path) == value


def test_transport_response_converts_to_a_viewer_payload():
    detector = DetectorArray(
        [[0., 0., -1.], [1., 0., 1.]], [[0., 0., 1.], [0., 0., 1.]], .05,
        lambda x: np.ones_like(np.asarray(x, float)),
        lambda w: np.ones_like(np.asarray(w, float)),
        identifiers={"cluster": [2, 2], "string": [4, 5], "channel": [7, 8]})
    components = np.arange(2 * 3 * 3, dtype=float).reshape(2, 3, 3) / 100
    response = SimpleNamespace(
        detector=detector, relative_time_edges_ns=np.array([-1., 0., 1., 2.]),
        time_origin_ns=np.array([10., 11.]), active=np.array([True, False]),
        components_pe=components, charge_components_pe=components.sum(axis=1),
        metadata={"method": "isotropic"})
    source = IsotropicFlash.monochromatic([0., 0., 0.], 1e5, 450.)
    value = viewer_payload(response, source=source, event_id="flash",
                           label="test flash")
    validate_event_viewer(value)
    assert value["detector"]["cluster_id"] == [2, 2]
    assert value["detector"]["string_id"] == [4, 5]
    assert value["detector"]["module_id"] == [7, 8]
    assert value["events"][0]["sources"][0]["type"] == "point"
    assert value["events"][0]["label"] == "test flash"


def test_self_contained_viewer_and_safe_json(tmp_path):
    value = payload()
    value["events"][0]["label"] = "</script><script>bad()</script>"
    path = write_event_viewer(value, tmp_path / "viewer.html")
    text = path.read_text()
    assert "__VIEWER_JS__" not in text
    assert "Plotly" in text
    assert "</script><script>bad()" not in text
    assert json.loads((tmp_path / "viewer.json").read_text())["schema"] == SCHEMA


def test_viewer_accepts_payload_without_optional_medium_and_catches_async_errors(tmp_path):
    value = payload()
    value.pop("medium")
    path = write_event_viewer(value, tmp_path / "viewer.html")
    text = path.read_text()
    assert "data.medium?.provenance" in text
    assert 'load(JSON.parse($("result").textContent)).catch' in text


def test_viewer_avoids_spread_maximum_on_full_array_heatmaps(tmp_path):
    value = payload()
    count, bins = 2304, 160
    positions = np.zeros((count, 3))
    positions[:, 0] = np.arange(count)
    value["detector"]["positions_m"] = positions.tolist()
    value["detector"]["cluster_id"] = (np.arange(count) // 288).tolist()
    value["detector"]["string_id"] = (np.arange(count) // 36 % 8).tolist()
    value["detector"]["module_id"] = (np.arange(count) % 36).tolist()
    value["readout"]["relative_time_edges_ns"] = np.arange(bins + 1).tolist()
    value["events"][0]["components"] = np.zeros((count, bins, 3)).tolist()
    value["events"][0]["charge_components"] = np.zeros((count, 3)).tolist()
    value["events"][0]["time_origin_ns"] = np.zeros(count).tolist()
    value["events"][0]["active"] = np.zeros(count, bool).tolist()
    path = write_event_viewer(value, tmp_path / "large-viewer.html")
    text = path.read_text()
    assert "function maxAbs" in text
    assert "Math.max(...raw.flat()" not in text
    assert "Math.max(...z.flat()" not in text


def test_compatible_events_merge_and_duplicate_ids_are_rejected():
    first = payload()
    second = payload()
    second["events"][0]["event_id"] = "event-2"
    second["events"][0]["label"] = "second pose"
    merged = merge_event_viewers(first, second)
    assert [event["event_id"] for event in merged["events"]] == ["event", "event-2"]

    second["events"][0]["event_id"] = "event"
    with pytest.raises(ValueError, match="unique"):
        merge_event_viewers(first, second)


def test_incompatible_viewer_payloads_do_not_merge():
    first = payload()
    second = payload()
    second["readout"]["relative_time_edges_ns"][-1] = 3.0
    with pytest.raises(ValueError, match="readout"):
        merge_event_viewers(first, second)


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(schema="wrong"),
    lambda x: x["events"][0].update(components=[]),
    lambda x: x["events"][0].update(time_origin_ns=[0.0]),
    lambda x: x["events"][0].update(active=[True]),
    lambda x: x["events"].append(dict(x["events"][0])),
])
def test_invalid_payload_is_rejected(mutation):
    value = payload()
    mutation(value)
    with pytest.raises(ValueError):
        validate_event_viewer(value)
