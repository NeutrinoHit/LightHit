import json

import numpy as np
import pytest

from lighthit.viewer import (SCHEMA, load_event_result, save_event_result,
                             validate_event_viewer, write_event_viewer)


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


def test_self_contained_viewer_and_safe_json(tmp_path):
    value = payload()
    value["events"][0]["label"] = "</script><script>bad()</script>"
    path = write_event_viewer(value, tmp_path / "viewer.html")
    text = path.read_text()
    assert "__VIEWER_JS__" not in text
    assert "Plotly" in text
    assert "</script><script>bad()" not in text
    assert json.loads((tmp_path / "viewer.json").read_text())["schema"] == SCHEMA


@pytest.mark.parametrize("mutation", [
    lambda x: x.update(schema="wrong"),
    lambda x: x["events"][0].update(components=[]),
    lambda x: x["events"][0].update(time_origin_ns=[0.0]),
    lambda x: x["events"].append(dict(x["events"][0])),
])
def test_invalid_payload_is_rejected(mutation):
    value = payload()
    mutation(value)
    with pytest.raises(ValueError):
        validate_event_viewer(value)
