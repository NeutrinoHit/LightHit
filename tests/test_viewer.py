import json
import shutil
import subprocess
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr

from lighthit import CherenkovTrack, DetectorArray, IsotropicFlash
from lighthit.viewer import (SCHEMA, load_event_result, save_event_result,
                             merge_event_viewers, source_timing, validate_event_viewer,
                             viewer_payload, write_event_viewer)


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
    assert "__VIEWER_SMOOTHING_JS__" not in text
    assert "Plotly" in text
    assert "3 ns Gaussian" in text
    assert "0+1 · prompt" in text
    assert "function smoothComponents" in text
    assert "</script><script>bad()" not in text
    assert json.loads((tmp_path / "viewer.json").read_text())["schema"] == SCHEMA


def test_viewer_uses_english_controls_and_small_source_arrows(tmp_path):
    path = write_event_viewer(payload(), tmp_path / "viewer.html")
    text = path.read_text()
    assert '<html lang="en">' in text
    assert "Open JSON" in text
    assert "Geometry and signal evolution" in text
    assert "Time histograms for all OMs" in text
    assert "sizeref:Math.max(2,Math.min(6,length*.03))" in text


def test_track_marker_uses_midpoint_and_half_length():
    value = payload()
    detector = DetectorArray(
        value["detector"]["positions_m"], [[0., 0., 1.]] * 2, .05,
        lambda x: np.ones_like(np.asarray(x, float)),
        lambda w: np.ones_like(np.asarray(w, float)))
    response = SimpleNamespace(
        detector=detector, relative_time_edges_ns=np.array([-1., 0., 1., 2.]),
        time_origin_ns=np.array([10., 11.]), active=np.array([True, False]),
        components_pe=np.asarray(value["events"][0]["components"]),
        charge_components_pe=np.asarray(value["events"][0]["charge_components"]),
        metadata={})
    track = CherenkovTrack.centered([1., 2., 3.], [0., 0., 1.], 10.)
    source = viewer_payload(response, source=track)["events"][0]["sources"][0]
    assert source["position_m"] == [1., 2., 3.]
    assert source["extent_m"] == 5.


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_gaussian_display_preserves_signed_bin_integrals():
    edges = np.arange(-15., 26., 5.)
    raw = np.zeros((1, len(edges) - 1, 3))
    raw[0, 3] = [1., -2., 3.]
    helper = Path(__file__).resolve().parents[1] / "src/lighthit/assets/event_viewer_smoothing.js"
    script = (
        f"const s=require({json.dumps(str(helper))});"
        f"const x={json.dumps(raw.tolist())};"
        f"console.log(JSON.stringify([s.smoothComponents(x,{json.dumps(edges.tolist())},3),x]));"
    )
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True,
                            check=True)
    smoothed, unchanged = json.loads(result.stdout)
    np.testing.assert_array_equal(unchanged, raw)
    expected = np.array([
        quad(lambda time: (ndtr((right - time) / 3)
                           - ndtr((left - time) / 3)) / 5,
             0, 5)[0]
        for left, right in zip(edges[:-1], edges[1:], strict=True)])
    np.testing.assert_allclose(np.asarray(smoothed)[0],
                               expected[:, None] * [1., -2., 3.], atol=3e-7)


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


def _timed(payload_value, **timing):
    source = payload_value["events"][0]["sources"][0]
    source["timing"] = {"s_m": [-2.0, 0.0, 2.0], "t_ns": [0.0, 6.0, 12.0],
                        "weight": [1.0, 1.0, 1.0], **timing}
    return payload_value


def test_track_payload_carries_emission_timing_on_the_detector_clock():
    detector = DetectorArray(
        payload()["detector"]["positions_m"], [[0., 0., 1.]] * 2, .05,
        lambda x: np.ones_like(np.asarray(x, float)),
        lambda w: np.ones_like(np.asarray(w, float)))
    value = payload()["events"][0]
    response = SimpleNamespace(
        detector=detector, relative_time_edges_ns=np.array([-1., 0., 1., 2.]),
        time_origin_ns=np.array([10., 11.]), active=np.array([True, False]),
        components_pe=np.asarray(value["components"]),
        charge_components_pe=np.asarray(value["charge_components"]), metadata={})
    track = CherenkovTrack.centered([1., 2., 3.], [0., 0., 1.], 30., time_ns=5.)
    timing = viewer_payload(response, source=track)["events"][0]["sources"][0]["timing"]
    assert timing["s_m"][0] == -15. and timing["s_m"][-1] == 15.
    assert timing["t_ns"][0] == 5.  # start of the track, at time_ns
    np.testing.assert_allclose(np.diff(timing["t_ns"]) / np.diff(timing["s_m"]),
                               1 / 0.299792458)


def test_source_timing_is_a_relativistic_front_weighted_by_photon_yield():
    z = np.linspace(-3., 3., 40)
    elements = SimpleNamespace(
        midpoints_m=np.column_stack((np.zeros_like(z), np.zeros_like(z), z)),
        photons=np.exp(-0.5 * z ** 2), length_m=np.full(len(z), .15),
        start_ns=20. + z / 0.299792458, end_ns=20. + (z + .15) / 0.299792458)
    timing = source_timing(elements, [0., 0., 0.], [0., 0., 1.], 3.)
    slope = np.polyfit(timing["s_m"], timing["t_ns"], 1)[0]
    assert slope == pytest.approx(1 / 0.299792458, rel=1e-6)
    assert max(timing["weight"]) == 1.
    assert timing["s_m"][np.argmax(timing["weight"])] == pytest.approx(0., abs=.1)


@pytest.mark.parametrize("bad", [
    {"t_ns": [0.0, 6.0]}, {"t_ns": [0.0, 9.0, 3.0]}, {"s_m": [0.0, 0.0, 2.0]},
    {"weight": [1.0, -1.0, 1.0]}, {"t_ns": [0.0, float("nan"), 3.0]}])
def test_invalid_source_timing_is_rejected(bad):
    with pytest.raises(ValueError, match="timing"):
        validate_event_viewer(_timed(payload(), **bad))
    validate_event_viewer(_timed(payload()))  # and the good block passes
    validate_event_viewer(_timed(payload(), t_ns=[12.0, 6.0, 0.0]))  # reversed motion


def test_viewer_has_a_detector_clock_and_source_animation(tmp_path):
    path = write_event_viewer(_timed(payload()), tmp_path / "viewer.html")
    text = path.read_text()
    assert 'id="clock"' in text and "Detector clock" in text
    assert "LightHitViewerSource" in text and "__VIEWER_SOURCE_JS__" not in text
    assert json.loads((tmp_path / "viewer.json").read_text())[
        "events"][0]["sources"][0]["timing"]["t_ns"] == [0.0, 6.0, 12.0]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is unavailable")
def test_source_animation_helper_moves_a_head_and_bins_on_the_detector_clock():
    helper = Path(__file__).resolve().parents[1] / "src/lighthit/assets/event_viewer_source.js"
    source = {"position_m": [0, 0, 0], "direction": [0, 0, 1], "shape": "track",
              "timing": {"s_m": [-2, 0, 2], "t_ns": [12, 6, 0]}}  # moves against +z
    script = (
        f"const s=require({json.dumps(str(helper))});const p=s.prepare({json.dumps(source)});"
        "const f=s.frame(p,3);"
        "console.log(JSON.stringify({rev:p.reversed,head:s.head(p,3),before:s.head(p,-1),"
        "after:s.head(p,13),hidden:s.frame(p,-Infinity).size,last:f.size.at(-1),"
        "bins:[-1,0,4.9,5,10].map(x=>s.binAt([0,5,10],x)),"
        "range:s.clockRange([0,10],[[0,1,0],[0,0,2]],[0,5,10],[3,20])}));")
    out = json.loads(subprocess.run(["node", "-e", script], capture_output=True, text=True,
                                    check=True).stdout)
    assert out["rev"] is True
    np.testing.assert_allclose(out["head"], [0, 0, 1])  # t=3 is halfway from s=2 to s=0
    assert out["before"] is None and out["after"] is None
    assert set(out["hidden"]) == {0} and out["last"] > 0
    assert out["bins"] == [-1, 0, 0, 1, 2]
    assert out["range"]["t0"] < 3 < 20 < out["range"]["t1"]


def test_straight_muon_selection_shows_a_track_not_a_spindle():
    z = np.linspace(-100., 100., 30)
    elements = SimpleNamespace(
        midpoints_m=np.column_stack((np.zeros_like(z), np.zeros_like(z), z)),
        photons=np.ones_like(z), length_m=np.full(len(z), 7.), start_ns=20. + z / .3,
        end_ns=20. + (z + 7.) / .3)
    from lighthit.viewer import _display_sources
    shower = SimpleNamespace(elements=elements)
    assert _display_sources(None, {}) == []
    plain = {"type": "axis", "shape": "spindle", "label": "G4 shower"}
    import lighthit.viewer as viewer
    original = viewer._display_source
    viewer._display_source = lambda source: [dict(plain)]
    try:
        assert _display_sources(shower, {})[0]["shape"] == "spindle"
        assert _display_sources(shower, {"track_uid": None})[0]["shape"] == "spindle"
        for metadata in ({"track_uid": 0}, {"automatic_track_selection": {"reason": "x"}}):
            assert _display_sources(shower, metadata)[0]["shape"] == "track"
    finally:
        viewer._display_source = original
