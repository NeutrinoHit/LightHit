"""The default WebGL event viewer: build, data safety and its display model."""
import copy
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lighthit.viewer import SCHEMA, write_event_viewer

ASSETS = Path(__file__).resolve().parents[1] / "src/lighthit/assets"
MODEL = ASSETS / "studio_model.js"
NODE = shutil.which("node")


def payload():
    edges = [-10.0, 0.0, 10.0, 20.0, 30.0]
    return {
        "schema": SCHEMA,
        "units": {"position": "m", "time": "ns", "signal": "photoelectrons"},
        "detector": {"label": "two strings", "positions_m": [[0.0, 0.0, 0.0], [0.0, 0.0, 15.0], [40.0, 0.0, 0.0]],
                     "cluster_id": [0, 0, 1], "string_id": [0, 0, 0], "module_id": [0, 1, 0]},
        "readout": {"relative_time_edges_ns": edges},
        "events": [{
            "event_id": "track", "label": "</script><script>bad()</script>",
            "sources": [{"type": "axis", "shape": "track", "position_m": [10.0, 0.0, 0.0],
                         "direction": [0.0, 0.0, 1.0], "extent_m": 50.0,
                         "timing": {"s_m": [-50.0, 0.0, 50.0], "t_ns": [0.0, 166.78, 333.56],
                                    "weight": [1.0, 1.0, 1.0]}}],
            "time_origin_ns": [100.0, 150.0, 400.0], "active": [True, True, False],
            "components": [[[0, 0, 0], [1.0, 0.5, 0.25], [0.5, -0.1, 0.25], [0, 0, 0]],
                           [[0, 0, 0], [0, 0, 0], [2.0, 0, 0], [0, 1.0, 0]],
                           [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]],
            "charge_components": [[1.5, 0.4, 0.5], [2.0, 1.0, 0.0], [0, 0, 0]],
            "computed_orders": [0, 1, 2], "diagnostics": {"elapsed_seconds": 1.5},
        }],
    }


def test_default_viewer_is_self_contained_webgl_and_needs_no_plotly(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.offline", None)
    value = payload()
    before = copy.deepcopy(value)
    path = write_event_viewer(value, tmp_path / "viewer.html")
    text = path.read_text()
    assert not re.search(r"__[A-Z_]+_(JS|CSS)__|__RESULT_JSON__", text)
    for marker in ("LightHitStudioGL", "LightHitStudioModel", "LightHitMovie", "LightHitViewerSmoothing",
                   'getContext("webgl2"', "Render MP4"):
        assert marker in text
    assert "Plotly" not in text
    assert "<script src=" not in text and "<link" not in text  # offline: no external resources
    assert "</script><script>bad()" not in text
    assert value == before
    assert json.loads((tmp_path / "viewer.json").read_text()) == before
    with pytest.raises(ImportError, match="legacy"):
        write_event_viewer(payload(), tmp_path / "legacy.html", legacy=True)


def test_legacy_viewer_remains_available(tmp_path):
    pytest.importorskip("plotly")
    text = write_event_viewer(payload(), tmp_path / "legacy.html", legacy=True).read_text()
    assert "Plotly" in text and "Time histograms for all OMs" in text
    assert "LightHitStudioGL" not in text


def test_every_element_the_app_uses_exists_in_the_page():
    app = (ASSETS / "studio_app.js").read_text()
    html = (ASSETS / "studio.html").read_text()
    ids = set(re.findall(r'\$\("([a-z0-9-]+)"\)', app)) | set(re.findall(r'\$\(`([a-z0-9-]+)-value`\)', app))
    ids |= {f"{name}-value" for name in re.findall(r'put\("([a-z-]+)"', app)}
    page = set(re.findall(r'id="([a-z0-9-]+)"', html))
    assert not sorted(ids - page)
    # every persisted control is a real input
    controls = re.search(r"const CONTROLS = \{(.*?)\};", app, re.S).group(1)
    for name in re.findall(r'"?([a-z0-9-]+)"?: "', controls):
        assert f'id="{name}"' in html, name


def node(script):
    result = subprocess.run([NODE, "-e", f"const M=require({json.dumps(str(MODEL))});{script}"],
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.skipif(NODE is None, reason="Node.js is unavailable")
def test_state_on_the_detector_clock_spreads_each_bin_uniformly_and_keeps_data():
    data = payload()
    out = node(f"""
const d={json.dumps(data)};const before=JSON.stringify(d);
const m=M.build(d,0);
const at=t=>{{const s=M.stateAt(m,t);return {{q:[...s.q],q3:[...s.q3],rate:[...s.rate]}}}};
console.log(JSON.stringify({{relevant:m.relevant,window:[...m.window],first:[...m.first],t0:m.t0,t1:m.t1,
  before:at(80),mid:at(105),end:at(1e6),prompt:[...M.stateAt(M.build(d,0,{{part:"prompt"}}),1e6).q],
  same:JSON.stringify(d)===before}}))""")
    assert out["same"] is True
    assert out["relevant"] == [0, 1]
    # OM 0: bin [0,10) relative = [100,110) detector; at 105 half of (1, .5, .25)
    np.testing.assert_allclose(out["mid"]["q3"][:3], [0.5, 0.25, 0.125])
    np.testing.assert_allclose(out["mid"]["rate"][0], 1.75 / 10)
    assert out["before"]["q"][:2] == [0, 0]
    np.testing.assert_allclose(out["end"]["q"][:2], [2.4, 3.0])
    np.testing.assert_allclose(out["window"][:2], [2.4, 3.0])
    np.testing.assert_allclose(out["prompt"][:2], [1.9, 3.0])
    # first light: 10 % of the positive total, inside the first lit bin
    np.testing.assert_allclose(out["first"][0], 100 + 10 * 0.24 / 1.75)  # 10 % of 2.4
    np.testing.assert_allclose(out["first"][1], 150 + 10 + 10 * 0.3 / 2.0)
    assert out["t0"] <= 0 and out["t1"] > 190


@pytest.mark.skipif(NODE is None, reason="Node.js is unavailable")
def test_pacing_is_monotone_invertible_and_linear_without_strength():
    out = node("""
const grid=Array.from({length:100},(_,k)=>k+0.5), act=grid.map(t=>t>40&&t<45?10:0);
const p=M.pacing(grid,act,0.8), q=M.pacing(grid,act,0);
const us=Array.from({length:41},(_,k)=>k/40);
console.log(JSON.stringify({t:us.map(p.time),back:us.map(u=>p.progress(p.time(u))),lin:us.map(q.time),
  share:p.progress(45)-p.progress(40)}))""")
    t = np.array(out["t"])
    assert np.all(np.diff(t) >= 0) and t[0] == pytest.approx(0) and t[-1] == pytest.approx(100)
    np.testing.assert_allclose(out["back"], np.linspace(0, 1, 41), atol=1e-9)
    np.testing.assert_allclose(out["lin"], np.linspace(0, 100, 41), atol=1e-9)
    assert out["share"] == pytest.approx(0.8 + 0.2 * 5 / 100)


@pytest.mark.skipif(NODE is None, reason="Node.js is unavailable")
def test_light_front_angle_and_emission_estimate():
    out = node("""
const same=M.frontAngle(1.34,1.34), real=M.frontAngle(1.34,1.36);
const pos=[[30,0,0],[0,40,0],[0,0,-50]], t0=17, ng=1.36, edges=[0,1,2,3,4];
const origins=pos.map(p=>t0+Math.hypot(...p)*ng/M.C_M_PER_NS-0.5);
const rows=pos.map(()=>[[0,0,0],[1,0,0],[0,1,0],[0,0,1]]);
console.log(JSON.stringify({same,real,est:M.estimateEmission(pos,rows,edges,origins,[0,0,0],ng)}))""")
    theta = np.arccos(1 / 1.34)
    assert out["same"] == pytest.approx(np.pi / 2 - theta)
    expected = np.arctan2(np.sin(theta) / 1.36, 1 - np.cos(theta) / 1.36)
    assert out["real"] == pytest.approx(expected)
    # first light 5 % into the bin [1, 2): origin + 1.05 - 0.5 offset
    assert out["est"] == pytest.approx(17 + 0.55)


@pytest.mark.skipif(NODE is None, reason="Node.js is unavailable")
def test_reversed_source_timing_moves_the_head_against_the_stored_direction():
    out = node("""
const s=M.sourceInfo({type:"axis",shape:"track",position_m:[0,0,0],direction:[0,0,1],extent_m:2,
  timing:{s_m:[-2,0,2],t_ns:[12,6,0]}});
console.log(JSON.stringify({dir:s.direction,mid:M.headAt(s,3),before:M.headAt(s,-1),after:M.headAt(s,20),start:s.start}))""")
    assert out["dir"] == [0, 0, -1] or out["dir"] == [-0.0, -0.0, -1]
    np.testing.assert_allclose(out["mid"]["xyz"], [0, 0, 1])
    assert out["before"] is None and out["after"]["done"] is True
    np.testing.assert_allclose(out["start"], [0, 0, 2])
