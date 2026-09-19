"""Self-contained interactive viewer for LightHit detector-array events.

The format is deliberately source-agnostic: a stored G4 shower, a fitted
track, a point flash and a future source model all provide the same three
response components on the same detector geometry.  Signed Fourier-inversion
bins are preserved; the viewer applies a signed-log colour transform only for
display and never changes stored values.
"""
from importlib.resources import files
from pathlib import Path
import json

import numpy as np


SCHEMA = "lighthit/event-viewer/1"


def validate_event_viewer(result):
    """Validate the portable JSON-compatible event-viewer payload."""
    if result.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported viewer schema; expected {SCHEMA}")
    detector = result.get("detector", {})
    positions = np.asarray(detector.get("positions_m"), float)
    if (positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions)
            or not np.isfinite(positions).all()):
        raise ValueError("detector.positions_m must be a finite nonempty (N,3) array")
    count = len(positions)
    for name in ("cluster_id", "string_id", "module_id"):
        values = np.asarray(detector.get(name))
        if values.shape != (count,) or values.dtype.kind not in "iuf":
            raise ValueError(f"detector.{name} must contain one numeric value per module")
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError(f"detector.{name} must contain finite integers")
    edges = np.asarray(result.get("readout", {}).get("relative_time_edges_ns"), float)
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("relative_time_edges_ns must be finite and strictly increasing")
    bins = len(edges) - 1
    events = result.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("viewer payload needs at least one event")
    identifiers = set()
    for event in events:
        identifier = str(event.get("event_id", ""))
        if not identifier or identifier in identifiers:
            raise ValueError("event_id values must be nonempty and unique")
        identifiers.add(identifier)
        components = np.asarray(event.get("components"), float)
        charge = np.asarray(event.get("charge_components"), float)
        origin = np.asarray(event.get("time_origin_ns"), float)
        if components.shape != (count, bins, 3) or not np.isfinite(components).all():
            raise ValueError(f"event {identifier}: components must have shape {(count, bins, 3)}")
        if charge.shape != (count, 3) or not np.isfinite(charge).all():
            raise ValueError(f"event {identifier}: charge_components must have shape {(count, 3)}")
        if origin.shape != (count,) or not np.isfinite(origin).all():
            raise ValueError(f"event {identifier}: time_origin_ns must have shape {(count,)}")
        for source in event.get("sources", []):
            if source.get("type") not in ("axis", "point"):
                raise ValueError(f"event {identifier}: unsupported source display type")
            position = np.asarray(source.get("position_m"), float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f"event {identifier}: invalid source position")
            if source["type"] == "axis":
                direction = np.asarray(source.get("direction"), float)
                if (direction.shape != (3,) or not np.isfinite(direction).all()
                        or not np.isclose(np.linalg.norm(direction), 1, atol=1e-10)):
                    raise ValueError(f"event {identifier}: invalid source direction")
    return result


def save_event_result(result, path):
    """Write validated viewer data as portable JSON."""
    validate_event_viewer(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False,
                               separators=(",", ":")), encoding="utf-8")
    return path


def load_event_result(path):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_event_viewer(result)


def write_event_viewer(result, output_path, *, json_path=None):
    """Build a standalone HTML viewer and its portable JSON companion."""
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install lighthit[viewer] to build the event viewer") from exc
    if isinstance(result, (str, Path)):
        result = load_event_result(result)
    else:
        validate_event_viewer(result)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    companion = Path(json_path) if json_path is not None else output.with_suffix(".json")
    save_event_result(result, companion)
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    assets = files("lighthit").joinpath("assets")
    html = assets.joinpath("event_viewer.html").read_text(encoding="utf-8")
    for token, value in {
        "__VIEWER_CSS__": assets.joinpath("event_viewer.css").read_text(encoding="utf-8"),
        "__PLOTLY_JS__": get_plotlyjs(),
        "__RESULT_JSON__": payload,
        "__VIEWER_JS__": assets.joinpath("event_viewer.js").read_text(encoding="utf-8"),
    }.items():
        html = html.replace(token, value)
    output.write_text(html, encoding="utf-8")
    return output


__all__ = ["SCHEMA", "validate_event_viewer", "save_event_result",
           "load_event_result", "write_event_viewer"]
