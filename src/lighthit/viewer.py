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


def _identifier(detector, *names, default):
    for name in names:
        if name in detector.identifiers:
            return np.asarray(detector.identifiers[name]).tolist()
    return np.asarray(default, int).tolist()


def _display_source(source):
    """Portable viewer description for public source objects we can show exactly."""
    if source is None:
        return []
    from .sources import (CherenkovTrack, G4Shower, IsotropicFlash,
                          SyntheticShower)
    if isinstance(source, IsotropicFlash):
        return [{"type": "point", "position_m": source.position_m.tolist(),
                 "label": "isotropic flash"}]
    if isinstance(source, CherenkovTrack):
        direction = np.asarray(source.direction, float)
        direction /= np.linalg.norm(direction)
        return [{"type": "axis", "shape": "track",
                 "position_m": (np.asarray(source.start_m, float)
                                + 0.5 * source.length_m * direction).tolist(),
                 "direction": direction.tolist(), "extent_m": float(source.length_m) / 2,
                 "label": "Cherenkov track"}]
    if isinstance(source, (G4Shower, SyntheticShower)):
        return [{"type": "axis", "shape": "spindle",
                 "position_m": source.centroid_m.tolist(),
                 "direction": source.principal_axis.tolist(),
                 "extent_m": float(source.extent_m),
                 "label": ("G4 shower" if isinstance(source, G4Shower)
                           else "synthetic shower")}]
    return []


def viewer_payload(response, *, source=None, event_id="event", label=None,
                   detector_label="detector"):
    """Convert one :class:`TransportResponse` into portable viewer data.

    Unknown source types remain fully viewable as detector responses; only the
    optional source marker is omitted. Additional events can be appended to the
    returned ``events`` list before passing it to :func:`write_event_viewer`.
    """
    detector = response.detector
    count = len(detector.positions_m)
    payload = {
        "schema": SCHEMA,
        "units": {"position": "m", "time": "ns", "signal": "photoelectrons"},
        "detector": {
            "label": str(detector_label),
            "positions_m": detector.positions_m.tolist(),
            "cluster_id": _identifier(detector, "cluster_id", "cluster",
                                      default=np.zeros(count, int)),
            "string_id": _identifier(detector, "string_id", "string", "subcluster",
                                     default=np.zeros(count, int)),
            "module_id": _identifier(detector, "module_id", "module", "channel",
                                     default=np.arange(count)),
        },
        "readout": {
            "relative_time_edges_ns": response.relative_time_edges_ns.tolist(),
        },
        "events": [{
            "event_id": str(event_id),
            "label": str(label if label is not None else event_id),
            "sources": _display_source(source),
            "time_origin_ns": response.time_origin_ns.tolist(),
            "active": response.active.tolist(),
            "components": response.components_pe.tolist(),
            "charge_components": response.charge_components_pe.tolist(),
            "diagnostics": dict(response.metadata),
        }],
    }
    return validate_event_viewer(payload)


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
        active = np.asarray(event.get("active", np.ones(count, bool)))
        if active.shape != (count,) or active.dtype.kind != "b":
            raise ValueError(f"event {identifier}: active must contain one boolean per module")
        for source in event.get("sources", []):
            if source.get("type") not in ("axis", "point"):
                raise ValueError(f"event {identifier}: unsupported source display type")
            position = np.asarray(source.get("position_m"), float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError(f"event {identifier}: invalid source position")
            if source["type"] == "axis":
                if source.get("shape", "line") not in ("line", "track", "spindle"):
                    raise ValueError(f"event {identifier}: invalid source display shape")
                direction = np.asarray(source.get("direction"), float)
                if (direction.shape != (3,) or not np.isfinite(direction).all()
                        or not np.isclose(np.linalg.norm(direction), 1, atol=1e-10)):
                    raise ValueError(f"event {identifier}: invalid source direction")
    return result


def merge_event_viewers(*payloads):
    """Combine compatible one- or multi-event payloads into one viewer.

    Detector geometry, units and time edges must match exactly.  This keeps a
    multi-pose study honest: only events evaluated on the same readout grid are
    placed behind one selector.
    """
    if not payloads:
        raise ValueError("at least one viewer payload is required")
    values = [validate_event_viewer(value) for value in payloads]
    result = json.loads(json.dumps(values[0], ensure_ascii=False))
    result["events"] = []
    reference = values[0]
    for value in values:
        for section in ("units", "detector", "readout"):
            if value.get(section) != reference.get(section):
                raise ValueError(f"viewer payloads have different {section}")
        result["events"].extend(value["events"])
    return validate_event_viewer(result)


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
        "__VIEWER_SMOOTHING_JS__": assets.joinpath("event_viewer_smoothing.js").read_text(encoding="utf-8"),
        "__VIEWER_JS__": assets.joinpath("event_viewer.js").read_text(encoding="utf-8"),
    }.items():
        html = html.replace(token, value)
    output.write_text(html, encoding="utf-8")
    return output


__all__ = ["SCHEMA", "viewer_payload", "validate_event_viewer", "merge_event_viewers",
           "save_event_result", "load_event_result", "write_event_viewer"]
