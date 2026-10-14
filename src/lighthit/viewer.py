"""Self-contained interactive viewer for LightHit detector-array events.

The format is deliberately source-agnostic: a stored G4 shower, a fitted
track, a point flash and a future source model all provide the same three
response components on the same detector geometry.  Signed Fourier-inversion
bins are preserved; the viewer applies a signed-log colour transform only for
display and never changes stored values.

Axis sources may carry an optional ``timing`` block (emission time along the
axis on the detector clock, the same clock as ``time_origin_ns``).  The viewer
uses it to animate a moving track head or a growing shower spindle in step
with the detector response; older payloads without it stay valid and static.
"""
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np


SCHEMA = "lighthit/event-viewer/1"


def _identifier(detector, *names, default):
    for name in names:
        if name in detector.identifiers:
            return np.asarray(detector.identifiers[name]).tolist()
    return np.asarray(default, int).tolist()


LIGHT_SPEED_M_PER_NS = 0.299792458


def source_timing(elements, centre_m, axis, half_m, bins=48):
    """Axial emission timing of light elements, for the viewer animation.

    Returns ``s_m`` (bin centres along ``axis`` from ``centre_m``), ``t_ns``
    (emission time on the detector clock) and ``weight`` (relative photon
    yield, maximum 1).  The time is a weighted straight-line fit of the element
    emission times against ``s``: a relativistic source is a front moving along
    its axis, and the fit keeps that front monotonic. Spectral ``coefficient0``
    is already integrated over each step and is used without another length
    factor. Display data only.
    """
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    s = (elements.midpoints_m - np.asarray(centre_m, float)) @ axis
    photons = (np.asarray(elements.photons, float) if hasattr(elements, "photons")
               else np.asarray(elements.coefficient0, float))
    time = 0.5 * (np.asarray(elements.start_ns, float) + np.asarray(elements.end_ns, float))
    edges = np.linspace(-half_m, half_m, bins + 1)
    index = np.clip(np.digitize(s, edges) - 1, 0, bins - 1)
    weight = np.bincount(index, photons, bins)
    if np.count_nonzero(weight) < 2:
        return None
    centres = 0.5 * (edges[1:] + edges[:-1])
    filled = weight > 0
    mean_s = np.bincount(index, photons * s, bins)[filled] / weight[filled]
    mean_time = np.bincount(index, photons * time, bins)[filled] / weight[filled]
    slope, offset = np.polyfit(mean_s, mean_time, 1, w=np.sqrt(weight[filled]))
    if abs(slope) < 1e-6:  # all light in one instant: fall back to a beta = 1 front
        slope = 1.0 / LIGHT_SPEED_M_PER_NS
    return {"s_m": centres.tolist(), "t_ns": (offset + slope * centres).tolist(),
            "weight": (weight / weight.max()).tolist()}


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
        half = float(source.length_m) / 2
        along = np.linspace(-half, half, 64)
        return [{"type": "axis", "shape": "track",
                 "position_m": (np.asarray(source.start_m, float)
                                + half * direction).tolist(),
                 "direction": direction.tolist(), "extent_m": half,
                 "label": "Cherenkov track",
                 "timing": {"s_m": along.tolist(),
                            "t_ns": (source.time_ns + (along + half)
                                     / (source.beta * LIGHT_SPEED_M_PER_NS)).tolist(),
                            "weight": [1.0] * len(along)}}]
    if isinstance(source, (G4Shower, SyntheticShower)):
        display = {"type": "axis", "shape": "spindle",
                   "position_m": source.centroid_m.tolist(),
                   "direction": source.principal_axis.tolist(),
                   "extent_m": float(source.extent_m),
                   "label": ("G4 shower" if isinstance(source, G4Shower)
                             else "synthetic shower")}
        timing = source_timing(source.elements, source.centroid_m,
                               source.principal_axis, source.extent_m)
        if timing is not None:
            display["timing"] = timing
        return [display]
    return []


def _display_track_uid(metadata):
    """Read an actual selected UID, including nested hybrid response metadata."""
    metadata = metadata or {}
    values = []
    components = metadata.get("muon_components")
    basis = metadata.get("ge2_basis")
    automatic = metadata.get("automatic_track_selection")
    if isinstance(components, dict):
        values.append(components.get("primary_uid"))
    if isinstance(basis, dict):
        values.append(basis.get("track_uid"))
    values.append(metadata.get("track_uid"))
    if isinstance(automatic, dict):
        values.extend(automatic.get(key) for key in ("track_uid", "primary_uid", "uid"))
    for value in values:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return int(value)
    return None


def _display_primary_track(source, track_uid):
    """Fit a display axis and clock using only the selected track's elements.

    This is a straight-axis animation, not an integration of the broken G4
    trajectory. Neither secondary emission times nor secondary yield influence
    its geometry or head timing; all physics arrays remain untouched.
    """
    elements = getattr(source, "elements", None)
    if elements is None or not hasattr(elements, "uid"):
        return None
    selected = np.asarray(elements.uid) == track_uid
    if not np.any(selected):
        return None
    starts = np.asarray(elements.start_m, float)[selected]
    directions = np.asarray(elements.direction, float)[selected]
    lengths = np.asarray(elements.length_m, float)[selected]
    ends = starts + lengths[:, None]*directions
    midpoints = .5*(starts+ends)
    # C0 already includes the true step length; multiplying by length again
    # would overweight long steps and distort both the geometry and clock.
    weights = (np.asarray(elements.photons, float)[selected] if hasattr(elements, "photons")
               else np.asarray(elements.coefficient0, float)[selected])
    if not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        return None
    share = weights/weights.sum()
    centre = np.sum(share[:, None]*midpoints, axis=0)
    offset = midpoints-centre
    covariance = np.einsum("i,ij,ik->jk", share, offset, offset)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    mean_direction = np.sum(share[:, None]*directions, axis=0)
    axis = (mean_direction if np.max(np.abs(eigenvalues)) <= 1e-24
            else eigenvectors[:, np.argmax(eigenvalues)])
    norm = np.linalg.norm(axis)
    if not np.isfinite(norm) or norm == 0:
        return None
    axis = axis/norm
    if axis @ mean_direction < 0:
        axis = -axis
    along = (np.concatenate((starts, ends))-centre) @ axis
    low, high = float(along.min()), float(along.max())
    half = .5*(high-low)
    if half <= 0:
        return None
    # The visible guide spans the projected endpoints, not a symmetric radius
    # about a potentially asymmetric photon-weighted centroid.
    centre = centre+.5*(low+high)*axis
    timing_elements = SimpleNamespace(midpoints_m=midpoints, photons=weights,
                                      start_ns=np.asarray(elements.start_ns, float)[selected],
                                      end_ns=np.asarray(elements.end_ns, float)[selected])
    timing = source_timing(timing_elements, centre, axis, half)
    display = {"type": "axis", "shape": "track", "label": "G4 track",
               "track_uid": int(track_uid), "position_m": centre.tolist(),
               "direction": axis.tolist(), "extent_m": half}
    if timing is not None:
        display["timing"] = timing
    return display


def _display_sources(source, metadata):
    """Display a selected primary muon independently of its secondary light."""
    track_uid = _display_track_uid(metadata)
    if track_uid is not None:
        track = _display_primary_track(source, track_uid)
        if track is not None:
            info = metadata or {}
            components = info.get("muon_components")
            automatic = info.get("automatic_track_selection")
            if ((isinstance(components, dict) and components.get("primary_uid") == track_uid)
                    or (isinstance(automatic, dict)
                        and automatic.get("reason") == "dominant_straight_muon")):
                track["label"] = "G4 muon track"
            return [track]
    return _display_source(source)


def viewer_payload(response, *, source=None, event_id="event", label=None,
                   detector_label="detector"):
    """Convert one :class:`TransportResponse` into portable viewer data.

    Unknown source types remain fully viewable as detector responses; only the
    optional source marker is omitted. Additional events can be appended to the
    returned ``events`` list before passing it to :func:`write_event_viewer`.
    """
    detector = response.detector
    count = len(detector.positions_m)
    computed_orders = list(getattr(response, "computed_orders", (0, 1, 2)))
    if computed_orders != [0, 1, 2]:
        components, charge_components, computed_orders = response.viewer_components()
    else:
        components = response.components_pe
        charge_components = response.charge_components_pe
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
            "sources": _display_sources(source, response.metadata),
            "time_origin_ns": response.time_origin_ns.tolist(),
            "active": response.active.tolist(),
            "components": np.asarray(components).tolist(),
            "charge_components": np.asarray(charge_components).tolist(),
            "computed_orders": computed_orders,
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
        orders = event.get("computed_orders", [0, 1, 2])
        if (not isinstance(orders, list) or not orders
                or sorted(set(orders)) != orders or not set(orders) <= {0, 1, 2}):
            raise ValueError(f"event {identifier}: computed_orders must be a sorted subset of [0, 1, 2]")
        for order in {0, 1, 2} - set(orders):
            # an order that was not computed is a placeholder, never a value
            if np.any(components[:, :, order] != 0) or np.any(charge[:, order] != 0):
                raise ValueError(f"event {identifier}: order {order} is not computed "
                                 "and must carry an exact zero placeholder")
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
                if source.get("timing") is not None:
                    _validate_timing(source["timing"], identifier)
    return result


def _validate_timing(timing, identifier):
    """Optional axial timing of a source: ``s_m`` ascending, ``t_ns`` monotonic."""
    s = np.asarray(timing.get("s_m"), float)
    t = np.asarray(timing.get("t_ns"), float)
    weight = np.asarray(timing.get("weight", np.ones_like(s)), float)
    if (s.ndim != 1 or len(s) < 2 or t.shape != s.shape or weight.shape != s.shape
            or not (np.isfinite(s).all() and np.isfinite(t).all()
                    and np.isfinite(weight).all())):
        raise ValueError(f"event {identifier}: source timing needs finite s_m, t_ns "
                         "and weight arrays of one length of at least two")
    step = np.diff(t)
    if (np.any(np.diff(s) <= 0) or np.any(weight < 0)
            or not (np.all(step >= 0) or np.all(step <= 0))):
        raise ValueError(f"event {identifier}: source timing needs increasing s_m, "
                         "monotonic t_ns and nonnegative weights")


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


def _embedded_payload(result):
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return payload.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _fill(template, tokens):
    for token, value in tokens.items():
        template = template.replace(token, value)
    return template


def write_event_viewer(result, output_path, *, json_path=None, legacy=False):
    """Build a standalone HTML viewer and its portable JSON companion.

    The default viewer is a three.js (WebGL2) scene with analysis panels and
    frame-exact MP4 export; three.js is inlined from the package, so the page
    works offline and needs no Python plotting package.  ``legacy=True`` builds
    the previous Plotly viewer, which requires ``lighthit[viewer]``.
    """
    get_plotlyjs = None
    if legacy:
        try:
            from plotly.offline import get_plotlyjs
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install lighthit[viewer] to build the legacy event viewer") from exc
    if isinstance(result, (str, Path)):
        result = load_event_result(result)
    else:
        validate_event_viewer(result)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    companion = Path(json_path) if json_path is not None else output.with_suffix(".json")
    save_event_result(result, companion)
    assets = files("lighthit").joinpath("assets")
    read = lambda *parts: assets.joinpath(*parts).read_text(encoding="utf-8")
    payload = _embedded_payload(result)
    if legacy:
        html = _fill(read("legacy", "event_viewer.html"), {
            "__VIEWER_CSS__": read("legacy", "event_viewer.css"),
            "__PLOTLY_JS__": get_plotlyjs(),
            "__RESULT_JSON__": payload,
            "__VIEWER_SMOOTHING_JS__": read("event_viewer_smoothing.js"),
            "__VIEWER_SOURCE_JS__": read("event_viewer_source.js"),
            "__VIEWER_JS__": read("legacy", "event_viewer.js"),
        })
    else:
        html = _fill(read("studio.html"), {
            "__STUDIO_CSS__": read("studio.css"),
            "__RESULT_JSON__": payload,
            "__VIEWER_SMOOTHING_JS__": read("event_viewer_smoothing.js"),
            "__STUDIO_MODEL_JS__": read("studio_model.js"),
            "__THREE_JS__": read("vendor", "three.min.js"),
            "__STUDIO_GL_JS__": read("studio_gl.js"),
            "__STUDIO_MOVIE_JS__": read("studio_movie.js"),
            "__STUDIO_APP_JS__": read("studio_app.js"),
        })
    output.write_text(html, encoding="utf-8")
    return output


__all__ = ["SCHEMA", "source_timing", "viewer_payload", "validate_event_viewer", "merge_event_viewers",
           "save_event_result", "load_event_result", "write_event_viewer"]
