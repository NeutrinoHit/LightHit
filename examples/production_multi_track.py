"""Transport several explicitly posed Cherenkov tracks through one prepared kernel."""
from pathlib import Path
import argparse
import json
import tomllib

import numpy as np
import lighthit as lh


def load_tracks(path):
    specification = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    entries = specification.get("tracks", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("tracks TOML must contain at least one [[tracks]] table")
    tracks = []
    seen = set()
    for item in entries:
        identifier = str(item["id"])
        if (not identifier or identifier in {".", ".."}
                or Path(identifier).name != identifier or identifier in seen):
            raise ValueError(f"invalid or duplicate track id: {identifier!r}")
        seen.add(identifier)
        pose = lh.SourcePose(item["position_m"], item["direction"],
                             item.get("time_ns", 0.))
        length = float(item["length_m"])
        beta = float(item.get("beta", 1.))
        if (not np.isfinite([length, beta]).all() or length <= 0
                or not 0 < beta <= 1):
            raise ValueError(f"{identifier}: length_m must be positive and 0 < beta <= 1")
        source = lh.CherenkovTrack.centered(
            pose.position_m, pose.direction, length,
            beta=beta, time_ns=pose.time_ns)
        tracks.append((identifier, str(item.get("label", identifier)), source))
    return tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bgvd-model", required=True)
    parser.add_argument("--tracks", required=True,
                        help="TOML file with numeric midpoint position and direction per track")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bin-ns", type=float, default=5.)
    parser.add_argument("--frequencies", type=int, default=321)
    parser.add_argument("--omega-max", type=float, default=1.2)
    parser.add_argument("--spectral-folded", dest="spectral_folded", action="store_true",
                        default="auto", help="require the wavelength-integrated cache")
    parser.add_argument("--reference", dest="spectral_folded", action="store_false",
                        help="use the original nine-wavelength calculation")
    parser.add_argument("--receiver-block", type=int, default=1)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    if args.frequencies < 2 or args.omega_max <= 0 or args.bin_ns <= 0:
        parser.error("--frequencies must be >= 2, --omega-max and --bin-ns positive")
    tracks = load_tracks(args.tracks)

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bgvd = lh.load_bgvd_model(args.bgvd_model, dataset="2021")
    config = lh.KernelConfig(
        omega_per_ns=np.linspace(0., args.omega_max, args.frequencies),
        relative_time_edges_ns=np.arange(-60., 740. + args.bin_ns, args.bin_ns),
        receiver_block=args.receiver_block,
        spectral_folded_cache=args.spectral_folded,
        cache_directory=args.cache, cache_policy="require")
    kernel = bgvd.kernel(config)

    payloads, summary = [], []
    for identifier, label, source in tracks:
        midpoint = np.asarray(source.start_m) + 0.5 * source.length_m * source.direction
        print(identifier, "midpoint_m:", midpoint.tolist(),
              "direction:", source.direction.tolist(), "length_m:", source.length_m)
        response = kernel.transport(source, method="track")
        active = int(response.active.sum())
        if active == 0 and not args.allow_empty:
            raise SystemExit(
                f"{identifier}: no active modules for midpoint_m={midpoint.tolist()} "
                f"and direction={source.direction.tolist()}")
        response_path = response.save(output / f"{identifier}.npz")
        payloads.append(lh.viewer_payload(
            response, source=source, event_id=identifier, label=label,
            detector_label="Baikal-GVD 2021"))
        summary.append({"event_id": identifier, "midpoint_m": midpoint.tolist(),
                        "direction": source.direction.tolist(),
                        "length_m": source.length_m, "beta": source.beta,
                        "time_ns": source.time_ns, "active_modules": active,
                        "total_charge_pe": float(response.charge_pe.sum()),
                        "response": str(response_path)})
        print(identifier, "active:", active,
              "charge [p.e.]:", float(response.charge_pe.sum()))

    viewer = lh.write_event_viewer(
        lh.merge_event_viewers(*payloads), output / "viewer.html")
    (output / "manifest.json").write_text(
        json.dumps({"events": summary, "viewer": str(viewer)}, indent=2),
        encoding="utf-8")
    print("viewer:", viewer)


if __name__ == "__main__":
    main()
