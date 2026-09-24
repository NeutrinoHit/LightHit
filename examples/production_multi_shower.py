"""Transport several explicitly posed G4 showers through one prepared kernel."""
from pathlib import Path
import argparse
import json
import tomllib

import numpy as np
import lighthit as lh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bgvd-model", required=True)
    parser.add_argument("--events", required=True,
                        help="TOML file with numeric position_m and direction per event")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bin-ns", type=float, default=5.)
    parser.add_argument("--frequencies", type=int, default=321)
    parser.add_argument("--omega-max", type=float, default=1.2)
    parser.add_argument("--spectral-folded", dest="spectral_folded", action="store_true",
                        default="auto", help="require the wavelength-integrated cache")
    parser.add_argument("--reference", dest="spectral_folded", action="store_false",
                        help="use the original nine-wavelength calculation")
    parser.add_argument("--azimuthal-degree", type=int, default=4)
    parser.add_argument("--receiver-block", type=int, default=8)
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    if args.frequencies < 2 or args.omega_max <= 0:
        parser.error("--frequencies must be >= 2 and --omega-max must be positive")

    specification = tomllib.loads(Path(args.events).read_text(encoding="utf-8"))
    events = specification.get("events", [])
    if not events:
        raise SystemExit("events TOML must contain at least one [[events]] table")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bgvd = lh.load_bgvd_model(args.bgvd_model, dataset="2021")
    config = lh.KernelConfig(
        omega_per_ns=np.linspace(0., args.omega_max, args.frequencies),
        relative_time_edges_ns=np.arange(-60., 740. + args.bin_ns, args.bin_ns),
        azimuthal_degree=args.azimuthal_degree,
        receiver_block=args.receiver_block,
        spectral_folded_cache=args.spectral_folded,
        cache_directory=args.cache, cache_policy="require")
    kernel = bgvd.kernel(config)
    payloads, summary = [], []
    for item in events:
        identifier = str(item["id"])
        pose = lh.SourcePose(item["position_m"], item["direction"],
                             item.get("time_ns", 0.))
        source = lh.G4Shower.from_hdf5(
            item["file"], event=item.get("event", 0)).placed(pose)
        distances = np.linalg.norm(
            bgvd.detector.positions_m - source.centroid_m[None, :], axis=1)
        print(identifier, "pose:", pose.position_m.tolist(), pose.direction.tolist(),
              "distance range_m:", [float(distances.min()), float(distances.max())])
        response = kernel.transport(source)
        active = int(response.active.sum())
        if active == 0 and not args.allow_empty:
            raise SystemExit(
                f"{identifier}: no active modules for explicit pose "
                f"position_m={pose.position_m.tolist()}, "
                f"direction={pose.direction.tolist()}")
        response_path = response.save(output / f"{identifier}.npz")
        payloads.append(lh.viewer_payload(
            response, source=source, event_id=identifier,
            label=item.get("label", identifier), detector_label="Baikal-GVD 2021"))
        summary.append({"event_id": identifier, "input": source.input_path,
                        "input_event": source.event,
                        "position_m": pose.position_m.tolist(),
                        "direction": pose.direction.tolist(),
                        "active_modules": active,
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
