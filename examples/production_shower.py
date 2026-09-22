"""Stored G4 shower on BGVD geometry with selectable transport method."""
import argparse
from pathlib import Path
import numpy as np
import lighthit as lh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bgvd-model", required=True)
    parser.add_argument("--g4", required=True)
    parser.add_argument("--event", type=int, default=5)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--position", type=float, nargs=3, required=True,
                        metavar=("X", "Y", "Z"),
                        help="target photon-weighted shower centroid in detector coordinates [m]")
    parser.add_argument("--direction", type=float, nargs=3, required=True,
                        metavar=("DX", "DY", "DZ"),
                        help="target principal shower direction")
    parser.add_argument("--time-ns", type=float, default=0.,
                        help="target earliest emission time")
    parser.add_argument("--bin-ns", type=float, default=5.)
    parser.add_argument("--method", default="axial",
                        choices=["axial", "axial_full", "axial_centroid"])
    parser.add_argument("--output", default="shower-response.npz")
    parser.add_argument("--viewer", default=None,
                        help="optional standalone viewer HTML path")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    bgvd = lh.load_bgvd_model(args.bgvd_model, dataset="2021")
    config = lh.KernelConfig(
        relative_time_edges_ns=np.arange(-60., 740. + args.bin_ns, args.bin_ns),
        cache_directory=args.cache, cache_policy="require")
    kernel = bgvd.kernel(config)
    source = lh.G4Shower.from_hdf5(args.g4, event=args.event).placed(
        lh.SourcePose(args.position, args.direction, args.time_ns))
    geometry = lh.describe_geometry(bgvd.detector)
    distances = np.linalg.norm(
        bgvd.detector.positions_m - source.centroid_m[None, :], axis=1)
    print("source centroid_m:", source.centroid_m.tolist())
    print("source direction:", source.principal_axis.tolist())
    print("source extent_m:", source.extent_m)
    print("detector bounds_min_m:", geometry.bounds_min_m.tolist())
    print("detector bounds_max_m:", geometry.bounds_max_m.tolist())
    print("centroid-to-OM distance range_m:",
          [float(distances.min()), float(distances.max())])
    response = kernel.transport(
        source, method=args.method, allow_experimental=args.method != "axial")
    if not np.any(response.active) and not args.allow_empty:
        raise SystemExit(
            "No active modules. Check --position/--direction and threshold.\n"
            f"source centroid_m={source.centroid_m.tolist()}\n"
            f"detector bounds_min_m={geometry.bounds_min_m.tolist()}\n"
            f"detector bounds_max_m={geometry.bounds_max_m.tolist()}\n"
            "Pass --allow-empty only when this is intentional.")
    response.save(args.output)
    if args.viewer is not None:
        payload = lh.viewer_payload(
            response, source=source, event_id=f"g4-event-{args.event}",
            label=f"G4 shower, event {args.event}", detector_label="Baikal-GVD 2021")
        viewer = lh.write_event_viewer(payload, Path(args.viewer))
        print("viewer:", viewer)
    print("OM angular model:", response.metadata["detector_angular_model"])
    print("active OMs:", response.active.sum(), "total expected p.e.:", response.charge_pe.sum())


if __name__ == "__main__":
    main()
