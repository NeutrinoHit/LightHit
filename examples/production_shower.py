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
    parser.add_argument("--frequencies", type=int, default=321)
    parser.add_argument("--omega-max", type=float, default=1.2)
    parser.add_argument("--spectral-folded", dest="spectral_folded", action="store_true",
                        default="auto", help="require the wavelength-integrated cache")
    parser.add_argument("--reference", dest="spectral_folded", action="store_false",
                        help="use the original nine-wavelength calculation")
    parser.add_argument("--azimuthal-degree", type=int, default=4,
                        help="source azimuthal modes; 0 is faster but must be checked against 4")
    parser.add_argument("--receiver-block", type=int, default=8)
    parser.add_argument("--method", default="axial",
                        choices=["axial", "axial_full", "axial_centroid"])
    parser.add_argument("--output", default="shower-response.npz")
    parser.add_argument("--viewer", default=None,
                        help="optional standalone viewer HTML path")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    if args.frequencies < 2 or args.omega_max <= 0:
        parser.error("--frequencies must be >= 2 and --omega-max must be positive")
    bgvd = lh.load_bgvd_model(args.bgvd_model, dataset="2021")
    config = lh.KernelConfig(
        omega_per_ns=np.linspace(0., args.omega_max, args.frequencies),
        relative_time_edges_ns=np.arange(-60., 740. + args.bin_ns, args.bin_ns),
        azimuthal_degree=args.azimuthal_degree,
        receiver_block=args.receiver_block,
        spectral_folded_cache=args.spectral_folded,
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
    print("source azimuthal degree:", args.azimuthal_degree)
    if "compiled_source_cells" in response.metadata:
        print("compiled source:", response.metadata["compiled_source_cells"],
              "cells,", response.metadata["compiled_source_channels"],
              "channels,", round(response.metadata["compiled_source_memory_mib"], 1),
              "MiB for both fields")
    print("timings_s:", {key: round(float(response.metadata.get(key, 0.)), 3)
                         for key in ("compile_seconds", "cache_seconds",
                                     "prepass_seconds", "zero_apply_seconds",
                                     "apply_seconds", "full_prepare_seconds",
                                     "full_apply_seconds", "elapsed_seconds")})
    print("active OMs:", response.active.sum(), "total expected p.e.:", response.charge_pe.sum())


if __name__ == "__main__":
    main()
