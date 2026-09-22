"""Prepare reusable private-BGVD transport tables before event processing."""
import argparse
import numpy as np
import lighthit as lh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bgvd-model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--kind", choices=("laser", "events", "all"), default="events")
    parser.add_argument("--wavelength", type=float, default=532.)
    parser.add_argument("--event-method", default="axial",
                        choices=("axial", "axial_full", "axial_centroid"))
    parser.add_argument("--bin-ns", type=float, default=5.)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    bgvd = lh.load_bgvd_model(args.bgvd_model, dataset="2021")
    config = lh.KernelConfig(
        relative_time_edges_ns=np.arange(-60., 740. + args.bin_ns, args.bin_ns),
        cache_directory=args.cache)
    kernel = bgvd.kernel(config)
    geometry = lh.describe_geometry(bgvd.detector)
    print("modules:", geometry.array.modules)
    print("centroid_m:", geometry.centroid_m.tolist())
    print("bounds_min_m:", geometry.bounds_min_m.tolist())
    print("bounds_max_m:", geometry.bounds_max_m.tolist())
    if args.kind in ("laser", "all"):
        kernel.build([args.wavelength], method="isotropic", force=args.force,
                     progress="console")
    if args.kind in ("events", "all"):
        kernel.build(method=args.event_method, force=args.force, progress="console")


if __name__ == "__main__":
    main()
