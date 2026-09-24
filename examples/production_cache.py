"""Prepare reusable private-BGVD transport tables before event processing."""
import argparse
from pathlib import Path
import numpy as np
import lighthit as lh
from laser_cache import laser_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bgvd-model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--kind", choices=("laser", "events", "all"), default="events")
    parser.add_argument("--wavelength", type=float, default=532.)
    parser.add_argument("--max-radius-m", type=float, default=None,
                        help="laser cache radius [m]; default is fixed by detector geometry")
    parser.add_argument("--position", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"), help=argparse.SUPPRESS)
    parser.add_argument("--event-method", default="axial",
                        choices=("axial", "axial_full", "axial_centroid"))
    parser.add_argument("--bin-ns", type=float, default=5.)
    parser.add_argument("--frequencies", type=int, default=321)
    parser.add_argument("--omega-max", type=float, default=1.2)
    parser.add_argument("--fold-spectral", action="store_true",
                        help="prepare source-independent Cherenkov wavelength folds after event caches")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.frequencies < 2 or args.omega_max <= 0:
        parser.error("--frequencies must be >= 2 and --omega-max must be positive")
    if args.position is not None:
        parser.error("--position is a source property, not a cache property; "
                     "use --max-radius-m only if the shared laser cache needs a larger radius")
    if args.max_radius_m is not None and args.kind == "events":
        parser.error("--max-radius-m applies to the laser cache, not events")
    if args.fold_spectral and args.kind == "laser":
        parser.error("--fold-spectral requires event directional caches")
    if args.fold_spectral and args.event_method == "axial_centroid":
        parser.error("--fold-spectral requires the directional axial event method")
    bgvd = lh.load_bgvd_model(args.bgvd_model, dataset="2021")
    config = lh.KernelConfig(
        omega_per_ns=np.linspace(0., args.omega_max, args.frequencies),
        relative_time_edges_ns=np.arange(-60., 740. + args.bin_ns, args.bin_ns),
        cache_directory=args.cache)
    kernel = bgvd.kernel(config)
    geometry = lh.describe_geometry(bgvd.detector)
    print("modules:", geometry.array.modules)
    print("centroid_m:", geometry.centroid_m.tolist())
    print("bounds_min_m:", geometry.bounds_min_m.tolist())
    print("bounds_max_m:", geometry.bounds_max_m.tolist())
    if args.kind in ("laser", "all"):
        laser_settings = laser_config(
            bgvd.detector, cache_directory=args.cache, bin_ns=args.bin_ns,
            omega_per_ns=config.omega_per_ns,
            max_radius_m=args.max_radius_m)
        print("laser cache profile: source-position independent")
        print("laser radial cache range_m:", list(laser_settings.radial_range_m))
        print("laser spatial k panel_per_m:", laser_settings.k_panel_per_m)
        bgvd.kernel(laser_settings).build(
            [args.wavelength], method="isotropic", force=args.force,
            progress="console")
    if args.kind in ("events", "all"):
        print("event radial cache range_m:", list(config.radial_range_m))
        ready = args.fold_spectral and not args.force and all(
            (Path(args.cache).expanduser() /
             f"directional-{kernel._directional_cache_key(float(wavelength))}.npz").is_file()
            for wavelength in kernel.wavelength.wavelength_nm)
        if ready:
            print("Directional wavelength caches ready; folding without loading all into RAM.")
        else:
            kernel.build(method=args.event_method, force=args.force, progress="console")
        if args.fold_spectral:
            kernel.build_folded_directional(progress=True)


if __name__ == "__main__":
    main()
