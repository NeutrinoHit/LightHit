"""Monochromatic isotropic calibration flash on the full BGVD 2021 geometry."""
import argparse
import shlex

import numpy as np
import lighthit as lh
from laser_cache import laser_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bgvd-model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", default="laser-response.npz")
    parser.add_argument("--wavelength", type=float, default=532.0)
    parser.add_argument("--photons", type=float, default=1e12)
    parser.add_argument("--position", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="laser position [m]; default is the printed detector centroid")
    parser.add_argument("--max-radius-m", type=float, default=None,
                        help="shared laser cache radius [m]; use the same value when building it")
    parser.add_argument("--bin-ns", type=float, default=5.)
    parser.add_argument("--frequencies", type=int, default=321)
    parser.add_argument("--omega-max", type=float, default=1.2)
    parser.add_argument("--viewer", default=None)
    args = parser.parse_args()
    if args.frequencies < 2 or args.omega_max <= 0:
        parser.error("--frequencies must be >= 2 and --omega-max must be positive")
    bgvd = lh.load_bgvd_model(args.bgvd_model, dataset="2021")
    geometry = lh.describe_geometry(bgvd.detector)
    centre = geometry.centroid_m if args.position is None else np.asarray(args.position)
    config = laser_config(bgvd.detector, cache_directory=args.cache,
                          bin_ns=args.bin_ns, cache_policy="require",
                          max_radius_m=args.max_radius_m,
                          omega_per_ns=np.linspace(0., args.omega_max,
                                                   args.frequencies))
    kernel = bgvd.kernel(config)
    print("laser position_m:", centre.tolist())
    print("shared radial cache range_m:", list(config.radial_range_m))
    print("spatial k panel_per_m:", config.k_panel_per_m)
    source = lh.IsotropicFlash.monochromatic(
        centre, args.photons, args.wavelength)
    try:
        response = kernel.transport(source)
    except FileNotFoundError as exc:
        command = shlex.join([
            "python", "examples/production_cache.py",
            "--bgvd-model", args.bgvd_model, "--cache", args.cache,
            "--kind", "laser", "--wavelength", str(args.wavelength),
            "--frequencies", str(args.frequencies),
            "--omega-max", str(args.omega_max),
        ] + ([] if args.max_radius_m is None else
             ["--max-radius-m", str(args.max_radius_m)]))
        raise SystemExit(
            f"The shared laser cache profile is not ready. Prepare it with:\n{command}"
        ) from exc
    except ValueError as exc:
        if "outside radial_range_m" not in str(exc):
            raise
        distance = np.linalg.norm(
            bgvd.detector.positions_m - centre[None, :], axis=1)
        suggested = int(np.ceil(1.05 * float(distance.max()) / 10) * 10)
        raise SystemExit(
            f"{exc}\nTo retain full scattered spectra for all OMs, rebuild the laser "
            f"cache with --max-radius-m {suggested} and pass the same value "
            "to this command. The cache remains reusable at other source positions."
        ) from exc
    response.save(args.output)
    if args.viewer is not None:
        payload = lh.viewer_payload(
            response, source=source, event_id="laser", label="calibration laser",
            detector_label="Baikal-GVD 2021")
        print("viewer:", lh.write_event_viewer(payload, args.viewer))
    omitted = int(response.metadata["omitted_outside_range"])
    print("OMs:", len(bgvd.detector), "active:", int(response.active.sum()),
          "total expected p.e.:", response.charge_pe.sum())
    if omitted:
        print("OMs outside radial cache:", omitted,
              "(direct light retained; scattered spectrum omitted after threshold screening)")


if __name__ == "__main__":
    main()
