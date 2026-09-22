"""Monochromatic isotropic calibration flash on the full BGVD 2021 geometry."""
import argparse
from pathlib import Path

import numpy as np
import lighthit as lh


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
    parser.add_argument("--bin-ns", type=float, default=5.)
    parser.add_argument("--viewer", default=None)
    args = parser.parse_args()
    bgvd = lh.load_bgvd_model(args.bgvd_model, dataset="2021")
    config = lh.KernelConfig(
        relative_time_edges_ns=np.arange(-60., 740. + args.bin_ns, args.bin_ns),
        cache_directory=args.cache, cache_policy="require")
    kernel = bgvd.kernel(config)
    geometry = lh.describe_geometry(bgvd.detector)
    centre = geometry.centroid_m if args.position is None else np.asarray(args.position)
    print("laser position_m:", centre.tolist())
    source = lh.IsotropicFlash.monochromatic(
        centre, args.photons, args.wavelength)
    response = kernel.transport(source)
    response.save(args.output)
    if args.viewer is not None:
        payload = lh.viewer_payload(
            response, source=source, event_id="laser", label="calibration laser",
            detector_label="Baikal-GVD 2021")
        print("viewer:", lh.write_event_viewer(payload, args.viewer))
    print("OMs:", len(bgvd.detector), "total expected p.e.:", response.charge_pe.sum())


if __name__ == "__main__":
    main()
