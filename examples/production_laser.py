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
    args = parser.parse_args()
    bgvd = lh.load_bgvd_model(args.bgvd_model, dataset="2021")
    kernel = bgvd.kernel(lh.KernelConfig(cache_directory=args.cache)).build([args.wavelength])
    centre = bgvd.detector.positions_m.mean(axis=0)
    source = lh.IsotropicFlash.monochromatic(
        centre, args.photons, args.wavelength)
    response = kernel.transport(source)
    response.save(args.output)
    print("OMs:", len(bgvd.detector), "total expected p.e.:", response.charge_pe.sum())


if __name__ == "__main__":
    main()
