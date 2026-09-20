"""Finite Cherenkov track: two-field spectral axial source, exact directional OM."""
import argparse
import numpy as np
import lighthit as lh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bgvd-model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", default="track-response.npz")
    args = parser.parse_args()
    bgvd = lh.load_bgvd_model(args.bgvd_model, dataset="2021")
    kernel = bgvd.kernel(lh.KernelConfig(cache_directory=args.cache)).build(
        method="track")
    centre = bgvd.detector.positions_m.mean(axis=0)
    direction = np.array([0.7, 0.3, 0.64]); direction /= np.linalg.norm(direction)
    source = lh.CherenkovTrack(centre - 60 * direction, direction, 120.0)
    response = kernel.transport(source, method="track")
    response.save(args.output)
    print("OM angular model:", response.metadata["detector_angular_model"])
    print("active OMs:", response.active.sum(), "total expected p.e.:", response.charge_pe.sum())


if __name__ == "__main__":
    main()
