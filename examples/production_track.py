"""Finite Cherenkov track: two-field spectral axial source, exact directional OM."""
import argparse
import numpy as np
import lighthit as lh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bgvd-model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", default="track-response.npz")
    parser.add_argument("--position", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="track midpoint [m]; default is the printed detector centroid")
    parser.add_argument("--direction", type=float, nargs=3, default=[.7, .3, .64],
                        metavar=("DX", "DY", "DZ"))
    parser.add_argument("--length-m", type=float, default=120.)
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
    direction = np.asarray(args.direction, float); direction /= np.linalg.norm(direction)
    print("track midpoint_m:", centre.tolist())
    print("track direction:", direction.tolist())
    source = lh.CherenkovTrack.centered(centre, direction, args.length_m)
    response = kernel.transport(source, method="track")
    response.save(args.output)
    if args.viewer is not None:
        payload = lh.viewer_payload(
            response, source=source, event_id="track", label="Cherenkov track",
            detector_label="Baikal-GVD 2021")
        print("viewer:", lh.write_event_viewer(payload, args.viewer))
    print("OM angular model:", response.metadata["detector_angular_model"])
    print("active OMs:", response.active.sum(), "total expected p.e.:", response.charge_pe.sum())


if __name__ == "__main__":
    main()
