"""Stored G4 shower on BGVD geometry with selectable transport method."""
import argparse
import lighthit as lh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bgvd-model", required=True)
    parser.add_argument("--g4", required=True)
    parser.add_argument("--event", type=int, default=5)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--method", default="axial",
                        choices=["axial", "axial_full", "axial_centroid"])
    parser.add_argument("--output", default="shower-response.npz")
    args = parser.parse_args()
    bgvd = lh.load_bgvd_model(args.bgvd_model, dataset="2021")
    kernel = bgvd.kernel(lh.KernelConfig(cache_directory=args.cache)).build(
        method=args.method)
    source = lh.G4Shower.from_hdf5(args.g4, event=args.event)
    response = kernel.transport(
        source, method=args.method, allow_experimental=args.method != "axial")
    response.save(args.output)
    print("OM angular model:", response.metadata["detector_angular_model"])
    print("active OMs:", response.active.sum(), "total expected p.e.:", response.charge_pe.sum())


if __name__ == "__main__":
    main()
