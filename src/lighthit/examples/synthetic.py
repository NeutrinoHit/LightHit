"""Public Baikal-like demonstration with no private detector inputs."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json

import numpy as np

import lighthit as lh


def synthetic_medium_model():
    return lh.SpectralMedium(
        [400., 450., 500.], [.030, .020, .040], [.030, .022, .017],
        [1.344, 1.339, 1.336], [1.386, 1.374, 1.367], g=.9,
        provenance="synthetic Baikal-like demo; not a calibration")


def baikal_like_detector(*, clusters=1, strings=8, modules=6,
                         cluster_pitch_m=180., string_radius_m=25.,
                         vertical_spacing_m=15.):
    """Small deterministic array inspired by, but not equal to, Baikal-GVD."""
    cluster_centres = np.zeros((clusters, 3))
    cluster_centres[:, 0] = np.arange(clusters) * cluster_pitch_m
    azimuth = np.arange(strings) * 2 * np.pi / strings
    string_offsets = np.column_stack((string_radius_m * np.cos(azimuth),
                                      string_radius_m * np.sin(azimuth),
                                      np.zeros(strings)))
    heights = (np.arange(modules) - (modules - 1) / 2) * vertical_spacing_m
    positions = (cluster_centres[:, None, None, :]
                 + string_offsets[None, :, None, :]
                 + np.column_stack((np.zeros(modules), np.zeros(modules), heights))
                 [None, None, :, :]).reshape(-1, 3)
    count = len(positions)
    orientations = np.tile([0., 0., -1.], (count, 1))

    def angular_acceptance(cosine):
        x = np.asarray(cosine, float)
        return .45 + .45 * x + .10 * x * x

    def spectral_efficiency(wavelength_nm):
        wavelength = np.asarray(wavelength_nm, float)
        return .20 * np.exp(-0.5 * ((wavelength - 450.) / 90.) ** 2)

    return lh.DetectorArray(
        positions, orientations, np.pi * .20 ** 2,
        angular_acceptance, spectral_efficiency,
        identifiers={
            "cluster_id": np.repeat(np.arange(clusters), strings * modules),
            "string_id": np.tile(np.repeat(np.arange(strings), modules), clusters),
            "module_id": np.tile(np.arange(modules), clusters * strings),
        },
        provenance="synthetic Baikal-like geometry and OM; not a calibration")


def quick_config(cache_directory, *, require=False, bin_ns=5.):
    return lh.KernelConfig(
        omega_per_ns=np.linspace(0., .30, 81),
        relative_time_edges_ns=np.arange(-40., 320. + bin_ns, bin_ns),
        wavelength_nodes=2, scattering_degree=8, source_degree=8,
        azimuthal_degree=2, cell_m=.5, k_max_per_m=4.,
        k_panel_per_m=.1, k_order=6, radial_range_m=(2., 120.),
        radial_nodes=32, angular_backend="auto", threshold_pe=.001,
        cache_directory=cache_directory,
        cache_policy="require" if require else "build")


def standard_config(cache_directory, *, require=False, bin_ns=5.):
    return lh.KernelConfig(
        relative_time_edges_ns=np.arange(-60., 740. + bin_ns, bin_ns),
        cache_directory=cache_directory,
        cache_policy="require" if require else "build")


def example_sources(detector):
    centre = lh.describe_geometry(detector).centroid_m
    return {
        "laser": lh.IsotropicFlash.monochromatic(
            centre + np.array([0., 0., 30.]), 1e10, 450.),
        "track": lh.CherenkovTrack.centered(
            centre + np.array([15., 0., 0.]), [.25, .15, .956], 20., beta=.999),
        "shower": lh.SyntheticShower.gaussian(
            lh.SourcePose(centre + np.array([-12., 8., 0.]),
                          [.35, -.15, .925], 0.),
            charged_track_length_m=300., elements=48, seed=7),
    }


def _config(profile, cache, *, require):
    return (quick_config(cache, require=require) if profile == "quick"
            else standard_config(cache, require=require))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def geometry_command(arguments):
    detector = baikal_like_detector(clusters=arguments.clusters)
    print(json.dumps(lh.describe_geometry(detector).as_dict(), indent=2))


def cache_command(arguments):
    detector = baikal_like_detector(clusters=arguments.clusters)
    kernel = lh.TransportKernel(
        synthetic_medium_model(), detector,
        _config(arguments.profile, arguments.cache, require=False))
    sources = example_sources(detector)
    selected = sources if arguments.sources == ["all"] else {
        name: sources[name] for name in arguments.sources}
    for name, source in selected.items():
        print(f"Prepare {name} cache")
        kernel.build(source=source, force=arguments.force, progress="console")


def run_command(arguments):
    output = Path(arguments.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    detector = baikal_like_detector(clusters=arguments.clusters)
    config = _config(arguments.profile, arguments.cache, require=True)
    kernel = lh.TransportKernel(synthetic_medium_model(), detector, config)
    sources = example_sources(detector)
    selected = sources if arguments.sources == ["all"] else {
        name: sources[name] for name in arguments.sources}
    payloads = []
    rows = []
    for name, source in selected.items():
        print(f"Transport {name}")
        response = kernel.transport(source)
        active = int(response.active.sum())
        if active == 0 and not arguments.allow_empty:
            raise RuntimeError(
                f"{name}: no active modules; inspect the explicit source pose, "
                "threshold and detector bounds, or pass --allow-empty")
        response_path = response.save(output / f"{name}-response.npz")
        payloads.append(lh.viewer_payload(
            response, source=source, event_id=name, label=name,
            detector_label="synthetic Baikal-like detector"))
        rows.append({"event": name, "active_modules": active,
                     "total_charge_pe": float(response.charge_pe.sum()),
                     "response": str(response_path)})
    payload = lh.merge_event_viewers(*payloads)
    viewer = lh.write_event_viewer(payload, output / "viewer.html")
    manifest = {
        "schema": "lighthit/synthetic-demo/1",
        "version": lh.__version__, "profile": arguments.profile,
        "geometry": lh.describe_geometry(detector).as_dict(),
        "config": asdict(config), "events": rows, "viewer": str(viewer),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    print("viewer:", viewer)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    geometry = sub.add_parser("geometry", help="print explicit array bounds and centres")
    geometry.add_argument("--clusters", type=int, default=1)
    geometry.set_defaults(function=geometry_command)
    cache = sub.add_parser("cache", help="build or validate reusable transport tables")
    cache.add_argument("--cache", required=True)
    cache.add_argument("--profile", choices=("quick", "standard"), default="quick")
    cache.add_argument("--clusters", type=int, default=1)
    cache.add_argument("--sources", nargs="+", choices=("all", "laser", "track", "shower"),
                       default=["all"])
    cache.add_argument("--force", action="store_true")
    cache.set_defaults(function=cache_command)
    run = sub.add_parser("run", help="transport events using an already prepared cache")
    run.add_argument("--cache", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--profile", choices=("quick", "standard"), default="quick")
    run.add_argument("--clusters", type=int, default=1)
    run.add_argument("--sources", nargs="+", choices=("all", "laser", "track", "shower"),
                     default=["all"])
    run.add_argument("--allow-empty", action="store_true")
    run.set_defaults(function=run_command)
    return value


def main(argv=None):
    arguments = parser().parse_args(argv)
    arguments.function(arguments)


if __name__ == "__main__":
    main()
