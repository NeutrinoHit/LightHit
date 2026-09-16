"""Reproducible point-Green calculation; output stays in a caller-chosen directory."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import tomllib
import numpy as np
import scipy
from . import __version__
from .medium import Medium, synthetic_medium
from .green import PointGreenSolver, SolverSettings
from .providers import load_bgvd_water


def peak_rss_mib():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value / (1024**2 if sys.platform == "darwin" else 1024)
    except (ImportError, AttributeError):
        return None


def configure(path, preset=None):
    with open(path, "rb") as stream:
        config = tomllib.load(stream)
    m = config.get("medium", {})
    kind = m.get("kind", "synthetic")
    if kind == "synthetic":
        medium = synthetic_medium()
    elif kind == "parameters":
        medium = Medium(**{k: v for k, v in m.items() if k != "kind"})
    elif kind == "bgvd_water":
        medium = load_bgvd_water(m.get("path"), wavelength_nm=m.get("wavelength_nm", 450),
                                 g=m.get("g", 0.9))
    else:
        raise ValueError(f"Unknown medium kind: {kind}")
    settings = SolverSettings(**config.get("solver", {}))
    if preset is not None:
        settings = {"quick": SolverSettings.quick, "balanced": SolverSettings,
                    "refined": SolverSettings.refined}[preset]()
    return config, medium, settings


def save_figures(result, raw, smooth, output):
    import matplotlib.pyplot as plt
    directory = output / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    # Independent figures, default matplotlib colors. Rates are bin averages.
    for d in range(len(result.radii_m)):
        for profile, name in ((raw, "raw"), (smooth, "readout")):
            fig, ax = plt.subplots(figsize=(8, 4.8))
            for p, label in enumerate(("0 scattering", "1 scattering", ">=2 scattering")):
                ax.plot(profile.centers_ns, profile.components[d, :, p] / np.diff(profile.edges_ns), label=label)
            ax.plot(profile.centers_ns, profile.rate_per_m2_ns[d], label="total", linewidth=2)
            ax.axvline(result.front_time_ns[d], linestyle=":", label="r / c_g")
            ax.set(xlabel="Time since emission origin [ns]", ylabel="Response [m^-2 ns^-1]",
                   title=f"Point {d}; sigma = {profile.sigma_ns:g} ns")
            ax.legend(); fig.tight_layout(); fig.savefig(directory / f"{name}-{d}.png", dpi=150)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4.8))
        spectrum = result.spectrum[:, d] * np.exp(-1j * result.omega_per_ns * result.front_time_ns[d])
        ax.plot(result.omega_per_ns, spectrum.real, label="Re")
        ax.plot(result.omega_per_ns, spectrum.imag, label="Im")
        ax.set(xlabel="omega [rad/ns]", ylabel="Spectrum [m^-2]", title="Spectrum with light-front phase removed")
        ax.legend(); fig.tight_layout(); fig.savefig(directory / f"spectrum-{d}.png", dpi=150)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="LightHit: point Green function, per unit effective detector area")
    parser.add_argument("--config", default="examples/point-green.toml")
    parser.add_argument("--output", default=".build/point-green")
    parser.add_argument("--preset", choices=["quick", "balanced", "refined"])
    parser.add_argument("--repeat", type=int, default=1, help="Number of fresh solves; no hidden warm-cache runs")
    parser.add_argument("--charge-only", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    start = perf_counter()
    config, medium, settings = configure(args.config, args.preset)
    source = config.get("source", {})
    mode = source.get("kind", "directed")
    if mode not in ("directed", "isotropic"):
        raise ValueError("source.kind must be directed or isotropic")
    direction = source.get("direction", [0, 0, 1]) if mode == "directed" else None
    obs = config.get("detector", {}).get("displacement_m", [17.32050807568877, 0, 10])
    fr = config.get("frequency", {})
    omega = np.array([0.0]) if args.charge_only else np.linspace(0, fr.get("max_per_ns", 1.2), fr.get("nodes", 241))
    solver = PointGreenSolver(medium, settings)
    runs = []
    result = None
    for index in range(args.repeat):
        result = solver.solve(omega, obs, direction=direction, photons=source.get("photons", 1),
                              emission_time_ns=source.get("time_ns", 0))
        runs.append(result.timings_s)
        print(f"solve {index + 1}: {result.timings_s['total']:.6f} s", flush=True)
    report = dict(version=__version__, medium=asdict(medium), settings=asdict(settings),
                  source=source, displacement_m=result.displacement_m.tolist(),
                  charge_per_m2=result.charge_per_m2.tolist(),
                  charge_components_per_m2=result.components[0].real.tolist(),
                  front_time_ns=result.front_time_ns.tolist(),
                  solve_runs=runs, median_solve_s=float(np.median([r['total'] for r in runs])),
                  first_quadrature_error=result.first_quadrature_error,
                  environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                   platform=platform.platform(), machine=platform.machine(), processor=platform.processor()),
                  approximation="Exact 0 and full-HG 1; all >=2 use HG coefficients through L and exact free angular tail",
                  units="m^-2 per photon; detector efficiency=1; multiply by small effective area for expected counts",
                  cache="none; the angular solve is shared by observations within a call",
                  private_data_note="Results from a private medium must remain local")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result.save(output / "spectrum.npz")
    if not args.charge_only:
        tr = config.get("readout", {})
        front = float(np.min(result.front_time_ns))
        width = tr.get("bin_width_ns", 2.0)
        if not np.isfinite(width) or width <= 0:
            raise ValueError("readout.bin_width_ns must be positive")
        edges = np.arange(front + tr.get("start_relative_front_ns", -30),
                          front + tr.get("stop_relative_front_ns", 650) + width * 0.5, width)
        raw = result.readout(edges, 0)
        smooth = result.readout(edges, tr.get("sigma_ns", 3.0))
        report["raw"] = dict(elapsed_s=raw.elapsed_s, **raw.diagnostics)
        report["readout"] = dict(sigma_ns=smooth.sigma_ns, elapsed_s=smooth.elapsed_s, **smooth.diagnostics)
        np.savez_compressed(output / "profiles.npz", edges_ns=edges,
                            raw_components=raw.components, readout_components=smooth.components,
                            sigma_ns=smooth.sigma_ns)
        if args.plots:
            save_figures(result, raw, smooth, output)
    report["peak_process_rss_mib"] = peak_rss_mib()
    report["peak_rss_note"] = "Whole-process high-water mark, not incremental kernel memory"
    report["elapsed_before_report_write_s"] = perf_counter() - start
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("charge [m^-2]:", result.charge_per_m2)
    print("report:", output / "report.json")
    return 0
