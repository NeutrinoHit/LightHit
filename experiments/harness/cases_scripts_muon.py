"""The muon-route research scripts (scripts/shower_basis) as measured cases.

Each runs the ported script on the standard inputs (scripts/shower_basis/paths.py)
with its own defaults unless stated. Output folders keep the scripts' former
names below experiments/results/work, so that a script that consumes another
script's output finds it. Order matters where one script needs another's
output; experiments/README.md lists the order.
"""
from __future__ import annotations

from .record import RESULTS
from .scripts import script_case

W = RESULTS / "work"


def bundle(*parts):
    """A bundle path resolved when the case runs."""
    from .inputs import bundle_path
    return lambda: str(bundle_path(*parts))


def holdout():
    """The primary-only seed-43 holdout file, written on first use by paths.holdout_primary()."""
    import sys
    from .record import ROOT
    sys.path.insert(0, str(ROOT / "scripts" / "shower_basis"))
    import paths
    return str(paths.holdout_primary())


def w(*parts):
    return str(W.joinpath(*parts))


def out(idea, name):
    return w("scripts", idea, name)


S = "shower_basis/"

# --- secondary-light structure (source statistics, no transport)
script_case("muon-secondary-structure", "summary", [S + "analyze_muon_secondaries.py",
            "--output", w("muon-segments-20261009", "structure")],
            report=w("muon-segments-20261009", "structure", "summary.json"))
for _e in (100, 1000):
    script_case("muon-secondary-structure", f"capture-electron-{_e}GeV",
                [S + "analyze_muon_secondaries.py", "--capture-electron-energy", _e,
                 "--output", w("muon-segments-20261009", "structure")],
                notes="Geant4 electron reference in the same physics list (writes electron-current-<E>GeV.npz)")
script_case("muon-secondary-physics-basis", "angle-40", [S + "probe_secondary_physics_basis.py",
            "--output", w("muon-physics-20261009", "secondary")])
script_case("muon-secondary-physics-basis", "angle-300", [S + "probe_secondary_physics_basis.py",
            "--angle-bins", 300, "--output", w("muon-physics-20261009", "secondary", "angular300")])
script_case("muon-secondary-physics-basis", "cone-mixture", [S + "probe_cone_mixture_basis.py"])

# --- order 0
script_case("shower-order0-continuous-spectrum", "muon-1TeV-primary",
            [S + "benchmark_ballistic_interval.py", "--energy", 1000,
             "--output", out("shower-order0-continuous-spectrum", "muon-1TeV-primary")],
            report=out("shower-order0-continuous-spectrum", "muon-1TeV-primary") + "/report.json",
            clean=[out("shower-order0-continuous-spectrum", "muon-1TeV-primary")])
script_case("ballistic-spectral-fused", "muon-1TeV", [S + "probe_ballistic_spectral.py", "--energy", 1000,
            "--output", out("ballistic-spectral-fused", "muon-1TeV")],
            report=out("ballistic-spectral-fused", "muon-1TeV") + "/report.json")

# --- primary geometry and order 1
script_case("muon-primary-straightening", "wiggle-1TeV", [S + "probe_muon_wiggle.py", "--energy", 1000,
            "--output", out("muon-primary-straightening", "wiggle-1TeV"), "--interval-spectrum"])
script_case("muon-integral-fit", "reference-segment-1TeV", [S + "probe_muon_integral.py", "--energy", 1000,
            "--reference-method", "segment", "--receivers", 197, 198, 1241,
            "--output", w("muon-segments-20261009", "integral", "exact1000")])
script_case("muon-primary-straightening", "few-chords-1TeV", [S + "check_few_muon_segments.py",
            "--energy", 1000, "--output", out("muon-primary-straightening", "few-chords-1TeV")],
            report=out("muon-primary-straightening", "few-chords-1TeV") + "/report.json")
script_case("muon-primary-polyline", "1TeV", [S + "benchmark_primary_polyline.py", "--energy", 1000,
            "--output", out("muon-primary-polyline", "1TeV")])
script_case("primary-prompt", "1TeV", [S + "run_primary_stage.py", "--energy", 1000,
            "--output", out("primary-prompt", "1TeV")])
script_case("muon-first-charge-only", "300GeV-tail", [S + "probe_first_charge_only.py",
            "--output", out("muon-first-charge-only", "300GeV-tail")],
            clean=[out("muon-first-charge-only", "300GeV-tail")])

# --- primary atlases (probes and validation of the shipped tables)
script_case("track-first-atlas", "primitive-1TeV", [S + "probe_track_first_primitive.py", "--energy", 1000,
            "--output", out("track-first-atlas", "primitive-1TeV")])
script_case("track-ge2-atlas", "primitive-1TeV", [S + "probe_track_ge2_primitive.py", "--energy", 1000,
            "--output", out("track-ge2-atlas", "primitive-1TeV")])
script_case("track-ge2-atlas", "mini-atlas-1TeV", [S + "probe_track_ge2_primitive.py", "--energy", 1000,
            "--mini-atlas", "--output", out("track-ge2-atlas", "mini-atlas-1TeV")],
            clean=[out("track-ge2-atlas", "mini-atlas-1TeV")])
script_case("track-ge2-atlas", "primary-atlas-1TeV", [S + "probe_track_ge2_primitive.py", "--energy", 1000,
            "--primary-atlas", lambda: str(out("track-ge2-atlas", "mini-atlas-1TeV")) + "/atlas.npz",
            "--output", out("track-ge2-atlas", "primary-atlas-1TeV")],
            clean=[out("track-ge2-atlas", "primary-atlas-1TeV")])
script_case("track-ge2-atlas", "original-step-control-1TeV", [S + "probe_primary_ge2_original.py",
            "--energy", 1000, "--comparison", out("track-ge2-atlas", "primary-atlas-1TeV"),
            "--output", out("track-ge2-atlas", "original-step-control-1TeV")],
            clean=[out("track-ge2-atlas", "original-step-control-1TeV")])
script_case("first-beta-atlas", "response-rho30", [S + "probe_first_beta_response.py", "--rho", 30,
            "--output", out("first-beta-atlas", "response-rho30")])
script_case("ge2-beta-atlas", "tail-300GeV", [S + "probe_ge2_beta_response.py", "--energy", 300,
            "--output", out("ge2-beta-atlas", "tail-300GeV")], clean=[out("ge2-beta-atlas", "tail-300GeV")])
script_case("response-cubature", "fit", ["tools/tables/build_response_cubature.py",
            "--output", w("joint-response-20261010")])
script_case("response-cubature", "check-1TeV-300GeV", [S + "check_joint_response_basis.py",
            "--basis", w("joint-response-20261010", "basis-rank16.npz"), "--energies", 1000, 300,
            "--output", out("response-cubature", "check-1TeV-300GeV")],
            clean=[out("response-cubature", "check-1TeV-300GeV")])

# --- universal primary (route stage) against direct controls
#: the primary options MuonTransport passes (src/lighthit/muon/api.py)
ROUTE_PRIMARY = ["--ge2-beta-charge", bundle("tables", "muon", "ge2-beta-charge.npz"),
                 "--q1-beta-charge", bundle("tables", "muon", "first-beta-charge.npz"),
                 "--direct-ge2-quadrature", "filon", "--q1-charge-gauss-order", 6]
for _e in (1000, 300):
    script_case("universal-primary", f"controls-{_e}GeV", [S + "check_universal_primary.py",
                "--energy", _e, "--controls", 8, *ROUTE_PRIMARY,
                "--output", w("primary-universal-20261009", "results", f"seed42-{_e}")],
                notes="route primary options; direct controls of the top 8 predicted OMs and sentinels")
script_case("universal-primary", "controls-holdout43", [S + "check_universal_primary.py",
            "--source", holdout,
            "--controls", 8, *ROUTE_PRIMARY,
            "--output", w("primary-universal-20261009", "results", "holdout43")])
script_case("universal-primary", "original-steps-holdout43", [S + "check_original_primary_holdout.py"])
script_case("universal-primary", "time-check-1TeV", ["shower_basis/run_primary_time_check.sh", 1000])
script_case("universal-primary", "charge-check-1TeV", ["shower_basis/run_primary_charge_check.sh", 1000])
script_case("universal-primary", "filon-check-1TeV", ["shower_basis/run_primary_filon_check.sh", 1000])
script_case("muon-primary-direct-ge2-quadrature", "backends-300GeV", [S + "probe_direct_ge2_backend.py",
            "--energy", 300, "--output", out("muon-primary-direct-ge2-quadrature", "backends-300GeV")],
            clean=[out("muon-primary-direct-ge2-quadrature", "backends-300GeV")])
script_case("muon-primary-direct-ge2-quadrature", "backward-filon-holdout43", [S + "probe_backward_filon.py",
            "--result", w("primary-universal-20261009", "results", "holdout43"),
            "--output", out("muon-primary-direct-ge2-quadrature", "backward-filon-holdout43")],
            clean=[out("muon-primary-direct-ge2-quadrature", "backward-filon-holdout43")])
script_case("primary-holdout-generation", "seed43", [S + "generate_primary_holdout.py", "--output",
            out("primary-holdout-generation", "seed43")], clean=[out("primary-holdout-generation", "seed43")])

# --- backward Monte Carlo control variates of primary order 1
script_case("first-control-variate", "300GeV", [S + "probe_first_control_variate.py",
            "--output", out("first-control-variate", "300GeV")])
script_case("first-control-variate", "cached-residual-300GeV", [S + "probe_cached_first_residual.py",
            "--output", out("first-control-variate", "cached-residual-300GeV")])

# --- previews and OM selection
script_case("positive-fluence-preview", "segment-check", [S + "check_positive_segment.py"])
script_case("positive-fluence-preview", "screen-1TeV", [S + "probe_positive_map_screen.py", "--energy", 1000,
            "--atlas", bundle("tables", "muon", "preview-atlas.npz"),
            "--output", w("positive-map-20261009", "screen", "point-1000-4m")])
script_case("positive-fluence-preview", "fine-1TeV", [S + "run_positive_map_fine.py", "--energy", 1000,
            "--atlas", bundle("tables", "muon", "preview-atlas.npz"),
            "--output", out("positive-fluence-preview", "fine-1TeV")])
script_case("charge-preview", "first-collision-pack", [S + "probe_shower_first_cache.py",
            "--source", w("muon-segments-20261009", "structure", "electron-current-100GeV.npz")])
script_case("charge-preview", "preview-1TeV", [S + "probe_charge_preview.py", "--energy", 1000,
            "--output", out("charge-preview", "preview-1TeV")])
script_case("muon-charge-screen", "1TeV-l8", [S + "probe_muon_charge_screen.py", "--energy", 1000,
            "--output", out("muon-charge-screen", "1TeV-l8")])
script_case("smart-muon", "script-1TeV", [S + "run_smart_muon.py", "--energy", 1000,
            "--output", out("smart-muon", "script-1TeV")])

# --- stationary fluence maps
script_case("stationary-fluence", "mc", [S + "probe_stationary_fluence_mc.py"])
script_case("stationary-fluence", "causal-maxent", [S + "check_causal_fluence.py"])
script_case("stationary-fluence", "atlas", [S + "probe_fluence_atlas.py"])

# --- secondaries: order 1
script_case("muon-packets", "order1-1TeV", [S + "probe_muon_packets.py", "--energy", 1000,
            "--output", out("muon-packets", "order1-1TeV")])
script_case("joint-first-cdf", "pencil-cdf-1TeV", [S + "probe_joint_first.py",
            "--output", out("joint-first-cdf", "pencil-cdf-1TeV")])
script_case("joint-first-cdf", "atlas-1TeV", [S + "probe_joint_first_atlas.py",
            "--output", out("joint-first-cdf", "atlas-1TeV")])
script_case("joint-first-cdf", "route-atlas-1TeV", [S + "run_joint_muon.py", "--energy", 1000,
            "--output", out("joint-first-cdf", "route-atlas-1TeV")],
            notes="run_joint_muon.py with its own defaults: first_backend=atlas, ge2_backend=basis, "
                  "scalar float64 operator, no skips")
script_case("joint-first-cdf", "route-library-defaults-1TeV", [S + "run_joint_muon.py", "--energy", 1000,
            "--ge2-backend", "joint", "--ge2-operator", "blas", "--first-backend", "grouped",
            "--first-skip-epsilon", 1e-3, "--first-fuse-bands", "--ge2-skip-epsilon", 1e-3,
            "--ge2-batch-points", 2000, "--ge2-blas-precision", "float32",
            "--output", out("joint-first-cdf", "route-library-defaults-1TeV")],
            notes="run_joint_muon.py with the JointMuonConfig defaults (what MuonTransport runs)")
script_case("secondary-stationary-pencils", "1TeV", [S + "probe_secondary_stationary.py",
            "--output", out("secondary-stationary-pencils", "1TeV")])

# --- secondaries: order >=2
script_case("joint-source-ge2", "packets-1TeV", [S + "probe_joint_ge2.py", "--energy", 1000,
            "--output", out("joint-source-ge2", "packets-1TeV")])
script_case("point-ge2", "benchmark-1TeV", [S + "benchmark_point_ge2.py",
            "--output", out("point-ge2", "benchmark-1TeV")], clean=[out("point-ge2", "benchmark-1TeV")])
script_case("point-ge2", "angular-degree-1TeV", [S + "probe_secondary_angular_degree.py",
            "--sample", out("point-ge2", "benchmark-1TeV"), "--output", out("point-ge2", "angular-degree-1TeV")])
script_case("secondary-time-lattice", "1TeV", [S + "probe_secondary_time_lattice.py", "--energy", 1000,
            "--all-bands", "--cubic", "--steps-ns", .02, .25, .5,
            "--output", out("secondary-time-lattice", "1TeV")])
script_case("secondary-time-lattice", "three-energies", ["shower_basis/run_secondary_time_check.sh",
            out("secondary-time-lattice", "three-energies")])
script_case("secondary-spectral-validation", "1TeV", [S + "validate_secondary_spectral.py", "--energy", 1000,
            "--direct-backend", "point", "--chunk", 1024, "--carrier-cells-m", .5, .75, 1,
            "--output", out("secondary-spectral-validation", "1TeV")])
script_case("secondary-spectral-validation", "three-energies", ["shower_basis/run_secondary_spectral_check.sh",
            out("secondary-spectral-validation", "three-energies")], heavy=True)
script_case("secondary-spectral-validation", "causal-window-1TeV", [S + "check_secondary_causal_window.py",
            "--reference", out("secondary-spectral-validation", "1TeV"),
            "--output", out("secondary-spectral-validation", "causal-window-1TeV")],
            clean=[out("secondary-spectral-validation", "causal-window-1TeV")])
script_case("early-source-selection", "1TeV", [S + "probe_early_source_selection.py", "--energy", 1000,
            "--output", out("early-source-selection", "1TeV")], clean=[out("early-source-selection", "1TeV")])
script_case("secondary-spectral-validation", "direct-ge2-1TeV", [S + "probe_secondary_direct_ge2.py",
            "--reference", out("early-source-selection", "1TeV"),
            "--output", out("secondary-spectral-validation", "direct-ge2-1TeV")])
script_case("travelling-source", "1TeV", [S + "probe_travelling_source.py",
            "--reference", out("early-source-selection", "1TeV"),
            "--output", out("travelling-source", "1TeV")])

# --- earlier full-event muon routes (scripts)
script_case("muon-components", "script-1TeV", [S + "run_muon_components.py", "--energy", 1000,
            "--output", out("muon-components", "script-1TeV")])
script_case("ntsim-hybrid-muon-fast", "long-muon-300GeV", [S + "benchmark_long_muon.py", "run",
            "--source", w("muon-five-cluster-20261008", "muon300.npz"), "--frequencies", 65,
            "--output", w("muon-five-cluster-20261008", "reference300")])
script_case("ntsim-hybrid-muon-fast", "prompt-screen-300GeV", [S + "probe_prompt_screen.py",
            "--reference", w("muon-five-cluster-20261008", "reference300", "response-0.npz"),
            "--output", w("muon-five-cluster-20261008", "probe-screen300")])
