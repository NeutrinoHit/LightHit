"""The shower-basis study (scripts/shower_basis, cluster 1) and the older
top-level scripts (scripts/*.py) as measured cases.

The shower-basis study scripts read and write their files in the current
directory, experiments/results/work/shower-basis; run_hybrid.py first writes
the references the exp*.py prototypes read (ref_e100_5.npz, cell_0.5.npz).
Top-level scripts build their own small tables for synthetic media unless a
bundle input is named.
"""
from __future__ import annotations

from .record import RESULTS, ROOT
from .scripts import script_case

W = RESULTS / "work"
SB = W / "shower-basis"


def bundle(*parts):
    from .inputs import bundle_path
    return lambda: str(bundle_path(*parts))


def out(idea, name):
    return str(W / "scripts" / idea / name)


E100 = bundle("events", "geant4-electron-100GeV.h5")

# ---------------------------------------------------------------- shower-basis study
script_case("shower-basis-study", "reference-cic-0.12m", ["shower_basis/run_hybrid.py", "e100", 5, 0.12,
            "ref_e100_5.npz"], cwd=SB, notes="transport_hybrid cloud-in-cell, KernelConfig.cell_m=0.12")
script_case("shower-basis-study", "reference-cic-0.5m", ["shower_basis/run_hybrid.py", "e100", 5, 0.5,
            "cell_0.5.npz"], cwd=SB)
for _n in range(1, 7):
    script_case("shower-basis-study", f"exp{_n}", [f"shower_basis/exp{_n}.py"], cwd=SB)
script_case("shower-basis-study", "basis-e100", ["shower_basis/run_basis.py", "e100", "0,1,2,3,4,5,6,7,8,9",
            "bench-e100/basis"], cwd=SB)
script_case("shower-basis-study", "pose", ["shower_basis/run_pose.py", "e100", 5, -60, -330, 450,
            -0.3, 0.5, 0.81], cwd=SB, notes="a second pose inside cluster 1")
script_case("shower-basis-study", "angular-compression", ["shower_basis/study_angular.py"], cwd=SB)
script_case("shower-basis-study", "frequency-interpolation", ["shower_basis/study_frequency.py", "e100", 5],
            cwd=SB, notes="needs LIGHTHIT_RTE_CACHE_321 (321-frequency tables, not in the bundle)")
script_case("shower-basis-study", "benchmark", ["shower_basis_benchmark.py",
            "--bgvd-model", bundle("bgvd-model"), "--cache", bundle("tables", "rte-cache"),
            "--g4", lambda: f"e100={bundle('events', 'geant4-electron-100GeV.h5')()}",
            "--events", "e100:0,5", "--pack", bundle("tables", "ge2-pack"),
            "--output", str(SB / "benchmark")], report=str(SB / "benchmark" / "summary.json"))
script_case("shower-basis-track-channel", "muons-cells-vs-basis", ["shower_basis/run_muons.py", "mu100", 0],
            cwd=SB)
script_case("shower-basis-track-channel", "time-muons", ["shower_basis/time_muons.py",
            bundle("tables", "ge2-pack"), "mu100:0", "mu1000:0"], cwd=SB)
script_case("shower-basis-track-channel", "ntsim-long-muon", ["shower_basis/check_long_muon_ntsim.sh", 42,
            str(SB / "long-muon-seed42")])

# ---------------------------------------------------------------- top-level scripts
for _name, _script in [("angular-system", "run_angular_system_example.py"),
                       ("fourier-adjoint", "run_fourier_adjoint_example.py"),
                       ("scattering-orders", "run_scattering_orders_example.py"),
                       ("spectral-rte", "run_spectral_rte_example.py")]:
    script_case("angular-solver-checks", _name, [_script, "--output", out("angular-solver-checks", _name)])
script_case("om-acceptance-legendre", "example", ["run_optical_module_example.py",
            "--output", out("om-acceptance-legendre", "example")])
script_case("overview-track-example", "example", ["run_overview_example.py",
            "--output", out("overview-track-example", "example")])
script_case("directional-om-kernel", "bipolar-identity", ["probe_bipolar_identity.py",
            "--output", out("directional-om-kernel", "bipolar-identity")])
script_case("directional-om-kernel", "benchmark", ["benchmark_directional.py",
            "--output", out("directional-om-kernel", "benchmark")])
script_case("directional-om-kernel", "directionality", ["verify_directionality.py",
            "--output", out("directional-om-kernel", "directionality")])
script_case("directional-om-kernel", "bgvd-smoke", ["smoke_bgvd_directional.py", "--bgvd",
            bundle("bgvd-model"), "--output", out("directional-om-kernel", "bgvd-smoke")])
script_case("cone-segment-route", "segment-event", ["run_segment_event.py",
            "--output", out("cone-segment-route", "segment-event")])
script_case("cone-segment-route", "benchmark", ["benchmark_route.py", "--output", out("cone-segment-route", "benchmark")])
script_case("cone-segment-route", "verify", ["verify_segment.py", "--output", out("cone-segment-route", "verify")])
script_case("event-moments-route", "moments", ["run_shower_moments.py", "--input", E100,
            "--output", out("event-moments-route", "moments")])
script_case("event-moments-route", "frequency", ["run_shower_frequency.py", "--input", E100,
            "--output", out("event-moments-route", "frequency")])
script_case("event-moments-route", "block-channels", ["probe_block_channels.py", "--input", E100,
            "--output", out("event-moments-route", "block-channels")])
script_case("event-moments-route", "benchmark", ["benchmark_moment_route.py", "--input", E100,
            "--output", out("event-moments-route", "benchmark")])
script_case("axial-source-route", "axial-3d", ["run_shower_axial.py", "--input", E100,
            "--output", out("axial-source-route", "axial-3d")])
script_case("axial-source-route", "axial-bins", ["run_shower_axial_bins.py", "--input", E100,
            "--output", out("axial-source-route", "axial-bins")])
script_case("axial-source-route", "fast-compiler", ["benchmark_axial_fast.py", "--input", E100,
            "--output", out("axial-source-route", "fast-compiler"), "--threads", 1])
script_case("shower-source-structure", "axis", ["probe_shower_axis.py", "--input", E100,
            "--output", out("shower-source-structure", "axis")])
script_case("shower-source-structure", "joint-rank", ["probe_shower_joint_rank.py", "--input", E100,
            "--output", out("shower-source-structure", "joint-rank") + ".json"])
script_case("effective-segments", "scan", ["run_effective_segments.py", "--input", E100,
            "--output", out("effective-segments", "scan")])
script_case("method-survey", "e100", ["run_method_survey.py", "--input", E100,
            "--output", out("method-survey", "e100")])
script_case("signal-screen-viewer", "e100", ["run_event_viewer.py", "--input", E100,
            "--output", out("signal-screen-viewer", "e100"), "--threads", 1])
script_case("prompt-vs-full-order1", "refinement", ["prompt_full_refinement.py",
            "--output", out("prompt-vs-full-order1", "refinement") + ".json"])
script_case("prompt-vs-full-order1", "bgvd-benchmark", ["prompt_benchmark.py",
            "--bgvd-model", bundle("bgvd-model"), "--g4-electron", E100,
            "--cache", lambda: __import__("os").environ["LIGHTHIT_RTE_CACHE_321"],
            "--cases", "track,shower-1000,shower-10000,electron",
            "--output", out("prompt-vs-full-order1", "bgvd-benchmark")],
            clean=[out("prompt-vs-full-order1", "bgvd-benchmark")],
            notes="the full method needs 321-frequency tables ($LIGHTHIT_RTE_CACHE_321, not in the bundle); "
                  "the script's 'muon' case read a Geant4 muon HDF5 (--g4-muon) that exists neither in "
                  "the bundle nor elsewhere, so it is left out")
script_case("point-green-backends", "numpy-vs-numba", ["benchmark_backends.py",
            "--config", str(W / "point-green.toml"), "--output", out("point-green-backends", "numpy-vs-numba")],
            notes="examples/point-green.toml was deleted in ec553be; the case restores it from git first")
script_case("banded-flash-cache", "build", ["build_cache.py"])
script_case("banded-flash-cache", "baikal-clusters", ["run_baikal_clusters.py"])
script_case("shell-mc-reference", "absorption", ["verify_absorption.py", "--output", out("shell-mc-reference", "absorption")])
script_case("shell-mc-reference", "resolution", ["check_shell_resolution.py",
            "--output", out("shell-mc-reference", "absorption")])
script_case("readout-ringing", "study", ["study_time_ringing.py", "--output", out("readout-ringing", "study")])
