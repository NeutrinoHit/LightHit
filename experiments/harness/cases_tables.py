"""Rebuild checks of bundle tables with tools/tables (Phase 6 of the registry)."""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np

from .cases import case
from .record import RESULTS
from .scripts import run_script

W = RESULTS / "work" / "tables"


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while block := stream.read(1 << 22):
            digest.update(block)
    return digest.hexdigest()


@case("ge2-pack-tables", "charge-slice-rebuild", kind="metrics", warmup=False,
      notes="tools/tables/build_ge2_charge_pack.py --source <bundle>/tables/ge2-pack; sha256 of the "
            "rebuilt charge-0.npy against the bundle's")
def _(ctx):
    from .inputs import bundle_path
    output = W / "ge2-pack-charge"
    if output.exists():
        shutil.rmtree(output)
    source = bundle_path("tables", "ge2-pack")
    metrics = run_script(ctx, "ge2-pack-tables", "charge-slice-rebuild",
                         ["tools/tables/build_ge2_charge_pack.py", "--source", str(source),
                          "--output", str(output)])
    rebuilt, shipped = _sha(output / "charge-0.npy"), _sha(source / "charge-0.npy")
    metrics["headline"] = dict(sha256_identical=rebuilt == shipped)
    metrics.update(rebuilt_sha256=rebuilt, bundle_sha256=shipped)
    metrics.pop("report", None)
    return metrics


@case("positive-fluence-preview", "atlas-rebuild", kind="metrics", warmup=False,
      notes="tools/tables/build_positive_fluence.py with the bundle's recorded count 6e6, seed 81 and the "
            "grid implied by its edges (96 rho bins, 384 z bins, 500 m); arrays compared with the bundle")
def _(ctx):
    from .inputs import bundle_path
    output = W / "preview-atlas-seed81.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    for stale in (output, output.with_suffix(".json")):
        stale.unlink(missing_ok=True)
    metrics = run_script(ctx, "positive-fluence-preview", "atlas-rebuild",
                         ["tools/tables/build_positive_fluence.py", "--count", 6000000, "--seed", 81,
                          "--rho-bins", 96, "--z-bins", 384, "--extent-m", 500, "--output", str(output)])
    shipped = bundle_path("tables", "muon", "preview-atlas.npz")
    compared = {}
    with np.load(output, allow_pickle=False) as a, np.load(shipped, allow_pickle=False) as b:
        for key in sorted(set(a.files) & set(b.files)):
            x, y = a[key], b[key]
            if key == "metadata" or x.dtype.kind in "US":
                continue
            same_shape = x.shape == y.shape
            compared[key] = dict(identical=bool(same_shape and np.array_equal(x, y)),
                                 max_abs_difference=(float(np.max(np.abs(x - y))) if same_shape and x.size else None))
        only = sorted(set(a.files) ^ set(b.files))
    metrics["headline"] = dict(arrays_identical=all(v["identical"] for v in compared.values()),
                               arrays_compared=len(compared))
    metrics.update(arrays=compared, keys_in_one_file_only=only,
                   rebuilt_sha256=_sha(output), bundle_sha256=_sha(shipped))
    metrics.pop("report", None)
    return metrics
