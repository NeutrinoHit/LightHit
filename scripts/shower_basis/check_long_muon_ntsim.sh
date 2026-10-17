#!/usr/bin/env bash
# Reproduce the local-window NTSim muon check against the unsplit shower basis.
set -euo pipefail

seed=${1:-42}
repository=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${NTSIM_PYTHON:-python}
: "${LIGHTHIT_DATA:?set LIGHTHIT_DATA to the data bundle}"
cache_dir=${NTSIM_LH_CACHE:-"$LIGHTHIT_DATA/tables/rte-cache"}
pack_dir=${NTSIM_LH_PACK:-"$LIGHTHIT_DATA/tables/ge2-pack"}
work=${LIGHTHIT_WORK:-"$repository/experiments/results/work"}
output_dir=${2:-"$work/ntsim-check/long-muon-validation/seed-$seed"}

if [[ ! -d "$cache_dir" ]]; then
    echo "Missing prepared RTE cache: $cache_dir" >&2
    exit 2
fi
export PYTHONPATH="$repository/src${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd "$repository"
"$python_bin" scripts/shower_basis/verify_ntsim.py --method basis --pdg 13 \
    --energy 100 --seed "$seed" --cache "$cache_dir" --pack "$pack_dir" \
    --output "$output_dir/full" --repeat 1
"$python_bin" scripts/shower_basis/verify_ntsim.py --method basis --pdg 13 \
    --energy 100 --seed "$seed" --track-uid 0 --cache "$cache_dir" \
    --pack "$pack_dir" --output "$output_dir/track" --repeat 1
"$python_bin" scripts/shower_basis/verify_ntsim.py --method compare \
    --reference "$output_dir/full/basis-0.npz" \
    --candidate "$output_dir/track/basis-0.npz" \
    --output "$output_dir/comparison.json"

"$python_bin" - "$output_dir/comparison.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["active_lost"] == report["active_gained"] == 0, report
assert report["prompt_charge_bitwise"] and report["prompt_bins_bitwise"], report
assert report["ge2"]["charge_max_rel_bright"] < 0.01, report
assert report["ge2"]["bins_l1_max_bright"] < 0.03, report
assert report["total"]["bins_l1_max_bright"] < 0.005, report
print("Long-muon comparison passed:", sys.argv[1])
PY
