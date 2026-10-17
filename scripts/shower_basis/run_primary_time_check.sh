#!/usr/bin/env bash
# Apply frozen optical atlases to saved five-cluster G4 primary sources.
set -euo pipefail
cd "$(dirname "$0")/../.."
energy="${1:-300}"
case "$energy" in
  300|1000|3000) ;;
  *) printf 'Choose energy 300, 1000 or 3000 GeV.\n' >&2; exit 2 ;;
esac
: "${LIGHTHIT_DATA:?set LIGHTHIT_DATA to the data bundle}"
work="${LIGHTHIT_WORK:-$PWD/experiments/results/work}"
output="${2:-$work/primary-time-20261010/manual-${energy}-$(date +%Y%m%d-%H%M%S)}"
python_bin="${LH_PYTHON:-python}"
export NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" scripts/shower_basis/check_universal_primary.py \
  --energy "$energy" --output "$output" \
  --q1-response-cubature "$work/joint-response-20261010/basis-rank16.npz" \
  --ge2-beta-charge "$LIGHTHIT_DATA/tables/muon/ge2-beta-charge.npz" \
  --q1-beta-charge "$LIGHTHIT_DATA/tables/muon/first-beta-charge.npz" \
  --direct-ge2-quadrature "${LH_GE2_METHOD:-adaptive}" \
  --fallback-pack "${LH_FALLBACK_PACK:-$LIGHTHIT_DATA/tables/ge2-pack}" \
  --q1-charge-gauss-order "${LH_Q1_GAUSS_ORDER:-0}"
"$python_bin" scripts/shower_basis/compare_universal_primary.py \
  --reference "$work/primary-universal-20261009/results/seed42-$energy" \
  --candidate "$output"
