#!/usr/bin/env bash
# Original G4 chord points versus secondary source bases; five clusters.
set -euo pipefail
cd "$(dirname "$0")/../.."
task_python="${LIGHTHIT_PYTHON:-python}"
task_output="${1:-${LIGHTHIT_WORK:-$PWD/experiments/results/work}/secondary-spectral-check-$(date +%Y%m%d-%H%M%S)}"
export NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
for task_energy in 300 1000 3000; do
  "$task_python" scripts/shower_basis/validate_secondary_spectral.py \
    --energy "$task_energy" --direct-backend point --chunk 1024 \
    --carrier-cells-m .5 .75 1 \
    --output "$task_output/$task_energy"
done
