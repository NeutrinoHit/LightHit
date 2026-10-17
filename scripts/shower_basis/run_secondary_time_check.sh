#!/usr/bin/env bash
# Saved G4 sources, five-cluster geometry; selected descendant families/OMs.
set -euo pipefail
cd "$(dirname "$0")/../.."
task_python="${LIGHTHIT_PYTHON:-python}"
task_output="${1:-${LIGHTHIT_WORK:-$PWD/experiments/results/work}/secondary-time-check-$(date +%Y%m%d-%H%M%S)}"
export NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
for task_energy in 300 1000 3000; do
  "$task_python" scripts/shower_basis/probe_secondary_time_lattice.py \
    --energy "$task_energy" --all-bands --cubic --steps-ns .02 .25 .5 \
    --output "$task_output/$task_energy"
done
