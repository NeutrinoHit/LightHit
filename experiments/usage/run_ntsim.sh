#!/usr/bin/env bash
# NTSim's run_lighthit.py scenarios under the usage tracer, one after another.
#
#   NTSIM=/path/to/ntsim NTSIM_PYTHON="$(pyenv prefix ntsim)/bin/python" \
#   LIGHTHIT_DATA=/path/to/lighthit-bgvd-data experiments/usage/run_ntsim.sh [name ...]
#
# The lighthit package is taken from this checkout (PYTHONPATH=src).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
: "${NTSIM:?set NTSIM to the NTSim checkout}"
: "${NTSIM_PYTHON:?set NTSIM_PYTHON to the NTSim environment python}"
: "${LIGHTHIT_DATA:?set LIGHTHIT_DATA to the data bundle}"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
results="$repo/experiments/results"
runner="$NTSIM/run/run_lighthit.py"
common=(1 --cache "$LIGHTHIT_DATA/tables/rte-cache" --clusters 1 2 3 4 5)

run() {
  local name="$1"; shift
  echo "== ntsim-$name $(date +%T) load $(sysctl -n vm.loadavg 2>/dev/null || cat /proc/loadavg)"
  LIGHTHIT_USAGE_SCENARIO="ntsim-$name" "$NTSIM_PYTHON" "$here/usage_trace.py" \
    "$results/usage/ntsim-$name.json" -- "$runner" "$@" --output "$results/ntsim/$name" \
    > "$results/logs/usage-ntsim-$name.log" 2>&1
  grep -E "^(event|laser|peak_rss)" "$results/logs/usage-ntsim-$name.log" | cut -c1-400 || true
}

mkdir -p "$results/usage" "$results/ntsim" "$results/logs"
names=("${@:-electron muon nue-cc numu-cc laser}")
for name in ${names[@]}; do
  case "$name" in
    electron) run electron "${common[@]}" --pdg 11 --energy 100 \
                --position 5 -269.5 330 --direction 0.35 -0.20 0.915 ;;
    muon)     run muon "${common[@]}" --pdg 13 --energy 1000 \
                --position -100 -950 350 --direction 0 1 0 ;;
    nue-cc)   run nue-cc "${common[@]}" --position 5 -269.5 330 \
                --particle 11 70 0.35 -0.20 0.915 --particle 211 25 0.55 -0.10 0.83 \
                --particle 2212 5 0.20 -0.40 0.89 ;;
    numu-cc)  run numu-cc "${common[@]}" --position 5 -269.5 200 \
                --particle 13 800 0.30 -0.15 0.94 --particle 211 150 0.50 -0.30 0.81 \
                --particle 2212 50 0.10 0.20 0.97 ;;
    laser)    run laser 1 --cache "$LIGHTHIT_DATA/tables/rte-cache" --clusters 1 2 3 4 5 \
                --laser 1e15 532 --laser-cache "$LIGHTHIT_DATA/tables/laser-cache" \
                --position -62 -385 361 ;;
    *) echo "unknown scenario $name" >&2; exit 2 ;;
  esac
done
