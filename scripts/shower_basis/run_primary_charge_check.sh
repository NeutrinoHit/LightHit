#!/usr/bin/env bash
# Latest stationary Q1 candidate; same saved G4 sources and optical payloads.
set -euo pipefail
export LH_Q1_GAUSS_ORDER="${LH_Q1_GAUSS_ORDER:-6}"
bash "$(dirname "$0")/run_primary_time_check.sh" "$@"
