#!/usr/bin/env bash
# Same optical caches, bounded spectral-plane fallback and exact omega-zero pack.
set -euo pipefail
export LH_GE2_METHOD="${LH_GE2_METHOD:-filon}"
: "${LIGHTHIT_DATA:?set LIGHTHIT_DATA to the data bundle}"
export LH_FALLBACK_PACK="${LH_FALLBACK_PACK:-$LIGHTHIT_DATA/tables/ge2-pack}"
bash "$(dirname "$0")/run_primary_charge_check.sh" "$@"
