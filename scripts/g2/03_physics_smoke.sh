#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_isaac_python
output="${1:-${G2_OUTPUT_ROOT}/g2_physics_startup_ladder}"
refuse_existing_output "${output}"
exec "${G2_ISAAC_PYTHON}" "${G2_REPO_ROOT}/scripts/diagnostics/run_g2_candidate_a_startup_isolation.py" \
  --output "${output}" --timeout-s 75 --full-repetitions 1 --execute-live
