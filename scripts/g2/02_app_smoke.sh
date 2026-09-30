#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_isaac_python
output="${1:-${G2_OUTPUT_ROOT}/g2_app_smoke}"
refuse_existing_output "${output}"
exec "${G2_ISAAC_PYTHON}" "${G2_REPO_ROOT}/scripts/diagnostics/run_g2_applauncher_constructor_probe.py" \
  --output "${output}" --runs 3 --timeout-s 45 --execute-live
