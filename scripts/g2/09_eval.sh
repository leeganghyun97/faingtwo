#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_isaac_python
checkpoint="${1:?usage: $0 CHECKPOINT.pt [EPISODES] [OUTPUT_DIR]}"
episodes="${2:-10}"
output="${3:-${G2_OUTPUT_ROOT}/playback_$(basename "${checkpoint}" .pt)}"
refuse_existing_output "${output}"
exec "${G2_ISAAC_PYTHON}" "${G2_REPO_ROOT}/scripts/run_grasp_checkpoint_playback.py" \
  --checkpoint "${checkpoint}" --episodes "${episodes}" --seed 123 --deterministic --output-dir "${output}"
