#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"
require_isaac_python
output="${G2_OUTPUT_ROOT}/g2_v31_lateral_off_fsm_advisory_25env_15k"
while [[ $# -gt 0 ]]; do
  case "$1" in --output-root) output="$2"; shift 2 ;; *) echo "usage: $0 [--output-root PATH]" >&2; exit 2 ;; esac
done
refuse_existing_output "${output}"
"${G2_REPO_ROOT}/scripts/preflight.sh" --profile live --method F
exec "${G2_ISAAC_PYTHON}" "${G2_REPO_ROOT}/scripts/run_g2_stage1a_vector_runtime.py" \
  --execute-live --output-dir "${output}" --report "${output}/STAGE1A_VECTOR_REPORT.json" \
  --seed 42 --num-envs 25 --accepted-transitions 15000 \
  --runtime-variant V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_15K_25ENV \
  --frozen-student-advisory-checkpoint "${GENIESIM_FROZEN_STUDENT_CHECKPOINT:?missing frozen student}" \
  --frozen-student-advisory-sha256 1834e70db6fc3a56e77350248961c24d3606dc1ce9385ead462ee5349d31bff2 \
  --wandb --wandb-project "${WANDB_PROJECT:-geniesim-g2-stage1a}" \
  --wandb-run-name "$(basename "${output}")" --wandb-group portable-stage1a-15k \
  --wandb-mode "${WANDB_MODE:-online}"
