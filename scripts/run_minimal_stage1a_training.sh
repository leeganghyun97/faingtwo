#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ ! -f "${root}/.env" ]] || { set -a; source "${root}/.env"; set +a; }

isaac_python="${GENIESIM_ISAAC_PYTHON:?GENIESIM_ISAAC_PYTHON must point to Isaac Python}"
output_root="${GENIESIM_OUTPUT_ROOT:-${root}/output}/minimal_stage1a_v3_current_3k_seed42"
seed=42
wandb_mode="${WANDB_MODE:-offline}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) output_root="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    --wandb-mode) wandb_mode="$2"; shift 2 ;;
    *) echo "usage: $0 [--output-root PATH] [--seed N] [--wandb-mode online|offline|disabled]" >&2; exit 2 ;;
  esac
done

case "${wandb_mode}" in online|offline|disabled) ;; *) echo "invalid --wandb-mode" >&2; exit 2 ;; esac

"${root}/scripts/preflight.sh" --profile live --method A
exec "${isaac_python}" "${root}/scripts/diagnostics/run_g2_stage1a_current_retry_supervisor.py" \
  --output-root "${output_root}" \
  --python "${isaac_python}" \
  --runner "${root}/scripts/run_g2_stage1a_vector_runtime.py" \
  --seed "${seed}" \
  --num-envs 10 \
  --accepted-transitions 3000 \
  --runtime-variant V3_BASELINE \
  --wandb-project "${WANDB_PROJECT:-geniesim-stage1a-minimal}" \
  --wandb-run-name "minimal-stage1a-v3-current-3k-seed${seed}" \
  --wandb-group portable-minimal-training \
  --wandb-mode "${wandb_mode}"
