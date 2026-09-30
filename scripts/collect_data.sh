#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ ! -f "${repo_root}/.env" ]] || { set -a; source "${repo_root}/.env"; set +a; }
execute_live=0
collection_root=""
episode_id=""
extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute-live) execute_live=1; shift ;;
    --collection-root) collection_root="${2:?missing --collection-root value}"; shift 2 ;;
    --episode-id) episode_id="${2:?missing --episode-id value}"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
usage: collect_data.sh --collection-root PATH --episode-id ID [--execute-live] [Keyboard-v3 options]

Default is dry-run. Live mode opens the canonical Keyboard-v3 terminal and
requires an initialized collection root, DISPLAY, Isaac Python, and external
Candidate-A authorities. No real-robot command path is enabled by this script.
EOF
      exit 0 ;;
    *) extra+=("$1"); shift ;;
  esac
done
if [[ -z "${collection_root}" || -z "${episode_id}" ]]; then
  echo "COLLECTION_FAIL: --collection-root and --episode-id are required" >&2
  exit 2
fi
command=("${repo_root}/scripts/run_g2_keyboard_v3_terminal_collection.sh" --collection-root "${collection_root}" --episode-id "${episode_id}" "${extra[@]}")
printf 'COLLECTION_MODE=%s\n' "$([[ ${execute_live} -eq 1 ]] && echo LIVE || echo DRY_RUN)"
printf 'COMMAND='; printf '%q ' "${command[@]}"; printf '\n'
if [[ ${execute_live} -eq 0 ]]; then
  exit 0
fi
if [[ -z "${GENIESIM_ISAAC_PYTHON:-}" || ! -x "${GENIESIM_ISAAC_PYTHON}" ]]; then
  echo "COLLECTION_FAIL: configure executable GENIESIM_ISAAC_PYTHON in .env" >&2
  exit 1
fi
if ! "${repo_root}/scripts/preflight.sh" --profile live --collection; then
  echo "COLLECTION_FAIL: canonical G2 teleop/URDF/action preflight failed" >&2
  exit 1
fi
export G2_ISAACLAB_PYTHON="${GENIESIM_ISAAC_PYTHON}"
exec "${command[@]}"
