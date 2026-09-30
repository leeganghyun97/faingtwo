#!/usr/bin/env bash
set -euo pipefail

G2_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${G2_REPO_ROOT}/source${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -f "${G2_REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${G2_REPO_ROOT}/.env"
  set +a
fi
G2_ISAAC_PYTHON="${GENIESIM_ISAAC_PYTHON:-}"
G2_OUTPUT_ROOT="${GENIESIM_OUTPUT_ROOT:-${G2_REPO_ROOT}/output}"

require_isaac_python() {
  if [[ -z "${G2_ISAAC_PYTHON}" || ! -x "${G2_ISAAC_PYTHON}" ]]; then
    echo "GENIESIM_ISAAC_PYTHON must name an executable Isaac Python" >&2
    exit 2
  fi
}

refuse_existing_output() {
  if [[ -e "$1" ]]; then
    echo "OUTPUT_REFUSES_OVERWRITE:$1" >&2
    exit 2
  fi
}
