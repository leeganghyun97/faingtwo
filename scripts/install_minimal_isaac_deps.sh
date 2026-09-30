#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ ! -f "${root}/.env" ]] || { set -a; source "${root}/.env"; set +a; }
isaac_python="${GENIESIM_ISAAC_PYTHON:?GENIESIM_ISAAC_PYTHON must point to Isaac Python}"

"${isaac_python}" -m pip install -r "${root}/requirements-minimal-training.txt"
"${isaac_python}" -c 'import h5py, torch, wandb; print("MINIMAL_ISAAC_RUNTIME_DEPS: PASS")'
