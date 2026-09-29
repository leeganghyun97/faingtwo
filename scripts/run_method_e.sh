#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ ! -f "${root}/.env" ]] || { set -a; source "${root}/.env"; set +a; }
exec python3 "${root}/scripts/repro/run_method.py" --method E "$@"
