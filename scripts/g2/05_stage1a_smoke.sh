#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
method="${1:-A}"
shift || true
method_lower="$(printf '%s' "${method}" | tr '[:upper:]' '[:lower:]')"
case "${method_lower}" in a|b|c|d|e|f|g) ;; *) echo "method must be A..G" >&2; exit 2 ;; esac
exec "${root}/scripts/run_method_${method_lower}.sh" --num-envs 10 --smoke --execute-live "$@"
