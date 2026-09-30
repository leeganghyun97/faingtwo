#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${root}/scripts/run_method_a.sh" --num-envs 10 --smoke --execute-live "$@"
