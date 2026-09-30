#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${root}/scripts/run_minimal_stage1a_training.sh" "$@"
