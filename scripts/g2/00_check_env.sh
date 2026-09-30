#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "${1:-}" == "--live" ]]; then
  exec "${root}/scripts/preflight.sh" --profile live --method "${2:-A}"
fi
exec "${root}/scripts/preflight.sh" --profile static
