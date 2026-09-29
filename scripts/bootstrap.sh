#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="install"
if [[ "${1:-}" == "--check-only" ]]; then
  mode="check"
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--check-only]" >&2
  exit 2
fi

if [[ ! -f "${repo_root}/requirements-lock.txt" ]]; then
  echo "BOOTSTRAP_FAIL: requirements-lock.txt missing" >&2
  exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "BOOTSTRAP_FAIL: python3 is required" >&2
  exit 1
fi

if [[ "${mode}" == "check" ]]; then
  python3 -c 'import sys; assert sys.version_info >= (3, 10)'
  echo "BOOTSTRAP_CHECK: PASS"
  echo "NOTE: Isaac Sim, Isaac Lab, ROS 2 and GPU drivers are external installs."
  exit 0
fi

if [[ ! -d "${repo_root}/.venv" ]]; then
  python3 -m venv "${repo_root}/.venv"
fi
"${repo_root}/.venv/bin/python" -m pip install --upgrade pip
"${repo_root}/.venv/bin/python" -m pip install -r "${repo_root}/requirements-lock.txt"
echo "BOOTSTRAP: portable analysis environment ready at ${repo_root}/.venv"
echo "NEXT: copy .env.example to .env, configure external inputs, then run scripts/preflight.sh"
