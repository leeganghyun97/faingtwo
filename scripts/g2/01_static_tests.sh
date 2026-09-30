#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${root}/scripts/g2/_common.sh"

python_bin="${GENIESIM_PYTHON:-${G2_ISAAC_PYTHON:-${root}/.venv/bin/python}}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="$(command -v python3)"
fi
if ! PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${python_bin}" -c 'import pytest' 2>/dev/null; then
  echo "PYTEST_NOT_AVAILABLE:${python_bin}" >&2
  echo "Set GENIESIM_PYTHON or GENIESIM_ISAAC_PYTHON to the prepared environment." >&2
  exit 2
fi
"${root}/scripts/preflight.sh" --profile static
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${python_bin}" -m pytest -q \
  "${root}/tests/test_reproducibility_release.py" \
  "${root}/tests/test_g2_stage1a_grasp_evaluation_contract.py" \
  "${root}/tests/test_g2_stage1a_reset_open_restore.py" \
  "${root}/tests/test_g2_stage1a_close_readiness_advisory.py"
