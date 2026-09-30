#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${root}/source${PYTHONPATH:+:${PYTHONPATH}}"
python_bin="${GENIESIM_PYTHON:-${root}/.venv/bin/python}"
[[ -x "${python_bin}" ]] || python_bin="$(command -v python3)"
"${root}/scripts/preflight.sh" --profile static
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${python_bin}" -m pytest -q \
  "${root}/tests/test_reproducibility_release.py" \
  "${root}/tests/test_g2_stage1a_grasp_evaluation_contract.py" \
  "${root}/tests/test_g2_stage1a_reset_open_restore.py" \
  "${root}/tests/test_g2_stage1a_close_readiness_advisory.py"
