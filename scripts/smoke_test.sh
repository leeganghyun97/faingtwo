#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
smoke_python="${GENIESIM_PYTHON:-${repo_root}/.venv/bin/python}"
if [[ ! -x "${smoke_python}" ]]; then
  smoke_python="$(command -v python3)"
fi
isaac_method=""
if [[ "${1:-}" == "--isaac" ]]; then
  isaac_method="${2:?usage: smoke_test.sh --isaac A..G}"
  shift 2
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--isaac A|B|C|D|E|F|G]" >&2
  exit 2
fi
if [[ $# -gt 0 ]]; then
  echo "unexpected arguments: $*" >&2
  exit 2
fi

"${repo_root}/scripts/bootstrap.sh" --check-only
"${repo_root}/scripts/preflight.sh" --profile static
"${repo_root}/scripts/validate_dataset.sh"
for method in a b c d e f g; do
  "${repo_root}/scripts/run_method_${method}.sh" --accepted-transitions 6000 >/dev/null
done
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "${smoke_python}" -m pytest -q \
  "${repo_root}/tests/test_reproducibility_release.py"
echo "STATIC_SMOKE_TEST: PASS"

if [[ -n "${isaac_method}" ]]; then
  normalized="$(printf '%s' "${isaac_method}" | tr '[:upper:]' '[:lower:]')"
  case "${normalized}" in a|b|c|d|e|f|g) ;; *) echo "method must be A..G" >&2; exit 2 ;; esac
  "${repo_root}/scripts/preflight.sh" --profile live --method "$(printf '%s' "${normalized}" | tr '[:lower:]' '[:upper:]')"
  exec "${repo_root}/scripts/run_method_${normalized}.sh" --smoke --execute-live
fi
