#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ ! -f "${repo_root}/.env" ]] || { set -a; source "${repo_root}/.env"; set +a; }
collection_root=""
output=""
split_seed="42"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --collection-root) collection_root="${2:?missing value}"; shift 2 ;;
    --output) output="${2:?missing value}"; shift 2 ;;
    --split-seed) split_seed="${2:?missing value}"; shift 2 ;;
    -h|--help)
      cat <<'EOF'
usage: validate_dataset.sh [--collection-root PATH --output RECEIPT.json] [--split-seed 42]

Without --collection-root, validates the tracked synthetic fixture. With a
collection root, runs the canonical Keyboard-v3 training-readiness validator.
EOF
      exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
export PYTHONPATH="${repo_root}/source${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${collection_root}" ]]; then
  exec python3 "${repo_root}/scripts/repro/validate_sample_dataset.py" \
    --root "${repo_root}/tests/fixtures/reproducibility/sample_dataset"
fi
if [[ -z "${output}" ]]; then
  echo "DATASET_VALIDATION_FAIL: real dataset validation requires --output" >&2
  exit 2
fi
python_bin="${GENIESIM_ISAAC_PYTHON:-python3}"
exec "${python_bin}" "${repo_root}/scripts/validate_g2_keyboard_v3_dataset.py" \
  --collection-root "${collection_root}" --output "${output}" --split-seed "${split_seed}"
