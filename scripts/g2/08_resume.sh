#!/usr/bin/env bash
set -euo pipefail
checkpoint="${1:-}"
if [[ -z "${checkpoint}" || ! -f "${checkpoint}" ]]; then
  echo "usage: $0 CHECKPOINT.pt" >&2
  exit 2
fi
sha256sum "${checkpoint}"
echo "RESUME_AUTHORITY: NOT_IMPLEMENTED_FAIL_CLOSED" >&2
echo "The canonical vector runtime does not restore optimizer+replay state; start a hash-matched fresh bounded run." >&2
exit 3
