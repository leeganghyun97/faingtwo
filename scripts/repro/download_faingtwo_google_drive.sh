#!/usr/bin/env bash
set -euo pipefail

# Download the complete FainGTwo Google Drive tree and reconstruct the r13
# portable archive. The Google Drive folder itself is the authority; future
# A--G comparison bundles placed below it are downloaded by the same command.

readonly DRIVE_FOLDER_ID="${FAINGTWO_DRIVE_FOLDER_ID:-1Xwjii48_z4dZDo6-9kEp4O5JT1Wd2W7L}"
readonly RCLONE_REMOTE="${FAINGTWO_RCLONE_REMOTE:-gdrive}"
readonly OUTPUT_DIR="${1:-${PWD}/FainGTwo-GDrive}"
readonly RELEASE_DIR="${OUTPUT_DIR}/Stage1A Portable Release r13"
readonly ARCHIVE="faingtwo-stage1a-data-models-20260930-r13.tar.gz"

command -v rclone >/dev/null 2>&1 || {
  echo "RCLONE_REQUIRED: install rclone, then run 'rclone config' once." >&2
  exit 2
}

if ! rclone listremotes | grep -Fxq "${RCLONE_REMOTE}:"; then
  echo "RCLONE_REMOTE_MISSING:${RCLONE_REMOTE}" >&2
  echo "Run: rclone config" >&2
  echo "Create a Google Drive remote named '${RCLONE_REMOTE}', then rerun." >&2
  exit 3
fi

mkdir -p "${OUTPUT_DIR}"
rclone copy \
  "${RCLONE_REMOTE},root_folder_id=${DRIVE_FOLDER_ID}:" \
  "${OUTPUT_DIR}" \
  --progress \
  --checkers 8 \
  --transfers 4 \
  --create-empty-src-dirs

if [[ -f "${RELEASE_DIR}/SHA256SUMS_DRIVE_PARTS" ]]; then
  (
    cd "${RELEASE_DIR}"
    grep '\.part$' SHA256SUMS_DRIVE_PARTS | sha256sum -c -
    cat "${ARCHIVE}."*.part > "${ARCHIVE}"
    sha256sum -c SHA256SUMS_DRIVE_PARTS
  )
fi

echo "FAINGTWO_DRIVE_DOWNLOAD: PASS"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
