#!/usr/bin/env bash
set -euo pipefail

# Canonical Keyboard-V3 launcher.  Isaac and its stdin live in a newly-created
# GNOME Terminal, matching the operational model of the ROS2 TurtleBot3
# keyboard examples.  The child shell deliberately remains open after Isaac
# exits so a native finalization failure cannot erase the visible diagnostics.

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${G2_ISAACLAB_PYTHON:-${GENIESIM_ISAAC_PYTHON:-}}"
collection_root=""
episode_id=""
translation_step_m="0.001"
motion_min_interval_s="0.050"
maximum_steps="1600"
seed="42"
target_residual_mm="auto"
smoothing_steps="5"
session_episodes="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --collection-root)
      collection_root="${2:?missing --collection-root value}"
      shift 2
      ;;
    --episode-id)
      episode_id="${2:?missing --episode-id value}"
      shift 2
      ;;
    --translation-step-m)
      translation_step_m="${2:?missing --translation-step-m value}"
      shift 2
      ;;
    --motion-min-interval-s)
      motion_min_interval_s="${2:?missing --motion-min-interval-s value}"
      shift 2
      ;;
    --maximum-steps)
      maximum_steps="${2:?missing --maximum-steps value}"
      shift 2
      ;;
    --seed)
      seed="${2:?missing --seed value}"
      shift 2
      ;;
    --target-residual-mm)
      target_residual_mm="${2:?missing --target-residual-mm value}"
      shift 2
      ;;
    --smoothing-steps)
      smoothing_steps="${2:?missing --smoothing-steps value}"
      shift 2
      ;;
    --episodes)
      session_episodes="${2:?missing --episodes value}"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
usage: run_g2_keyboard_v3_terminal_collection.sh \
  --collection-root PATH --episode-id ID \
  [--translation-step-m 0.001] [--motion-min-interval-s 0.050] \
  [--maximum-steps 1600] [--seed 42] [--target-residual-mm auto|16..22] \
  [--smoothing-steps 5] [--episodes 0]

The default motion contract is 1.0 mm per accepted key pulse and at most one
XYZ pulse every 50 ms (20 mm/s requested maximum).  The 4.5 mm action bound is
unchanged. Each XYZ pulse is raised-cosine shaped over 5 control steps by
default. --episodes 0 keeps one Isaac process alive and collects episodes until
X/Esc is used; a positive value bounds the session. A validated cuRobo-dataset
robot/cube state is loaded directly; live startup planning/execution is disabled.
EOF
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${collection_root}" || -z "${episode_id}" ]]; then
  echo "--collection-root and --episode-id are required" >&2
  exit 2
fi
if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
  echo "GENIESIM_ISAAC_PYTHON (or G2_ISAACLAB_PYTHON) must name an executable Isaac Python" >&2
  exit 2
fi

if [[ "${target_residual_mm}" == "auto" ]]; then
  target_residual_mm="$(${python_bin} -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]) / "result_receipts"
targets = tuple(range(16, 23))
counts = {target: 0 for target in targets}
for path in root.glob("*.result.json"):
    payload = json.loads(path.read_text(encoding="utf-8"))
    target = int(payload["target_residual_mm"])
    if target not in counts:
        raise SystemExit(f"receipt target outside 16--22 mm: {target}")
    counts[target] += 1
print(min(targets, key=lambda target: (counts[target], target)))
' "${collection_root}")"
fi
case "${target_residual_mm}" in
  16|17|18|19|20|21|22) ;;
  *) echo "--target-residual-mm must be auto or an integer 16..22" >&2; exit 2 ;;
esac
case "${smoothing_steps}" in
  1|2|3|4|5|6|7|8|9|10) ;;
  *) echo "--smoothing-steps must be an integer 1..10" >&2; exit 2 ;;
esac
if ! [[ "${session_episodes}" =~ ^[0-9]+$ ]]; then
  echo "--episodes must be 0 or a positive integer" >&2
  exit 2
fi
target_residual_m="$(${python_bin} -c 'import sys; print(f"{int(sys.argv[1])/1000.0:.3f}")' "${target_residual_mm}")"
if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is not set; cannot create Isaac/terminal GUI" >&2
  exit 2
fi
if ! command -v gnome-terminal >/dev/null 2>&1; then
  echo "gnome-terminal is required for the separate Keyboard-V3 console" >&2
  exit 2
fi
if [[ ! -f "${collection_root}/COLLECTION_MANIFEST.json" \
   || ! -d "${collection_root}/episodes" \
   || ! -d "${collection_root}/rejected_episodes" \
   || ! -d "${collection_root}/result_receipts" ]]; then
  echo "incomplete Keyboard-V3 collection root: ${collection_root}" >&2
  exit 2
fi

report_path="${collection_root}/KEYBOARD_V3_LIVE_REPORT_${episode_id}.json"
process_receipt="${collection_root}/KEYBOARD_V3_PROCESS_${episode_id}.txt"
if [[ -e "${report_path}" || -e "${process_receipt}" ]]; then
  echo "episode report/receipt already exists; refusing overwrite" >&2
  exit 2
fi

runner="${repository_root}/scripts/diagnostics/run_g2_curobo_planner_live_smoke.py"
candidate_asset="${repository_root}/artifacts/g2_bounded_passive_range_qualification_20260921/candidates_v2/A_source_min_q3_10deg_q4_11p25deg/robot_fix.usda"
candidate_hash="d1ffc70c1e7ee628348742e6f2ed824e15cbeb5e790f06400b8b28d7abe03a28"

read -r -d '' child_body <<'EOF' || true
receipt="$1"
report="$2"
repository_root="$3"
shift 3
cd "${repository_root}" || exit 125
set +e
"$@"
status=$?
temporary="${receipt}.tmp.$$"
{
  printf 'PROCESS_EXIT_CODE=%s\n' "${status}"
  printf 'FUNCTIONAL_REPORT=%s\n' "${report}"
  printf 'FUNCTIONAL_REPORT_PRESERVED=%s\n' "$(test -f "${report}" && echo YES || echo NO)"
  if [[ "${status}" -eq 139 ]]; then
    printf 'PROCESS_CLASSIFICATION=KNOWN_OR_UNRESOLVED_NATIVE_FINALIZATION_SIGSEGV\n'
  elif [[ "${status}" -eq 0 ]]; then
    printf 'PROCESS_CLASSIFICATION=CLEAN_EXIT\n'
  else
    printf 'PROCESS_CLASSIFICATION=NONZERO_EXIT\n'
  fi
} > "${temporary}"
mv "${temporary}" "${receipt}"
echo
echo "Keyboard V3 child exited with status ${status}."
echo "Process receipt: ${receipt}"
echo "This terminal is intentionally kept open for inspection. Type 'exit' to close it."
exec bash --noprofile --norc
EOF

gnome-terminal \
  --title="G2 Keyboard V3 — ${episode_id}" \
  -- \
  bash -lc "${child_body}" g2-keyboard-v3-child \
  "${process_receipt}" "${report_path}" "${repository_root}" \
  env "PYTHONPATH=${repository_root}/source${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" "${runner}" \
  --execute-live \
  --gui \
  --seed "${seed}" \
  --output "${report_path}" \
  --episode-horizon-steps "${maximum_steps}" \
  --ordinary-tracking-cap 20 \
  --critical-tracking-cap 20 \
  --final-settle-cap 20 \
  --planner-active-velocity-limit-rad-s 0.8 \
  --diagnostic-asset-variant custom \
  --diagnostic-asset-path "${candidate_asset}" \
  --diagnostic-asset-sha256 "${candidate_hash}" \
  --keyboard-v3-direct-pregrasp-v2 \
  --keyboard-v3-collection-root "${collection_root}" \
  --keyboard-v3-episode-id "${episode_id}" \
  --keyboard-v3-backoff-m "${target_residual_m}" \
  --keyboard-v3-maximum-steps "${maximum_steps}" \
  --keyboard-v3-translation-step-m "${translation_step_m}" \
  --keyboard-v3-motion-min-interval-s "${motion_min_interval_s}" \
  --keyboard-v3-terminal-smoothing-steps "${smoothing_steps}" \
  --keyboard-v3-continuous-session \
  --keyboard-v3-session-max-episodes "${session_episodes}"

echo "Spawned separate G2 Keyboard-V3 terminal for ${episode_id}."
echo "Selected direct-init nominal residual target: ${target_residual_mm} mm."
echo "Startup cuRobo planning/execution: DISABLED (validated dataset state load)."
echo "Continuous session: episodes=${session_episodes} (0 means until X/Esc)."
echo "Terminal motion smoothing: ${smoothing_steps} control steps per pulse."
echo "The new terminal owns keyboard focus; Isaac viewport focus is not required."
