# Security and repository exclusions

## Excluded material

- `.env`, API keys, W&B credentials, SSH/private keys and local account data.
- `artifacts/`, `output/`, `outputs/`, `wandb/`, checkpoints and replay buffers.
- HDF5 datasets, rosbags, camera video, real robot data and operator records.
- `G2 도면 파일/` and its ZIP archive: private engineering drawings with
  unverified redistribution rights.
- local clones/worktrees (`RLinf/`, `openpi/`, `genie_sim/`, `workspace/`).
- the untracked full `source/geniesim/assets/` pack and Candidate-A overlays.
- caches, build products, Isaac/Kit caches and native crash artifacts.

The upstream repository already tracks three binary wheels under `3rdparty/`
and records relevant third-party licenses. This reproducibility branch does
not add or replace those binaries.

## Secret scan policy

`scripts/preflight.sh --profile static` scans tracked files for private-key
headers and common token assignments without printing secret values. It also
rejects tracked `.env` files and newly tracked runtime data. A clean result is
necessary but not sufficient for a public release.

The configured upstream `origin` is an AgiBotTech repository and must be
treated as public unless proven otherwise. Reproducibility changes may only be
pushed to a remote whose visibility has been independently confirmed PRIVATE.

## Current audit findings

- No private-key file was found outside ignored runtime/vendor trees.
- Two scene-language constants files contain empty API-key defaults, not a
  populated credential.
- Hundreds of existing absolute `/home/fain` and `/data/fain-data` references
  remain in legacy research scripts and reports. Canonical release wrappers do
  not depend on those values; the static preflight limits its absolute-path
  gate to the reproducibility entrypoints and configs. Legacy files are listed
  as migration debt rather than rewritten en masse.
- Four upstream tracked symlinks resolve only inside the optional teleop/native
  SDK layout (`source/geniesim/teleop/tasks`, two G2 model links, and
  `libpinocchio_casadi.so`). They are documented external exclusions and are
  not in the Stage-1A canonical closure. Any other broken tracked symlink fails
  preflight.
