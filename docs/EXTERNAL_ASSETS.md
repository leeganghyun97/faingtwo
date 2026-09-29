# External assets and learned artifacts

No production USD, mesh pack, real dataset, recording, or checkpoint is added
by this reproducibility work. Paths and known hashes live in
`configs/reproducibility/external_assets.json`.

## Required inputs

| Input | Local environment variable | Expected SHA-256 / authority | Use |
|---|---|---|---|
| Genie Sim asset pack | `GENIESIM_ASSET_ROOT` | official/internal manifest | Isaac scene and robot composition |
| Candidate-A `robot_fix.usda` | `GENIESIM_CANDIDATE_A_USD` | `d1ffc70c…03a28` | Stage-1A collection and A–G live runtime |
| Human-grasp GRU checkpoint | `GENIESIM_GRU_CHECKPOINT` | `fdec68ae…d510` | hybrid collection/runtime paths |
| Conditional-CORAL frozen student | `GENIESIM_FROZEN_STUDENT_CHECKPOINT` | `1834e70d…bff2` | Methods F and G |
| Far-reach BC checkpoint | `GENIESIM_FAR_REACH_BC_CHECKPOINT` | `bcb1b7ed…14bc` | cuRobo/BC nominal reach used by live methods |
| Stage-1A residual actor | `GENIESIM_RESIDUAL_ACTOR_CHECKPOINT` | `bbc32ed9…59d` | residual SAC initialization |
| Isaac Python | `GENIESIM_ISAAC_PYTHON` | version checks in preflight | all live Isaac paths |

The complete hashes are in the JSON manifest. `scripts/preflight.sh --profile
live` verifies them. A hash mismatch is fail-closed; do not edit the manifest
to accept an unexplained file.

## Acquisition

Base Genie Sim assets must come from the official distribution linked by the
upstream README or an authorized internal mirror. Candidate-A overlays and
learned checkpoints must be copied from the authorized project archive. Their
redistribution rights are not established by this repository.

Do not add these files to Git LFS merely to bypass Git size limits. LFS is not
used here because size is not the blocker—redistribution authority is.

## Expected local layout

External files may live anywhere. Set `.env` variables to their absolute local
paths. For compatibility with legacy source-freeze code, `scripts/preflight.sh
--materialize-legacy-links` can only *report* the expected repository-relative
locations; it never copies or links files automatically.
