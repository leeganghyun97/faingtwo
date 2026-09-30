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
| Pregrasp source HDF5 | `GENIESIM_PREGRASP_HDF5` | `5c18ac12…04e3` | immutable vector initial states |
| Pregrasp source report | `GENIESIM_PREGRASP_REPORT` | `36ba5925…57451d` | source-state provenance |
| Isaac Python | `GENIESIM_ISAAC_PYTHON` | version checks in preflight | all live Isaac paths |

Repository-owned G2 URDF/config and action-scale authorities are not external
assets. They are frozen by
`configs/reproducibility/g2_training_authority.json`. The external asset pack
must contain
`robot/curobo_robot/assets/robot/G2/G2_omnipicker_fixed_dual.urdf`, and live
preflight requires it to be byte-identical to the repository-owned training
URDF.

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

External files may live anywhere when the corresponding runtime path is
environment-injected. Set `.env` variables to their absolute local paths.
Legacy source-freeze receipts that require repository-relative layout are
reported explicitly by live preflight. The supported materialization path is
the local-only bundle below.

For an authorized workstation-to-workstation move, build the bundle on the
source workstation:

```bash
python3 scripts/repro/minimal_training_bundle.py export \
  --bundle /data/fain-data/geniesim_stage1a_minimal_bundle_v2 \
  --far-reach-checkpoint /path/to/fold_02/best.pt
```

After cloning on the target and installing Isaac Sim/Lab, transfer that
directory and install it into the clone. Every file is verified before copy;
`--write-env` creates only the ignored local `.env`:

```bash
python3 scripts/repro/minimal_training_bundle.py install \
  --bundle /path/to/geniesim_stage1a_minimal_bundle_v2 \
  --write-env \
  --isaac-python /path/to/isaac/python \
  --output-root /path/to/output
./scripts/install_minimal_isaac_deps.sh
./scripts/preflight.sh --profile live --method A
./scripts/run_minimal_stage1a_training.sh
```

The bundle excludes Isaac, is marked `LOCAL_AUTHORIZED_TRANSFER_ONLY`, and
must not be committed or redistributed without the asset owner's permission.

The optional native ROS teleoperation stack under `source/teleop/` is distinct
from the canonical Isaac Lab Keyboard-v3 training environment. Vendor-library
or platform symlinks needed only by native teleoperation are not valid
substitutes for the Stage-1A teleoperation files in the training-authority
manifest.
