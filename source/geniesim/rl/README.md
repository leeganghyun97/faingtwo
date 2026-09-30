# GenieSim RL — Reinforcement Learning with GenieSim Simulation

> The no-ROS direct-PhysX path is separate from the legacy
> ROS/MuJoCo/SpaceMouse stack described below. Stage 1 reproduces the audited
> fixed-payload Place environment; Stage 2 connects a 76-D state-only,
> reward-only SAC learner to that environment. See
> [`docs/STAGE1_PLACE.md`](../../../docs/STAGE1_PLACE.md) and
> [`docs/STAGE2_SAC_PLACE.md`](../../../docs/STAGE2_SAC_PLACE.md).
> The authoritative current train/eval/play status and commands are in
> [`docs/TRAIN_PLAY_COMMANDS.md`](../../../docs/TRAIN_PLAY_COMMANDS.md).
>
> Stage 2 is project Step 2 but still **Curriculum 0**: its object state is the
> fixed `gripper_r_base_link` payload proxy. It does not yet provide a free
> object, physical grasp/contact/release, cuRobo collision checking, or full
> Pick-and-Place. The gripper policy dimension is masked, leaving six effective
> Cartesian action dimensions.
>
> Stage 3 adds the source-pinned `G2_omnipicker/robot_fix.usda`, independent
> dynamic block 088, kinematic table 019, actual filtered PhysX contact,
> angular OmniPicker control, and a complete seven-action SAC lifecycle. It is
> still gated by Stage 2 actual acceptance and a measured frozen Stage 3
> profile. See [`docs/STAGE3_GRASP_LIFT.md`](../../../docs/STAGE3_GRASP_LIFT.md).
>
> Stage 1~3 share the capability-based `blackwell_sm120_16gb` runtime contract;
> it validates measured compute capability 12.0, driver, VRAM, CUDA 12.8, and
> compiled `sm_120` without trusting a 5080/5080 Ti marketing string. See
> [`docs/RTX5080_GPU_PROFILE.md`](../../../docs/RTX5080_GPU_PROFILE.md).
>
> The current planner-guided task-development runtime is right-arm-only. SAC
> controls right-arm/wrist motion plus the right OmniPicker master; torso,
> head and the left side are held and measured after actions. Perception uses
> synchronized head/right-wrist RGB-D and creates no left-wrist render product.
> Reset candidates and paths outside the measured right-arm IK, collision and
> joint-limit workspace are discarded before replay insertion.
> The Level-1 torso is fixed at q=0 and is never an action. Its links remain in
> head-camera/arm FK and self-collision geometry. The shared V12 task reset
> keeps both checked-in head and right-wrist RGB-D streams active. Visibility
> is an OR gate: at least four projected-ROI cube pixels in either camera are
> sufficient, and partial visibility is valid. Only simultaneous loss in both
> views fails pre-contact execution. A fingertip-covered center ray remains
> diagnostic instead of rejecting a visible cube, while
> post-contact occlusion uses contact/proprioception instead of privileged
> object pose in the actor.

## G2 exhibition teacher/student contract

The current Isaac Lab diagnostic path separates the privileged teacher from
the deployment student:

```text
head RGB -> RGB encoder -----+
head D+mask -> depth encoder-+
wrist RGB -> RGB encoder ----+-> fusion -> GRU -> 6-D arm + 1-D gripper
wrist D+mask -> depth encoder+              -> torso proposal/gate (disabled)
arm/hand/torso state --------+
previous action + frame age -+
```

Head/wrist and RGB/depth weights are separate by default; sharing is an
explicit ablation option.  Full-image reconstruction is not used.  Relative
object pose, bilateral contact, depth validity, and failure class are
training-only auxiliary targets.  Privileged cube/contact/phase state is not
part of the student forward input.  Sequences carry episode id, contiguous
step id, padding mask, sequence length, burn-in, hidden-state reset, camera
timestamp, and frame age.

The recurrent teacher, online SAC actor and offline student also share two
visual objectives. Symmetric cross-camera InfoNCE aligns simultaneous head
and right-wrist tokens (near-identical privileged poses are learner-side
false-negative masks only). A temporal pose-residual objective supervises
the change in object-in-EE pose between valid consecutive samples and is
masked across padding, reset and episode boundaries. Their default weights
are `0.02` and `0.05`; both raw losses are emitted to W&B separately.

Keyboard and teacher samples use one canonical 7-D policy action (6-D SE(3)
plus gripper).  The raw 8-D keyboard command is retained for audit, but its
extra elbow-nullspace channel is disabled before control by default.  This
ensures that the action stored as the expert label is also the action actually
sent through the common teacher-compatible controller.  Enabling the elbow
channel is allowed only as redundancy-control data and masks that row out of
7-D behavior cloning.

The 1.2 kg live split-encoder and reverse-start probes are short diagnostics,
not long-training approval.  See
[`docs/G2_EXHIBITION_TEACHER_STUDENT_AUDIT_20260903.md`](../../../docs/G2_EXHIBITION_TEACHER_STUDENT_AUDIT_20260903.md).

### Reproducible pre-loss v2 baseline

The strongest strict-lift historical baseline is preserved as the named
`preloss_v2_gru256x16_fresh` recipe in
`configs/model/g2_recurrent_visual_preloss_v2.json`. It uses GRU 256,
sequence/burn-in/stride 16/4/12, 2-pixel RGB-D shift and disables the
experimental cross-camera contrastive and temporal pose-residual losses. The
historical 120k checkpoint is an evaluation reference only; this entrypoint
always creates a fresh actor, critics and replay.

After cloning, obtain binary assets and verify the repository contract:

```bash
git lfs pull
python scripts/check_g2_preloss_v2_clone_readiness.py --require-tracked
```

Human demonstration HDF5 and training outputs are deliberately not committed
to Git. Point the runner at hash-audited dataset copies and the installed Isaac
Python environment:

```bash
export G2_VISUAL_TEACHER_PYTHON=/absolute/path/to/isaac/python
export G2_VISUAL_TEACHER_OUTPUT_ROOT=/absolute/path/to/output
export G2_DEMONSTRATION_DATASETS=/absolute/a.hdf5:/absolute/b.hdf5

bash scripts/run_g2_recurrent_visual_teacher_preloss_v2.sh \
  --wandb --wandb-mode online \
  --wandb-project geniesim-g2-recurrent-visual-teacher \
  --wandb-run-name g2-preloss-v2-fresh
```

The exact asset hashes, required source roots and external-data boundary are
sealed in `configs/repro/g2_recurrent_visual_preloss_v2_clone_manifest.json`.

Direct-PhysX Stage 2 commands:

```bash
cd /home/fain/genie_sim
bash scripts/run_g2_place_stage2.sh test
bash scripts/run_g2_place_stage2.sh preflight
bash scripts/run_g2_place_stage2.sh smoke --device cpu
bash scripts/run_g2_place_stage2.sh train --total-transitions 100000 \
  --seed 42 --eval-seed 1000042 --device cpu
bash scripts/run_g2_place_stage2.sh eval --checkpoint CHECKPOINT.pt \
  --episodes 100 --eval-seed 1000042 --device cpu
bash scripts/run_g2_place_stage2.sh play --checkpoint CHECKPOINT.pt \
  --episodes 1 --eval-seed 1000042 --device cpu
```

The shared Blackwell profile reports the Stage 2 8 GiB free-VRAM reference and
never stops another GPU process. Stage 2 free-memory thresholds are advisory;
hardware identity, driver/CUDA compatibility, and actual allocation failures
remain strict.
A Stage 2 smoke checks collection, replay, gradient updates, logging, and
checkpoint plumbing; it is not evidence that the policy converged.
The runner does not yet aggregate multi-seed train/eval artifacts into
`output/stage2/acceptance_gate.json`, so a successful Stage 2 run cannot by
itself authorize Stage 3. Do not manufacture that gate by hand.

Direct-PhysX Stage 3 commands:

```bash
cd /home/fain/genie_sim
bash scripts/run_g2_grasp_lift_stage3.sh test
bash scripts/run_g2_grasp_lift_stage3.sh preflight

# Evidence-only actual-PhysX source-pose/contact diagnostic. The capacity is
# deliberately explicit; this command does not create a frozen profile.
bash scripts/run_g2_grasp_lift_stage3.sh calibrate \
  --stage2-gate output/stage2/acceptance_gate.json \
  --max-contact-count MEASURED_CAPACITY

bash scripts/run_g2_grasp_lift_stage3.sh smoke \
  --stage2-gate output/stage2/acceptance_gate.json \
  --reward-profile output/stage3/calibration/reward_profile.json
bash scripts/run_g2_grasp_lift_stage3.sh train \
  --stage2-gate output/stage2/acceptance_gate.json \
  --reward-profile output/stage3/calibration/reward_profile.json \
  --total-transitions 100000 --seed 42 --eval-seed 1000042
bash scripts/run_g2_grasp_lift_stage3.sh eval \
  --stage2-gate output/stage2/acceptance_gate.json \
  --reward-profile output/stage3/calibration/reward_profile.json \
  --checkpoint CHECKPOINT.pt --episodes 100 --eval-seed 1000042
bash scripts/run_g2_grasp_lift_stage3.sh play \
  --stage2-gate output/stage2/acceptance_gate.json \
  --reward-profile output/stage3/calibration/reward_profile.json \
  --checkpoint CHECKPOINT.pt --episodes 1 --eval-seed 1000042
```

These Stage 3 simulator commands are currently syntax templates, not an open
execution path: no validated Stage 2 acceptance-gate producer or Stage 3 frozen
reward-profile producer exists. `calibrate` writes diagnostic evidence only.
Both producers and their actual measured artifacts are required before using
the learning/evaluation/play commands.

Stage 4, Stage 5, and Stage 9 currently expose strict **host-only** contract,
reward-oracle, and replay checks:

```bash
cd /home/fain/genie_sim
bash scripts/run_g2_reach_stage4.sh test
bash scripts/run_g2_pick_place_stage5.sh test
bash scripts/run_g2_her_stage9.sh test
bash scripts/run_g2_stage4_5_9_supervisor.sh test

# Stops at the first missing or rejected actual acceptance artifact.
bash scripts/run_g2_stage4_5_9_supervisor.sh check
```

The sequential checker enforces Stage 4 host check -> actual Stage 4 PASS gate
-> Stage 5 host check -> actual Stage 5/6/7/8 PASS chain -> Stage 9 host check.
It never treats a host-check exit code of zero as acceptance. There are no
Stage 4/5/9 `smoke`, `train`, `eval`, or `play` commands yet: the corresponding
direct-PhysX/SAC integrations, empirical profiles, predecessor gates, and
validated Stage 4--9 GPU memory budgets do not exist. See
[`docs/STAGE4_REACH.md`](../../../docs/STAGE4_REACH.md),
[`docs/STAGE5_PICK_PLACE.md`](../../../docs/STAGE5_PICK_PLACE.md),
[`docs/STAGE9_HER.md`](../../../docs/STAGE9_HER.md), and
[`docs/STAGE4_5_9_SUPERVISOR.md`](../../../docs/STAGE4_5_9_SUPERVISOR.md).

Stage 9 currently validates an exact 85-D dense replay/oracle boundary, but it
does not permit HER relabeling. The accepted desired goal would be a
collision-bearing physical target/bin pose while the achieved goal is the
object pose; changing the former cannot reuse recorded contact/collision
dynamics. A fully valid Stage 5--8 chain therefore exits with
`HER_BLOCKED_NONRELABELABLE_PHYSICAL_TARGET`, not Stage 9 acceptance. Opening
HER requires a new virtual-goal/physical-context schema and Stage 5--8
re-acceptance.

The legacy RLinf/MuJoCo/ROS/SpaceMouse material below documents the audited
upstream integration only. It is not imported or executed by the Stage 1–3
reward-only direct-PhysX path.

This module connects [GenieSim](https://github.com/AgibotTech/genie_sim) with
[RLinf](https://github.com/RLinf/RLinf) for robot reinforcement learning,
featuring **Isaac Sim + MuJoCo dual-simulator** architecture and
**SpaceMouse human-in-the-loop** training.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  RLinf (training framework)                                          │
│    Task Env (e.g. PlaceWorkpieceEnv)                                 │
│      ├── _extract_states()   52-dim SHM state → 26-dim model state   │
│      ├── _expand_actions()    7-dim model action → 14-dim SHM action │
│      └── _compute_reward()   dense reward from body poses            │
│    GenieSimBaseEnv  →  GenieSimShmClient                             │
└──────────────────────────┬───────────────────────────────────────────┘
                           │  Shared Memory (SHM)
                           │  ├── Frame SHM  (camera images)
                           │  ├── Ctrl SHM   (per-env state/action/info)
                           │  └── Step SHM   (request-reply sync)
┌──────────────────────────┴───────────────────────────────────────────┐
│  sim_server.py  (GenieSim side)                                      │
│    GenieSimVectorEnv  ← manages MuJoCo lifecycle + signal-based sync │
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐       ┌──────────────────────┐  │
│  │ MuJoCo env_0 │   │ MuJoCo env_1 │  ...  │ Isaac Sim renderer   │  │
│  │ 1000 Hz      │   │ 1000 Hz      │       │ 30 Hz (GridCloner)   │  │
│  └──────────────┘   └──────────────┘       └──────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Design Choices

- **MuJoCo** handles physics at 1000 Hz per environment — one process per env, isolated by ROS 2 namespace
- **Isaac Sim** provides photo-realistic rendering via `GridCloner`
- **Shared memory (SHM)** is the only data channel between RLinf and sim (zero-copy for camera images)
- **EE control mode** — IK (damped Jacobian) runs inside MuJoCo; actions are delta EE pose targets
- **Task Env pattern** — SHM transports full-dimensional state/action; each task maps to/from a smaller model space via `_extract_states()` / `_expand_actions()`

### Reward Design

The `place_workpiece` task uses three reward components:

| Component | Description |
|-----------|-------------|
| `r_alive` | Exponentially decaying reward based on 3D distance and orientation error to target |
| `r_below` | Penalty when workpiece drops below target height |
| `r_success` | Sparse one-time reward when workpiece is placed at target and held still |

---

## Quick Start

### Prerequisites

- NVIDIA GPU (RTX 3090+, VRAM ≥ 24GB)
- Docker with NVIDIA Container Toolkit
- 3Dconnexion SpaceMouse (for data collection)

### 1. Clone Repositories

```bash
mkdir workspace && cd workspace
git clone https://github.com/AgibotTech/genie_sim.git
git clone -b dev/geniesim https://github.com/RLinf/RLinf.git
```

For GenieSim installation and asset downloads, refer to the
[GenieSim documentation](https://agibot-world.com/sim-evaluation/docs/#/v3).

### 2. Build Docker Images

**Base image** (GenieSim + Isaac Sim + MuJoCo + ROS 2):

```bash
bash genie_sim/scripts/build_geniesim_rlinf_image.sh
```

**Training image** (RLinf + PyTorch + training dependencies):

```bash
cd RLinf
docker build \
  --build-arg BUILD_TARGET=embodied-geniesim \
  -t geniesim-rlinf-train:latest \
  .
```

### 3. Download Pretrained Weights

```bash
cd RLinf/examples/embodiment/config
# For mainland China: export HF_ENDPOINT=https://hf-mirror.com
hf download RLinf/RLinf-ResNet10-pretrained --local-dir .
```

### 4. Collect Demonstrations

Connect the SpaceMouse via USB, then:

```bash
cd workspace
bash RLinf/rlinf/envs/geniesim/scripts/run.sh collect --num-demos 50
```

SpaceMouse controls:

| Action | Effect |
|--------|--------|
| Translate device | Move right arm end-effector (x/y/z) |
| Rotate device | Rotate right arm end-effector (roll/pitch/yaw) |
| Press left button | Save demo → environment resets |
| Press right button | Discard demo → environment resets |

Demos are saved to `genie_sim/sac_demo/`.

### 5. Convert Demonstrations

```bash
bash RLinf/rlinf/envs/geniesim/scripts/run.sh convert
```

### 6. Start Training

```bash
bash RLinf/rlinf/envs/geniesim/scripts/run.sh train
```

During training, env_0 accepts real-time SpaceMouse intervention while remaining
environments are driven by the policy.

Override Hydra parameters:

```bash
# Adjust discount factor
bash RLinf/rlinf/envs/geniesim/scripts/run.sh train algorithm.gamma=0.97

# Adjust BC regularization
bash RLinf/rlinf/envs/geniesim/scripts/run.sh train algorithm.bc_coef=5.0
```

### 7. Monitor Training

```bash
tensorboard --logdir workspace/results/
```

Key metrics: `critic_loss`, `q_values`, `eval/success_rate`, `entropy`, `bc_loss`.

### 8. Debug Shell

```bash
bash RLinf/rlinf/envs/geniesim/scripts/run.sh shell
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `run.sh collect --num-demos N` | Collect N demonstrations |
| `run.sh convert` | Convert demos to replay buffer |
| `run.sh train` | Start SAC + SpaceMouse HIL training |
| `run.sh shell` | Interactive container shell |
| `run.sh help` | Show all commands |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `PermissionError` on `/dev/shm/geniesim_*` | Run `bash RLinf/rlinf/envs/geniesim/scripts/cleanup_stale.sh` |
| Stale `.geniesim_idle` causing hang | Same cleanup script above |
| Isaac Sim startup timeout | Increase `startup_timeout_sec` in env YAML |
| GPU out of memory | Reduce `env.train.total_num_envs` via Hydra override |

## License

Mozilla Public License Version 2.0 — see `LICENSE` in the repository root.
## G2 head-depth ablation profiles

`scripts/run_g2_head_depth_ablation.sh` keeps the established wrist RGB-D,
45-D deployable proprioception, cross-camera attention, GRU-256 and SAC
contracts fixed while changing only the head-depth path:

- `phase1-hrnet-fpn`: high-resolution/FPN fusion followed by spatial softmax.
- `phase2-coordinate-depth`: `[D_normalized, M_valid, X_pixel, Y_pixel]`,
  GroupNorm + SiLU, no MaxPool/GAP, and at most 4x downsampling.
- `phase3-unet-aux`: phase 2 plus a learner-only lightweight U-Net decoder and
  validity-masked normalized-depth L1 loss. The decoder is not called during
  rollout or deployment.

Set `G2_DEMONSTRATION_DATASETS` to colon-separated HDF5 paths and run the
profiles in the order above. Each invocation creates a fresh output directory;
checkpoint and replay restoration are rejected.
