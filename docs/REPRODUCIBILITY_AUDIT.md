# Stage-1A reproducibility audit

Audit date: 2026-09-29

This receipt describes the explicit reproducibility profile on branch
`chore/reproducible-a-to-g-release`. It does not certify excluded production
assets, real datasets, or robot hardware.

## Scope and result

| Area | Result | Evidence |
|---|---|---|
| Method A–G definitions | PASS (7/7) | Code-derived matrix and JSON registry; no inferred variants |
| Canonical dependency closure | PASS | 116 repository-local Python/shell files resolved from the live roots |
| G2 teleop/training authority | PASS | 24 hash-pinned URDF/SRDF/config/action/runtime files; scale and rate parity |
| Static preflight | PASS | Secrets, canonical absolute paths, runtime data, G2 authority, config and fixture checks |
| Current shell live preflight | NOT_READY | `.env` is not configured; external checkpoints remain fail-closed |
| Required static clean-export smoke | PASS | bootstrap check, preflight, sample validation, A–G dry-run, 6 tests |
| Extended clean-export regression | PARTIAL | 206 passed; 13 require excluded source assets/catalogs or noncanonical legacy entrypoints |
| Method A–G GPU/Isaac execution | NOT_RUN | Long live experiments were not started by this packaging audit |
| Real robot runtime | NOT_RUN | The release wrappers do not authorize hardware |
| Remote delivery | BLOCKED | Configured project fork is publicly visible; private-only push policy applied |

## Security and repository contents

- The staged release set contains 203 files and approximately 4.5 MB.
- No newly staged file is 10 MB or larger.
- No dataset, rosbag, checkpoint, video, W&B run, raw camera artifact, private
  drawing, `.env`, or credential is staged.
- The tracked-file scanner found no populated token assignment or private-key
  header. Two upstream constants with empty API-key defaults are not secrets.
- Four upstream teleop/native-SDK symlinks are documented external exclusions.
  The Stage-1A canonical dependency closure does not use them; every
  unexpected broken tracked symlink remains a preflight failure.
- The public upstream binary wheels are pre-existing upstream content. This
  release adds no third-party binary or model.

## Clean validation detail

The required `scripts/smoke_test.sh` succeeds using only the selected release
files and an existing Python interpreter:

1. bootstrap contract check;
2. static preflight and secret scan;
3. synthetic dataset validation;
4. Method A–G config/dry-run dispatch;
5. reproducibility release tests (`6 passed`), including exact G2 authority
   hashes and teleop/action-scale separation.

The Stage-1A training authority now distinguishes the 22.5 mm teleop
normalization divisor, 4.5 mm final metric action bound, and 0.45 mm effective
Residual-SAC authority. It also binds the latest training URDF to the cuRobo
asset-pack mirror by SHA-256. See `docs/G2_TELEOP_TRAINING_AUTHORITY.md`.

The larger selected regression run reached `206 passed`. The remaining tests
were deliberately not promoted to PASS because they require either:

- the excluded Candidate-A production asset/source catalog; or
- broader historical collection/audit files outside the canonical A–G
  runtime dependency closure.

After authorized asset migration, run live preflight before any Isaac smoke.
Missing or mismatched files remain fail-closed and must not be replaced by
relaxed hashes.

## Verified workstation reference

- Ubuntu 22.04.5 LTS; ROS 2 Humble
- Python 3.12.14 in the Isaac environment
- NVIDIA driver 580.178.04; RTX 5080 (16,303 MiB)
- PyTorch 2.10.0+cu128; torchvision 0.25.0+cu128
- Isaac Sim 6.0.1.0; Isaac Lab 6.1.14
- nvidia-curobo 0.7.7.post1.dev5; W&B 0.26.1

These versions are a validated compatibility point, not guessed minimums.

## Remote visibility decision

The configured `g2-preloss-fork` is accessible through the authenticated SSH
remote, but its GitHub repository API also returns HTTP 200 without
authentication. It is therefore PUBLIC. No commit from this release branch
may be pushed there. A new or existing confirmed-private remote is required
for upload.
