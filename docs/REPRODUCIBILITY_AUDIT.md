# Stage-1A reproducibility audit

Audit date: 2026-09-30

This receipt covers the portable G2 Stage-1A release on branch
`stage1a-portable-training`. It certifies the repository source and the
separately hash-verified transfer bundle; Isaac Sim/Lab remain host-installed.

## Scope and result

| Area | Result | Evidence |
|---|---|---|
| Method A–G definitions | PASS (7/7) | JSON registry and clean-clone dry-run dispatch |
| Canonical dependency closure | PASS | Static preflight: 71/71 checks |
| G2 teleop/training authority | PASS | Hash-pinned URDF/SRDF/config/action/runtime files |
| Clean-clone static regression | PASS | 19 tests |
| AppLauncher smoke | PASS | 3/3 constructor probes; 0 hangs |
| Physics smoke | PASS | first step and finite observation/action/reward/termination |
| Stage-1A 10-env smoke | PASS | 100-transition preflight; reset OPEN parity 10/10 |
| External transfer bundle | PASS | 58 files; internal `SHA256SUMS` verified |
| Keyboard-v3 ROS2-free source | PASS | separate repository; 25 tests |
| Method A–G long training | NOT_RUN | Packaging validation does not claim new performance results |
| Real robot runtime | NOT_RUN | Release entrypoints do not authorize hardware |

## Security and repository contents

- Large datasets, checkpoints, raw RGB-D, W&B runs, Kit/Omniverse caches,
  local environments, ROS2 workspaces and credentials are excluded from Git.
- Static preflight found no populated token/private-key material, canonical
  user-specific absolute path, or tracked runtime dataset.
- External Candidate-A assets, BC policies, student/checkpoint artifacts and
  optional Keyboard-v3 BC datasets are delivered by the transfer archive and
  verified by SHA-256.
- Four upstream native teleop symlinks are documented external exclusions and
  are not dependencies of the canonical Stage-1A simulator path.

## Clean-clone validation

The authoritative clean clone was checked out from:

```text
https://github.com/leeganghyun97/genie_sim.git
branch: stage1a-portable-training
```

Validation completed with:

1. static preflight and source/asset authority checks;
2. 19 selected static tests;
3. Method A–G 10-env/6K command resolution;
4. AppLauncher 3/3 bounded constructor probes;
5. first physics step with finite vector receipts;
6. 10-env measured OPEN restore and reset parity;
7. no runtime hard-stop, forbidden collision, action-bound violation,
   gripper-authority violation, or premature pre-CLOSE contact in the smoke.

The 100-transition Stage-1A run is a wiring/reset/safety smoke, not a grasp
performance benchmark. A–G long training must still be run on the destination
GPU after live preflight passes.

## Verified workstation reference

- Ubuntu 22.04.5 LTS
- Python 3.12.14 in the Isaac environment
- Isaac Sim 6.0.1.0; Isaac Lab 6.1.14
- PyTorch 2.10.0+cu128; CUDA 12.8
- NVIDIA driver 580.178.04; RTX 5080 16 GB
- W&B 0.26.1

ROS2 is not required by the portable Stage-1A simulator or the separate
Keyboard-v3 collection repository.

## Known limitations

- AppLauncher may intermittently wait in startup; live wrappers use a fresh
  top-level process and bounded retries.
- Isaac can exit with SIGSEGV/139 after functional receipts are durably saved;
  this is tracked separately and never used to fabricate a PASS.
- Optimizer/replay resume is not certified. `scripts/g2/08_resume.sh` therefore
  fails closed; use fresh bounded runs with matching source/config hashes.
- Missing or mismatched external hashes remain fail-closed.
