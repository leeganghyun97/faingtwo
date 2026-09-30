# Stage-1A Method A–G matrix

This matrix is derived from `METHODS_6K` and `METHODS` in
`scripts/run_g2_stage1a_reset_fixed_7p5k_comparison.py` and the runtime variant
dispatch in `source/geniesim/rl/sac/stage1a_isaac_vector_smoke.py`. It does not
rename or infer methods from experiment nicknames.

All seven methods share the measured-state OPEN restore gate, fresh geometry
receipt, HER_FORCE replay, residual XYZ action contract, and episode-level
evaluation. `GRASP_SUCCESS` means Stable bilateral contact for 10 control
steps; Contact and Bilateral are not success.

All seven also share the exact teleoperation/robot/action authority in
`configs/reproducibility/g2_training_authority.json`: the current G2 training
URDF/config, robot-root metric action frame, 4.5 mm final XYZ bound, and 0.45 mm
effective Residual-SAC authority. The 22.5 mm teleop value is a normalization
divisor and is not a Method A–G per-step action bound.

| Method | Purpose | Input | Output | Data source | Entry point | Main config | Dependency modules | Validation | Current status |
|---|---|---|---|---|---|---|---|---|---|
| A — V3 CURRENT | Current Reward-V3/HER_FORCE control baseline | deployable RGB-D/robot state and residual state | SAC checkpoints, replay receipt, episode metrics | current live rollout only | `scripts/run_method_a.sh` | `V3_CURRENT_HER_FORCE_RESET_FIXED_6K_25ENV` | vector smoke, Reward V3, real SAC coordinator, reset gate | dry-run; source freeze; OPEN restore; physics smoke; 6K report | IMPLEMENTED; live fair-run result pending |
| B — V3.1 | Historical V3.1 with lateral hard CLOSE admission | same deployable observation | same runtime artifacts | current live rollout only | `scripts/run_method_b.sh` | `V3_1_HER_FORCE_RESET_FIXED_6K_25ENV` | A plus V3.1 CLOSE persistence | same checks as A | IMPLEMENTED; known over-conservative lateral gate |
| C — V3.1 lateral-off | Remove lateral from hard CLOSE gate while retaining telemetry | same deployable observation | same runtime artifacts | current live rollout only | `scripts/run_method_c.sh` | `V3_1_LATERAL_OFF_HER_FORCE_RESET_FIXED_6K_25ENV` | A plus lateral-off FSM and post-CLOSE micro correction | same checks as A | IMPLEMENTED; strongest historical candidate, old results reset-confounded |
| D — V3.2 | Multi-change phase/reward ablation | same deployable observation | same runtime artifacts | current live rollout only | `scripts/run_method_d.sh` | `V3_2_HER_FORCE_RESET_FIXED_6K_25ENV` | V3.2 reward config and phase telemetry | same checks as A | IMPLEMENTED; no equal-budget superiority established |
| E — Privileged geometry hard-gate | Diagnostic comparison using exact geometry as CLOSE authority | deployable observation plus privileged geometry at runtime | diagnostic/runtime comparison report | current live rollout only | `scripts/run_method_e.sh` | `V3_PRIVILEGED_GEOMETRY_HARD_GATE_HER_FORCE_RESET_FIXED_6K_25ENV` | privileged geometry oracle and close gate | same checks plus privileged receipt | IMPLEMENTED, DIAGNOSTIC ONLY; non-deployable authority |
| F — FSM advisory | C authority plus frozen Conditional-CORAL student logging; student cannot veto FSM | deployable Wrist RGB-D/features; external frozen student checkpoint | SAC artifacts, advisory metrics, sequence receipts | current live rollout only; frozen checkpoint is sidecar | `scripts/run_method_f.sh` | `V3_1_LATERAL_OFF_FSM_ADVISORY_HER_FORCE_RESET_FIXED_6K_25ENV` | C plus frozen spatial student and FSM sequence writer | same checks plus checkpoint hash and advisory receipt | IMPLEMENTED; advisory-only contract |
| G — CURRENT GRU + Privileged | Causal GRU auxiliary learns binary/signed-margin targets without changing action/CLOSE ownership | 128-D actor feature plus frozen deployable score; privileged targets only in auxiliary loss | SAC artifacts, GRU auxiliary checkpoint/metrics | current live rollout for SAC; exact teacher target is not SAC replay input | `scripts/run_method_g.sh` | `V3_CURRENT_GRU_PRIVILEGED_HER_FORCE_RESET_FIXED_6K_25ENV` | `stage1a_current_gru_privileged.py`, frozen student, vector runtime | update-free smoke first, then reset/source/safety checks | IMPLEMENTED; full fair 6K result pending |

## Shared authority boundaries

- SAC replay contains current rollout `(s, a, r, s')` only.
- Offline/boundary/heldout teacher data is never inserted into SAC replay.
- Privileged geometry is runtime authority only for Method E. In F it is not
  an input. In G it is a supervision target only.
- Methods F/G require the hash-pinned frozen student checkpoint listed in
  `configs/reproducibility/external_assets.json`.
- The canonical runners presently define fair 25-env budgets of 6K and 7.5K.
  Other env/budget combinations are not silently remapped by the wrappers.

## Common evaluation schema

- `ANY_CONTACT_RATE`: at least one primary pad contacts the cube.
- `BILATERAL_CONTACT_CANDIDATE_RATE`: both primary pads contact concurrently.
- `GRASP_SUCCESS_STABLE_RATE`: bilateral contact persists for 10 control steps.
- `CONTACT_TO_BILATERAL`, `BILATERAL_TO_STABLE`, and
  `POST_BILATERAL_CONTACT_LOSS_RATE` remain separate.
