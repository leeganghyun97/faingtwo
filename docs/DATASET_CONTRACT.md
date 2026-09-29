# Dataset contract

## Canonical Keyboard-v3 episodes

- one HDF5 file per episode under `<collection-root>/episodes/`;
- rejected episodes and immutable result receipts are separate;
- translations use SI metres in `robot_root`;
- quaternion order is XYZW;
- physics/control/RGB-D rates are 500/50/25 Hz;
- timestamps, episode ID, env ID and reset generation are durable;
- post-CLOSE outcome fields never become pre-CLOSE student inputs;
- episode-level train/validation/heldout splits are disjoint.

Stage-1A sequence rows additionally preserve deployable RGB-D/features, robot
state, previous action, residual action, FSM state, teacher-only signed margin,
and contact/bilateral/stable outcomes. Teacher geometry is supervision only
unless a method is explicitly the diagnostic privileged hard-gate (Method E).

## Evaluation authority

`GRASP_SUCCESS = STABLE_GRASP` only: bilateral primary-pad contact must persist
for 10 control steps at 50 Hz (about 200 ms). Contact and instantaneous
bilateral contact are reported separately.

## Integrity

Generate a portable manifest with:

```bash
find "$GENIESIM_DATA_ROOT" -type f -print0 | sort -z \
  | xargs -0 sha256sum > DATASET_SHA256SUMS.txt
```

Do not commit the files or their absolute paths. A shareable manifest should
use paths relative to the dataset root and be reviewed for sensitive names.

The synthetic fixture under `tests/fixtures/reproducibility/sample_dataset`
contains no camera or robot data and validates only portable schema handling.
