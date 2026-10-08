# Simulation Migration

## Migrated

- Local `vendor/booster_mjlab`: minimized K1 model, actuators, sensors, velocity
  task, PPO runner, ONNX export, viewer, and pinned dependency lock.
- `run.py`: task-aware train, play, list, and configuration checks.
- `scripts/`: Delta and DeltaAI training entry points.
- `tests/`: pure prediction and CPU environment smoke checks.

The vendored directory is intentionally unchanged and excluded from Git.
Project-specific work lives under `src/goalkeeper_sim` and imports it as a
local dependency. Course users restore it with `scripts/setup_course_vendor.sh`.

## Not Migrated

- HW3 reward exercises and their grading tests.
- HW0/HW2 code, notebooks, logs, checkpoints, and W&B runs.
- Course handouts and cluster account configuration.
- Optional motion-policy and AMP code removed from the course snapshot.

## Licensing

The course snapshot and its upstream repository have no declared top-level
software license. See `BOOSTER_MJLAB_PROVENANCE.md` and the two vendor manifests
in this directory. The source and K1 assets are therefore kept out of this
public repository and treated as course-internal until redistribution
permission or a suitable upstream license is confirmed.
