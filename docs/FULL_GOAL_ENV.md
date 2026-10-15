# Full-width rolling-ball/goal environment

This is an opt-in scene contribution from the local
`2026-10-15_K1_ball_goal_env_v1` environment copy. It does not replace
`Goalkeeper-K1-Sim`, change `run.py`, or include trained policies or detector
weights. No Booster source, K1 meshes, vendor directory or virtual environment
is redistributed. Restore the course dependency using the existing setup guide.

## Scene definition

Coordinates are environment-local: +X is the initial K1 forward direction, +Y
is left, +Z is up. The robot starts at X=Y=0 with a 0.005 m root-height reset
offset. The guard region ends an episode when the root exceeds X=+/-0.30 m or
Y=+/-1.35 m; these are software boundaries, not walls.

| Parameter | Value |
|---|---|
| Goal line | X=-0.8 m |
| Goal nominal width / bar center height | 2.4 / 1.0 m |
| Post and bar diameter | 0.05 m |
| Goal depth / net | Not modeled |
| Goal collision | Disabled; the goal cannot save a shot |
| Ball radius / mass | 0.11 m / 0.43 kg |
| Ball slide/spin/roll geom friction parameters | 0.6 / 0.005 / 0.0001 |
| Initial ball center height | 0.112 m |
| Launch X / Y | [1.8, 2.6] / [-1.4, 1.4] m |
| Goal target Y | [-1.07, 1.07] m, sampled independently of the origin |
| Initial total rolling speed | [0.35, 0.50] m/s |
| Physics / control step | 0.005 / 0.020 s |
| Maximum episode duration | 16 simulated seconds |

The velocity points from the sampled origin to the sampled point on the goal
line. Angular velocity is initialized for pure rolling. Only the reset event
writes ball state; there is no sustained steering, attraction or attachment.
The +/-1.07 m convention uses the nominal goal width, not the clear space
between post surfaces. Posts are visual-only. This preserves the saved scene's
definition rather than silently changing its shot distribution.

`configs/full_goal_scene.json` contains all exposed scene settings:

```python
from goalkeeper_sim.full_goal.settings import FullGoalSettings
settings = FullGoalSettings.from_json("configs/full_goal_scene.json")
```

## Check and view

Use the restored course environment, as with the original task:

```bash
uv run --project vendor/booster_mjlab --locked --no-dev python tests/check_full_goal.py
uv run --project vendor/booster_mjlab --locked --no-dev python tests/smoke_full_goal.py
uv run --project vendor/booster_mjlab --locked --no-dev python scripts/run_full_goal.py check
uv run --project vendor/booster_mjlab --locked --no-dev python scripts/run_full_goal.py play \
  --agent zero --num-envs 1 --device cpu --viewer viser
```

Zero actions are only a scene preview, not a trained goalkeeper. This task's
default observations still include the existing six-dimensional exact ball
state, and its default reward/command design is the existing standing baseline.
It does not reproduce the local RGB-guided AMP demonstration or claim a save
rate. `train` is an available upstream entry point, not a completed goalkeeper
reward design.

For a GPU RGB sensor preview:

```bash
uv run --project vendor/booster_mjlab --locked --no-dev python scripts/run_full_goal.py play \
  --camera --agent zero --num-envs 1 --device cuda:0 --viewer viser
uv run --project vendor/booster_mjlab --locked --no-dev python tests/smoke_full_goal.py --camera
```

`--camera` chooses the separate `Goalkeeper-K1-FullGoal-RGB` task. It adds a
128x96 RGB sensor with 100-degree vertical FOV, mounted on Head_2 at
(0.12, 0, 0.07) m and pitched down 50 degrees. It does not add a detector or
change policy observations to RGB. Camera rendering requires a supported GPU
runtime; the camera-free task supports CPU smoke checks.

## Reuse with a new K1 policy

```python
from goalkeeper_sim.full_goal.env_cfg import full_goal_env_cfg

cfg = full_goal_env_cfg(
    policy_env_cfg=my_new_k1_task_cfg,
    settings=settings,
    num_envs=128,
    enable_camera=False,
)
```

The supplied config is deep-copied. Its observations, actions, commands, rewards,
curriculum and additional sensors are retained. The scene adapter adds the ball
and goal and installs the fixed resets, launch and outcome terms. The base task
must use the K1 body names and define `feet_ground_contact` and
`non_foot_ground_contact`. Add any new sensors required by custom observations.
Choose explicitly whether ball observations are RGB estimates, privileged
training state, or something else; launch target diagnostics are not inserted
into observations. Use a new runner/checkpoint/output directory for new training.

The scene has no hard-coded 75-input pretrained actor or frozen AMP dependency.
To train ground-level two-hand collection after stopping the ball, remove or
replace `cfg.terminations["ball_stopped"]` in the new task. Otherwise success
ends the episode before collection.

## Outcome semantics and limits

- A valid touch requires a non-hand robot/ball collision while upright, with
  foot support and no fall, guard-region exit, illegal ground contact or goal.
- A stop requires prior valid touch, ball/ground contact, speed below 0.12 m/s,
  distance below 0.75 m, the whole ball in front of the line, and standing
  support continuously for 0.4 s.
- A goal requires the whole ball behind X=-0.8 and overlap with the nominal
  goal mouth in Y. This is the saved rolling-shot convention; no airborne
  crossbar-height classification is implemented.
- Goals are sticky within an episode. Failed standing support cannot receive
  stop credit. Timeout is unresolved, never automatically a defensive save.
- Before auto-reset, `env._full_goal_outcome.last` records goal/touch/stop/failure
  flags. `env._full_goal_shots` is launch metadata for diagnostics only.

Local source and asset snapshots remain local. The user's frozen demonstration
v1 has not been modified by this contribution.

## Validation

Validated on WSL2/Linux with Python 3.13, mjlab 1.6.0, MuJoCo 3.11.0,
PyTorch 2.11.0 and rsl-rl-lib 5.5.0. Both the full upstream Booster source and
the existing minimized course source passed the new two-environment CPU smoke
check. The optional RGB variant also passed on an RTX 4060 Laptop GPU.

Checks cover 10,000 sampled trajectories, independent launch/target sampling,
seed reproducibility, negative success cases, partial resets, pure-rolling
initial velocity, goal termination/auto-reset, and RGB tensor shape. The
original prediction, registry and two-environment CPU smoke checks also pass.
These are implementation checks, not goalkeeper performance evaluation.

The inherited terrain uses zero geom friction with contact mixing; MuJoCo Warp
emits a minimum-friction warning, also present in the original task. CCD shape
and camera/raycast geom-group warnings may also appear. No terrain/contact
retuning is included here; longer training and numerical stability have not
been certified by these short checks.
