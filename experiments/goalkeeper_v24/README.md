# Booster K1 goalkeeper v24

This package runs the trained K1 goalkeeper in a MuJoCo simulation. It handles balls rolling along the ground. The robot uses its head RGB camera to estimate where the ball will cross its defensive line, side-steps into the path, brakes, and tries to collect the ball with both hands.

## What was built

- **Ball perception.** A small convolutional network segments the ball in 128 × 96 RGB images. The controller projects the detection onto the ball-center plane and fits recent observations to estimate velocity, an interception point, and arrival time. Near the robot, it checks the visible outline against the known ball size and uses recent reliable observations through brief occlusions. Simulator ball state supplies training labels and evaluation measurements. The running policy uses RGB estimates, robot state, and contact feedback.
- **Side-step and braking.** The lower-level controller uses a PPO policy trained with an AMP motion prior. Reference side-step clips guide the gait, and physical rewards encourage foot placement, speed tracking, and a stable stop.
- **Receiving.** A second PPO policy makes bounded adjustments to the reference receiving motion. For v24, the lower-level gait stays fixed while the higher-level policy is fine-tuned for 200 updates in 16 parallel environments. The controller can begin opening the arms during the final braking step and adjust the transition to the receiving posture using the estimated time until the ball arrives.
- **Interactive comparison.** The browser viewer takes a launch speed, start point, and aim point. Comparison mode runs the older v20 goalkeeper with its original gait and the v24 goalkeeper in the same scene. It can repeat the previous shot with the other policy and checks that the ball and robot start in the same physical state.

## Prerequisites

Use Linux or WSL2 with an NVIDIA GPU and working CUDA and OpenGL support. The tested setup used Python 3.13, `mjlab` 1.6.0, MuJoCo 3.11.0, PyTorch 2.11.0, and Viser 1.1.0. The commands below use GLFW. Under WSL2, use a working WSLg display.

Install the team repository's course-provided `booster_mjlab` snapshot on your machine. Run these commands from the **team repository root**, with this package stored in `experiments/goalkeeper_v24/`.

```bash
bash scripts/setup_course_vendor.sh /absolute/path/to/course
uv sync --project vendor/booster_mjlab --locked --no-dev
export BOOSTER_PY="$PWD/vendor/booster_mjlab/.venv/bin/python"
cd experiments/goalkeeper_v24
```

If the course environment is already installed elsewhere, set `BOOSTER_PY` to that environment's Python executable and start in this package directory. The package resolves the K1 model and meshes from the installed `booster_mjlab` package. Course simulator source, its environment, and its robot assets stay in the local vendor installation.

## Run the trained models

Run both commands from this package directory.

```bash
# Verify the model files and run 12 real policy/physics steps per version.
MUJOCO_GL=glfw "$BOOSTER_PY" -B custom_shot_viewer.py --compare --smoke-steps 12

# Start the v20 / v24 custom-shot comparison.
MUJOCO_GL=glfw "$BOOSTER_PY" -B custom_shot_viewer.py --compare --port 8156
```

Open [the local viewer](http://127.0.0.1:8156/). Enter an initial speed in meters per second, a start `(X, Y)`, and an aim `(X, Y)`. The robot starts near `X = 0`, and the goal line is `X = -0.80 m`. The aim point sets the initial direction. The ball then moves under normal simulation physics, including contact with the robot. Select **Launch with v20** or **Launch with v24**, then select **Repeat with other version** to reuse the same shot. The panel keeps the two outcomes together. Simulation can run slower than real time.

For a starting example, use `0.50 m/s`, start `(3.50, 0.40)`, and aim `(-0.80, 0.40)`. To run v24 by itself, omit `--compare` and use the port printed by the server. Use a different port if that port is already in use.

## Re-run the evaluation

```bash
MUJOCO_GL=glfw "$BOOSTER_PY" -B evaluate_v24.py --suite matched
MUJOCO_GL=glfw "$BOOSTER_PY" -B evaluate_v24.py --suite fresh
```

Each command runs 32 seeded shots at 0.50–0.75 m/s and writes reports under `runs/`. The two seeds are `2026102723` and `2026102824`. Use `--episodes 2` for a shorter end-to-end check. It uses a smaller sample, so compare the archived counts only against the full suites.

A strict save requires the robot to stay stable and keep the ball on the ground under continuous two-hand control for at least two seconds. Ball contact is counted separately.

| Archived v24 evaluation | Interception contact | Strict saves | Physical failures | Timeouts |
| --- | ---: | ---: | ---: | ---: |
| Matched suite, 32 shots | 31 | 21 | 6 | 5 |
| Fresh suite, 32 shots | 32 | 23 | 7 | 2 |

These are results from the original simulation runs. The live v20/v24 comparison is for individual shots. A physical failure can end a trial before the ball reaches the goal, so a goal count alone cannot establish a save rate. The results cover ground balls in simulation. Re-runs may vary with GPU and renderer differences.

## Optional training reproduction

```bash
MUJOCO_GL=glfw "$BOOSTER_PY" -B train_v24.py --iterations 200 --num-envs 16 --seed 2026102624
```

This repeats the v24 high-level fine-tuning stage from the included v20 checkpoint and uses the included receiving gait. The trainer generates rolling shots in simulation and saves the run and checkpoint under `runs/`. To evaluate a newly trained checkpoint, pass its path with `evaluate_v24.py --checkpoint <path>`. The packaged vision and gait weights are the starting assets for this run.

## Repository layout

| Path | Contents |
| --- | --- |
| `custom_shot_viewer.py` | Live browser simulation and same-shot comparison |
| `evaluate_v24.py` | Reproduction of the seeded evaluation protocol |
| `train_v24.py` | Optional high-level PPO fine-tuning |
| `models/` | Frozen checkpoints, gait actor configurations, and SHA-256 checksums |
| `runtime/` | Project simulation, perception, and reference-pose modules |
| `runs/` | Local reports, traces, and training outputs, ignored by Git |

Commit the contents of this package as one directory in the team repository. The supplied checkpoints are used directly. Large recorded replays and generated runs can be shared separately from the source branch.
