# Vision-Based Goalkeeping for the Booster K1

A camera-guided goalkeeper that detects a rolling ball, predicts its arrival, and selects a safe interception action for the Booster K1 humanoid robot.

The current implementation provides the first simulation layer: a K1 on flat ground, a physical rolling ball, and an exact robot-local ball-state input.

## Goal

The system aims to stop a soft ball and keep it inside a marked capture zone for at least two seconds. Its central research question is whether arrival prediction improves interception over fixed-posture and purely reactive defenses.

```text
Camera -> Ball tracking -> Ground-plane position and velocity
       -> Arrival prediction -> Finite-state action selector -> K1
```

The planned action library includes holding position, reaching, a foot stop, and limited lateral adjustment.

## Simulation

```bash
./scripts/setup_course_vendor.sh "../course"
uv sync --project vendor/booster_mjlab --locked --no-dev
uv run --project vendor/booster_mjlab --locked --no-dev python run.py check
uv run --project vendor/booster_mjlab --locked --no-dev python tests/smoke_env.py
```

View the task with a zero-action controller:

```bash
uv run --project vendor/booster_mjlab --locked --no-dev \
  python run.py play Goalkeeper-K1-Sim \
  --agent zero --num-envs 1 --device cpu --viewer viser
```

Project code lives in `src/goalkeeper_sim`. The course dependency is restored locally under `vendor/booster_mjlab` and is not published with this repository because its upstream source has no declared software license.

## Evaluation

We compare three controllers:

- **Fixed:** holds a predefined defensive posture.
- **Reactive:** responds to the current ball position.
- **Predictive:** acts on the estimated crossing position and time.

The core target is at least **70% control success across 30 near-center trials** at approximately **0.2-0.4 m/s**. The final study evaluates save rate, control rate, arrival error, response latency, and aborted trials across three arrival regions and two speed bands.
