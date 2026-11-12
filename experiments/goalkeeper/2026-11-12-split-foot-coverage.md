# Separate aerial and rolling coverage for lateral guidance

## Scope and rollback

This experiment replaces the production `body_intercept` velocity-tracking
reward with signed root progress and separates aerial/rolling coverage gates.
The prior velocity-tracking stage was saved as commit `302e415` before edits.
That commit preserves its code and tests, including movement_weight=2.0.
Restore it in a separate checkout/worktree for a historical experiment rather
than overwriting unrelated newer work. No long training was run here.

The existing actor/critic/AMP inputs (92/95/22), 22 actions, task/AMP mixing,
goal cost -2, fall cost -5, physics, shot sampling, other rewards and outcome
classification are unchanged. Geometry coverage is NOT a successful save.
Existing checkpoints remain structurally loadable, but returns have changed.

## Aerial guidance

Use distance from the predicted interception target to the nearest hand site.
Feet and trunk cannot turn off aerial body guidance. Retain the 30cm root
lateral corridor, 30cm fade and 5cm hand-point stopping radius:

```
error = target_y - root_y
body_gate = clip((abs(error) - .30) / .30, 0, 1)
hand_gate = clip((nearest_hand_distance - .05) / .30, 0, 1)
raw_air = 2 * body_gate * hand_gate * clip(sign(error) * root_vy, -1, 1)
```

## Rolling coverage

Read the enabled `left_foot_collision` and `right_foot_collision` geoms from
the compiled simulation. These are meshes, not authored boxes. Cache their
compiled `geom_aabb` centers/half sizes in the geom-local frame and transform
the ball path using each geom's current world position/quaternion. In particular,
do not confuse the foot body frame with MuJoCo's recentered mesh frame.

Observed left bounds: center (0.003909, -0.000823, 0.008907)m and half sizes
(0.032909, 0.040107, 0.093008)m; right is approximately mirrored. The geom has
a nonidentity rotation. These dimensions are NOT world-aligned foot dimensions.

For each foot, extrapolate the current rolling ball velocity to that foot's x
position. Prediction is bounded to 2.5s. Inspect only a +/-0.20m segment along
the trajectory near that crossing, clipped to the future interval. Reject
outgoing, entirely passed and beyond-horizon paths for coverage. Ball z follows
its current linear velocity; no free-fall extrapolation is used for rolling.
The horizon supports slow distant rolling shots; the inspected segment remains
short. This is a constant-velocity prediction, not a friction/contact simulator.

Transform both segment endpoints into each geom frame. Compute minimum distance
to the box by vectorized convex 1D minimization (20 ternary iterations plus both
endpoints). A box distance for point p is:

```
distance(p) = norm(max(abs(p - box_center) - half_size, 0))
surface_gap = min_over_feet_and_segment(distance) - actual_configured_ball_radius
```

Each foot is tested separately; a gap between feet is not a filled support box.
Yaw, pitch and raised feet change coverage. A foot in front of the fixed
interception plane can cover the future ball path. No extra overlap margin is
added: a nonpositive surface gap turns off guidance; positive gaps fade over
0.10m. This fade is an initial tuning parameter, not a validated tolerance.

```
direction = clip((target_y - root_y) / .05, -1, 1)
foot_gate = clip(surface_gap / .10, 0, 1)
raw_rolling = 2 * foot_gate * clip(direction * root_vy, -1, 1)
```

The 30cm root corridor does not apply to rolling shots. The 5cm direction fade
smooths the sign through zero; it is not a dead zone. At exactly zero lateral
root error, the term does not choose a foot or arbitrarily select a stepping
direction. Foot positioning at that point remains a limitation of this root
guidance and is not solved by declaring geometric coverage.

## Shared lifecycle and scaling

Both branches retain the existing valid incoming/pre-interception-plane active
mask and upright factor. Stop/success proxy and resolved shots close the term;
coverage never changes those flags. Stationary root gets zero; progress toward
the target gets positive feedback, movement away gets negative feedback. Covered
states get zero, not a target-velocity braking penalty. Near-field geometry does
not imply infinite-horizon safety or an actual ball contact.

RewardTerm weight remains 10; ordinary dt scaling and task factor .6 apply.
At dt=.02 the absolute mixed contribution is bounded by .24/step. At full gates
and .3m/s directed velocity the mixed contribution is .072/step. Aerial and
rolling rewards select via the physical `is_rolling` shot flag, not region ID.

## Limitations

- Mesh bounding boxes overapproximate the collision hull, especially rounded
  corners. A positive geometric overlap is only an approximate guidance gate.
- Feet are assumed fixed over prediction, and rolling drag/bounces are ignored.
- The 30cm aerial corridor is retained as an experimental baseline, not a
  calibrated reachability model. Only rolling guidance removes it.
- Existing eereach/success point-based rewards remain unchanged in this commit.
  This is not a complete redesign of foot accuracy rewards.
- Compared with the prior stage this changes formula AND coverage; training
  improvements cannot be attributed to either independently without ablation.

## Files and verification

Production: `mdp/foot_coverage.py`, `mdp/rewards.py`, `config/k1/env_cfgs.py`.
Diagnostics add `body_coverage_kind` and `rolling_foot_surface_gap` (1e6 means
no valid predicted foot segment); legacy command columns are zero in this mode.
Historical pure velocity-tracking functions/tests remain available, but the old
production-wiring velocity test is replaced by the split-coverage environment test.

Passed scripts:

1. `test_goalkeeper_foot_coverage.py`: independent distance calculations,
   segment interior, sphere radius, gaps between feet, yaw/pitch, raised and
   forward feet, horizon, translations, mirrors and invalid ball states.
2. `test_goalkeeper_split_coverage_env.py`: compiled real K1 bounds, rolling
   inside the 30cm corridor, aerial/rolling separation, no stopping penalty,
   direction signs, dt/mixing, stage closure, partial reset and a real step.
3. `test_goalkeeper_movement_env.py`: real root/arm distinction and coverage,
   timing, immutable plane, dt and reset regression.
4. `test_goalkeeper_movement_training.py`: eight transitions and one real
   PPO/AMP update with expert sampling; finite losses and updated parameters.
5. `test_goalkeeper_fall_and_movement_env.py`: fall/timeout/recovery costs.
6. `test_goalkeeper_timed_interception.py`: prediction/timing regressions.

Ruff on new geometry, configuration, diagnostic and new tests, plus
`git diff --check`, passed. A short frozen-policy diagnostic uses the new
columns and asserts unchanged policy/AMP parameters; this is a smoke test, not
evidence of trained improvement. Raw artifacts/logs remain local under
`artifacts/goalkeeper_split_coverage_20261112/` and
`artifacts/goalkeeper_split_coverage_smoke_20261112/`.

For training, preserve the previous stepping checkpoint as baseline and compare
directed body/foot displacement, ball-arrival coverage error, safe saves and
falls; reward growth alone is not a success criterion.
