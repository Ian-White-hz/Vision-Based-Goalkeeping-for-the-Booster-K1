# G1 goalkeeper motions to K1

The offline converter produces six K1 reference motions from the adjacent
`Humanoid-Goalkeeper` checkout. It does not transfer G1 policy weights. The result
is a kinematic motion library; torque-controlled tracking is a separate validation
step before using these motions on hardware.

## Run

```bash
uv run retarget-goalkeeper \
  --source-repo ../Humanoid-Goalkeeper \
  --output-dir artifacts/goalkeeper_k1
```

For a single motion, add `--clips lefthand`. Outputs are protected against
overwriting; use a different directory or explicitly add `--overwrite`.
To slow playback while retaining every frame, use `--playback-speed 0.75`.
The default source FPS is 30, taken from upstream `G129Cfg.dataset.frame_rate`;
the `.pt` files themselves contain no FPS field. Supply `--fps` for other exports.

Optional offscreen snapshots require a working EGL/OpenGL driver:

```bash
MUJOCO_GL=egl uv run retarget-goalkeeper \
  --output-dir artifacts/goalkeeper_k1_preview --render-preview
```

## Outputs

```text
artifacts/goalkeeper_k1/
  serial/{lefthand,righthand,leftjump,rightjump,leftstep,rightstep}.pkl
  parallel/{lefthand,righthand,leftjump,rightjump,leftstep,rightstep}.pkl
  reports/<clip>.json
  reports/<clip>.npz
  previews/<clip>_<variant>.png  # only with --render-preview
  manifest.json
```

The pickle schema matches `MotionFile`: `fps`, `root_pos` (T,3), `root_rot`
(T,4, **xyzw**), and `dof_pos` (T,22 serial / T,34 parallel). MuJoCo qpos uses
**wxyz**; conversion happens only when writing qpos. Head angles are held at zero.
The joint order matches `K1_JOINT_ORDER` or `K1_PARALLEL_JOINT_ORDER`.
There are six separate clips, so future region-conditioned AMP can retain the
upstream association: 0 left hand, 1 right hand, 2 left jump, 3 right jump,
4 left step, 5 right step. These are motion labels, not K1 goal-region dimensions.

Browse either library using the existing viewer (non-AMP tasks avoid a dependency
on the default Hugging Face dataset for resets):

```bash
uv run visualize-motions --dataset artifacts/goalkeeper_k1/serial \
  --task Mjlab-Velocity-Flat-Booster-K1 --device cpu

uv run visualize-motions --dataset artifacts/goalkeeper_k1/parallel \
  --task Mjlab-Velocity-Flat-Booster-K1-Parallel --device cpu
```

## What the converter does

1. Load tensor-only `.pt` files using `torch.load(weights_only=True)` and read
   `joint_id.txt`. Published clips have **21**, not 29, joint columns. The eight
   missing G1 waist/wrist angles are zero, matching upstream `MotionLib`.
2. Reconstruct named G1 links with URDF forward kinematics, including joint-origin
   rotations and fixed joints. Verify six knee/ankle/elbow trajectories against
   the published unnamed 17-link tracks. Reject incompatible FK (>3 cm error).
   No unverified 17-link index is used as an IK target.
3. Scale upper arm, forearm, thigh and shank segments separately using the two
   robots' geometry. Transfer torso orientation (including G1 waist yaw) to the
   K1 root, normalize the initial heading, and retain scaled root translation.
4. Solve 20 non-head K1 joints with bounded least squares, MuJoCo analytic
   Jacobians, hand/ankle and elbow/knee targets, foot orientation, temporal
   regularization and hand/arm collision-distance penalties. G1 shoulder yaw is
   not assigned to K1 elbow yaw. Joint-angle mapping is only an initial guess.
5. Infer stance from source ankle height and horizontal speed, and anchor stance
   targets in world coordinates. Contact inference is a heuristic: source data
   has no contact labels. Lift the root only as needed to keep the actual foot
   collision meshes above the ground; preserve the source flight clearance.
6. Use existing parallel-ankle IK to expand 22 joints to 34. Record how many
   ankle frames required workspace projection, and independently validate the
   parallel geometry rather than assuming its contacts equal the serial model.

## Read the report

`manifest.json` records source hashes, options and all clip reports. Each report
includes initial/final IK error, mean/max hand and foot errors, joint limits,
motor speed peaks, acceleration, inferred stance slip, self/ground contacts,
ankle projection counts and MuJoCo equality residuals.

`kinematic_checks_passed` requires no >1 mm self/other-body ground penetration,
no joint-limit or motor-speed violation, no ankle projection, max hand error
<=10 cm, inferred stance slip <=0.15 m/s, and converged IK frames. These are
screening thresholds, not guarantees of balance, actuator torque feasibility,
or successful goalkeeping. No-load speed limits alone do not validate the
torque-speed envelope. Inspect the previews and run dynamic motion tracking
before choosing the final training reference set.

The NPZ diagnostics preserve pre-contact-anchor local targets, foot orientations,
inferred contact masks and the actual errors against the targets used by IK.
If a motion fails, the output is still saved with `needs_review` and specific
issues, so the failure remains visible rather than silently discarding frames.

## Tests

```bash
uv run python -m unittest discover -s tests -p 'test_goalkeeper_retarget.py' -v
```

The numerical tests check analytic Jacobians against finite differences, speed
bounds, URDF conventions, invalid inputs and parallel closure. A data integration
test checks all six original clips when the adjacent checkout is available.

## Attribution

Source motions: [InternRobotics/Humanoid-Goalkeeper](https://github.com/InternRobotics/Humanoid-Goalkeeper),
"Humanoid Goalkeeper: Learning from Position Conditioned Task-Motion Constraints".
Upstream README identifies the project as CC BY-NC-SA 4.0. Converted artifacts
remain derived from these motions; preserve attribution and the applicable
license when distributing them. No upstream source code is vendored by this tool.
