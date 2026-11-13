"""CPU MuJoCo checks of ball contacts and six-region flights, without a policy."""

import mujoco
import numpy as np
from booster_mjlab.tasks.goalkeeper.source.physics import ball_spec
from booster_mjlab.tasks.goalkeeper.source.settings import EVAL_REGIONS
from booster_mjlab.tasks.goalkeeper.scene.assets import goal_spec
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings

spec = ball_spec()
spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], friction=[0, 0, 0]
)
spec.option.timestep = 0.005
m = spec.compile()
d = mujoco.MjData(m)
goal = goal_spec(FullGoalSettings()).compile()
assert not goal.geom_contype.any() and not goal.geom_conaffinity.any()


def reset(pos, vel):
    mujoco.mj_resetData(m, d)
    d.qpos[:3] = pos
    d.qpos[3] = 1
    d.qvel[:3] = vel


reset([0, 0, 1], [0, 0, 0])
touched = False
peak = 0.0
for _ in range(240):
    mujoco.mj_step(m, d)
    touched |= d.ncon > 0
    if touched:
        peak = max(peak, float(d.qpos[2]))
assert 0.12 < peak < 1, peak
print("drop rebound peak (ball centre):", peak)
reset([0, 0, 0.1], [1, 0, 0])
for _ in range(200):
    mujoco.mj_step(m, d)
assert 0.1 < d.qpos[0] < 1.1
assert np.linalg.norm(d.qvel[:3]) < 1.01
print("sliding-to-rolling distance at 1s:", d.qpos[0])
# Deterministic low/middle/high launch heights, all six targets.
# Ablations report crossing error; only unperturbed trajectories must match.
for mode in ("none", "drag", "force", "velocity", "all"):
    errors = []
    misses = 0
    rng = np.random.default_rng(42)
    for bounds in EVAL_REGIONS:
        for height in (0.15, 1.0, 1.75):
            start = np.array([4.0, 0.0, height])
            end = np.array(
                [-0.3, (bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2]
            )
            t = 0.8
            v = (end - start) / t
            v[2] += 0.5 * 9.81 * t
            tc = (0.1 - start[0]) / v[0]
            expected = start + v * tc
            expected[2] -= 0.5 * 9.81 * tc**2
            reset(start, v)
            crossing = None
            for i in range(400):
                if i % 4 == 0:
                    force = np.zeros(3)
                    if mode in ("drag", "all"):
                        vel = d.qvel[:3]
                        force -= (
                            0.5
                            * 1.225
                            * 0.47
                            * np.pi
                            * 0.1**2
                            * np.linalg.norm(vel)
                            * vel
                        )
                    if mode in ("force", "all"):
                        force += rng.uniform(-0.5, 0.5, 3)
                    d.xfrc_applied[1, :3] = force
                    if mode in ("velocity", "all") and i > 0 and i % 100 == 0:
                        d.qvel[:3] += rng.uniform(-0.5, 0.5, 3)
                mujoco.mj_step(m, d)
                if d.qpos[0] <= 0.1:
                    crossing = d.qpos[:3].copy()
                    break
            if crossing is None:
                misses += 1
            else:
                errors.append(float(np.linalg.norm(crossing[1:] - expected[1:])))
    print(mode, "misses", misses, "max crossing error", max(errors, default=0))
    if mode == "none":
        assert misses == 0 and max(errors) < 0.06
print("Ball contact and six-region deterministic checks passed")
