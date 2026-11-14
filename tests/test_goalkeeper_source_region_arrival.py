"""Verify production sampling with actual free-flight simulation and goal origins."""

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from booster_mjlab.tasks.goalkeeper.source.config import env_cfg
from booster_mjlab.tasks.goalkeeper.source.events import sample_region_shots
from booster_mjlab.tasks.goalkeeper.source.physics import ball_spec
from booster_mjlab.tasks.goalkeeper.source.settings import (
    REGIONS,
    GOAL_WIDTH,
    GOAL_HEIGHT,
    GOAL_X,
    INTERCEPT_X,
    SHOT_INTERCEPT_TIME,
    BALL_RADIUS,
    SIM_DT,
)
from booster_mjlab.tasks.goalkeeper.source.state import state

torch.set_num_threads(1)
torch.manual_seed(42)
ids = torch.arange(600) % 6
bounds = torch.tensor(REGIONS)[ids]
starts, velocities, targets = sample_region_shots(bounds, ids)
spec = ball_spec()
spec.option.timestep = SIM_DT
spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], friction=[0, 0, 0]
)
m = spec.compile()
d = mujoco.MjData(m)
# Keeper stands in front of the goal: heels (~-0.1) clear of the goal line.
assert GOAL_X <= -0.3
assert bounds[:, 2].min() > BALL_RADIUS
errors, bounced = [], np.zeros(6)
for shot_id, (start, velocity, target, bound) in enumerate(
    zip(starts, velocities, targets, bounds)
):
    mujoco.mj_resetData(m, d)
    d.qpos[:3] = start.numpy()
    d.qpos[3] = 1
    d.qvel[:3] = velocity.numpy()
    at_robot = at_goal = touched = False
    for _ in range(300):
        previous = d.qpos[:3].copy()
        mujoco.mj_step(m, d)
        if d.ncon:
            assert at_robot, "Ground contact before robot plane"
            touched = True
        if not at_robot and previous[0] > INTERCEPT_X >= d.qpos[0]:
            f = (INTERCEPT_X - previous[0]) / (d.qpos[0] - previous[0])
            p = previous + f * (d.qpos[:3] - previous)
            assert bound[0] <= p[1] <= bound[1] and bound[2] <= p[2] <= bound[3], (
                p,
                bound,
            )
            t = d.time - SIM_DT + f * SIM_DT
            assert (
                SHOT_INTERCEPT_TIME[0] - SIM_DT <= t <= SHOT_INTERCEPT_TIME[1] + SIM_DT
            )
            errors.append(np.linalg.norm(p - target.numpy()))
            at_robot = True
        if previous[0] > GOAL_X >= d.qpos[0]:
            f = (GOAL_X - previous[0]) / (d.qpos[0] - previous[0])
            p = previous + f * (d.qpos[:3] - previous)
            # Would score: ball fully inside posts and under the bar.
            assert abs(p[1]) <= GOAL_WIDTH / 2 - BALL_RADIUS, p
            assert p[2] <= GOAL_HEIGHT - BALL_RADIUS, p
            bounced[shot_id % 6] += touched
            at_goal = True
            break
    assert at_robot and at_goal
assert max(errors) < 0.005, max(errors)
print(
    "600/600 robot-plane crossings inside assigned region and goal mouth; "
    f"max target error {max(errors):.4f} m; bounce-before-goal per region "
    f"{(bounced / 100).round(2).tolist()}"
)
cfg = env_cfg(6, play=True, randomize=False)
e = ManagerBasedRlEnv(cfg, device="cpu")
try:
    e.reset(seed=42)
    robot = e.scene["robot"]
    for name in ("Trunk", "Head_2", "left_hand_end_ball"):
        ids, _ = robot.find_bodies(name)
        if ids:
            print(
                name,
                "default position",
                (
                    robot.data.body_link_pos_w[0, ids[0]] - e.scene.env_origins[0]
                ).tolist(),
            )
    gid = e.scene["goal"].indexing.mocap_id
    pos = e.sim.data.mocap_pos[:, gid, :]
    torch.testing.assert_close(pos, e.scene.env_origins)
    assert torch.unique(pos, dim=0).shape[0] == 6
    assert state(e).end_regions.tolist() == list(range(6))
    print("6 distinct goal origins aligned with 6 environments and regions")
finally:
    e.close()
