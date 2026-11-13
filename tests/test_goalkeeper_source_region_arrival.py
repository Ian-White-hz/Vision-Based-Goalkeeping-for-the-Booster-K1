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
    INTERCEPT_HEIGHTS,
    INTERCEPT_X,
)
from booster_mjlab.tasks.goalkeeper.source.state import state

torch.set_num_threads(1)
torch.manual_seed(42)
bounds = torch.tensor(REGIONS).repeat(100, 1)
starts, velocities, targets = sample_region_shots(bounds)
spec = ball_spec()
spec.option.timestep = 0.005
spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], friction=[0, 0, 0]
)
m = spec.compile()
d = mujoco.MjData(m)
assert bounds[:, :2].abs().max() + 0.1 < GOAL_WIDTH / 2
assert bounds[:, 3].max() + 0.1 < GOAL_HEIGHT
errors = []
for shot_id, (start, velocity, target, bound) in enumerate(
    zip(starts, velocities, targets, bounds)
):
    mujoco.mj_resetData(m, d)
    d.qpos[:3] = start.numpy()
    d.qpos[3] = 1
    d.qvel[:3] = velocity.numpy()
    crossed = False
    for _ in range(250):
        previous = d.qpos[:3].copy()
        mujoco.mj_step(m, d)
        assert d.ncon == 0, "Ground contact before goal crossing"
        if previous[0] > INTERCEPT_X >= d.qpos[0]:
            f = (INTERCEPT_X - previous[0]) / (d.qpos[0] - previous[0])
            z = previous[2] + f * (d.qpos[2] - previous[2])
            lo, hi = INTERCEPT_HEIGHTS[shot_id % 6]
            assert lo <= z <= hi, (z, lo, hi)
            assert 0.695 <= d.time - 0.005 + f * 0.005 <= 0.905
        if d.qpos[0] <= GOAL_X:
            f = (GOAL_X - previous[0]) / (d.qpos[0] - previous[0])
            p = previous + f * (d.qpos[:3] - previous)
            assert bound[0] <= p[1] <= bound[1] and bound[2] <= p[2] <= bound[3], (
                p,
                bound,
            )
            errors.append(np.linalg.norm(p - target.numpy()))
            crossed = True
            break
    assert crossed
print("600/600 actual goal crossings inside assigned region; max error:", max(errors))
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
