"""All five allowed blockers share one target metric, independent of region."""

from types import SimpleNamespace as NS

import mjlab  # noqa: F401
import torch
from mjlab.managers import SceneEntityCfg

from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    ee_reach_from_env,
    target_success_from_env,
)
from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    get_blocker_target_distance,
    reset_shot_outcome_state,
)


class Scene(dict):
    pass


torch.set_num_threads(1)
batch = 30
region = torch.arange(6).repeat_interleave(5)
selected = torch.arange(5).repeat(6)
origins = torch.arange(batch).float()[:, None] * torch.tensor([[3.0, -2.0, 0.0]])
targets = torch.tensor([[0.1, 0.2, 0.3]]).repeat(batch, 1)
# Store selected sites/bodies out of order, with excluded points on the target.
site_ids, body_ids = [2, 0], [3, 0, 2]
sites_e = targets[:, None, :].repeat(1, 3, 1)
bodies_e = targets[:, None, :].repeat(1, 4, 1)
sites_e[:, site_ids] += 2.0
bodies_e[:, body_ids] += 2.0
for row, point in enumerate(selected.tolist()):
    points, index = (sites_e, site_ids[point]) if point < 2 else (
        bodies_e, body_ids[point - 2]
    )
    points[row, index] = targets[row] + torch.tensor([0.0, 0.1, 0.0])

scene = Scene(
    robot=NS(data=NS(
        site_pos_w=sites_e + origins[:, None, :],
        body_link_pos_w=bodies_e + origins[:, None, :],
        body_link_lin_vel_w=torch.zeros_like(bodies_e),
        site_lin_vel_w=torch.zeros_like(sites_e),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(batch, 1),
    )),
    ball=NS(data=NS(
        root_link_pos_w=origins + torch.tensor([[1.0, 0.0, 1.0]]),
        root_link_lin_vel_w=torch.tensor([[-4.0, 0.0, 0.0]]).repeat(batch, 1),
    )),
)
scene.env_origins = origins
env = NS(
    num_envs=batch, device="cpu", common_step_counter=0, scene=scene,
    _full_goal_shots={
        "region": region,
        "target_position_e": targets,
        "velocity": scene["ball"].data.root_link_lin_vel_w.clone(),
    },
    _goalkeeper_outcome_state=NS(shot_resolved=torch.zeros(batch, dtype=torch.bool)),
)
cfg = SceneEntityCfg("robot", site_ids=site_ids, body_ids=body_ids)
reset_shot_outcome_state(env)
distance = get_blocker_target_distance(env, cfg)
torch.testing.assert_close(distance, torch.full((batch,), 0.1), atol=5e-6, rtol=1e-5)
reach = ee_reach_from_env(env, cfg)
success = target_success_from_env(env, cfg)
torch.testing.assert_close(reach, torch.sigmoid(torch.full((batch,), 0.5)), atol=5e-6, rtol=1e-5)
torch.testing.assert_close(success, torch.ones(batch))
assert not env._goalkeeper_shot_state.success_flag.any(), "Proximity must not mark a save"

# Mirror the full geometry, including nonzero environment origins.
for points in (scene["robot"].data.site_pos_w, scene["robot"].data.body_link_pos_w):
    points[..., 1] *= -1
scene.env_origins[:, 1] *= -1
targets[:, 1] *= -1
torch.testing.assert_close(get_blocker_target_distance(env, cfg), distance)
torch.testing.assert_close(ee_reach_from_env(env, cfg), reach)
torch.testing.assert_close(target_success_from_env(env, cfg), success)

# Moving only the winning point outside the strict radius removes attainment.
for row, point in enumerate(selected.tolist()):
    points, index = (scene["robot"].data.site_pos_w, site_ids[point]) if point < 2 else (
        scene["robot"].data.body_link_pos_w, body_ids[point - 2]
    )
    points[row, index] = targets[row] + origins[row] + torch.tensor([0.0, 0.16, 0.0])
assert not target_success_from_env(env, cfg).any()
torch.testing.assert_close(
    ee_reach_from_env(env, cfg), torch.sigmoid(torch.full((batch,), 0.2)),
    atol=5e-6, rtol=1e-5,
)

# Resolution closes both rewards even if every target is attained.
env._goalkeeper_outcome_state.shot_resolved[:] = True
assert not ee_reach_from_env(env, cfg).any()
assert not target_success_from_env(env, cfg, strict_threshold=0.2).any()
print("30 cases: five blockers × six regions; exclusions, offsets, mirror and resolution passed")
