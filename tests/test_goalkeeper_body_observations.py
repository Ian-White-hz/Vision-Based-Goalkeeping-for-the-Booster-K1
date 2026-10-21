from types import SimpleNamespace as NS

import torch
from mjlab.envs import mdp as common_mdp
from mjlab.scene import SceneCfg

from booster_mjlab.tasks.goalkeeper.goalkeeper_env_cfg import (
    make_goalkeeper_env_cfg,
)

from booster_mjlab.robots.booster_k1.k1_constants import (
    K1_JOINT_ORDER,
    get_k1_default_joint_pos,
)

cfg = make_goalkeeper_env_cfg(
    scene=SceneCfg(num_envs=2),
    decimation=4,
    episode_length_s=3.0,
)

gravity = torch.tensor([
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
])

angular_velocity = torch.tensor([
    [0.0, 0.0, 0.0],
    [0.1, 0.2, 0.3],
])

num_joints = len(K1_JOINT_ORDER)
assert num_joints == 22

default_joint_pos = get_k1_default_joint_pos().repeat(2, 1)
default_joint_vel = torch.zeros_like(default_joint_pos)

joint_offset = torch.zeros_like(default_joint_pos)
joint_offset[1, :] = 0.1

joint_velocity = torch.zeros_like(default_joint_pos)
joint_velocity[1, :] = 0.2

last_action = torch.zeros((2, num_joints))
last_action[1, :] = 0.3

env = NS(scene={
    "robot": NS(data=NS(
        projected_gravity_b=gravity,
        root_link_ang_vel_b=angular_velocity,
        joint_pos=default_joint_pos + joint_offset,
        default_joint_pos=default_joint_pos,
        joint_vel=joint_velocity,
        default_joint_vel=default_joint_vel,
    )),
})

env.action_manager = NS(action=last_action)

actor_terms = cfg.observations["actor"].terms

expected = {
    "projected_gravity": gravity,
    "base_ang_vel": angular_velocity,
    "joint_pos": joint_offset,
    "joint_vel": joint_velocity,
    "actions": last_action,
}

assert list(actor_terms) == list(expected)

assert actor_terms["joint_pos"].func is common_mdp.joint_pos_rel
assert actor_terms["joint_vel"].func is common_mdp.joint_vel_rel
assert actor_terms["actions"].func is common_mdp.last_action

results = []
for name, term in actor_terms.items():
    value = term.func(env, **term.params)

    assert value.shape == expected[name].shape
    assert torch.isfinite(value).all()
    assert torch.allclose(value, expected[name], atol=1e-6)

    results.append(value)
    print(name, tuple(value.shape))

body_observation = torch.cat(results, dim=-1)
assert body_observation.shape == (2, 72)
assert torch.allclose(
    body_observation[:, -num_joints:],
    last_action,
)

critic_terms = cfg.observations["critic"].terms
assert set(critic_terms) == set(actor_terms) | {"ball_position"}
assert critic_terms["joint_pos"] is not actor_terms["joint_pos"]

print("全身状态观测检查通过")