from types import SimpleNamespace as NS

import torch
from mjlab.managers import SceneEntityCfg

from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    stopball_from_env,
    target_success_from_env,
)
from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    reset_shot_outcome_state,
)


class FakeScene(dict):
    pass


targets = torch.tensor([[0.1, 0.0, 0.5]]).repeat(2, 1)

hand_positions = torch.full((2, 2, 3), 10.0)

# 环境 0 使用左手，距离目标 0.1 m。
hand_positions[0, 0] = targets[0] + torch.tensor([0.0, 0.1, 0.0])

# 环境 1 使用右手，距离目标 0.2 m。
hand_positions[1, 1] = targets[1] + torch.tensor([0.0, 0.2, 0.0])

scene = FakeScene(
    robot=NS(data=NS(site_pos_w=hand_positions)),
    ball=NS(
        data=NS(
            root_link_pos_w=torch.tensor([
                [0.5, 0.0, 0.11],
                [0.5, 0.0, 0.11],
            ]),
            root_link_lin_vel_w=torch.tensor([
                [-1.0, 0.0, 0.0],
                [-4.0, 0.0, 0.0],
            ]),
        ),
    ),
)
scene.env_origins = torch.zeros(2, 3)

env = NS(
    num_envs=2,
    device="cpu",
    common_step_counter=0,
    scene=scene,
    _full_goal_shots={
        "target_position_e": targets,
        "region": torch.tensor([0, 1]),
        "velocity": torch.tensor([[-4.0, 0.0, 0.0]]).repeat(2, 1),
    },
)

hand_cfg = SceneEntityCfg("robot", site_ids=[0, 1])

# 顺序一：success → stopball。
reset_shot_outcome_state(env, None)

success_first = target_success_from_env(env, hand_cfg).clone()
stop_after = stopball_from_env(env).clone()

torch.testing.assert_close(success_first, torch.tensor([1.0, 0.0]))
torch.testing.assert_close(stop_after, torch.tensor([1.0, 0.0]))

# 顺序二：清理状态后，stopball → success。
reset_shot_outcome_state(env, None)

stop_first = stopball_from_env(env).clone()
success_after = target_success_from_env(env, hand_cfg).clone()

torch.testing.assert_close(stop_first, stop_after)
torch.testing.assert_close(success_after, success_first)

# 下一步，环境 0 已有成功标志，但不再重复给 stopball 奖励。
env.common_step_counter += 1

torch.testing.assert_close(
    target_success_from_env(env, hand_cfg),
    torch.tensor([2.0, 0.0]),
)
torch.testing.assert_close(
    stopball_from_env(env),
    torch.zeros(2),
)

print("success 适配、跨步状态和调用顺序检查通过")