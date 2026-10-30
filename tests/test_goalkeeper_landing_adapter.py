from types import SimpleNamespace as NS

import torch
from mjlab.managers import RewardTermCfg

from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    JumpLandingReward,
    jump_landing_reward,
)


class FakeScene(dict):
    pass


root_position = torch.zeros(3, 3)
root_position[:, 2] = 0.51

default_state = torch.zeros(3, 13)
default_state[:, 2] = 0.51

found = torch.ones(3, 2)
force = torch.zeros(3, 2, 3)
force[..., 2] = -100.0

scene = FakeScene(
    robot=NS(
        data=NS(
            root_link_pos_w=root_position,
            default_root_state=default_state,
        ),
    ),
    feet_ground_contact=NS(
        data=NS(found=found, force=force),
    ),
)
scene.env_origins = torch.zeros(3, 3)

env = NS(
    num_envs=3,
    device="cpu",
    common_step_counter=0,
    scene=scene,
    _full_goal_shots={
        "region": torch.tensor([2, 3, 0]),
    },
)

cfg = RewardTermCfg(
    func=JumpLandingReward,
    weight=4.0,
    params={
        "robot_name": "robot",
        "sensor_name": "feet_ground_contact",
        "minimum_rise": 0.08,
        "force_sign": -1.0,
        "contact_threshold": 1.0,
    },
)
term = JumpLandingReward(cfg, env)


def check(expected):
    result = term(env, **cfg.params).clone()
    torch.testing.assert_close(result, torch.tensor(expected))

    # 同一步再次读取，结果不变。
    torch.testing.assert_close(
        term(env, **cfg.params),
        result,
    )
    print("landing:", result)
    env.common_step_counter += 1


# 初始双脚支撑，没有腾空历史。
check([0.0, 0.0, 0.0])

# 双脚离地，但只升高 1 cm，不算达到起跳门控。
found.zero_()
force.zero_()
root_position[:, 2] = 0.52
check([0.0, 0.0, 0.0])

# 升高 14 cm，两个跳跃类别建立腾空历史。
root_position[:, 2] = 0.65
check([1.0, 1.0, 0.0])

# 单脚落地。
found[:, 0] = 1
force[:, 0, 2] = -100.0
check([0.0, 0.0, 0.0])

# 双脚落地。
found.fill_(1)
force[..., 2] = -100.0
check([6.0, 6.0, 0.0])

# 只重置环境 0，环境 1 的历史仍然保留。
term.reset(torch.tensor([0]))
root_position[0, 2] = 0.51
check([0.0, 6.0, 0.0])

# 原公式的严格阈值边界。
boundary_force = torch.zeros(1, 2, 3)
boundary_force[0, :, 2] = torch.tensor([1.0, 10.0])

torch.testing.assert_close(
    jump_landing_reward(
        boundary_force,
        has_in_air=torch.tensor([True]),
        jump_mask=torch.tensor([True]),
    ),
    torch.tensor([1.0]),
)

print("落地奖励适配、缓存和局部 reset 检查通过")