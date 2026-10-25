import math
from types import SimpleNamespace as NS

import torch

from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    post_orientation_exp,
    post_ang_vel_exp,
    post_lin_vel_exp,
)


robot = NS(
    data=NS(
        projected_gravity_b=torch.tensor([
            [0.0, 0.0, -1.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]),
        root_link_ang_vel_b=torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
            [1.0, -2.0, 8.0],
        ]),
        root_link_lin_vel_b=torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 3.0],
            [0.5, 0.0, 0.0],
        ]),
    )
)

env = NS(scene={"robot": robot})
recovery_mask = torch.tensor([False, True, True])

cases = [
    (
        "postorientation",
        post_orientation_exp,
        torch.tensor([0.0, 1.0, math.exp(-3.0)]),
    ),
    (
        "postangvel",
        post_ang_vel_exp,
        torch.tensor([0.0, 1.0, math.exp(-15.0)]),
    ),
    (
        "postlinvel",
        post_lin_vel_exp,
        torch.tensor([0.0, 1.0, math.exp(-0.75)]),
    ),
]

for name, function, expected in cases:
    reward = function(env, recovery_mask=recovery_mask)

    assert reward.shape == (3,)
    assert torch.isfinite(reward).all()
    torch.testing.assert_close(reward, expected)

    # 全部关闭恢复阶段时，所有奖励必须为零。
    disabled_reward = function(
        env,
        recovery_mask=torch.zeros(3, dtype=torch.bool),
    )
    torch.testing.assert_close(
        disabled_reward,
        torch.zeros(3),
    )

    print(name, reward)

print("三个恢复阶段奖励的计算检查通过")