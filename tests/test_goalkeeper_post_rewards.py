import math
from types import SimpleNamespace as NS

import torch

from mjlab.managers import(
    SceneEntityCfg
)

from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    post_orientation_exp,
    post_ang_vel_exp,
    post_lin_vel_exp,
    post_recovery_from_env,
    post_upper_joint_position_from_env
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

class FakeScene(dict):
    pass


scene = FakeScene(
    robot=robot,
    ball=NS(
        data=NS(
            root_link_pos_w=torch.tensor([
                [1.0, 0.0, 0.1],
                [-0.1, 0.0, 0.1],
                [1.0, 0.0, 0.1],
            ]),
            root_link_lin_vel_w=torch.tensor([
                [-3.0, 0.0, 0.0],
                [-3.0, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
            ]),
        ),
    ),
)
scene.env_origins = torch.zeros(3, 3)

adapter_env = NS(
    scene=scene,
    _full_goal_shots={
        "velocity": torch.tensor([[-3.0, 0.0]]).repeat(3, 1),
    },
)

kinds = (
    "orientation",
    "angular_velocity",
    "linear_velocity",
)

# 复用前面 cases 中的预期计算结果。
for kind, (_, _, expected) in zip(kinds, cases, strict=True):
    reward = post_recovery_from_env(
        adapter_env,
        kind=kind,
    )

    assert reward.shape == (3,)
    assert torch.isfinite(reward).all()
    torch.testing.assert_close(reward, expected)

    print("恢复奖励适配:", kind, reward)

print("三个恢复奖励的适配检查通过")

default_position = torch.zeros(3, 6)
default_position[:, 1] = 0.2
default_position[:, 3] = -0.4

joint_position = default_position.clone()

# 未选中的关节，即使偏离很大，也不应该影响奖励。
joint_position[:, 0] = 100.0

# 第三个环境，两个选中关节分别偏离 ±0.5 rad。
joint_position[2, 1] += 0.5
joint_position[2, 3] -= 0.5

robot.data.joint_pos = joint_position
robot.data.default_joint_pos = default_position

reward = post_upper_joint_position_from_env(
    adapter_env,
    asset_cfg=SceneEntityCfg(
        "robot",
        joint_ids=[1, 3],
    ),
)

expected = torch.tensor([
    0.0,
    1.0,
    math.exp(-0.5),
])

assert reward.shape == (3,)
assert torch.isfinite(reward).all()
torch.testing.assert_close(reward, expected)

print("上肢恢复奖励:", reward)
print("上肢恢复奖励适配检查通过")