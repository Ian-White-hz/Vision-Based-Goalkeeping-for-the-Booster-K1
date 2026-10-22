from types import SimpleNamespace

import torch
from mjlab.envs import mdp as common_mdp
from mjlab.scene import SceneCfg

from booster_mjlab.tasks.goalkeeper.goalkeeper_env_cfg import (
    make_goalkeeper_env_cfg,
)

from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    base_ang_vel_xy_l2,
    base_backward_velocity_penalty,
    base_line_deviation,
    knee_height_below_minimum,
)

from copy import deepcopy



cfg = make_goalkeeper_env_cfg(
    scene=SceneCfg(num_envs=3),
    decimation=4,
    episode_length_s=3.0,
)

term = cfg.rewards["smoothness"]
assert term.func is common_mdp.action_acc_l2
assert term.weight == -0.1
assert cfg.scale_rewards_by_dt is True

# 每行是一个环境，每列是一个 action 维度。
previous_previous = torch.zeros(3, 22)
previous = torch.zeros(3, 22)
current = torch.zeros(3, 22)

# 环境 0：0 -> 0 -> 0
# 保持全部为零。

# 环境 1：0 -> 0.1 -> 0.2
previous[1, 0] = 0.1
current[1, 0] = 0.2

# 环境 2：0 -> 0.1 -> 0
previous[2, 0] = 0.1

env = SimpleNamespace(
    action_manager=SimpleNamespace(
        action=current,
        prev_action=previous,
        prev_prev_action=previous_previous,
    )
)

raw_value = term.func(env, **term.params)

assert raw_value.shape == (3,)
assert torch.isfinite(raw_value).all()
torch.testing.assert_close(
    raw_value,
    torch.tensor([0.0, 0.0, 0.04]),
)

weighted_value = raw_value * term.weight
torch.testing.assert_close(
    weighted_value,
    torch.tensor([0.0, 0.0, -0.004]),
)

print("原始 smoothness:", raw_value)
print("乘权重后，未乘 dt:", weighted_value)
print("动作二阶平滑奖励检查通过")

velocity_term = cfg.rewards["dof_vel"]
assert velocity_term.func is common_mdp.joint_vel_l2
assert velocity_term.weight == -5e-4

joint_velocity = torch.zeros(3, 22)
joint_velocity[1, :2] = torch.tensor([1.0, -2.0])
joint_velocity[2, :2] = torch.tensor([-1.0, 2.0])

velocity_env = SimpleNamespace(
    scene={
        "robot": SimpleNamespace(
            data=SimpleNamespace(
                joint_vel=joint_velocity,
            )
        )
    }
)

velocity_cost = velocity_term.func(
    velocity_env,
    **velocity_term.params,
)

assert velocity_cost.shape == (3,)
assert torch.isfinite(velocity_cost).all()

torch.testing.assert_close(
    velocity_cost,
    torch.tensor([0.0, 5.0, 5.0]),
)

torch.testing.assert_close(
    velocity_cost * velocity_term.weight,
    torch.tensor([0.0, -0.0025, -0.0025]),
)

print("原始 dof_vel:", velocity_cost)
print("关节速度奖励检查通过")


angular_term = cfg.rewards["ang_vel_xy"]
assert angular_term.func is base_ang_vel_xy_l2
assert angular_term.weight == -0.1
assert angular_term.params == {"robot_name": "robot"}

angular_velocity_b = torch.tensor([
    [0.0, 0.0, 5.0],
    [1.0, -2.0, 0.0],
    [-1.0, 2.0, 8.0],
])

angular_env = SimpleNamespace(
    scene={
        "robot": SimpleNamespace(
            data=SimpleNamespace(
                root_link_ang_vel_b=angular_velocity_b,
            )
        )
    }
)

angular_cost = angular_term.func(
    angular_env,
    **angular_term.params,
)

assert angular_cost.shape == (3,)
assert torch.isfinite(angular_cost).all()

torch.testing.assert_close(
    angular_cost,
    torch.tensor([0.0, 5.0, 5.0]),
)

torch.testing.assert_close(
    angular_cost * angular_term.weight,
    torch.tensor([0.0, -0.5, -0.5]),
)

print("原始 ang_vel_xy:", angular_cost)
print("基座局部角速度奖励检查通过")


position_limit_term = cfg.rewards["dof_pos_limits"]
assert position_limit_term.func is common_mdp.joint_pos_limits
assert position_limit_term.weight == -3.0

joint_position = torch.zeros(3, 22)

# 环境 0：恰好处于边界，不惩罚。
joint_position[0, :2] = torch.tensor([-1.0, 1.0])

# 环境 1：超过上限 0.2。
joint_position[1, 0] = 1.2

# 环境 2：分别超出下限 0.3、上限 0.4。
joint_position[2, :2] = torch.tensor([-1.3, 1.4])

# Shape: (环境数, 关节数, 下限/上限)。
soft_limits = torch.zeros(3, 22, 2)
soft_limits[:, :, 0] = -1.0
soft_limits[:, :, 1] = 1.0

position_limit_env = SimpleNamespace(
    scene={
        "robot": SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=joint_position,
                soft_joint_pos_limits=soft_limits,
            )
        )
    }
)

position_limit_cost = position_limit_term.func(
    position_limit_env,
    **position_limit_term.params,
)

assert position_limit_cost.shape == (3,)
assert torch.isfinite(position_limit_cost).all()

torch.testing.assert_close(
    position_limit_cost,
    torch.tensor([0.0, 0.2, 0.7]),
)

torch.testing.assert_close(
    position_limit_cost * position_limit_term.weight,
    torch.tensor([0.0, -0.6, -2.1]),
)

print("原始 dof_pos_limits:", position_limit_cost)
print("关节位置软限位奖励检查通过")


retreat_term = cfg.rewards["noretreat"]
assert retreat_term.func is base_backward_velocity_penalty
assert retreat_term.weight == -2.0
assert retreat_term.params == {"robot_name": "robot"}

linear_velocity_b = torch.tensor([
    [1.0, 0.0, 0.0],
    [-0.4, 0.0, 0.0],
    [-2.0, 3.0, 4.0],
])

retreat_env = SimpleNamespace(
    scene={
        "robot": SimpleNamespace(
            data=SimpleNamespace(
                root_link_lin_vel_b=linear_velocity_b,
            )
        )
    }
)

retreat_cost = retreat_term.func(
    retreat_env,
    **retreat_term.params,
)

assert retreat_cost.shape == (3,)
assert torch.isfinite(retreat_cost).all()

torch.testing.assert_close(
    retreat_cost,
    torch.tensor([0.0, 0.4, 1.0]),
)

torch.testing.assert_close(
    retreat_cost * retreat_term.weight,
    torch.tensor([0.0, -0.8, -2.0]),
)

print("原始 noretreat:", retreat_cost)
print("局部后退速度奖励检查通过")


class MockSpatialScene(dict):
    pass


origins = torch.tensor([
    [0.0, 0.0, 0.0],
    [10.0, 0.0, 1.0],
    [-10.0, 0.0, 2.0],
])

root_position = torch.tensor([
    [0.1, 0.0, 0.6],
    [10.7, 0.0, 1.6],
    [-11.5, 0.0, 2.6],
])

# 三个 body：非膝部、左膝、右膝。
body_position = torch.zeros(3, 3, 3)
body_position[:, :, 2] = origins[:, 2:3] + torch.tensor([
    [0.0, 0.15, 0.25],
    [0.0, 0.10, 0.30],
    [0.0, 0.20, 0.30],
])

spatial_scene = MockSpatialScene(
    robot=SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=root_position,
            body_link_pos_w=body_position,
        )
    )
)
spatial_scene.env_origins = origins
spatial_env = SimpleNamespace(scene=spatial_scene)

# 参考平面惩罚。
line_term = cfg.rewards["stayonline"]
assert line_term.func is base_line_deviation
assert line_term.weight == -2.0

line_cost = line_term.func(spatial_env, **line_term.params)

assert line_cost.shape == (3,)
assert torch.isfinite(line_cost).all()
torch.testing.assert_close(
    line_cost,
    torch.tensor([0.0, 0.5, 1.0]),
)
torch.testing.assert_close(
    line_cost * line_term.weight,
    torch.tensor([0.0, -1.0, -2.0]),
)

# 膝高惩罚。
knee_term = cfg.rewards["penalize_kneeheight"]
assert knee_term.func is knee_height_below_minimum
assert knee_term.weight == -100.0
assert knee_term.params["asset_cfg"].body_names == (
    "Left_Shank", "Right_Shank"
)

# 模拟真实环境初始化时的名称→索引解析，不修改原配置。
knee_params = deepcopy(knee_term.params)
knee_params["asset_cfg"].body_ids = [1, 2]

knee_cost = knee_term.func(spatial_env, **knee_params)

assert knee_cost.shape == (3,)
assert torch.isfinite(knee_cost).all()
torch.testing.assert_close(
    knee_cost,
    torch.tensor([0.0, 1.0, 0.0]),
)
torch.testing.assert_close(
    knee_cost * knee_term.weight,
    torch.tensor([0.0, -100.0, 0.0]),
)

print("原始 stayonline:", line_cost)
print("原始 penalize_kneeheight:", knee_cost)
print("位置约束奖励检查通过")