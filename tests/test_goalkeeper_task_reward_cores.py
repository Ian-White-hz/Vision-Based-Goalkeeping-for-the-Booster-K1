import math

import torch

from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    ee_reach_reward,
    target_success_reward,
    stopball_transition,
    post_upper_joint_position_exp,
)

batch_size = 10

target = torch.zeros(batch_size, 3)
target[:, 2] = 0.5

torso = torch.zeros(batch_size, 3)
torso[:, 2] = 0.5

# 环境 0：远球阶段，横向误差 0.6，高度项 0.3。
target[0, 1] = 0.6
torso[0, 2] = 0.8

velocity = torch.zeros(batch_size, 3)
velocity[1, 1] = 0.5
velocity[2, 1] = -0.5
velocity[3:5, 2] = 0.5
velocity[5, 1] = 0.5
velocity[6, 1] = -0.5

gravity = torch.zeros(batch_size, 3)
gravity[:, 2] = -1.0
gravity[9] = torch.tensor([0.0, 1.0, 0.0])

ball_x = torch.ones(batch_size)
ball_x[0] = 2.0
ball_x[7] = -0.1

launch_vx = torch.full((batch_size,), -4.0)
ball_vx = launch_vx.clone()
ball_vx[8] = -1.0

region = torch.tensor([0, 0, 1, 2, 3, 4, 5, 0, 0, 0])

reach_inputs = dict(
    distance=torch.full((batch_size,), 0.2),
    target_position_e=target,
    torso_position_e=torso,
    body_velocity_w=velocity,
    projected_gravity_b=gravity,
    ball_position_x_e=ball_x,
    ball_velocity_x_w=ball_vx,
    launch_velocity_x_w=launch_vx,
    region=region,
)

reach_reward = ee_reach_reward(**reach_inputs)

assert reach_reward.shape == (batch_size,)
assert torch.isfinite(reach_reward).all()

torch.testing.assert_close(
    reach_reward,
    torch.tensor([
        0.55,
        1.25, 1.25, 1.25, 1.25, 1.25, 1.25,
        1.0, 1.0, 0.0,
    ]),
)

# 课程等级增加时，跳跃速度激励增强。
advanced_reward = ee_reach_reward(
    **reach_inputs,
    curriculum_level=1.0,
)
torch.testing.assert_close(
    advanced_reward[3:5],
    torch.tensor([2.0, 2.0]),
)

print("eereach:", reach_reward)

distance_reward = ee_reach_reward(
    distance=torch.tensor([0.1, 0.2, 0.3]),
    target_position_e=target[:3],
    torso_position_e=torso[:3],
    body_velocity_w=torch.zeros(3, 3),
    projected_gravity_b=gravity[:3],
    ball_position_x_e=torch.ones(3),
    ball_velocity_x_w=torch.full((3,), -4.0),
    launch_velocity_x_w=torch.full((3,), -4.0),
    region=torch.zeros(3, dtype=torch.long),
)

assert distance_reward[0] > distance_reward[1]
assert distance_reward[1] > distance_reward[2]

success_reward = target_success_reward(
    distance=torch.tensor([0.1, 0.1, 0.15, 0.2]),
    success_flag=torch.tensor([False, True, True, True]),
)

torch.testing.assert_close(
    success_reward,
    torch.tensor([1.0, 2.0, 0.0, 0.0]),
)

print("success:", success_reward)

stop_flag = torch.tensor([False, True, False, False, False])
success_flag = torch.tensor([False, True, False, False, False])

old_stop = stop_flag.clone()
old_success = success_flag.clone()

stop_inputs = dict(
    ball_position_x_e=torch.tensor([0.5, 0.5, -0.1, 0.5, 0.0]),
    ball_velocity_x_w=torch.tensor([-1.0, -1.0, -1.0, -2.0, -1.0]),
    launch_velocity_x_w=torch.full((5,), -4.0),
)

stop_reward, next_stop, next_success = stopball_transition(
    **stop_inputs,
    stop_flag=stop_flag,
    success_flag=success_flag,
)

torch.testing.assert_close(
    stop_reward,
    torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]),
)
torch.testing.assert_close(
    next_stop,
    torch.tensor([True, True, True, False, False]),
)
torch.testing.assert_close(
    next_success,
    torch.tensor([True, True, False, False, False]),
)

# 输入没有被原地修改。
assert torch.equal(stop_flag, old_stop)
assert torch.equal(success_flag, old_success)

# 提交新标记后，相同事件不再给奖。
repeated_reward, _, _ = stopball_transition(
    **stop_inputs,
    stop_flag=next_stop,
    success_flag=next_success,
)
torch.testing.assert_close(repeated_reward, torch.zeros(5))

print("stopball:", stop_reward)

default_position = torch.full((3, 4), 0.2)
joint_position = default_position.clone()

joint_position[0] += 1.0
joint_position[2, 0] += 0.5
joint_position[2, 1] -= 0.5

upper_reward = post_upper_joint_position_exp(
    joint_position=joint_position,
    default_joint_position=default_position,
    recovery_mask=torch.tensor([False, True, True]),
)

torch.testing.assert_close(
    upper_reward,
    torch.tensor([0.0, 1.0, math.exp(-0.5)]),
)

print("postupperdofpos:", upper_reward)
print("最后四项奖励计算核心检查通过")