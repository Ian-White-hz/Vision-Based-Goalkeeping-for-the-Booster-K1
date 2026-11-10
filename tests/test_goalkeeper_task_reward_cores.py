import math

import mjlab
import torch

from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    intercept_proximity_reward,
    target_success_reward,
    stopball_transition,
    post_upper_joint_position_exp,
)


# 新接近奖励：距离越小越好，阶段关闭后为零。
distance = torch.tensor([0.1, 0.2, 0.3, 0.1])
active = torch.tensor([True, True, True, False])

reach_reward = intercept_proximity_reward(
    distance,
    active,
)

torch.testing.assert_close(
    reach_reward,
    torch.tensor([
        0.6224593,
        0.5,
        0.3775407,
        0.0,
    ]),
)

assert reach_reward[0] > reach_reward[1] > reach_reward[2]

print("统一接近奖励:", reach_reward)


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