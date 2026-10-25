import math

import torch

from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    feet_slippage_exp,
    excessive_foot_contact,
    jump_landing_reward,
)


velocity = torch.zeros(4, 2, 3)
velocity[0] = 3.0             # 无接触，速度不计入。
velocity[1, 1, 0] = 5.0      # 只有未接触的右脚运动。
velocity[2, 0, 0] = 0.1
velocity[2, 1, 1] = 0.2
velocity[3, 0, 0] = 10.0

force = torch.zeros(4, 2, 3)
force[1, 0, 2] = 10.0
force[2, :, 2] = 10.0
force[3, 0, 2] = 1.0         # 严格 > 1 才计入。

slippage_reward = feet_slippage_exp(velocity, force)

assert slippage_reward.shape == (4,)
assert torch.isfinite(slippage_reward).all()
torch.testing.assert_close(
    slippage_reward,
    torch.tensor([1.0, 1.0, math.exp(-3.0), 1.0]),
)

print("feet_slippage:", slippage_reward)

sharp_force = torch.zeros(4, 2, 3)
sharp_force[..., 2] = torch.tensor([
    [1000.0, 1000.0],
    [2000.0, 0.0],
    [2200.0, 0.0],
    [1200.0, 1200.0],
])

sharp_cost = excessive_foot_contact(sharp_force)

assert sharp_cost.shape == (4,)
assert torch.isfinite(sharp_cost).all()
torch.testing.assert_close(
    sharp_cost,
    torch.tensor([0.0, 0.0, 1.0, 1.0]),
)

print("penalize_sharpcontact:", sharp_cost)

landing_force = torch.zeros(6, 2, 3)
landing_force[..., 2] = torch.tensor([
    [10.0, 10.0],   # 没有跳起历史。
    [0.0, 0.0],     # 曾跳起，无脚接触。
    [10.0, 0.0],    # 曾跳起，单脚接触。
    [10.0, 10.0],   # 曾跳起，双脚接触。
    [10.0, 10.0],   # 非跳跃区域。
    [1.0, 1.0],     # 等于阈值：不属于 >1 或 <1。
])

has_in_air = torch.tensor([
    False, True, True, True, True, True,
])
jump_mask = torch.tensor([
    True, True, True, True, False, True,
])

landing_reward = jump_landing_reward(
    landing_force,
    has_in_air=has_in_air,
    jump_mask=jump_mask,
)

assert landing_reward.shape == (6,)
assert torch.isfinite(landing_reward).all()
torch.testing.assert_close(
    landing_reward,
    torch.tensor([0.0, 1.0, 0.0, 6.0, 0.0, 1.0]),
)

print("successland:", landing_reward)
print("三个接触相关奖励的计算检查通过")