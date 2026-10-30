from types import SimpleNamespace

import torch

from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    get_recovery_mask,
    compute_rolling_shot_target,
    get_hand_target_distance
)

from mjlab.managers import SceneEntityCfg

class FakeScene(dict):
    pass


origins = torch.zeros(6, 3)
origins[:, 0] = torch.tensor([
    0.0, 10.0, -10.0, 20.0, -20.0, 30.0,
])

ball_position_w = origins.clone()
ball_position_w[:, 0] += torch.tensor([
    1.0, -0.1, 1.0, -0.1, 0.0, 1.0,
])

ball_velocity_w = torch.zeros(6, 3)
ball_velocity_w[:, 0] = torch.tensor([
    -3.0, -3.0, -0.9, -0.9, -3.0, -1.0,
])

scene = FakeScene(
    ball=SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=ball_position_w,
            root_link_lin_vel_w=ball_velocity_w,
        ),
    ),
)
scene.env_origins = origins

env = SimpleNamespace(
    scene=scene,
    _full_goal_shots={
        "velocity": torch.tensor([[-3.0, 0.0]]).repeat(6, 1),
    },
    _goalkeeper_outcome_state=SimpleNamespace(
        shot_resolved=torch.zeros(6, dtype=torch.bool),
    ),
)

mask = get_recovery_mask(env)

expected = torch.tensor([
    False,  # 球在前方，速度没变。
    True,   # 球已经经过 x=0。
    True,   # vx 增加了 2.1。
    True,   # 两个条件都满足。
    False,  # x 恰好等于 0，不满足严格小于。
    False,  # vx 恰好增加 2，不满足严格大于。
])

assert mask.shape == (6,)
assert mask.dtype == torch.bool
torch.testing.assert_close(mask, expected)

# 确认它重新读取当前数据，而不是锁存之前的 True。
ball_position_w[1, 0] = origins[1, 0] + 1.0
assert not get_recovery_mask(env)[1].item()

print("recovery_mask:", mask)
print("恢复阶段判断检查通过")

target_position_e, region = compute_rolling_shot_target(
    start_x=torch.tensor([2.1, 2.1, 2.1]),
    start_y=torch.tensor([1.0, -1.0, 0.0]),
    goal_y=torch.tensor([0.0, 0.0, 0.0]),
    goal_x=-1.9,
    ball_radius=0.11,
)

expected_target = torch.tensor([
    [0.1, 0.5, 0.11],
    [0.1, -0.5, 0.11],
    [0.1, 0.0, 0.11],
])

torch.testing.assert_close(
    target_position_e,
    expected_target,
    atol=1e-6,
    rtol=1e-5,
)
torch.testing.assert_close(
    region,
    torch.tensor([4, 5, 4]),
)
assert region.dtype == torch.long

print("扑救目标:", target_position_e)
print("区域编号:", region)
print("滚地球目标与区域计算通过")

hand_origins = torch.zeros(6, 3)
hand_origins[:, 0] = torch.arange(6) * 10.0

targets = torch.tensor([[0.1, 0.0, 0.5]]).repeat(6, 1)

# 故意设置为：右手、无关 site、左手。
site_position_e = torch.zeros(6, 3, 3)
site_position_e[:, 0] = targets + torch.tensor([0.0, -2.0, 0.0])
site_position_e[:, 1] = 100.0
site_position_e[:, 2] = targets + torch.tensor([0.0, 1.0, 0.0])

hand_scene = FakeScene(
    robot=SimpleNamespace(
        data=SimpleNamespace(
            site_pos_w=(
                site_position_e + hand_origins[:, None, :]
            ),
        ),
    ),
)
hand_scene.env_origins = hand_origins

hand_env = SimpleNamespace(
    scene=hand_scene,
    _full_goal_shots={
        "region": torch.arange(6),
        "target_position_e": targets,
    },
)

distance = get_hand_target_distance(
    hand_env,
    hand_cfg=SceneEntityCfg(
        "robot",
        site_ids=[2, 0],  # 指定左手、右手的顺序。
    ),
)

torch.testing.assert_close(
    distance,
    torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0]),
)

print("六类指定手到目标的距离:", distance)
print("手部选择与距离计算通过")