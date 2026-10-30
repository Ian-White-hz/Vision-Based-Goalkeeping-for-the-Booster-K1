import torch

from booster_mjlab.tasks.goalkeeper.mdp.outcomes import (
    goal_crossing_mask,
    GoalkeeperOutcomeState
)


previous = torch.tensor([
    [-0.70, 0.00, 0.20],  # 正常进球。
    [-0.70, 0.00, 0.20],  # 当前只有一部分球过线。
    [-0.70, 1.30, 0.20],  # 从侧面射偏。
    [-0.70, 0.00, 1.20],  # 从横梁上方经过。
    [-1.00, 0.00, 0.20],  # 上一步已经完整过线。
    [-1.00, 0.00, 0.20],  # 从门后向门前运动。
    [-0.70, 0.00, 0.20],  # 过线时在门内，终点已移动到侧面。
])

current = torch.tensor([
    [-1.00, 0.00, 0.20],
    [-0.85, 0.00, 0.20],
    [-1.00, 1.30, 0.20],
    [-1.00, 0.00, 1.20],
    [-1.20, 0.00, 0.20],
    [-0.70, 0.00, 0.20],
    [-1.00, 1.40, 0.20],
])

result = goal_crossing_mask(
    previous,
    current,
    goal_x=-0.8,
    goal_width=2.4,
    goal_height=1.0,
    ball_radius=0.11,
)

expected = torch.tensor([
    True,
    False,
    False,
    False,
    False,
    False,
    True,
])

assert result.shape == (7,)
assert result.dtype == torch.bool
torch.testing.assert_close(result, expected)

print("进球判定:", result)
print("进球几何判定检查通过")

state = GoalkeeperOutcomeState(
    num_envs=2,
    device="cpu",
    goal_x=-0.8,
    goal_width=2.4,
    goal_height=1.0,
    ball_radius=0.11,
    ball_x_bounds=(-2.0, 5.0),
    ball_y_limit=3.0,
)

initial_position = torch.tensor([
    [-0.70, 0.0, 0.20],
    [-0.70, 0.0, 0.20],
])

state.reset(initial_position)

position = torch.tensor([
    [-1.00, 0.0, 0.20],
    [-0.85, 0.0, 0.20],
])

# 第 10 步，只有环境 0 完整过线。
state.update(position, step=10)

assert state.goal_conceded.tolist() == [True, False]
assert state.goal_conceded_step.tolist() == [10, -1]

# 重复读取同一位置，不修改首次失球时间。
state.update(position, step=10)

assert state.goal_conceded.tolist() == [True, False]
assert state.goal_conceded_step.tolist() == [10, -1]

# 第 11 步，环境 0 的球返回门前，环境 1 失球。
position = torch.tensor([
    [-0.60, 0.0, 0.20],
    [-1.00, 0.0, 0.20],
])

state.update(position, step=11)

assert state.goal_conceded.tolist() == [True, True]
assert state.goal_conceded_step.tolist() == [10, 11]

# 只重置环境 0，环境 1 的记录和历史位置都应保留。
previous_env1 = state.previous_ball_position_e[1].clone()

state.reset(
    initial_position,
    env_ids=torch.tensor([0]),
)

assert state.goal_conceded.tolist() == [False, True]
assert state.goal_conceded_step.tolist() == [-1, 11]

torch.testing.assert_close(
    state.previous_ball_position_e[0],
    initial_position[0],
)
torch.testing.assert_close(
    state.previous_ball_position_e[1],
    previous_env1,
)

print("失球锁存、首次发生时间、局部 reset 检查通过")


outcome = GoalkeeperOutcomeState(
    num_envs=3,
    device="cpu",
    goal_x=-0.8,
    goal_width=2.4,
    goal_height=1.0,
    ball_radius=0.11,
    ball_x_bounds=(-2.0, 5.0),
    ball_y_limit=3.0,
)

start = torch.tensor([
    [-0.70, 0.0, 0.20],
    [-0.70, 0.0, 0.20],
    [-0.70, 0.0, 0.20],
])
outcome.reset(start)

position = torch.tensor([
    [-1.00, 0.0, 0.20],  # 进球。
    [0.00, 3.2, 0.20],   # 出界。
    [1.00, 1.4, 0.20],   # 在球门侧面，但没有出界。
])
outcome.update(position, step=20)

assert outcome.goal_conceded.tolist() == [True, False, False]
assert outcome.ball_out_of_bounds.tolist() == [False, True, False]
assert outcome.shot_resolved.tolist() == [True, True, False]
assert outcome.shot_resolved_step.tolist() == [20, 20, -1]

position = torch.tensor([
    [-2.20, 0.0, 0.20],  # 先前已进球，不再改记为出界。
    [-1.00, 0.0, 0.20],  # 先前已出界，不再改记为进球。
    [1.00, 3.05, 0.20],  # 球仍有一部分留在有效区域。
])
outcome.update(position, step=21)

assert outcome.goal_conceded.tolist() == [True, False, False]
assert outcome.ball_out_of_bounds.tolist() == [False, True, False]
assert outcome.shot_resolved_step.tolist() == [20, 20, -1]

# 只重置出界的环境，其他环境结果应保持。
outcome.reset(start, env_ids=torch.tensor([1]))

assert outcome.goal_conceded.tolist() == [True, False, False]
assert outcome.ball_out_of_bounds.tolist() == [False, False, False]
assert outcome.shot_resolved.tolist() == [True, False, False]
assert outcome.shot_resolved_step.tolist() == [20, -1, -1]

print("出界、射门结果锁存和局部重置检查通过")