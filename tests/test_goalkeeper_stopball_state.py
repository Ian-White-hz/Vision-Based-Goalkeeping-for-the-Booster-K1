from types import SimpleNamespace

import torch

from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    reset_shot_outcome_state,
    get_shot_outcome_state,
)


class FakeScene(dict):
    pass


position = torch.tensor([
    [0.5, 0.0, 0.1],
    [0.5, 0.0, 0.1],
    [-0.1, 0.0, 0.1],
])
velocity = torch.tensor([
    [-1.0, 0.0, 0.0],
    [-4.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
])

scene = FakeScene(
    ball=SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=position,
            root_link_lin_vel_w=velocity,
        ),
    ),
)
scene.env_origins = torch.zeros(3, 3)

env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    common_step_counter=0,
    scene=scene,
    _full_goal_shots={
        "velocity": torch.tensor([[-4.0, 0.0, 0.0]]).repeat(3, 1),
    },
)

reset_shot_outcome_state(env, None)
state = get_shot_outcome_state(env)

torch.testing.assert_close(
    state.stopball_reward, torch.tensor([1.0, 0.0, 0.0])
)
assert torch.equal(
    state.stop_flag, torch.tensor([True, False, True])
)
assert torch.equal(
    state.success_flag, torch.tensor([True, False, False])
)
assert not state.success_before_step.any()

# 同一步再次读取，返回相同缓存。
state = get_shot_outcome_state(env)
torch.testing.assert_close(
    state.stopball_reward, torch.tensor([1.0, 0.0, 0.0])
)

# 下一步，同一射门不再给奖。
env.common_step_counter += 1
state = get_shot_outcome_state(env)
torch.testing.assert_close(state.stopball_reward, torch.zeros(3))
assert state.success_before_step[0]

# 环境 1 此时首次满足条件。
env.common_step_counter += 1
velocity[1, 0] = -1.0
state = get_shot_outcome_state(env)
torch.testing.assert_close(
    state.stopball_reward, torch.tensor([0.0, 1.0, 0.0])
)

# 同一步重新发射环境 0，不影响环境 1 的缓存。
reset_shot_outcome_state(env, torch.tensor([0]))
velocity[0, 0] = -4.0
state = get_shot_outcome_state(env)

torch.testing.assert_close(
    state.stopball_reward, torch.tensor([0.0, 1.0, 0.0])
)
assert not state.stop_flag[0]
assert state.success_flag[1]

print("stopball 状态、缓存和局部 reset 检查通过")