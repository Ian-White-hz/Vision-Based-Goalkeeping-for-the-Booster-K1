"""Slow-ball saves, friction rejection, reward caching and recovery routing."""

from types import SimpleNamespace as NS

import mjlab  # noqa: F401 -- initialize task imports before the reward modules
import torch

from booster_mjlab.tasks.goalkeeper.mdp.metrics import category_save
from booster_mjlab.tasks.goalkeeper.mdp.rewards import stopball_from_env
from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    get_recovery_mask,
    get_shot_outcome_state,
    reset_shot_outcome_state,
    stopball_transition,
)


# Stopped/reflected rolling balls, slight slowdown, friction-only stop,
# behind/exactly on the plane, unchanged ballistic rules, invalid launch/data.
velocity = torch.tensor(
    [0.0, 0.4, -0.38, 0.0, 0.0, 0.0, -1.0, -2.0, 0.0, 0.0, float("nan")]
)
launch = torch.tensor([-0.4] * 6 + [-4.0, -4.0, -0.4, 0.0, -0.4])
position = torch.full_like(velocity, 0.1)
position[4:6] = torch.tensor([-0.1, 0.0])
rolling = torch.tensor([True] * 6 + [False, False, False, True, True])
contact = torch.ones_like(rolling)
contact[3] = False
flags = torch.zeros_like(rolling)
inputs = dict(
    ball_position_x_e=position,
    ball_velocity_x_w=velocity,
    launch_velocity_x_w=launch,
    is_rolling=rolling,
    robot_ball_contact=contact,
)
reward, stopped, success = stopball_transition(
    **inputs,
    stop_flag=flags,
    success_flag=flags,
)
expected = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
torch.testing.assert_close(reward, expected)
assert torch.equal(success, expected.bool())
assert not flags.any(), "Pure transition must not mutate input flags"
again, _, _ = stopball_transition(
    **inputs,
    stop_flag=stopped,
    success_flag=success,
)
assert not again.any(), "One shot must not earn repeated stopball rewards"

for fraction in (0.0, -0.1, 1.1, float("nan")):
    try:
        stopball_transition(
            **inputs,
            stop_flag=flags,
            success_flag=flags,
            rolling_velocity_change_fraction=fraction,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid rolling fraction was accepted")

without_contact = dict(inputs)
without_contact.pop("robot_ball_contact")
try:
    stopball_transition(**without_contact, stop_flag=flags, success_flag=flags)
except ValueError:
    pass
else:
    raise AssertionError("Rolling saves must require explicit contact evidence")

print("Rolling stop/reflection, friction rejection and ballistic regression passed")


class FakeScene(dict):
    pass


# Only a middle substep had contact; the current found values are all zero.
# Distinct origins also exercise environment-local x coordinates.
origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
positions = origins + torch.tensor([0.1, 0.0, 0.11])
velocities = torch.zeros(3, 3)
velocities[1, 0] = 0.4
history = torch.zeros(3, 1, 4, 3)
history[:2, 0, 2, 0] = 1.0
found = torch.zeros(3, 1)
scene = FakeScene(
    ball=NS(data=NS(root_link_pos_w=positions, root_link_lin_vel_w=velocities)),
    robot_ball_contact=NS(data=NS(found=found, force_history=history)),
)
scene.env_origins = origins
env = NS(
    scene=scene,
    num_envs=3,
    device="cpu",
    common_step_counter=10,
    _full_goal_shots={
        "velocity": torch.tensor([[-0.4, 0.0, 0.0]]).repeat(3, 1),
        "is_rolling": torch.ones(3, dtype=torch.bool),
        "region": torch.tensor([4, 5, 4]),
    },
    _goalkeeper_outcome_state=NS(shot_resolved=torch.zeros(3, dtype=torch.bool)),
)
reset_shot_outcome_state(env)

# Recovery can be evaluated before reward/metrics without consuming the reward.
assert get_recovery_mask(env).tolist() == [True, True, False]
state = get_shot_outcome_state(env)
assert not state.success_before_step.any()
torch.testing.assert_close(stopball_from_env(env), torch.tensor([1.0, 1.0, 0.0]))
torch.testing.assert_close(stopball_from_env(env), torch.tensor([1.0, 1.0, 0.0]))
torch.testing.assert_close(category_save(env, 4), torch.tensor([1.0, 0.0, 0.0]))
torch.testing.assert_close(category_save(env, 5), torch.tensor([0.0, 1.0, 0.0]))

# A partial reset must not erase another environment's same-step reward.
reset_shot_outcome_state(env, torch.tensor([1]))
history[1] = 0.0
velocities[1, 0] = -0.4
torch.testing.assert_close(stopball_from_env(env), torch.tensor([1.0, 0.0, 0.0]))
assert state.success_flag.tolist() == [True, False, False]

# A slight contact is not enough; friction stopping the ball later is not a save.
env.common_step_counter += 1
history.zero_()
found[1] = 1.0
velocities[1, 0] = -0.38
assert not stopball_from_env(env).any()
assert state.success_before_step.tolist() == [True, False, False]
env.common_step_counter += 1
found.zero_()
velocities[1, 0] = 0.0
assert not stopball_from_env(env).any()
assert state.success_flag.tolist() == [True, False, False]
assert get_recovery_mask(env).tolist() == [True, False, False]

env.common_step_counter += 1
del scene["robot_ball_contact"]
try:
    get_shot_outcome_state(env)
except RuntimeError as error:
    assert "robot-ball contact sensor" in str(error)
else:
    raise AssertionError("Missing rolling-shot contact sensor must fail clearly")

print("Substep contact, save metrics, recovery, caching and partial reset passed")
