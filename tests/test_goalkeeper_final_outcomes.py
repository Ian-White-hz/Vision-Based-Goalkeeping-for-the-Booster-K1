"""Final saves, held stops, late goals, repeated reads and event reward scale."""

from types import SimpleNamespace as NS

import mjlab  # noqa: F401
import torch

from booster_mjlab.amp.runners.reward_mixing import mix_amp_rewards
from booster_mjlab.tasks.goalkeeper.mdp.metrics import (
    category_contact,
    category_goal,
    category_miss,
    category_save,
    category_unresolved,
)
from booster_mjlab.tasks.goalkeeper.mdp.outcomes import GoalkeeperOutcomeState
from booster_mjlab.tasks.goalkeeper.mdp.rewards import goal_conceded_cost
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings


def make_state(count):
    return GoalkeeperOutcomeState(
        count,
        "cpu",
        goal_x=-0.8,
        goal_width=2.4,
        goal_height=1.0,
        ball_radius=0.11,
        ball_x_bounds=(-2.0, 5.0),
        ball_y_limit=3.0,
    )


state = make_state(8)
start = torch.tensor([[0.5, 0.0, 0.112]]).repeat(8, 1)
start[0, 0] = -0.7
state.reset(start)
position = start.clone()
position[0, 0] = -1.0  # Contact never overrides an actual goal.
position[1:3, 1] = 3.2  # Touched clearance vs unassisted miss.
position[6, 0] = -0.2  # Stopping behind the intercept plane is not a held save.
velocity = torch.zeros(8, 3)
velocity[5, 0] = -1.0  # Touch/slowdown alone is not a save.
velocity[7, 0] = float("nan")
contact = torch.tensor([True, True, False, True, False, True, True, True])

for step in range(10, 30):
    inputs = dict(ball_velocity_w=velocity, robot_ball_contact=contact, step_dt=0.02)
    state.update(position, step, **inputs)
    count = state.safe_stop_steps.clone()
    for _ in range(4):
        state.update(position, step, **inputs)
        torch.testing.assert_close(state.safe_stop_steps, count)
    if step < 29:
        assert not state.saved[3]
    contact.zero_()  # Touch may be brief; held-stop confirmation uses its history.

assert state.saved.tolist() == [False, True, False, True, False, False, False, False]
assert state.goal_conceded.tolist() == [
    True,
    False,
    False,
    False,
    False,
    False,
    False,
    False,
]
assert state.safe_stop_steps[3].item() == 20
assert not state.shot_resolved[3], (
    "Held-stop metrics must not change episode termination"
)

# Moving again invalidates the held-stop status; a later goal must still be seen.
velocity[3, 0] = -1.0
state.update(
    position, 30, ball_velocity_w=velocity, robot_ball_contact=contact, step_dt=0.02
)
assert not state.saved[3]
position[3, 0] = -1.0
state.update(
    position, 31, ball_velocity_w=velocity, robot_ball_contact=contact, step_dt=0.02
)
assert state.goal_conceded[3] and not state.saved[3]
assert state.goal_conceded_step[3].item() == 31

# No late contact may relabel a previously unassisted miss as a clearance.
contact[2] = True
state.update(
    position, 32, ball_velocity_w=velocity, robot_ball_contact=contact, step_dt=0.02
)
assert state.robot_contact_seen[2] and not state.saved[2]

# Partial resets clear only those environments, including contact and timer state.
keep = state.robot_contact_seen[3:].clone()
state.reset(start, env_ids=torch.tensor([0, 1, 2]))
assert not state.robot_contact_seen[:3].any()
assert not state.saved[:3].any()
assert (state.last_update_step[:3] == -1).all()
torch.testing.assert_close(state.robot_contact_seen[3:], keep)
print("最终解围、安全停球、后续失球、重复读取和局部重置检查通过")


class Scene(dict):
    pass


def fake_env(count=4, dt=0.02, scaled=True):
    origins = torch.zeros(count, 3)
    origins[:, 0] = torch.arange(count) * 10.0
    pos = torch.tensor([[0.5, 0.0, 0.112]]).repeat(count, 1)
    history = torch.zeros(count, 1, 4, 3)
    scene = Scene(
        ball=NS(
            data=NS(
                root_link_pos_w=pos + origins, root_link_lin_vel_w=torch.zeros(count, 3)
            )
        ),
        robot_ball_contact=NS(
            data=NS(found=torch.zeros(count, 1), force_history=history)
        ),
    )
    scene.env_origins = origins
    result = NS(
        scene=scene,
        common_step_counter=10,
        step_dt=dt,
        cfg=NS(scale_rewards_by_dt=scaled),
        num_envs=count,
        device="cpu",
        _full_goal_shots={"region": torch.zeros(count, dtype=torch.long)},
        _goalkeeper_outcome_state=make_state(count),
    )
    result._goalkeeper_outcome_state.reset(pos)
    return result


env = fake_env()
env.scene["ball"].data.root_link_pos_w[:, 0] += torch.tensor([-1.5, 0.0, 0.0, 0.0])
env.scene["ball"].data.root_link_pos_w[1:3, 1] = 3.2
env.scene["robot_ball_contact"].data.force_history[:2, 0, 1, 0] = 1.0
# Exercise a metric-before-reward order and earlier-substep contact detection.
torch.testing.assert_close(category_save(env, 0), torch.tensor([0.0, 1.0, 0.0, 0.0]))
torch.testing.assert_close(category_contact(env, 0), torch.tensor([1.0, 1.0, 0.0, 0.0]))
torch.testing.assert_close(category_goal(env, 0), torch.tensor([1.0, 0.0, 0.0, 0.0]))
torch.testing.assert_close(category_miss(env, 0), torch.tensor([0.0, 0.0, 1.0, 0.0]))
torch.testing.assert_close(
    category_unresolved(env, 0), torch.tensor([0.0, 0.0, 0.0, 1.0])
)
for _ in range(3):
    torch.testing.assert_close(
        goal_conceded_cost(env), torch.tensor([50.0, 0.0, 0.0, 0.0])
    )
env.common_step_counter += 1
assert not goal_conceded_cost(env).any()
assert env._goalkeeper_outcome_state.goal_conceded[0]
print("触球、速度代理以外的最终状态分类及奖励读取顺序检查通过")


penalty = FullGoalSettings().goal_penalty
for dt in (0.01, 0.02, 0.04):
    for scaled in (True, False):
        env = fake_env(count=1, dt=dt, scaled=scaled)
        env.scene["ball"].data.root_link_pos_w[0, 0] = -1.0
        raw = goal_conceded_cost(env)
        task = -penalty * raw * (dt if scaled else 1.0)
        torch.testing.assert_close(task, torch.tensor([-20.0]))
        total, mixed_task, style = mix_amp_rewards(
            task,
            torch.zeros(1),
            style_weight=0.4,
            step_dt=dt,
            scale_rewards_by_dt=scaled,
        )
        torch.testing.assert_close(total, torch.tensor([-12.0]))
        torch.testing.assert_close(mixed_task, total)
        assert not style.any()
        env.common_step_counter += 1
        assert not goal_conceded_cost(env).any()

try:
    FullGoalSettings(goal_penalty=0.0)
except ValueError:
    pass
else:
    raise AssertionError("Goal penalty must be positive")
print("一次性失球任务贡献 -20、AMP 混合贡献 -12，跨 dt/缩放模式检查通过")
