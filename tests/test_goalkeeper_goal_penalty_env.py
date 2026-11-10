"""Real environment goal cost, prior stop flags, episode logs and auto-reset."""

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.amp.runners.reward_mixing import mix_amp_rewards
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_env_cfg,
)
from booster_mjlab.tasks.goalkeeper.mdp.metrics import (
    category_save,
    category_velocity_change,
)
from booster_mjlab.tasks.goalkeeper.mdp.rewards import goal_conceded_cost
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings


torch.set_num_threads(1)
cfg = booster_k1_goalkeeper_env_cfg(
    num_envs=2, settings=FullGoalSettings(recovery_duration_s=0.06)
)
env = ManagerBasedRlEnv(cfg, device="cpu")
try:
    env.reset(seed=42)
    assert env.scene.env_origins.abs().any(), "Exercise environment-local coordinates"
    outcome = env._goalkeeper_outcome_state
    proxy = env._goalkeeper_shot_state
    region = int(env._full_goal_shots["region"][0])
    motion_name = (
        "lefthand",
        "righthand",
        "leftjump",
        "rightjump",
        "leftstep",
        "rightstep",
    )[region]

    # Reachable regression state: a previous contact/velocity event was recorded,
    # but the current ball trajectory goes on to concede. Preserve row 1 history
    # to verify row 0 auto-reset cannot erase another environment's state.
    outcome.robot_contact_seen[:] = True
    proxy.success_flag[0] = True
    proxy.stop_flag[0] = True
    ball_state = torch.zeros((2, 13))
    ball_state[:, 3] = 1
    ball_state[:, :3] = (
        torch.tensor([[-1.0, 0.0, 0.2], [-1.0, 1.6, 0.2]]) + env.scene.env_origins
    )
    env.scene["ball"].write_root_state_to_sim(ball_state)
    env.sim.forward()
    env.scene.update(dt=0.0)
    action = torch.zeros((2, env.action_manager.total_action_dim))
    _, task, done, truncated, _ = env.step(action)
    assert not done.any() and not truncated.any()
    assert outcome.goal_conceded.tolist() == [True, False]
    torch.testing.assert_close(category_save(env, region), torch.zeros(2))
    assert category_velocity_change(env, region)[0].item() == 1.0

    raw = goal_conceded_cost(env)
    torch.testing.assert_close(raw, torch.tensor([50.0, 0.0]))
    goal_term = env.reward_manager.get_term_cfg("goal_conceded")
    task_penalty = raw * goal_term.weight * env.step_dt
    torch.testing.assert_close(task_penalty, torch.tensor([-20.0, 0.0]))
    index = env.reward_manager.active_terms.index("goal_conceded")
    actual = env.reward_manager._step_reward[:, index] * env.step_dt
    torch.testing.assert_close(actual, task_penalty)
    total, _, _ = mix_amp_rewards(
        task,
        torch.zeros(2),
        style_weight=0.4,
        step_dt=env.step_dt,
        scale_rewards_by_dt=True,
    )
    without_goal, _, _ = mix_amp_rewards(
        task - task_penalty,
        torch.zeros(2),
        style_weight=0.4,
        step_dt=env.step_dt,
        scale_rewards_by_dt=True,
    )
    torch.testing.assert_close(total - without_goal, torch.tensor([-12.0, 0.0]))

    # A second read cannot consume the event; later steps must not repeat it.
    torch.testing.assert_close(goal_conceded_cost(env), raw)
    for _ in range(2):
        _, _, done, truncated, _ = env.step(action)
        assert not done.any() and not truncated.any()
        assert not goal_conceded_cost(env).any()

    _, _, done, truncated, extras = env.step(action)
    assert done.tolist() == [True, False] and not truncated.any()
    log = extras["log"]
    assert log[f"Episode_Metrics/Goalkeeper/{motion_name}/goal"].item() == 1.0
    assert log[f"Episode_Metrics/Goalkeeper/{motion_name}/save"].item() == 0.0
    assert (
        log[f"Episode_Metrics/Goalkeeper/{motion_name}/velocity_change"].item() == 1.0
    )
    assert not outcome.goal_conceded[0] and not outcome.robot_contact_seen[0]
    assert outcome.robot_contact_seen[1]
    assert outcome.goal_conceded_step[0].item() == -1
    assert outcome.safe_stop_steps[0].item() == 0
    assert not goal_conceded_cost(env).any()
    print("真实双环境：先停球代理后失球，goal=-20 / mixed=-12，仅惩罚一次")
    print("最终 save=0、velocity_change=1、goal=1，自动局部 reset 和日志检查通过")
finally:
    env.close()
