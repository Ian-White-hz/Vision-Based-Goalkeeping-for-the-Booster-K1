import math

import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_env_cfg,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import (
    FullGoalSettings,
)


def check_observations(observations):
    assert observations["actor"].shape == (1, 92)
    assert observations["critic"].shape == (1, 95)

    for value in observations.values():
        assert torch.isfinite(value).all()

cfg = booster_k1_goalkeeper_env_cfg(
    num_envs=1,
    settings=FullGoalSettings(
        recovery_duration_s=0.06,
    ),
)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

try:
    assert math.isclose(env.step_dt, 0.02)
    assert env.action_manager.total_action_dim == 22

    joint_action = env.action_manager.get_term("joint_pos")
    assert tuple(joint_action.target_names) == K1_JOINT_ORDER

    observations, _ = env.reset(seed=42)
    check_observations(observations)

    outcome = env._goalkeeper_outcome_state

    assert not outcome.shot_resolved.any()

    torch.testing.assert_close(
        outcome.previous_ball_position_e,
        env._full_goal_shots["start_position_e"],
    )

    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )

    # 少量物理推进，检查实际数据链路。
    for _ in range(5):
        observations, reward, terminated, truncated, _ = env.step(action)

        check_observations(observations)

        assert reward.shape == (1,)
        assert torch.isfinite(reward).all()
        assert not terminated.any()
        assert not truncated.any()

        robot = env.scene["robot"]
        ball = env.scene["ball"]

        for value in (
            robot.data.joint_pos,
            robot.data.joint_vel,
            robot.data.root_link_pos_w,
            robot.data.root_link_quat_w,
            ball.data.root_link_pos_w,
            ball.data.root_link_lin_vel_w,
        ):
            assert torch.isfinite(value).all()

    # 将球放到球门后，验证下一次 env.step 自动记录失球。
    ball_state = torch.zeros((1, 13), device=env.device)

    ball_state[:, :3] = (
        torch.tensor(
            [[-1.0, 0.0, 0.2]],
            device=env.device,
        )
        + env.scene.env_origins
    )
    ball_state[:, 3] = 1.0

    env.scene["ball"].write_root_state_to_sim(ball_state)

    observations, reward, terminated, truncated, _ = env.step(action)

    assert outcome.goal_conceded[0].item()
    assert outcome.shot_resolved[0].item()
    assert (
        outcome.shot_resolved_step[0].item()
        == env.common_step_counter
    )

    # 刚记录失球，尚未消耗恢复窗口。
    assert not terminated.any()
    assert not truncated.any()

    # 经过 1、2 个控制步，还不能结束。
    for _ in range(2):
        observations, reward, terminated, truncated, _ = env.step(action)

        assert not terminated.any()
        assert not truncated.any()
        assert outcome.shot_resolved[0].item()

    # 第 3 个控制步：达到 0.06 秒，任务终止并自动 reset。
    observations, reward, terminated, truncated, _ = env.step(action)

    assert terminated.all()
    assert not truncated.any()
    assert (env.episode_length_buf == 0).all()

    assert not outcome.shot_resolved.any()
    assert not outcome.goal_conceded.any()
    assert not outcome.ball_out_of_bounds.any()
    assert (outcome.shot_resolved_step == -1).all()

    check_observations(observations)

    print("恢复窗口结束及自动重置检查通过")
    
    # 仅推进计数器到超时前一步，验证超时和自动重置。
    env.episode_length_buf.fill_(env.max_episode_length - 1)

    observations, reward, terminated, truncated, _ = env.step(action)

    assert truncated.all()
    assert not terminated.any()
    assert torch.isfinite(reward).all()
    assert (env.episode_length_buf == 0).all()
    check_observations(observations)

    print("控制周期:", env.step_dt)
    print("Action 维数:", env.action_manager.total_action_dim)
    print("Actor shape:", tuple(observations["actor"].shape))
    print("Critic shape:", tuple(observations["critic"].shape))
    print("实际环境 reset、step、超时自动重置检查通过")

    
    assert not outcome.shot_resolved.any()
    assert not outcome.goal_conceded.any()
    assert not outcome.ball_out_of_bounds.any()
    assert (outcome.shot_resolved_step == -1).all()

finally:
    env.close()