"""Real multi-env pose/rate costs, partial reset and checkpoint input shapes."""

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_amp_env_cfg,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings


torch.set_num_threads(1)
settings = FullGoalSettings()
env = ManagerBasedRlEnv(booster_k1_goalkeeper_amp_env_cfg(num_envs=2), device="cpu")
try:
    obs, _ = env.reset(seed=42)
    assert obs["actor"].shape == (2, 92)
    assert obs["critic"].shape == (2, 95)
    assert obs["amp"].shape == (2, 22)
    assert env.scene.env_origins.abs().any()
    heading = env.reward_manager.get_term_cfg("heading_deviation")
    rate = env.reward_manager.get_term_cfg("yaw_rate")
    assert heading.weight == -settings.heading_penalty
    assert rate.weight == -settings.yaw_rate_penalty
    assert not heading.func(env, **heading.params).any()
    assert not rate.func(env, **rate.params).any()
    robot = env.scene["robot"]

    # Same yaw magnitude and opposite vertical angular speeds, in separate worlds.
    state = torch.cat(
        (robot.data.root_link_pose_w, robot.data.root_link_vel_w), dim=-1
    ).clone()
    angle = torch.deg2rad(torch.tensor([55.0, -55.0])) / 2
    state[:, 3:7] = torch.stack(
        (angle.cos(), angle * 0, angle * 0, angle.sin()), dim=-1
    )
    state[:, 7:] = 0
    state[:, 12] = torch.tensor([1.75, -1.75])
    robot.write_root_state_to_sim(state)
    env.sim.forward()
    env.scene.update(dt=0.0)
    torch.testing.assert_close(
        heading.func(env, **heading.params), torch.full((2,), 0.25)
    )
    torch.testing.assert_close(rate.func(env, **rate.params), torch.full((2,), 0.25))

    env.reward_manager.compute(dt=env.step_dt)
    terms = env.reward_manager.active_terms
    for name, expected in (("heading_deviation", -0.025), ("yaw_rate", -0.005)):
        value = env.reward_manager._step_reward[:, terms.index(name)] * env.step_dt
        torch.testing.assert_close(value, torch.full((2,), expected))

    # A partial reset restores only row 0; no reference from the previous
    # terminal pose is accidentally captured by a reset event.
    obs, _ = env.reset(env_ids=torch.tensor([0]))
    torch.testing.assert_close(
        heading.func(env, **heading.params), torch.tensor([0.0, 0.25])
    )
    torch.testing.assert_close(rate.func(env, **rate.params), torch.tensor([0.0, 0.25]))

    # A forward-facing lateral translation is free of both new costs.
    state[1, :3] = robot.data.default_root_state[1, :3] + env.scene.env_origins[1]
    state[1, 1] += 0.4
    state[1, 3:7] = robot.data.default_root_state[1, 3:7]
    state[1, 7:] = 0
    robot.write_root_state_to_sim(state[1:2], env_ids=torch.tensor([1]))
    env.sim.forward()
    env.scene.update(dt=0.0)
    assert not heading.func(env, **heading.params).any()
    assert not rate.func(env, **rate.params).any()
    print("真实环境：左右 55 度/±1.75rad/s 代价相同，task 每步 -0.025/-0.005")
    print("局部 reset、前向横移无新代价，Actor/Critic/AMP 维数保持 92/95/22")
finally:
    env.close()
