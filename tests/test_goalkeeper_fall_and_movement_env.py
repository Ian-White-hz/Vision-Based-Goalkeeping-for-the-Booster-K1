"""Real terminal reward before auto-reset: fall, timeout and recovery are distinct."""

import math

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.amp.runners.reward_mixing import mix_amp_rewards
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import booster_k1_goalkeeper_amp_env_cfg


torch.set_num_threads(1)
cfg = booster_k1_goalkeeper_amp_env_cfg(num_envs=3)
cfg.episode_length_s = .2
env = ManagerBasedRlEnv(cfg, device="cpu")
try:
    obs, _ = env.reset(seed=42)
    assert {k: obs[k].shape[1] for k in ("actor", "critic", "amp")} == {"actor": 92, "critic": 95, "amp": 22}
    assert env.action_manager.total_action_dim == 22
    manager = env.reward_manager
    reach = manager.get_term_cfg("eereach")
    body = manager.get_term_cfg("body_intercept")
    success = manager.get_term_cfg("success")
    fall = manager.get_term_cfg("fall")
    assert body.params["reach_threshold"] == .05
    assert body.params["lateral_reach"] == .3 and body.params["fade_distance"] == .3
    assert reach.params["reach_threshold"] == .2
    assert success.params["strict_threshold"] == .15
    assert body.weight == reach.weight == 10. and success.weight == 5. and fall.weight == -20.

    robot = env.scene["robot"]
    root = torch.cat((robot.data.root_link_pose_w, robot.data.root_link_vel_w), dim=-1).clone()
    angle = math.radians(75.) / 2
    root[0, 3:7] = torch.tensor([math.cos(angle), math.sin(angle), 0., 0.])
    root[0, 2] += .4  # Avoid a deep ground overlap in this terminal-pose probe.
    root[:, 7:] = 0
    robot.write_root_state_to_sim(root)
    ball = torch.zeros((3, 13))
    ball[:, :3] = env.scene.env_origins + torch.tensor([[1., 0., 1.2]])
    ball[:, 3] = 1.
    ball[:, 7] = -1.2
    env.scene["ball"].write_root_state_to_sim(ball)
    env._full_goal_shots["velocity"][:] = ball[:, 7:10]
    env.sim.forward()
    env.scene.update(dt=0.)

    # Lane 0 falls, lane 1 times out, lane 2 completes its recovery interval.
    env.episode_length_buf[1] = env.max_episode_length
    state = env._goalkeeper_outcome_state
    state.shot_resolved[2] = True
    state.shot_resolved_step[2] = env.common_step_counter - math.ceil(2. / env.step_dt) - 1
    action = torch.zeros((3, env.action_manager.total_action_dim))
    _, task, terminated, truncated, extras = env.step(action)
    assert terminated.tolist() == [True, False, True], terminated
    assert truncated.tolist() == [False, True, False], truncated
    index = manager.active_terms.index("fall")
    contribution = manager._step_reward[:, index] * env.step_dt
    torch.testing.assert_close(contribution, torch.tensor([-20., 0., 0.]))
    total, _, _ = mix_amp_rewards(task, torch.zeros(3), style_weight=.4,
                                 step_dt=env.step_dt, scale_rewards_by_dt=True)
    without, _, _ = mix_amp_rewards(task - contribution, torch.zeros(3), style_weight=.4,
                                   step_dt=env.step_dt, scale_rewards_by_dt=True)
    torch.testing.assert_close(total - without, torch.tensor([-12., 0., 0.]), atol=1e-6, rtol=1e-5)
    logged = extras["log"]["Episode_Reward/fall"]
    torch.testing.assert_close(logged, torch.tensor(-20. / 3 / cfg.episode_length_s))
    # Cached termination flags remain terminal after auto-reset in mjlab.
    # The reward must nevertheless read the reset pose and return zero.
    assert env.termination_manager.get_term("excessive_body_tilt")[0]
    assert not fall.func(env, **fall.params).any()
    _, _, terminated, truncated, _ = env.step(action)
    assert not terminated.any() and not truncated.any()
    assert not manager._step_reward[:, index].any()
    print("Real K1: fall/timeout/recovery task contributions [-20,0,0], mixed [-12,0,0]")
    print("Terminal episode log, auto-reset without stale cost, no next-step repeat; only body cutoff=5cm")
finally:
    env.close()
