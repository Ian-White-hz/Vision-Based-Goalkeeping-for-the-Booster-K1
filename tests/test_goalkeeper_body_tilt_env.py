"""Real K1 whole-episode tilt, recovery coexistence and partial reset."""

import math

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import booster_k1_goalkeeper_amp_env_cfg
from booster_mjlab.tasks.goalkeeper.mdp.rewards import post_recovery_from_env


torch.set_num_threads(1)
cfg = booster_k1_goalkeeper_amp_env_cfg(num_envs=3)
env = ManagerBasedRlEnv(cfg, device="cpu")
try:
    obs, _ = env.reset(seed=42)
    assert {k: obs[k].shape[1] for k in ("actor", "critic", "amp")} == {
        "actor": 92, "critic": 95, "amp": 22,
    }
    manager = env.reward_manager
    term = manager.get_term_cfg("body_tilt")
    assert "pre_recovery_body_tilt" not in manager.active_terms
    assert term.weight == -5.0
    assert manager.get_term_cfg("fall").weight == -5.0
    assert manager.get_term_cfg("body_intercept").params["reach_threshold"] == .05
    robot = env.scene["robot"]
    root = torch.cat((robot.data.root_link_pose_w, robot.data.root_link_vel_w), -1).clone()
    half = math.radians(40.) / 2
    root[:, 3:7] = torch.tensor([math.cos(half), 0., math.sin(half), 0.])
    root[:, 2] += .4
    root[:, 7:] = 0.
    robot.write_root_state_to_sim(root)
    ball = torch.zeros(3, 13)
    ball[:, :3] = env.scene.env_origins + torch.tensor([1., 0., 1.2])
    ball[1, 0] = env.scene.env_origins[1, 0] - .1  # Passed the robot.
    ball[:, 3] = 1.
    ball[:, 7] = -1.2
    env.scene["ball"].write_root_state_to_sim(ball)
    env._full_goal_shots["velocity"][:] = ball[:, 7:10]
    env.sim.forward()
    env.scene.update(dt=0.)
    env._goalkeeper_outcome_state.shot_resolved[2] = True

    # Lane 0: approaching; lane 1: behind; lane 2: already resolved.
    expected = torch.tensor([.25, .25, .25])
    for _ in range(2):
        torch.testing.assert_close(term.func(env, **term.params), expected)
    recovery = post_recovery_from_env(env, kind="orientation")
    assert recovery[0] == 0 and (recovery[1:] > 0).all()
    # A successful-contact proxy must not switch off the balance cost either.
    env._goalkeeper_shot_state.success_flag[0] = True
    torch.testing.assert_close(term.func(env, **term.params), expected)
    env._goalkeeper_shot_state.success_flag[0] = False

    index = manager.active_terms.index("body_tilt")
    # RewardManager owns integration: no event-style 1/dt compensation.
    for dt in (.01, .02, .04):
        manager.compute(dt)
        contribution = manager._step_reward[:, index] * dt
        torch.testing.assert_close(contribution, -5. * expected * dt)
        torch.testing.assert_close(contribution / dt, torch.full((3,), -1.25))

    env._reset_idx(torch.tensor([0], device=env.device))
    env.sim.forward()
    env.scene.update(dt=0.)
    torch.testing.assert_close(term.func(env, **term.params), torch.tensor([0., .25, .25]))
    # Episode summaries are emitted on reset, not on an ordinary running step.
    env.episode_length_buf[0] = env.max_episode_length
    _, task, _, _, extras = env.step(torch.zeros(3, env.action_manager.total_action_dim))
    assert torch.isfinite(task).all()
    assert "Episode_Reward/body_tilt" in extras["log"]
    assert "Episode_Reward/pre_recovery_body_tilt" not in extras["log"]
    print("Real K1: whole-episode tilt, recovery coexistence, dt scaling, partial reset and env.step passed")
finally:
    env.close()
