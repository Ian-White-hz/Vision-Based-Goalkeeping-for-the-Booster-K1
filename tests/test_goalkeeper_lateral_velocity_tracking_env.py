"""Real environment: air/rolling parity, stopping and lifecycle gating."""
import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import booster_k1_goalkeeper_amp_env_cfg

torch.set_num_threads(1)
env = ManagerBasedRlEnv(booster_k1_goalkeeper_amp_env_cfg(num_envs=2), device='cpu')
try:
    obs, _ = env.reset(seed=42)
    assert [obs[k].shape[1] for k in ('actor', 'critic', 'amp')] == [92, 95, 22]
    assert env.action_manager.total_action_dim == 22
    term = env.reward_manager.get_term_cfg('body_intercept')
    assert term.params['movement_weight'] == 2. and term.params['max_speed'] == .3
    assert term.params['lateral_reach'] == .3 and term.params['reach_threshold'] == .05
    assert env.reward_manager.get_term_cfg('fall').weight == -5
    assert env.reward_manager.get_term_cfg('goal_conceded').weight == -2
    robot = env.scene['robot']
    root = torch.cat((robot.data.root_link_pose_w, robot.data.root_link_vel_w), -1).clone()
    root[:, 7:] = 0
    root[:, 8] = .3
    robot.write_root_state_to_sim(root)
    ball = torch.zeros(2, 13)
    ball[:, :3] = env.scene.env_origins + torch.tensor([2., 0., .8])
    ball[:, 3] = 1
    ball[:, 7] = -1
    env.scene['ball'].write_root_state_to_sim(ball)
    env._full_goal_shots['velocity'][:] = ball[:, 7:10]
    env._full_goal_shots['target_position_e'][:] = torch.tensor([.1, .8, .5])
    env._full_goal_shots['is_rolling'][:] = torch.tensor([False, True])

    def refresh():
        env.sim.forward()
        env.scene.update(dt=0.)

    def read():
        return term.func(env, **term.params).clone()

    refresh()
    positive = read()
    assert (positive > 0).all()
    torch.testing.assert_close(positive[0], positive[1])  # Same geometry, different shot types.
    torch.testing.assert_close(read(), positive)
    for dt in (.01, .02, .04):
        env.reward_manager.compute(dt=dt)
        i = env.reward_manager.active_terms.index('body_intercept')
        actual = env.reward_manager._step_reward[:, i] * dt * .6
        torch.testing.assert_close(actual, positive * 10 * dt * .6)

    # Corridor stopping is still active, even though movement command is zero.
    env._full_goal_shots['target_position_e'][:, 1] = .2
    assert (read() < 0).all()
    env._full_goal_shots['target_position_e'][:, 1] = .8
    # Blocker at target also commands a stop. Root has nonzero field y speed.
    point = robot.data.site_pos_w[:, term.params['blocker_cfg'].site_ids[0]] - env.scene.env_origins
    env._full_goal_shots['target_position_e'][:] = point
    assert (read() < 0).all()
    env._full_goal_shots['target_position_e'][:] = torch.tensor([.1, .8, .5])

    # Success proxy, stopped-shot flag and final outcome each close the adapter.
    for obj, attr in [(env._goalkeeper_shot_state, 'success_flag'),
                      (env._goalkeeper_shot_state, 'stop_flag'),
                      (env._goalkeeper_outcome_state, 'shot_resolved')]:
        getattr(obj, attr)[0] = True
        assert read()[0] == 0 and read()[1] > 0
        getattr(obj, attr)[0] = False
    for position, vx in [(2., 1.), (0., -1.)]:
        ball[0, 0] = env.scene.env_origins[0, 0] + position
        ball[0, 7] = vx
        env.scene['ball'].write_root_state_to_sim(ball)
        refresh()
        assert read()[0] == 0 and read()[1] > 0
    before = read()[1]
    env.reset(env_ids=torch.tensor([0]))
    torch.testing.assert_close(read()[1], before)
    assert torch.isfinite(read()).all()
    env.step(torch.zeros(2, 22))
    print('PASS: air/rolling parity, corridor/coverage braking, active flags, dt/mixing, partial reset, step')
finally:
    env.close()
