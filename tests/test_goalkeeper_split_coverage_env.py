"""Real K1 compiled collision bounds, split guidance and no stopping penalty."""
import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import booster_k1_goalkeeper_amp_env_cfg
from booster_mjlab.tasks.goalkeeper.mdp.foot_coverage import rolling_coverage_from_env

torch.set_num_threads(1)
env = ManagerBasedRlEnv(booster_k1_goalkeeper_amp_env_cfg(num_envs=2), device='cpu')
try:
    obs, _ = env.reset(seed=42)
    assert [obs[k].shape[1] for k in ('actor', 'critic', 'amp')] == [92,95,22]
    term = env.reward_manager.get_term_cfg('body_intercept')
    robot = env.scene['robot']
    root = torch.cat((robot.data.root_link_pose_w, robot.data.root_link_vel_w), -1).clone()
    root[:, 7:] = 0
    root[:, 8] = .3
    robot.write_root_state_to_sim(root)
    ball = torch.zeros(2, 13)
    ball[:, :3] = env.scene.env_origins + torch.tensor([1., .6, .11])
    ball[:, 3] = 1
    ball[:, 7] = -1
    env._full_goal_shots['velocity'][:] = ball[:, 7:10]
    env._full_goal_shots['is_rolling'][:] = torch.tensor([False, True])
    env._full_goal_shots['target_position_e'][:] = torch.tensor([.1, .2, .11])

    def refresh():
        env.scene['ball'].write_root_state_to_sim(ball)
        env.sim.forward()
        env.scene.update(dt=0.)

    def raw():
        return term.func(env, **term.params).clone()

    def gaps():
        return rolling_coverage_from_env(env, term.params['foot_cfg'], 'ball', ball_radius=.11)

    refresh()
    assert (gaps() > 0).all()
    assert raw()[0] == 0 and raw()[1] > 0, 'Rolling must not inherit the 30cm corridor'
    torch.testing.assert_close(raw(), raw())
    for centers, halves in env._goalkeeper_foot_bounds.values():
        assert halves.shape == (2, 3) and (halves > 0).all()
        print('Compiled foot geom centers/half-sizes:', centers.tolist(), halves.tolist())
    # Feet cover the actual future path: no movement reward OR braking penalty.
    left = robot.data.body_link_pos_w[:, robot.body_names.index('left_foot_link')]
    ball[:, 1] = left[:, 1]
    refresh()
    assert (gaps() <= 0).all()
    assert raw()[1] == 0
    # Air guidance is independent of that foot-path coverage.
    env._full_goal_shots['target_position_e'][:] = torch.tensor([.1, .8, .6])
    assert raw()[0] > 0 and raw()[1] == 0
    ball[:, 1] = env.scene.env_origins[:, 1] + .6
    refresh()
    assert (raw() > 0).all()
    positive = raw()
    root[:, 8] = -.3
    robot.write_root_state_to_sim(root)
    refresh()
    torch.testing.assert_close(raw(), -positive)
    root[:, 8] = 0
    robot.write_root_state_to_sim(root)
    refresh()
    assert raw().eq(0).all()
    root[:, 8] = .3
    robot.write_root_state_to_sim(root)
    refresh()
    for dt in [.01,.02,.04]:
        expected = raw()*10*dt*.6
        env.reward_manager.compute(dt=dt)
        i=env.reward_manager.active_terms.index('body_intercept')
        torch.testing.assert_close(env.reward_manager._step_reward[:,i]*dt*.6, expected)
    for obj, attr in [(env._goalkeeper_shot_state,'success_flag'),
                      (env._goalkeeper_shot_state,'stop_flag'),
                      (env._goalkeeper_outcome_state,'shot_resolved')]:
        getattr(obj,attr)[1]=True
        assert raw()[1]==0
        getattr(obj,attr)[1]=False
    for x,vx in [(1.,1.),(0.,-1.)]:
        ball[1,0]=env.scene.env_origins[1,0]+x
        ball[1,7]=vx
        refresh()
        assert raw()[1]==0
    before=raw()[1]
    env.reset(env_ids=torch.tensor([0]))
    torch.testing.assert_close(raw()[1],before)
    _, reward, _, _, _ = env.step(torch.zeros(2,22))
    assert torch.isfinite(reward).all()
    assert env.reward_manager.get_term_cfg('fall').weight==-5
    assert env.reward_manager.get_term_cfg('goal_conceded').weight==-2
    print('PASS: mesh bounds, separate air/rolling gates, stationary zero, no braking, signs, dt and lifecycle')
finally:
    env.close()
