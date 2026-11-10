"""Real K1: arm motion cannot replace root progress; arrival coverage settles."""
import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from booster_mjlab.amp.runners.reward_mixing import mix_amp_rewards
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import booster_k1_goalkeeper_amp_env_cfg
from booster_mjlab.tasks.goalkeeper.mdp.interception import interception_features

torch.set_num_threads(1)
env = ManagerBasedRlEnv(booster_k1_goalkeeper_amp_env_cfg(num_envs=2), device="cpu")
try:
    obs, _ = env.reset(seed=42)
    assert {k: obs[k].shape[1] for k in ("actor", "critic", "amp")} == {"actor": 92, "critic": 95, "amp": 22}
    assert env.action_manager.total_action_dim == 22
    reach = env.reward_manager.get_term_cfg("eereach")
    body = env.reward_manager.get_term_cfg("body_intercept")
    success = env.reward_manager.get_term_cfg("success")
    assert reach.params["guidance"] == success.params["guidance"] == body.params["guidance"]
    assert reach.params["guidance"] is not success.params["guidance"]
    robot, ball = env.scene["robot"], env.scene["ball"]
    origin = env.scene.env_origins
    assert origin.abs().any()
    root = torch.cat((robot.data.root_link_pose_w, robot.data.root_link_vel_w), -1).clone()
    root[:, 7:] = 0
    ball_state = torch.zeros(2, 13)
    ball_state[:, :3] = origin + torch.tensor([[2., 0., 1.2]])
    ball_state[:, 3] = 1
    ball_state[:, 7] = -4
    env._full_goal_shots["velocity"][:] = ball_state[:, 7:10]
    env._full_goal_shots["region"][:] = torch.tensor([0, 1])
    env._full_goal_shots["is_rolling"][:] = False
    target = env._full_goal_shots["target_position_e"]
    target[:] = torch.tensor([[.1, .8, .5], [.1, -.8, .5]])
    def refresh():
        env.sim.forward()
        env.scene.update(dt=0.)
    def raw(term):
        return term.func(env, **term.params).clone()
    def arms(speed):
        v = torch.zeros_like(robot.data.joint_vel)
        for name in ("Left_Shoulder_Pitch", "Right_Shoulder_Pitch"):
            v[:, robot.joint_names.index(name)] = speed
        robot.write_joint_state_to_sim(robot.data.joint_pos.clone(), v)
        refresh()
    robot.write_root_state_to_sim(root)
    ball.write_root_state_to_sim(ball_state)
    refresh()
    arms(0.)
    torch.testing.assert_close(raw(body), torch.zeros(2))
    initial_reach = raw(reach)
    arms(8.)
    torch.testing.assert_close(raw(body), torch.zeros(2))
    torch.testing.assert_close(raw(reach), initial_reach)
    arms(0.)
    root[:, 8] = .5
    robot.write_root_state_to_sim(root)
    refresh()
    assert raw(body)[0] > 0 and raw(body)[1] < 0
    root[:, 8] = -.5
    robot.write_root_state_to_sim(root)
    refresh()
    assert raw(body)[0] < 0 and raw(body)[1] > 0
    # Near arrival, a fast sweep earns less than identical settled geometry.
    root[:, 8] = 0
    robot.write_root_state_to_sim(root)
    refresh()
    hand_ids = [robot.site_names.index("left_hand"), robot.site_names.index("right_hand")]
    points = robot.data.site_pos_w[torch.arange(2), hand_ids] - origin
    target[:] = points + torch.tensor([[0., .1, 0.]])
    nominal = target.clone()
    ball_state[:, :3] = target + origin
    ball_state[:, 0] += .08
    ball_state[:, 2] += .5 * 9.81 * .02**2
    ball.write_root_state_to_sim(ball_state)
    refresh()
    slow_reach, slow_success = raw(reach), raw(success)
    assert (slow_success > .8).all()
    arms(8.)
    assert (raw(reach) < slow_reach).all()
    assert (raw(success) < slow_success).all()
    torch.testing.assert_close(raw(body), torch.zeros(2))
    arms(0.)
    torch.testing.assert_close(raw(reach), slow_reach)
    torch.testing.assert_close(target, nominal)
    f = interception_features(env, reach.params["blocker_cfg"], guidance=reach.params["guidance"])
    torch.testing.assert_close(f["tau"], torch.full((2,), .02), atol=1e-6, rtol=1e-5)
    assert (f["target"][:, 0] == nominal[:, 0]).all()
    ball_state[:, 1] += .1
    ball.write_root_state_to_sim(ball_state)
    refresh()
    corrected = interception_features(env, reach.params["blocker_cfg"], guidance=reach.params["guidance"])
    assert corrected["active"].all()
    assert (corrected["target"][:, 1] > nominal[:, 1]).all()
    torch.testing.assert_close(corrected["target"][:, 0], nominal[:, 0])
    torch.testing.assert_close(target, nominal)
    ball_state[:, 1] -= .1
    ball.write_root_state_to_sim(ball_state)
    refresh()
    for term, name in ((reach, "eereach"), (success, "success"), (body, "body_intercept")):
        expected = raw(term) * term.weight
        env.reward_manager.compute(dt=env.step_dt)
        index = env.reward_manager.active_terms.index(name)
        torch.testing.assert_close(env.reward_manager._step_reward[:, index], expected)
    base = env.reward_manager.compute(dt=.02).clone()
    for dt in (.01, .04):
        total = env.reward_manager.compute(dt=dt).clone()
        torch.testing.assert_close(total, base * dt / .02, atol=1e-5, rtol=1e-5)
        mixed, task, style = mix_amp_rewards(total, torch.full((2,), .75), style_weight=.4,
                                            step_dt=dt, scale_rewards_by_dt=True)
        torch.testing.assert_close(task, .6 * total)
        torch.testing.assert_close(mixed, task + style)
    ball_state[0, 7] = 4
    ball.write_root_state_to_sim(ball_state)
    refresh()
    for term in (reach, success, body):
        assert raw(term)[0] == 0
    before = raw(reach)[1]
    env.reset(env_ids=torch.tensor([0]))
    torch.testing.assert_close(raw(reach)[1], before)
    assert all(torch.isfinite(raw(t)).all() for t in (reach, success, body))
    print("Real K1: root-only direction, sweep vs hold, immutable plane, dt/mixing and partial reset passed")
finally:
    env.close()
