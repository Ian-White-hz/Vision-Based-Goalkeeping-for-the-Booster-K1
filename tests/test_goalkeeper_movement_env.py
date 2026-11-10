"""Real K1: signed geometry, same-state reward comparison, scaling and resets."""

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.amp.runners.reward_mixing import mix_amp_rewards
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import booster_k1_goalkeeper_amp_env_cfg
from booster_mjlab.tasks.goalkeeper.mdp.rewards import intercept_movement_reward
from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    get_blocker_positions_e,
    get_blocker_target_distance,
    get_blocker_target_kinematics,
)


def legacy_reach(env, blocker_cfg, **params):
    """Pre-change formula, only for a same-state comparison on active probes."""
    robot = env.scene[blocker_cfg.name]
    target = env._full_goal_shots["target_position_e"]
    torso_id = blocker_cfg.body_ids[0]
    torso = robot.data.body_link_pos_w[:, torso_id] - env.scene.env_origins
    velocity = robot.data.body_link_lin_vel_w[:, torso_id]
    lateral = (target[:, 1] - torso[:, 1]).clamp(-1, 1)
    lateral = torch.where(lateral.abs() < .3, 0., lateral)
    vertical = (torso[:, 2] - target[:, 2].clamp(.3, 1.2)).clamp(0, 1)
    alignment = 1 - (vertical + lateral.abs()) / 2
    region = env._full_goal_shots["region"]
    gain = torch.ones_like(lateral)
    gain = torch.where((region == 0) | (region == 4), 1 + 3 * velocity[:, 1].clamp(0, 3), gain)
    gain = torch.where((region == 1) | (region == 5), 1 - 3 * velocity[:, 1].clamp(-3, 0), gain)
    gain = torch.where((region == 2) | (region == 3), 1 + 3 * velocity[:, 2].clamp(0, 3), gain)
    proximity = torch.sigmoid(params["distance_sigma"] * (
        params["reach_threshold"] - get_blocker_target_distance(env, blocker_cfg)
    ))
    ball_x = env.scene["ball"].data.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0]
    upright = 1 - robot.data.projected_gravity_b[:, :2].square().sum(-1).clamp(0, 1)
    return torch.where(ball_x > 1.5, alignment, proximity * gain) * upright


torch.set_num_threads(1)
env = ManagerBasedRlEnv(booster_k1_goalkeeper_amp_env_cfg(num_envs=2), device="cpu")
try:
    obs, _ = env.reset(seed=42)
    assert {k: obs[k].shape[1] for k in ("actor", "critic", "amp")} == {
        "actor": 92, "critic": 95, "amp": 22,
    }
    assert env.action_manager.total_action_dim == 22
    term = env.reward_manager.get_term_cfg("eereach")
    blocker = term.params["blocker_cfg"]
    assert term.weight == 10
    assert env.scene.env_origins.abs().any()
    robot, ball = env.scene["robot"], env.scene["ball"]
    origins = env.scene.env_origins
    root = torch.cat((robot.data.root_link_pose_w, robot.data.root_link_vel_w), dim=-1).clone()
    root[:, 7:] = 0
    root[:, 8] = .5
    ball_state = torch.zeros(2, 13)
    ball_state[:, :3] = origins + torch.tensor([[2., 0., 1.2]])
    ball_state[:, 3] = 1
    ball_state[:, 7] = -1.2

    def refresh():
        env.sim.forward()
        env.scene.update(dt=0.)

    robot.write_root_state_to_sim(root)
    ball.write_root_state_to_sim(ball_state)
    refresh()
    # Keep launch metadata consistent, so the old velocity-change proxy does
    # not mistake this controlled ball state for a successful stop.
    env._full_goal_shots["velocity"][:] = ball_state[:, 7:10]
    points = get_blocker_positions_e(env, blocker)
    target = env._full_goal_shots["target_position_e"]
    target[:] = torch.tensor([[.1, 0., .5]])
    target[0, 1] = points[0, :, 1].max() + .6
    target[1, 1] = points[1, :, 1].min() - .6
    # Deliberately contradict the geometry with the class labels.
    env._full_goal_shots["region"][:] = torch.tensor([1, 0])
    distance, closing = get_blocker_target_kinematics(env, blocker)
    torch.testing.assert_close(robot.data.site_lin_vel_w[:, blocker.site_ids],
                               torch.tensor([[[0., .5, 0.]]]).expand(2, 2, 3))
    assert closing[0] > 0 and closing[1] < 0
    raw = term.func(env, **term.params)
    assert raw[0] > 0 and raw[1] < 0
    torch.testing.assert_close(raw, term.func(env, **term.params))  # idempotent
    env._full_goal_shots["region"][:] = torch.tensor([4, 5])
    torch.testing.assert_close(raw, term.func(env, **term.params))

    # Verify the actual site/body velocity projection by translating the real
    # articulated robot, including a nonzero world-z angular velocity.
    root[:, 12] = .6
    robot.write_root_state_to_sim(root)
    refresh()
    site_pos = robot.data.site_pos_w[:, blocker.site_ids]
    torso_pos = robot.data.root_link_pos_w[:, None]
    offset = site_pos - torso_pos
    expected_site_v = torch.tensor([[[0., .5, 0.]]]) + torch.cross(
        torch.tensor([[[0., 0., .6]]]).expand_as(offset), offset, dim=-1,
    )
    torch.testing.assert_close(robot.data.site_lin_vel_w[:, blocker.site_ids], expected_site_v, atol=1e-5, rtol=1e-5)
    root[:, 12] = 0
    robot.write_root_state_to_sim(root)
    refresh()
    eps = .001
    shifted = root.clone()
    shifted[:, 1] += .5 * eps
    robot.write_root_state_to_sim(shifted)
    refresh()
    next_distance = get_blocker_target_distance(env, blocker)
    torch.testing.assert_close((distance - next_distance) / eps, closing, atol=.002, rtol=.005)
    robot.write_root_state_to_sim(root)
    refresh()

    # Reverse velocity, keeping the pose/target fixed; the feedback reverses.
    root[:, 8] = -.5
    robot.write_root_state_to_sim(root)
    refresh()
    reverse = term.func(env, **term.params)
    assert reverse[0] < 0 and reverse[1] > 0
    root[:, 8] = .5
    robot.write_root_state_to_sim(root)
    refresh()

    # Same-state old/new reward-manager comparison: only eereach changes.
    new_func = term.func
    index = env.reward_manager.active_terms.index("eereach")
    other = [i for i in range(len(env.reward_manager.active_terms)) if i != index]
    term.func = legacy_reach
    old_total = env.reward_manager.compute(dt=env.step_dt).clone()
    old_terms = env.reward_manager._step_reward.clone()
    term.func = new_func
    new_total = env.reward_manager.compute(dt=env.step_dt).clone()
    new_terms = env.reward_manager._step_reward.clone()
    torch.testing.assert_close(new_terms[:, other], old_terms[:, other])
    torch.testing.assert_close(new_total - old_total,
                               (new_terms[:, index] - old_terms[:, index]) * env.step_dt,
                               atol=1e-6, rtol=1e-5)
    print("Old/new raw eereach (left/right, vy=+.5):",
          (old_terms[:, index] / 10).tolist(), (new_terms[:, index] / 10).tolist())

    # Ordinary reward-rate scaling, with and without dt. The mixed change is
    # always task coefficient .6; AMP is an identical controlled score here.
    for scale_by_dt in (True, False):
        env.reward_manager._scale_by_dt = scale_by_dt
        for dt in (.01, .02, .04):
            total = env.reward_manager.compute(dt=dt).clone()
            scale = dt if scale_by_dt else 1
            torch.testing.assert_close(total, new_total * scale / env.step_dt, atol=1e-5, rtol=1e-5)
            mixed, task_component, style = mix_amp_rewards(
                total, torch.full((2,), .75), style_weight=.4,
                step_dt=dt, scale_rewards_by_dt=scale_by_dt,
            )
            torch.testing.assert_close(task_component, .6 * total)
            torch.testing.assert_close(mixed, task_component + style)
    env.reward_manager._scale_by_dt = True

    # There is no far/near switch at ball_x=1.5 anymore.
    for x in (1.49, 1.51):
        ball_state[:, 0] = origins[:, 0] + x
        ball.write_root_state_to_sim(ball_state)
        refresh()
        torch.testing.assert_close(term.func(env, **term.params), raw)

    # Any allowed part may attain the target; movement then contributes zero,
    # even if the torso is still translating.
    points = get_blocker_positions_e(env, blocker)
    for point in range(5):
        target[:] = points[:, point]
        d, c = get_blocker_target_kinematics(env, blocker)
        assert intercept_movement_reward(d, c).eq(0).all()
        torch.testing.assert_close(term.func(env, **term.params), torch.full((2,), .7310586))

    # Partial reset cannot give another env a spurious progress pulse. The
    # geometry is evaluated afresh, so there is no cross-episode history.
    target[:] = points[:, 0] + torch.tensor([[0., .7, 0.]])
    env._goalkeeper_outcome_state.shot_resolved[0] = True
    before = term.func(env, **term.params).clone()
    assert before[0] == 0
    env.reset(env_ids=torch.tensor([0]))
    after = term.func(env, **term.params)
    assert not env._goalkeeper_outcome_state.shot_resolved[0]
    torch.testing.assert_close(after[1], before[1])
    assert torch.isfinite(after).all()
    print("Real K1: signed left/right, site rotation, finite difference, label independence,")
    print("all other rewards, dt/mixing, all five blockers, partial reset and 92/95/22 inputs passed")
finally:
    env.close()
