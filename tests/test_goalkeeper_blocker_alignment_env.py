"""Real K1 counterfactual: only the target-attainment reward may change."""

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.amp.runners.reward_mixing import mix_amp_rewards
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_amp_env_cfg,
)
from booster_mjlab.tasks.goalkeeper.mdp.rewards import target_success_reward
from booster_mjlab.tasks.goalkeeper.mdp.interception import interception_features
from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    get_blocker_target_distance,
    get_hand_target_distance,
    get_shot_outcome_state,
)


def legacy_success(env, blocker_cfg, strict_threshold, guidance):
    """Previous region-selected-hand criterion, used only for comparison."""
    state = get_shot_outcome_state(env)
    reward = target_success_reward(
        get_hand_target_distance(env, blocker_cfg),
        state.success_before_step,
        strict_threshold,
    )
    f = interception_features(env, blocker_cfg, guidance=guidance)
    return torch.where(f["active"], reward * f["time_weight"] * f["settling"], 0.)


torch.set_num_threads(1)
env = ManagerBasedRlEnv(booster_k1_goalkeeper_amp_env_cfg(num_envs=2), device="cpu")
try:
    obs, _ = env.reset(seed=42)
    assert {name: obs[name].shape[1] for name in ("actor", "critic", "amp")} == {
        "actor": 92, "critic": 95, "amp": 22,
    }
    assert env.action_manager.total_action_dim == 22
    reach = env.reward_manager.get_term_cfg("eereach")
    success = env.reward_manager.get_term_cfg("success")
    reach_cfg, success_cfg = reach.params["blocker_cfg"], success.params["blocker_cfg"]
    assert reach_cfg is not success_cfg
    for attr in ("name", "site_names", "body_names", "site_ids", "body_ids"):
        assert getattr(reach_cfg, attr) == getattr(success_cfg, attr), attr
    assert tuple(success_cfg.site_names) == ("left_hand", "right_hand")
    assert tuple(success_cfg.body_names) == ("Trunk", "left_foot_link", "right_foot_link")
    assert reach.weight == 10.0 and success.weight == 5.0
    assert reach.params["distance_sigma"] == 5.0
    assert reach.params["reach_threshold"] == 0.2
    assert success.params["strict_threshold"] == 0.15

    # Place the ball clear of the robot. Targets are controlled probes, not
    # sampled physical trajectories or claims of a successful save.
    ball_state = torch.zeros((2, 13))
    ball_state[:, :3] = env.scene.env_origins + torch.tensor([[1.0, 0.0, 1.2]])
    ball_state[:, 3] = 1.0
    ball_state[:, 7] = -1.2
    env.scene["ball"].write_root_state_to_sim(ball_state)
    env.sim.forward()
    env.scene.update(dt=0.0)
    shots = env._full_goal_shots
    shots["region"][:] = torch.tensor([4, 5])
    shots["velocity"][:] = ball_state[:, 7:10]
    robot = env.scene["robot"]
    points_e = torch.cat((
        robot.data.site_pos_w[:, reach_cfg.site_ids],
        robot.data.body_link_pos_w[:, reach_cfg.body_ids],
    ), dim=1) - env.scene.env_origins[:, None, :]
    new_function = success.func
    names = env.reward_manager.active_terms
    index = names.index("success")
    other_indices = [i for i in range(len(names)) if i != index]
    assert env.scene.env_origins.abs().any()

    for point_index, label in enumerate(("left_hand", "right_hand", "Trunk", "left_foot", "right_foot")):
        shots["target_position_e"][:] = points_e[:, point_index]
        torch.testing.assert_close(get_blocker_target_distance(env, reach_cfg), torch.zeros(2))
        # Evaluate both definitions on exactly the same simulation state.
        success.func = legacy_success
        old_total = env.reward_manager.compute(dt=env.step_dt).clone()
        old_terms = env.reward_manager._step_reward.clone()
        success.func = new_function
        new_total = env.reward_manager.compute(dt=env.step_dt).clone()
        new_terms = env.reward_manager._step_reward.clone()
        torch.testing.assert_close(new_terms[:, other_indices], old_terms[:, other_indices])
        torch.testing.assert_close(new_terms[:, index], torch.full((2,), 1.0))
        torch.testing.assert_close(
            new_total - old_total,
            (new_terms[:, index] - old_terms[:, index]) * env.step_dt,
            atol=1e-6, rtol=1e-5,
        )
        old_mixed, _, _ = mix_amp_rewards(
            old_total, torch.full((2,), 0.75), style_weight=0.4,
            step_dt=env.step_dt, scale_rewards_by_dt=True,
        )
        new_mixed, _, _ = mix_amp_rewards(
            new_total, torch.full((2,), 0.75), style_weight=0.4,
            step_dt=env.step_dt, scale_rewards_by_dt=True,
        )
        torch.testing.assert_close(new_mixed - old_mixed, 0.6 * (new_total - old_total), atol=1e-6, rtol=1e-5)
        if point_index >= 2:
            torch.testing.assert_close(old_terms[:, index], torch.zeros(2))
            torch.testing.assert_close(new_mixed - old_mixed, torch.full((2,), 0.012), atol=1e-6, rtol=1e-5)
        assert not env._goalkeeper_shot_state.success_flag.any()
        assert not env._goalkeeper_outcome_state.saved.any()
        print(label, "old/new raw success:", (old_terms[:, index] / 5).tolist(),
              (new_terms[:, index] / 5).tolist(), "mixed delta:", (new_mixed - old_mixed).tolist())

    env._goalkeeper_outcome_state.shot_resolved[0] = True
    torch.testing.assert_close(success.func(env, **success.params), torch.tensor([0.0, .2]))
    assert reach.func(env, **reach.params)[0] == 0
    _, _ = env.reset(env_ids=torch.tensor([0]))
    assert not env._goalkeeper_outcome_state.shot_resolved[0]
    assert torch.isfinite(success.func(env, **success.params)).all()
    print("Real K1: shared config, all other rewards unchanged, dt/mixing, partial reset and 92/95/22 inputs passed")
finally:
    env.close()
