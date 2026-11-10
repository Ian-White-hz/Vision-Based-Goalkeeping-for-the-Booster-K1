import mjlab  # noqa: F401 -- initialize mjlab before task imports
import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_env_cfg,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import (
    FullGoalSettings,
)
from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    get_shot_outcome_state,
)


cfg = booster_k1_goalkeeper_env_cfg(
    num_envs=2,
    settings=FullGoalSettings(fixed_motion_id=4),
)
env = ManagerBasedRlEnv(cfg, device="cpu")

try:
    env.reset(seed=42)

    term = env.reward_manager.get_term_cfg("eereach")
    blocker_cfg = term.params["blocker_cfg"]

    assert tuple(blocker_cfg.site_names) == (
        "left_hand", "right_hand",
    )
    assert tuple(blocker_cfg.body_names) == (
        "Trunk", "left_foot_link", "right_foot_link",
    )

    initial = term.func(env, **term.params).clone()

    assert initial.shape == (2,)
    assert torch.isfinite(initial).all()
    assert (initial > 0.0).all()
    assert (initial <= 1.0).all()

    robot = env.scene[blocker_cfg.name]
    origins = env.scene.env_origins

    assert origins.abs().any(), "测试需要包含非零环境原点"

    points_w = torch.cat(
        (
            robot.data.site_pos_w[:, blocker_cfg.site_ids, :],
            robot.data.body_link_pos_w[:, blocker_cfg.body_ids, :],
        ),
        dim=1,
    )

    targets_w = (
        env._full_goal_shots["target_position_e"] + origins
    )

    distance_w = torch.linalg.vector_norm(
        points_w - targets_w[:, None, :],
        dim=-1,
    ).amin(dim=1)

    expected = torch.sigmoid(
        term.params["distance_sigma"]
        * (term.params["reach_threshold"] - distance_w)
    )

    near_expected = expected.clone()
    torch.testing.assert_close(initial, expected)

    print("多环境坐标转换检查通过")

    # 只改变类别标签，不改变球、机器人或目标位置。
    env._full_goal_shots["region"].fill_(1)
    unchanged = term.func(env, **term.params)
    torch.testing.assert_close(unchanged, initial)
    ball = env.scene["ball"]

    # 位置与姿态 7 维，加上线速度与角速度 6 维。
    original_ball_state = torch.cat(
        (
            ball.data.root_link_pose_w,
            ball.data.root_link_vel_w,
        ),
        dim=1,
    ).clone()

    intercept_x = env._full_goal_shots["target_position_e"][0, 0]


    def set_ball_state(state):
        ball.write_root_state_to_sim(state)

        # 写入状态后刷新派生数据，不推进物理时间。
        env.sim.forward()
        env.scene.update(dt=0.0)


    cases = (
        # 相对拦截平面的 x 偏移、vx、是否应给奖励。
        (0.01, -0.4, True),
        (-0.01, -0.4, False),
        (0.01, 0.4, False),
        (0.01, 0.0, False),
    )

    for offset, velocity_x, should_be_active in cases:
        modified = original_ball_state.clone()

        modified[0, 0] = origins[0, 0] + intercept_x + offset
        modified[0, 7] = velocity_x

        set_ball_state(modified)
        reward = term.func(env, **term.params)

        expected_first = (
            near_expected[0]
            if should_be_active
            else torch.zeros_like(initial[0])
        )

        torch.testing.assert_close(reward[0], expected_first)

        # 环境 1 不应受到环境 0 的变化影响。
        torch.testing.assert_close(reward[1], initial[1])

        # 确认不是因为成功或射门结束，才导致奖励为零。
        state = get_shot_outcome_state(env)
        assert not state.stop_flag.any()
        assert not state.success_flag.any()
        assert not env._goalkeeper_outcome_state.shot_resolved.any()

    set_ball_state(original_ball_state)

    print("越过拦截平面、出射和停球的奖励关闭检查通过")

    # 成功后应退出接近目标奖励阶段。
    state = get_shot_outcome_state(env)
    state.stop_flag.fill_(True)
    state.success_flag.fill_(True)

    disabled = term.func(env, **term.params)
    torch.testing.assert_close(disabled, torch.zeros_like(initial))
    print("候选点:", blocker_cfg.site_names, blocker_cfg.body_names)
    print("接近奖励:", initial)
    print("真实环境全程距离引导、静止方向无加成和成功后关闭检查通过")

finally:
    env.close()
