import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_env_cfg,
)
from booster_mjlab.tasks.goalkeeper.scene.events import (
    sample_six_region_shots,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import (
    FullGoalSettings,
)

from mjlab.managers import SceneEntityCfg
from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    get_hand_target_distance,
)


settings = FullGoalSettings()
assert settings.rolling_probability_by_region == (0., 0., 0., 0., 1., 1.)

try:
    sample_six_region_shots(1, settings, regions=torch.tensor([0.0]))
except ValueError as error:
    assert "int64" in str(error)
else:
    raise AssertionError("Expected non-int64 shot categories to fail")

sample = sample_six_region_shots(
    6,
    settings,
    regions=torch.arange(6),
    generator=torch.Generator().manual_seed(42),
)

target = sample["target_position_e"]
goal_target = sample["goal_target_position_e"]
bounds = torch.tensor(settings.intercept_region_bounds)

torch.testing.assert_close(
    sample["region"],
    torch.arange(6),
)

assert (target[:, 1] >= bounds[:, 0]).all()
assert (target[:, 1] <= bounds[:, 1]).all()
assert (target[:, 2] >= bounds[:, 2]).all()
assert (target[:, 2] <= bounds[:, 3]).all()
assert (sample["velocity"][:, 0] < 0.0).all()
assert sample["is_rolling"].tolist() == [False, False, False, False, True, True]

for value in sample.values():
    assert torch.isfinite(value).all()

print("六类采样目标:", target)

torch.testing.assert_close(
    target[:, 0],
    torch.full_like(target[:, 0], settings.intercept_x),
    atol=1e-6,
    rtol=1e-6,
)

torch.testing.assert_close(
    goal_target[:, 0],
    torch.full_like(goal_target[:, 0], settings.goal_x),
)

assert (sample["intercept_time"] > 0.0).all()
assert (sample["intercept_time"] < sample["flight_time"]).all()

for position_key, time_key in (
    ("target_position_e", "intercept_time"),
    ("goal_target_position_e", "flight_time"),
):
    time = sample[time_key]

    reconstructed = sample["start_position_e"] + sample["velocity"] * time[:, None]
    ballistic = ~sample["is_rolling"]
    reconstructed[ballistic, 2] -= 0.5 * 9.81 * time[ballistic].square()

    torch.testing.assert_close(
        reconstructed,
        sample[position_key],
        atol=1e-5,
        rtol=1e-5,
    )

# Distribution-level contract: at least 10,000 valid shots per category.
many_regions = torch.arange(60_000) % 6
many = sample_six_region_shots(
    len(many_regions),
    settings,
    regions=many_regions,
    generator=torch.Generator().manual_seed(7),
)
many_bounds = bounds[many_regions]
many_target = many["target_position_e"]
assert (
    (many_target[:, 1] >= many_bounds[:, 0])
    & (many_target[:, 1] <= many_bounds[:, 1])
    & (many_target[:, 2] >= many_bounds[:, 2])
    & (many_target[:, 2] <= many_bounds[:, 3])
).all()
assert torch.equal(many["is_rolling"], many_regions >= 4)
rolling_speed = torch.linalg.vector_norm(
    many["velocity"][many["is_rolling"], :2], dim=-1
)
assert (rolling_speed >= settings.speed_range[0] - 1.0e-6).all()
assert (rolling_speed <= settings.speed_range[1] + 1.0e-6).all()

cfg = booster_k1_goalkeeper_env_cfg(
    num_envs=2,
    settings=settings,
    shot_mode="six_region",
)
env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

try:
    env.reset(seed=42)

    hand_cfg = SceneEntityCfg(
        "robot",
        site_names=("left_hand", "right_hand"),
        preserve_order=True,
    )
    hand_cfg.resolve(env.scene)

    distance = get_hand_target_distance(env, hand_cfg)

    assert distance.shape == (env.num_envs,)
    assert torch.isfinite(distance).all()
    assert (distance >= 0.0).all()

    print("手部 site 顺序:", hand_cfg.site_names)
    print("实际手到目标距离:", distance)

    ball = env.scene["ball"]
    shots = env._full_goal_shots

    torch.testing.assert_close(
        ball.data.root_link_pos_w - env.scene.env_origins,
        shots["start_position_e"],
    )
    torch.testing.assert_close(
        ball.data.root_link_lin_vel_w,
        shots["velocity"],
    )
    expected_ang_vel = torch.zeros_like(ball.data.root_link_ang_vel_w)
    rolling = shots["is_rolling"]
    expected_ang_vel[rolling, 0] = -shots["velocity"][rolling, 1] / settings.ball_radius
    expected_ang_vel[rolling, 1] = shots["velocity"][rolling, 0] / settings.ball_radius
    torch.testing.assert_close(ball.data.root_link_ang_vel_w, expected_ang_vel)
    assert shots["region"].dtype == torch.long

    # 局部 reset 不应改变另一个环境的球或记录。
    previous_position = ball.data.root_link_pos_w[1].clone()
    previous_velocity = ball.data.root_link_lin_vel_w[1].clone()
    previous_records = {name: value[1].clone() for name, value in shots.items()}

    env.reset(env_ids=torch.tensor([0], device=env.device))

    torch.testing.assert_close(
        ball.data.root_link_pos_w[1],
        previous_position,
    )
    torch.testing.assert_close(
        ball.data.root_link_lin_vel_w[1],
        previous_velocity,
    )

    for name, previous in previous_records.items():
        torch.testing.assert_close(shots[name][1], previous)

    print("实际发球区域:", shots["region"])
    print("实际扑救目标:", shots["target_position_e"])
    print("六类发球与局部重置检查通过")

    reach_cfg = env.reward_manager.get_term_cfg("eereach")

    success_cfg = env.reward_manager.get_term_cfg("success")

    assert tuple(reach_cfg.params["hand_cfg"].site_names) == ("left_hand", "right_hand")

    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )

    # 少量推进，只验证 reward 接口，不评估扑救效果。
    for _ in range(30):
        _, total_reward, terminated, truncated, _ = env.step(action)

        raw_reach = reach_cfg.func(env, **reach_cfg.params)

        assert raw_reach.shape == (env.num_envs,)
        assert torch.isfinite(raw_reach).all()
        assert (raw_reach >= 0.0).all()
        assert torch.isfinite(total_reward).all()

        if terminated.any().item() or truncated.any().item():
            break

        raw_success = success_cfg.func(env, **success_cfg.params)

        assert raw_success.shape == (env.num_envs,)
        assert torch.isfinite(raw_success).all()
        assert (
            (raw_success == 0.0) | (raw_success == 1.0) | (raw_success == 2.0)
        ).all()

    print("ee_reach 原始奖励:", raw_reach)
    print("ee_reach 环境接入检查通过")
    print("success 原始奖励:", raw_success)

    _, reset_extras = env.reset(env_ids=torch.tensor([0], device=env.device))
    metric_logs = {
        key: value
        for key, value in reset_extras["log"].items()
        if key.startswith("Episode_Metrics/Goalkeeper/")
    }
    assert len(metric_logs) == 24
    coverage = [
        metric_logs[f"Episode_Metrics/Goalkeeper/{name}/coverage"]
        for name in (
            "lefthand",
            "righthand",
            "leftjump",
            "rightjump",
            "leftstep",
            "rightstep",
        )
    ]
    torch.testing.assert_close(torch.stack(coverage).sum(), torch.tensor(1.0))
    print("六类 coverage/save/goal/fall episode metrics reset 日志检查通过")

finally:
    env.close()
