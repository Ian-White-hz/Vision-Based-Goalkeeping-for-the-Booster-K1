import torch
from mjlab.envs import ManagerBasedRlEnv

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_env_cfg,
)
from booster_mjlab.tasks.goalkeeper.scene.events import (
    sample_shots,
    compute_ballistic_launch_velocity
)
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings


settings = FullGoalSettings()

# 第一层：只检查采样，不创建仿真。
sample = sample_shots(
    128,
    settings,
    generator=torch.Generator().manual_seed(42),
)
repeated = sample_shots(
    128,
    settings,
    generator=torch.Generator().manual_seed(42),
)

for name in sample:
    assert torch.isfinite(sample[name]).all()
    torch.testing.assert_close(sample[name], repeated[name])

assert (sample["start_x"] >= settings.start_x_range[0]).all()
assert (sample["start_x"] <= settings.start_x_range[1]).all()
assert (sample["start_y"] >= settings.start_y_range[0]).all()
assert (sample["start_y"] <= settings.start_y_range[1]).all()
assert (sample["target_y"].abs() <= settings.target_half_width).all()

speed = torch.linalg.vector_norm(sample["velocity"], dim=1)
torch.testing.assert_close(speed, sample["speed"])

assert (speed >= settings.speed_range[0]).all()
assert (speed <= settings.speed_range[1]).all()
assert (sample["velocity"][:, 0] < 0).all()

# 按初始直线方向延伸，应该经过采样的球门目标。
travel_time = (
    settings.goal_x - sample["start_x"]
) / sample["velocity"][:, 0]

target_y = (
    sample["start_y"]
    + sample["velocity"][:, 1] * travel_time
)
torch.testing.assert_close(
    target_y,
    sample["target_y"],
    atol=1e-6,
    rtol=1e-5,
)

print("发球采样检查通过")

# 第二层：真实环境 reset。
cfg = booster_k1_goalkeeper_env_cfg(
    num_envs=2,
    settings=settings,
    shot_mode="rolling",
)

event_names = list(cfg.events)
assert event_names.index("reset_scene_to_default") < event_names.index(
    "full_goal_launch"
)

start_position_e = torch.tensor([
    [2.0, 0.0, 0.11],
    [2.0, 0.0, 0.11],
])

target_position_e = torch.tensor([
    [0.1, 0.4, 0.60],
    [0.1, -0.4, 0.85],
])

flight_time = torch.tensor([0.5, 0.6])

launch_velocity = compute_ballistic_launch_velocity(
    start_position_e=start_position_e,
    target_position_e=target_position_e,
    flight_time=flight_time,
)

duration = flight_time[:, None]
acceleration = torch.tensor([0.0, 0.0, -9.81])

reconstructed_target = (
    start_position_e
    + launch_velocity * duration
    + 0.5 * acceleration * duration.square()
)

assert launch_velocity.shape == (2, 3)
assert torch.isfinite(launch_velocity).all()
assert (launch_velocity[:, 0] < 0.0).all()

torch.testing.assert_close(
    reconstructed_target,
    target_position_e,
    atol=1e-5,
    rtol=1e-5,
)

print("空中发球初速度:", launch_velocity)
print("空中发球速度计算通过")

env = ManagerBasedRlEnv(cfg=cfg, device="cpu")

try:
    env.reset(seed=42)

    ball = env.scene["ball"]
    shots = env._full_goal_shots

    assert shots["target_position_e"].shape == (2, 3)
    assert shots["region"].shape == (2,)
    assert shots["region"].dtype == torch.long

    print("实际发球目标:", shots["target_position_e"])
    print("实际发球区域:", shots["region"])

    position_e = (
        ball.data.root_link_pos_w - env.scene.env_origins
    )

    torch.testing.assert_close(position_e[:, 0], shots["start_x"])
    torch.testing.assert_close(position_e[:, 1], shots["start_y"])
    torch.testing.assert_close(
        position_e[:, 2],
        torch.full((2,), settings.ball_radius + 0.002),
    )
    torch.testing.assert_close(
        ball.data.root_link_lin_vel_w[:, :2],
        shots["velocity"],
    )

    angular_velocity = torch.zeros(2, 3)
    angular_velocity[:, 0] = (
        -shots["velocity"][:, 1] / settings.ball_radius
    )
    angular_velocity[:, 1] = (
        shots["velocity"][:, 0] / settings.ball_radius
    )

    torch.testing.assert_close(
        ball.data.root_link_ang_vel_w,
        angular_velocity,
    )

    # 只重置环境 0，环境 1 的球和发射记录应保持不变。
    previous_position = ball.data.root_link_pos_w[1].clone()
    previous_velocity = ball.data.root_link_lin_vel_w[1].clone()
    previous_records = {
        name: value[1].clone()
        for name, value in shots.items()
    }

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
        torch.testing.assert_close(
            env._full_goal_shots[name][1],
            previous,
        )

    print("真实发球与局部重置检查通过")

finally:
    env.close()