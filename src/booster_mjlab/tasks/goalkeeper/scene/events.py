"""Independent origin/goal sampling and reset-only, pure-rolling launches."""

import torch

from .settings import FullGoalSettings
from ..mdp.shot_state import compute_rolling_shot_target


def compute_ballistic_launch_velocity(
    start_position_e: torch.Tensor,
    target_position_e: torch.Tensor,
    flight_time: torch.Tensor,
    gravity: float = 9.81,
) -> torch.Tensor:
    if (flight_time <= 0.0).any():
        raise ValueError("flight_time must be positive")

    duration = flight_time[:, None]

    velocity = (
        target_position_e - start_position_e
    ) / duration

    velocity[:, 2] += 0.5 * gravity * flight_time

    return velocity

def compute_ballistic_intercept(
    start_position_e: torch.Tensor,
    veloctiy: torch.Tensor,
    *,
    intercept_x: float,
    gravity: float = 9.81,
) -> tuple[torch.Tensor, torch.Tensor]:
    velocity_x = veloctiy[:, 0]

    if (velocity_x >= 0.0).any():
        raise ValueError("velocity_x must be negative for intercept")

    intercept_time = (
        intercept_x - start_position_e[:, 0]
    ) / velocity_x

    if (
        (~torch.isfinite(intercept_time)) | (intercept_time <= 0.0)
    ).any():
        raise ValueError("Invalid intercept parameters")

    position_e = (
        start_position_e
        + veloctiy * intercept_time[:, None]
    )

    position_e[:, 2] -= (
        0.5 * gravity * torch.square(intercept_time)
    )

    return position_e, intercept_time

def sample_six_region_shots(
    count,
    settings: FullGoalSettings,
    device="cpu",
    generator=None,
    regions=None,
    gravity: float = 9.81,
):
    '''采样六区域发球, shape (count, 3)'''
    if count < 0:
        raise ValueError("count must be nonnegative")

    def uniform(low, high):
        return low + (high - low) * torch.rand(
            count,
            device=device,
            generator=generator,
        )

    # 如果没指定 regions, 则随机采样
    if regions is None:
        region = torch.randint(
            0, 6, (count,), 
            device=device,
            generator=generator,
        )
    else:
        region = regions.to(device=device)
        if region.shape != (count,) or ((region < 0) | (region >= 6)).any():
            raise ValueError("regions must be a tensor of shape (count,) with values in [0, 5]")

    # 根据 region 采样起始位置和目标位置
    bounds = torch.tensor(
        settings.six_region_bounds,
        dtype=torch.float32,
        device=device,
    )[region]

    # 采样起始位置和目标位置
    start_position_e = torch.stack(
        (
            uniform(*settings.start_x_range),
            uniform(*settings.start_y_range),
            uniform(*settings.start_z_range),
        ),
        dim=-1
    )

    goal_target_position_e = torch.stack(
        (
            torch.full(
                (count,), 
                settings.goal_x, 
                device=device
            ),
            uniform(bounds[:, 0], bounds[:, 1]),
            uniform(bounds[:, 2], bounds[:, 3]),
        ),
        dim=-1
    )

    flight_time = uniform(*settings.flight_time_range)

    velocity = compute_ballistic_launch_velocity(
        start_position_e=start_position_e,
        target_position_e=goal_target_position_e,
        flight_time=flight_time,
        gravity=gravity,
    )

    target_position_e, intercept_time = compute_ballistic_intercept(
        start_position_e=start_position_e,
        veloctiy=velocity,
        intercept_x=settings.intercept_x,
        gravity=gravity,
    )

    if (intercept_time >= flight_time).any():
        raise ValueError("intercept_time must be less than flight_time")

    return {
        "start_position_e": start_position_e,
        "goal_target_position_e": goal_target_position_e,
        "target_position_e": target_position_e,
        "flight_time": flight_time,
        "intercept_time": intercept_time,
        "velocity": velocity,
        "region": region,
    }

def launch_six_region_shots(
    env,
    env_ids,
    settings: FullGoalSettings,
):
    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else env_ids
    )

    sample = sample_six_region_shots(
        len(ids),
        settings,
        device=env.device,
        gravity=-env.cfg.sim.mujoco.gravity[2],
    )

    if not hasattr(env, "_full_goal_shots"):
        env._full_goal_shots = {
            name: torch.zeros(
                (env.num_envs, *value.shape[1:]), 
                device=env.device, 
                dtype=value.dtype
            )
            for name, value in sample.items()
        }

    for name, value in sample.items():
        env._full_goal_shots[name][ids] = value

    # 设置球的状态
    state = torch.zeros(
        (len(ids), 13),
        device=env.device,
    )
    state[:, :3] = (
        sample["start_position_e"]
        + env.scene.env_origins[ids]
    )
    state[:, 3] = 1.0
    state[:, 7:10] = sample["velocity"]

    env.scene["ball"].write_root_state_to_sim(
        state,
        env_ids=ids,
    )



def sample_shots(count, settings: FullGoalSettings, device="cpu", generator=None):
    if count < 0:
        raise ValueError("count must be nonnegative")

    def uniform(low, high):
        return low + (high - low) * torch.rand(count, device=device, generator=generator)

    x = uniform(*settings.start_x_range)
    y = uniform(*settings.start_y_range)
    target_y = uniform(-settings.target_half_width, settings.target_half_width)
    speed = uniform(*settings.speed_range)
    direction = torch.stack((settings.goal_x - x, target_y - y), dim=1)
    velocity = direction / direction.norm(dim=1, keepdim=True) * speed[:, None]
    return dict(start_x=x, start_y=y, target_y=target_y, speed=speed, velocity=velocity)


def launch_shots(env, env_ids, settings: FullGoalSettings):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    sample = sample_shots(len(ids), settings, device=env.device)
    target_position_e, region = compute_rolling_shot_target(
        start_x=sample["start_x"],
        start_y=sample["start_y"],
        goal_y=sample["target_y"],
        goal_x=settings.goal_x,
        ball_radius=settings.ball_radius,
        intercept_x=settings.intercept_x,
    )

    sample["target_position_e"] = target_position_e
    sample["region"] = region
    # Diagnostics only: never automatically inserted into observations.
    if not hasattr(env, "_full_goal_shots"):
        env._full_goal_shots = {
            name: torch.zeros((env.num_envs, *value.shape[1:]), device=env.device, dtype=value.dtype)
            for name, value in sample.items()
        }
    for name, value in sample.items():
        env._full_goal_shots[name][ids] = value
    pose = torch.zeros((len(ids), 7), device=env.device)
    velocity = torch.zeros((len(ids), 6), device=env.device)
    pose[:, 0], pose[:, 1] = sample["start_x"], sample["start_y"]
    pose[:, 2], pose[:, 3] = settings.ball_radius + 0.002, 1.0
    pose[:, :3] += env.scene.env_origins[ids]
    velocity[:, :2] = sample["velocity"]
    velocity[:, 3] = -velocity[:, 1] / settings.ball_radius
    velocity[:, 4] = velocity[:, 0] / settings.ball_radius
    env.scene["ball"].write_root_state_to_sim(torch.cat((pose, velocity), dim=1), env_ids=ids)