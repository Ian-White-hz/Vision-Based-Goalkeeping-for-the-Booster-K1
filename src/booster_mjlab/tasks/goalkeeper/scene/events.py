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

    velocity = (target_position_e - start_position_e) / duration

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

    intercept_time = (intercept_x - start_position_e[:, 0]) / velocity_x

    if ((~torch.isfinite(intercept_time)) | (intercept_time <= 0.0)).any():
        raise ValueError("Invalid intercept parameters")

    position_e = start_position_e + veloctiy * intercept_time[:, None]

    position_e[:, 2] -= 0.5 * gravity * torch.square(intercept_time)

    return position_e, intercept_time


def sample_six_region_shots(
    count,
    settings: FullGoalSettings,
    device="cpu",
    generator=None,
    regions=None,
    gravity: float = 9.81,
    max_resample_rounds: int = 128,
):
    """Sample six motion-aligned shots defined at the robot intercept plane.

    Hand and jump categories (0--3) are ballistic. Step categories (4--5)
    are rolling shots. Every accepted shot also places the whole ball inside
    the goal aperture at ``goal_x``.
    """
    if count < 0:
        raise ValueError("count must be nonnegative")
    if not isinstance(max_resample_rounds, int) or max_resample_rounds <= 0:
        raise ValueError("max_resample_rounds must be a positive integer")
    if not torch.isfinite(torch.tensor(gravity)) or gravity <= 0.0:
        raise ValueError("gravity must be finite and positive")

    def uniform(low, high, size):
        return low + (high - low) * torch.rand(
            size,
            device=device,
            generator=generator,
        )

    # 如果没指定 regions, 则随机采样
    if regions is None:
        if settings.fixed_motion_id is None:
            region = torch.randint(0, 6, (count,), device=device, generator=generator)
        else:
            region = torch.full(
                (count,), settings.fixed_motion_id, dtype=torch.long, device=device
            )
    else:
        if not isinstance(regions, torch.Tensor) or regions.dtype != torch.int64:
            raise ValueError("regions must be an int64 tensor")
        region = regions.to(device=device)
        if region.shape != (count,) or ((region < 0) | (region >= 6)).any():
            raise ValueError("regions must have shape (count,) with values in [0, 5]")

    bounds = torch.tensor(
        settings.intercept_region_bounds,
        dtype=torch.float32,
        device=device,
    )[region]

    start_position_e = torch.empty((count, 3), dtype=torch.float32, device=device)
    goal_target_position_e = torch.empty_like(start_position_e)
    target_position_e = torch.empty_like(start_position_e)
    velocity = torch.empty_like(start_position_e)
    flight_time = torch.empty(count, dtype=torch.float32, device=device)
    intercept_time = torch.empty_like(flight_time)
    is_rolling = region >= 4

    pending = torch.arange(count, device=device)
    for _ in range(max_resample_rounds):
        if pending.numel() == 0:
            break
        size = pending.numel()
        pending_regions = region[pending]
        rolling = pending_regions >= 4

        start = torch.stack(
            (
                uniform(*settings.start_x_range, size),
                uniform(*settings.start_y_range, size),
                uniform(*settings.start_z_range, size),
            ),
            dim=-1,
        )
        candidate_bounds = bounds[pending]
        # Hand shots launch low enough to pass through the hand-height region;
        # jump shots keep the configured elevated source range.
        hand = pending_regions < 2
        if hand.any():
            start[hand, 2] = uniform(
                settings.ball_radius + 0.002,
                min(settings.start_z_range[1], 0.30),
                int(hand.sum().item()),
            )

        intercept = torch.stack(
            (
                torch.full((size,), settings.intercept_x, device=device),
                uniform(candidate_bounds[:, 0], candidate_bounds[:, 1], size),
                uniform(candidate_bounds[:, 2], candidate_bounds[:, 3], size),
            ),
            dim=-1,
        )
        fraction = (settings.intercept_x - start[:, 0]) / (
            settings.goal_x - start[:, 0]
        )
        goal_z = uniform(
            settings.ball_radius + settings.target_margin,
            settings.goal_height - settings.ball_radius,
            size,
        )
        goal_y = (intercept[:, 1] - (1.0 - fraction) * start[:, 1]) / fraction
        goal = torch.stack(
            (
                torch.full((size,), settings.goal_x, device=device),
                goal_y,
                goal_z,
            ),
            dim=-1,
        )
        height_above_chord = intercept[:, 2] - (
            (1.0 - fraction) * start[:, 2] + fraction * goal[:, 2]
        )
        duration_squared = (
            2.0 * height_above_chord / (gravity * fraction * (1.0 - fraction))
        )
        duration = torch.sqrt(torch.clamp_min(duration_squared, 0.0))
        candidate_velocity = compute_ballistic_launch_velocity(
            start, goal, torch.clamp_min(duration, 1.0e-6), gravity=gravity
        )
        candidate_intercept_time = duration * fraction

        if rolling.any():
            rolling_start = start[rolling]
            rolling_goal = goal[rolling]
            rolling_start[:, 2] = settings.ball_radius + 0.002
            rolling_goal[:, 2] = settings.ball_radius + 0.002
            direction = rolling_goal[:, :2] - rolling_start[:, :2]
            distance = torch.linalg.vector_norm(direction, dim=-1)
            speed = uniform(*settings.speed_range, int(rolling.sum().item()))
            rolling_velocity = direction / distance[:, None] * speed[:, None]
            rolling_duration = distance / speed
            rolling_fraction = (settings.intercept_x - rolling_start[:, 0]) / (
                settings.goal_x - rolling_start[:, 0]
            )
            rolling_intercept = rolling_start + rolling_fraction[:, None] * (
                rolling_goal - rolling_start
            )
            candidate_velocity[rolling] = torch.cat(
                (rolling_velocity, torch.zeros_like(speed[:, None])), dim=-1
            )
            duration[rolling] = rolling_duration
            candidate_intercept_time[rolling] = rolling_duration * rolling_fraction
            intercept[rolling] = rolling_intercept
            start[rolling] = rolling_start
            goal[rolling] = rolling_goal

        valid = (
            torch.isfinite(start).all(dim=-1)
            & torch.isfinite(goal).all(dim=-1)
            & torch.isfinite(candidate_velocity).all(dim=-1)
            & torch.isfinite(intercept).all(dim=-1)
            & torch.isfinite(duration)
            & torch.isfinite(candidate_intercept_time)
            & (candidate_intercept_time > 0.0)
            & (candidate_intercept_time < duration)
            & (duration >= settings.flight_time_range[0])
            & ((rolling) | (duration <= settings.flight_time_range[1]))
            & (intercept[:, 1] >= candidate_bounds[:, 0])
            & (intercept[:, 1] <= candidate_bounds[:, 1])
            & (intercept[:, 2] >= candidate_bounds[:, 2])
            & (intercept[:, 2] <= candidate_bounds[:, 3])
            & (goal[:, 1].abs() + settings.ball_radius <= settings.goal_width / 2)
            & (goal[:, 2] - settings.ball_radius >= 0.0)
            & (goal[:, 2] + settings.ball_radius <= settings.goal_height)
        )
        accepted = pending[valid]
        start_position_e[accepted] = start[valid]
        goal_target_position_e[accepted] = goal[valid]
        target_position_e[accepted] = intercept[valid]
        velocity[accepted] = candidate_velocity[valid]
        flight_time[accepted] = duration[valid]
        intercept_time[accepted] = candidate_intercept_time[valid]
        pending = pending[~valid]

    if pending.numel() > 0:
        missing = torch.bincount(region[pending], minlength=6).tolist()
        raise RuntimeError(
            "Unable to sample valid goalkeeper shots after "
            f"{max_resample_rounds} rounds; pending by category: {missing}"
        )

    return {
        "start_position_e": start_position_e,
        "goal_target_position_e": goal_target_position_e,
        "target_position_e": target_position_e,
        "flight_time": flight_time,
        "intercept_time": intercept_time,
        "velocity": velocity,
        "region": region,
        "is_rolling": is_rolling,
    }


def launch_six_region_shots(
    env,
    env_ids,
    settings: FullGoalSettings,
):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids

    sample = sample_six_region_shots(
        len(ids),
        settings,
        device=env.device,
        gravity=-env.cfg.sim.mujoco.gravity[2],
    )

    if not hasattr(env, "_full_goal_shots"):
        env._full_goal_shots = {
            name: torch.zeros(
                (env.num_envs, *value.shape[1:]), device=env.device, dtype=value.dtype
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
    state[:, :3] = sample["start_position_e"] + env.scene.env_origins[ids]
    state[:, 3] = 1.0
    state[:, 7:10] = sample["velocity"]
    rolling = sample["is_rolling"]
    state[rolling, 10] = -sample["velocity"][rolling, 1] / settings.ball_radius
    state[rolling, 11] = sample["velocity"][rolling, 0] / settings.ball_radius

    env.scene["ball"].write_root_state_to_sim(
        state,
        env_ids=ids,
    )


def sample_shots(count, settings: FullGoalSettings, device="cpu", generator=None):
    if count < 0:
        raise ValueError("count must be nonnegative")

    def uniform(low, high):
        return low + (high - low) * torch.rand(
            count, device=device, generator=generator
        )

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
            name: torch.zeros(
                (env.num_envs, *value.shape[1:]), device=env.device, dtype=value.dtype
            )
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
    env.scene["ball"].write_root_state_to_sim(
        torch.cat((pose, velocity), dim=1), env_ids=ids
    )
