"""Ball state writers and reset events."""

from collections.abc import Sequence

import torch

from mjlab.utils.lab_api.math import quat_apply


def _env_ids(env, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is not None:
        return env_ids
    return torch.arange(env.num_envs, device=env.device, dtype=torch.int64)


def _batch_vector(
    value: torch.Tensor | Sequence[float],
    batch_size: int,
    device: str,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.shape == (3,):
        tensor = tensor.unsqueeze(0).expand(batch_size, -1)
    if tensor.shape != (batch_size, 3):
        raise ValueError(f"{name} must have shape (3,) or ({batch_size}, 3)")
    return tensor


def set_ball_state(
    env,
    position: torch.Tensor | Sequence[float],
    linear_velocity: torch.Tensor | Sequence[float],
    env_ids: torch.Tensor | None = None,
    *,
    frame: str = "robot",
    angular_velocity: torch.Tensor | Sequence[float] = (0.0, 0.0, 0.0),
    ball_name: str = "ball",
    robot_name: str = "robot",
) -> None:
    """Set exact ball position and velocity in the robot or world frame."""
    ids = _env_ids(env, env_ids)
    position = _batch_vector(position, len(ids), env.device, "position")
    linear_velocity = _batch_vector(
        linear_velocity, len(ids), env.device, "linear_velocity"
    )
    angular_velocity = _batch_vector(
        angular_velocity, len(ids), env.device, "angular_velocity"
    )

    if frame == "robot":
        robot = env.scene[robot_name]
        robot_position = robot.data.root_link_pos_w[ids]
        robot_orientation = robot.data.root_link_quat_w[ids]
        position = robot_position + quat_apply(robot_orientation, position)
        linear_velocity = (
            robot.data.root_link_lin_vel_w[ids]
            + quat_apply(robot_orientation, linear_velocity)
        )
        angular_velocity = quat_apply(robot_orientation, angular_velocity)
    elif frame != "world":
        raise ValueError("frame must be 'robot' or 'world'")

    ball = env.scene[ball_name]
    root_state = ball.data.default_root_state[ids].clone()
    root_state[:, :3] = position
    root_state[:, 3:7] = root_state.new_tensor((1.0, 0.0, 0.0, 0.0))
    root_state[:, 7:10] = linear_velocity
    root_state[:, 10:13] = angular_velocity
    ball.write_root_state_to_sim(root_state, env_ids=ids)


def _sample_vectors(
    count: int,
    ranges: dict[str, tuple[float, float]],
    device: str,
) -> torch.Tensor:
    values = torch.empty((count, 3), device=device)
    for column, key in enumerate(("x", "y", "z")):
        low, high = ranges.get(key, (0.0, 0.0))
        values[:, column].uniform_(low, high)
    return values


def reset_ball_state(
    env,
    env_ids: torch.Tensor | None,
    position_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
) -> None:
    """Sample a rolling-ball launch around each fixed robot spawn on reset."""
    ids = _env_ids(env, env_ids)
    positions = _sample_vectors(len(ids), position_range, env.device)
    positions += env.scene.env_origins[ids]
    set_ball_state(
        env,
        position=positions,
        linear_velocity=_sample_vectors(len(ids), velocity_range, env.device),
        env_ids=ids,
        frame="world",
    )
