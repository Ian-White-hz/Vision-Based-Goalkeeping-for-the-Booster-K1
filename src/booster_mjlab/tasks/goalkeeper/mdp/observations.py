from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

'''
    Compute the position of the ball in the robot's root frame.
    Param:
        env: The environment containing the robot and the ball.
        robot_name: The name of the robot in the environment.
        ball_name: The name of the ball in the environment.
    Returns:
        A torch.Tensor of shape (B, 3) representing the ball's position in the robot's root frame.
'''
def ball_position_b(
    env: ManagerBasedRlEnv,
    robot_name: str = "robot",
    ball_name: str = "ball",
) -> torch.Tensor:
    robot = env.scene[robot_name]
    ball = env.scene[ball_name]

    relative_position_w = (
        ball.data.root_link_pos_w - robot.data.root_link_pos_w
    )

    return quat_apply_inverse(
        # Robot body frame expressed in world frame
        robot.data.root_link_quat_w,
        relative_position_w
    )

def simulated_ball_measurement_b(
    env: ManagerBasedRlEnv,
    robot_name: str = "robot",
    ball_name: str = "ball",
    max_range: float = 5.0,
) -> torch.Tensor:
    if max_range <= 0.0:
        raise ValueError("max_range must be positive.")

    position_b = ball_position_b(env, robot_name, ball_name)

    distance = torch.linalg.vector_norm(position_b, dim=-1)

    valid = (
        torch.isfinite(position_b).all(dim=-1) 
        & (distance <= max_range)
    )

    measured_position_b = torch.where(
        valid[:, None],
        position_b,
        torch.zeros_like(position_b)
    )

    return torch.cat(
        (
            measured_position_b,
            valid[:, None].to(dtype=position_b.dtype),
        ),
        dim=-1
    )