"""Ball observations expressed in the robot root frame."""

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse


def ball_state_b(
    env,
    robot_name: str = "robot",
    ball_name: str = "ball",
) -> torch.Tensor:
    """Return relative ball position and velocity as ``[x, y, z, vx, vy, vz]``."""
    robot = env.scene[robot_name]
    ball = env.scene[ball_name]

    relative_position_w = ball.data.root_link_pos_w - robot.data.root_link_pos_w
    relative_velocity_w = (
        ball.data.root_link_lin_vel_w - robot.data.root_link_lin_vel_w
    )
    position_b = quat_apply_inverse(robot.data.root_link_quat_w, relative_position_w)
    velocity_b = quat_apply_inverse(robot.data.root_link_quat_w, relative_velocity_w)
    return torch.cat((position_b, velocity_b), dim=-1)
