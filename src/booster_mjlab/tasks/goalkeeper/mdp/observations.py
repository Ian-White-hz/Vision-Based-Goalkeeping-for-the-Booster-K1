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