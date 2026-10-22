from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers import SceneEntityCfg

import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

def base_ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    robot_name: str = "robot",
) -> torch.Tensor:
    robot = env.scene[robot_name]
    angular_velocity_b = robot.data.root_link_ang_vel_b

    return torch.sum(
        torch.square(angular_velocity_b[:, :2]),
        dim=1,
    )

def base_backward_velocity_penalty(
    env: ManagerBasedRlEnv,
    robot_name: str = "robot",
) -> torch.Tensor:
    robot = env.scene[robot_name]
    velocity_x_b = robot.data.root_link_lin_vel_b[:, 0]

    return torch.clamp(
        -velocity_x_b,
        min=0.0,
        max=1.0,
    )

def base_line_deviation(
    env:ManagerBasedRlEnv,
    robot_name:str = "robot",
    reference_x: float = 0.0,
    deadband: float = 0.2,
    max_distance: float = 1.2,
):
    """Deviation from a reference plane along world x, shape (B,)"""
    robot = env.scene[robot_name]

    position_x = (
        robot.data.root_link_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )
    distance = torch.abs(position_x - reference_x)

    return torch.clamp(
        distance,
        min=deadband,
        max=max_distance,
    ) - deadband

def knee_height_below_minimum(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    minimum_height: float = 0.15,
):
    """Whether any selected knee is too low above the flat ground, shape (B,)."""
    robot = env.scene[asset_cfg.name]

    knee_height = (
        robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2]
        - env.scene.env_origins[:, 2:3]
    )

    return (
        torch.min(knee_height, dim=1).values < minimum_height
    ).float()