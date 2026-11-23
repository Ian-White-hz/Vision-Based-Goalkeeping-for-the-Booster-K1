"""Approximate self-occlusion of the ball by the arms, without rendering.

Each arm is a chain of spheres along shoulder -> elbow -> wrist -> hand tip.
For every sphere, the line of sight from the camera to the ball centre is
compared with the ball's cone radius at that depth; the overlap gives an
occluded fraction in [0, 1]. Head and trunk sit behind the camera.
"""

from __future__ import annotations

import torch

ARM_CHAINS = (
    ("Left_Arm_1", "Left_Arm_2", "Left_Arm_3", "left_hand_link", "left_hand_end_ball"),
    (
        "Right_Arm_1",
        "Right_Arm_2",
        "Right_Arm_3",
        "right_hand_link",
        "right_hand_end_ball",
    ),
)
SAMPLES_PER_LINK = 3


def occluder_points(body_pos: torch.Tensor, chain_ids) -> torch.Tensor:
    """Sphere centres [N, M, 3] interpolated along each body chain."""
    pts = []
    t = torch.linspace(0, 1, SAMPLES_PER_LINK + 1, device=body_pos.device)[:-1]
    for ids in chain_ids:
        a, b = body_pos[:, ids[:-1]], body_pos[:, ids[1:]]  # [N, L, 3]
        seg = a[:, :, None] + (b - a)[:, :, None] * t[None, None, :, None]
        pts.append(seg.flatten(1, 2))
        pts.append(body_pos[:, ids[-1:]])
    return torch.cat(pts, 1)


def occluded_fraction(
    cam_pos, ball_pos, spheres, sphere_radius: float, ball_radius: float
):
    """Largest overlap of any sphere with the ball's viewing cone, per env."""
    d = ball_pos - cam_pos
    dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    ray = d / dist
    rel = spheres - cam_pos[:, None]
    s = (rel * ray[:, None]).sum(-1)  # depth along the ray
    in_front = (s > 0.05) & (s < dist - ball_radius)
    perp = (rel - s[..., None] * ray[:, None]).norm(dim=-1)
    cone = ball_radius * s / dist  # ball silhouette radius at that depth
    overlap = ((sphere_radius + cone - perp) / (2 * cone).clamp_min(1e-6)).clamp(0, 1)
    return torch.where(in_front, overlap, 0.0).amax(-1)
