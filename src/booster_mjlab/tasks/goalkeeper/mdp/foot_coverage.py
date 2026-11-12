"""Reward-only rolling-ball coverage using compiled collision-mesh bounds.

The oriented bounds approximate the collision hull; this is not a contact or
save detector. Feet are held fixed during a bounded linear-path prediction.
"""
import math

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse


def segment_box_distance(start, end, center, half_size):
    """Minimum Euclidean distance along a segment to an axis-aligned box.

    Convex 1-D minimization, vectorized over leading dimensions. Include both
    endpoints; 20 ternary iterations give sub-mm precision for our short paths.
    """
    def distance(t):
        p = start + t[..., None] * (end - start)
        return (torch.abs(p - center) - half_size).clamp_min(0).norm(dim=-1)

    low, high = torch.zeros_like(start[..., 0]), torch.ones_like(start[..., 0])
    endpoint = torch.minimum(distance(low), distance(high))
    for _ in range(20):
        a, b = (2 * low + high) / 3, (low + 2 * high) / 3
        choose_left = distance(a) <= distance(b)
        low, high = torch.where(choose_left, low, a), torch.where(choose_left, b, high)
    return torch.minimum(endpoint, distance((low + high) / 2))


def rolling_foot_gap(ball_pos, ball_vel, geom_pos, geom_quat, centers, half_sizes,
                     *, ball_radius, prediction_horizon=2.5, path_half_length=.2):
    """Sphere surface gap to each foot's short future path neighborhood (B,F).

    Predict to each foot's x, bounded in time. Only inspect +/-20cm along the
    trajectory there, not an infinite line. No gravity for rolling trajectories.
    """
    for name, value in (("ball_radius", ball_radius), ("prediction_horizon", prediction_horizon),
                        ("path_half_length", path_half_length)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    valid_ball = torch.isfinite(ball_pos).all(-1) & torch.isfinite(ball_vel).all(-1)
    valid_ball &= ball_vel[:, 0] < -1e-6
    p, v = torch.nan_to_num(ball_pos), torch.nan_to_num(ball_vel)
    crossing = (p[:, None, 0] - geom_pos[..., 0]) / (-v[:, None, 0]).clamp_min(1e-6)
    window = path_half_length / v.norm(dim=-1).clamp_min(1e-6)
    lo = (crossing - window[:, None]).clamp(0, prediction_horizon)
    hi = (crossing + window[:, None]).clamp(0, prediction_horizon)
    valid = valid_ball[:, None] & (crossing + window[:, None] >= 0) & (crossing <= prediction_horizon)
    valid &= torch.isfinite(geom_pos).all(-1) & torch.isfinite(geom_quat).all(-1)
    start = p[:, None] + v[:, None] * lo[..., None]
    end = p[:, None] + v[:, None] * hi[..., None]
    local_start = quat_apply_inverse(geom_quat, start - geom_pos)
    local_end = quat_apply_inverse(geom_quat, end - geom_pos)
    gap = segment_box_distance(local_start, local_end, centers, half_sizes) - ball_radius
    return torch.where(valid, gap, torch.full_like(gap, torch.inf))


def rolling_coverage_from_env(env, foot_cfg, ball_name, **params):
    """Read current geom poses and static compiled bounds, cached per selection."""
    robot = env.scene[foot_cfg.name]
    ids = tuple(int(i) for i in foot_cfg.geom_ids)
    cache = getattr(env, "_goalkeeper_foot_bounds", {})
    key = (foot_cfg.name, ids)
    if key not in cache:
        global_ids = [int(robot.indexing.geom_ids[i]) for i in ids]
        model = env.sim.mj_model
        if any(not (model.geom_contype[i] or model.geom_conaffinity[i]) for i in global_ids):
            raise ValueError("Foot coverage must select enabled collision geometry")
        bounds = torch.as_tensor(model.geom_aabb[global_ids].copy(), device=env.device,
                                 dtype=torch.float32).reshape(-1, 2, 3)
        if not torch.isfinite(bounds).all() or not (bounds[:, 1] > 0).all():
            raise ValueError("Invalid compiled foot collision bounds")
        cache[key] = (bounds[:, 0], bounds[:, 1])
        env._goalkeeper_foot_bounds = cache
    centers, halves = cache[key]
    ball = env.scene[ball_name]
    return rolling_foot_gap(ball.data.root_link_pos_w, ball.data.root_link_lin_vel_w,
                            robot.data.geom_pos_w[:, ids], robot.data.geom_quat_w[:, ids],
                            centers, halves, **params).amin(-1)
