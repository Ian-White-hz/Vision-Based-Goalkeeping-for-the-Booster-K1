"""Time-aligned coverage and independent root lateral guidance.

All quantities are reward-side simulation state, not additional policy inputs.
No per-episode history, target mutation, or contact/save relabeling is needed.
"""

from dataclasses import dataclass
import math

import torch

from .shot_state import get_blocker_positions_e, get_shot_outcome_state


@dataclass(frozen=True)
class InterceptionCfg:
    arrival_window_s: float = .2
    preparation_weight: float = .2
    near_prediction_s: float = .2
    prediction_limit_m: float = .2
    settling_weight: float = .5
    settling_speed_m_s: float = 1.

    def __post_init__(self):
        for name in ("arrival_window_s", "near_prediction_s", "settling_speed_m_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.prediction_limit_m) or self.prediction_limit_m < 0:
            raise ValueError("prediction_limit_m must be finite and nonnegative")
        for name in ("preparation_weight", "settling_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")


def intercept_prediction(ball_pos, ball_vel, nominal_target, rolling, gravity, cfg):
    """Blend a short current-trajectory prediction into the immutable target.

    Time-to-arrival always uses the original plane. Rolling shots keep the
    nominal ground-level z; ballistic shots include gravity. Invalid inputs
    return inactive finite placeholders, never an unbounded prediction.
    """
    if ball_pos.ndim != 2 or ball_pos.shape[-1] != 3 or any(
        x.shape != ball_pos.shape for x in (ball_vel, nominal_target)
    ):
        raise ValueError("Ball position, velocity and target must have shape (B, 3)")
    if rolling.shape != ball_pos.shape[:1] or rolling.dtype != torch.bool:
        raise ValueError("rolling must be bool (B,)")
    finite = torch.isfinite(ball_pos).all(-1) & torch.isfinite(ball_vel).all(-1)
    finite &= torch.isfinite(nominal_target).all(-1)
    valid = finite & (ball_vel[:, 0] < -1e-6) & (ball_pos[:, 0] > nominal_target[:, 0])
    pos, vel, nominal = (torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
                         for x in (ball_pos, ball_vel, nominal_target))
    tau = (pos[:, 0] - nominal[:, 0]) / (-vel[:, 0]).clamp_min(1e-6)
    tau = torch.where(valid, tau.clamp_min(0), 0.)
    lookahead = tau.clamp(max=cfg.near_prediction_s)[:, None]
    prediction = pos + vel * lookahead + .5 * gravity * lookahead.square()
    prediction[:, 0] = nominal[:, 0]
    prediction[:, 2] = torch.where(rolling, nominal[:, 2], prediction[:, 2])
    blend = (1 - tau / cfg.near_prediction_s).clamp(0, 1)
    correction = (prediction - nominal).clamp(-cfg.prediction_limit_m, cfg.prediction_limit_m)
    target = nominal + blend[:, None] * correction
    return target, tau, valid


def coverage_factors(distance, speed, tau, cfg, *, reach_threshold=.2):
    """Preparation < arrival; a fast pass gets less reward than settled coverage."""
    if any(x.shape != distance.shape for x in (speed, tau)) or distance.ndim != 1:
        raise ValueError("Distance, speed and arrival time must have shape (B,)")
    if not math.isfinite(reach_threshold) or reach_threshold <= 0:
        raise ValueError("reach_threshold must be finite and positive")
    urgency = (1 - tau / cfg.arrival_window_s).clamp(0, 1)
    time_weight = cfg.preparation_weight + (1 - cfg.preparation_weight) * urgency.square()
    close = (1 - distance / reach_threshold).clamp(0, 1)
    ratio = speed.abs() / cfg.settling_speed_m_s
    # Stable even for very large finite speeds.
    speed_cost = 1 - 1 / (1 + ratio.square())
    settling = 1 - cfg.settling_weight * urgency * close * speed_cost
    return time_weight, settling


def body_lateral_progress(target_y, root_y, root_vy, blocker_distance, *,
                          lateral_reach=.3, movement_weight=.5,
                          fade_distance=.3, speed_limit=1., reach_threshold=.05):
    """Only real root translation earns movement reward; hands only gate it off.

    A reachable lateral corridor avoids forcing body motion for nearby balls.
    Overshoot changes the sign, and an attained blocker shuts guidance down.
    """
    for name, value in (("lateral_reach", lateral_reach), ("movement_weight", movement_weight)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name, value in (("fade_distance", fade_distance), ("speed_limit", speed_limit),
                        ("reach_threshold", reach_threshold)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    error = target_y - root_y
    excess = (error.abs() - lateral_reach).clamp_min(0)
    body_gate = (excess / fade_distance).clamp(0, 1)
    blocker_gate = ((blocker_distance - reach_threshold) / fade_distance).clamp(0, 1)
    speed = (error.sign() * root_vy).clamp(-speed_limit, speed_limit)
    return movement_weight * body_gate * blocker_gate * speed


def interception_features(env, blocker_cfg, *, ball_name="ball", guidance=None, reach_threshold=.2):
    cfg = InterceptionCfg(**(guidance or {}))
    robot, ball = env.scene[blocker_cfg.name], env.scene[ball_name]
    origins = env.scene.env_origins
    nominal = env._full_goal_shots["target_position_e"]
    rolling = env._full_goal_shots["is_rolling"]
    gravity = nominal.new_tensor(env.cfg.sim.mujoco.gravity)
    target, tau, valid = intercept_prediction(
        ball.data.root_link_pos_w - origins, ball.data.root_link_lin_vel_w,
        nominal, rolling, gravity, cfg,
    )
    points = get_blocker_positions_e(env, blocker_cfg)
    velocities = torch.cat((robot.data.site_lin_vel_w[:, blocker_cfg.site_ids],
                            robot.data.body_link_lin_vel_w[:, blocker_cfg.body_ids]), dim=1)
    distances = (points - target[:, None]).norm(dim=-1)
    distance = distances.amin(dim=1)
    # When two blockers tie, a stationary one can provide coverage even if the
    # other sweeps past. No arm velocity is ever added as a positive term.
    speed = velocities.norm(dim=-1).masked_fill(distances != distance[:, None], torch.inf).amin(dim=1)
    time_weight, settling = coverage_factors(distance, speed, tau, cfg, reach_threshold=reach_threshold)
    shot = get_shot_outcome_state(env)
    active = valid & ~shot.stop_flag & ~shot.success_flag & ~env._goalkeeper_outcome_state.shot_resolved
    active &= torch.isfinite(distance) & torch.isfinite(speed)
    upright = 1 - robot.data.projected_gravity_b[:, :2].square().sum(-1).clamp(0, 1)
    return dict(target=target, tau=tau, distance=distance, speed=speed,
                time_weight=time_weight, settling=settling, active=active, upright=upright)
