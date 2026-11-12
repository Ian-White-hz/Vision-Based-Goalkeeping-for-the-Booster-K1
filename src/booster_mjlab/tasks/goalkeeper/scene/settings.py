"""Scene parameters from the local 2026-10-15 ball/goal environment copy."""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class FullGoalSettings:
    goal_x: float = -0.8
    goal_width: float = 2.4
    goal_height: float = 1.0
    post_radius: float = 0.025
    ball_radius: float = 0.11
    ball_mass: float = 0.43
    ball_friction: tuple[float, float, float] = (0.6, 0.005, 0.0001)
    start_x_range: tuple[float, float] = (1.8, 2.6)
    start_y_range: tuple[float, float] = (-1.4, 1.4)
    # speed_range: tuple[float, float] = (0.35, 0.50)
    speed_range: tuple[float, float] = (1.0, 1.5)
    target_margin: float = 0.02
    robot_x_limit: float = 0.30
    robot_y_limit: float = 1.35
    root_reset_z_offset: float = 0.005
    stop_speed: float = 0.12
    stop_hold_s: float = 0.4
    stop_distance: float = 0.75
    goal_penalty: float = 2.0
    fall_penalty: float = 5.0
    heading_penalty: float = 5.0
    yaw_rate_penalty: float = 1.0
    heading_deadband_deg: float = 20.0
    heading_full_penalty_deg: float = 90.0
    yaw_rate_deadband: float = 0.5
    yaw_rate_full_penalty: float = 3.0
    intercept_x: float = 0.1
    start_z_range: tuple[float, float] = (0.55, 0.85)
    flight_time_range: tuple[float, float] = (0.5, 0.8)
    ball_x_bounds: tuple[float, float] = (-2.0, 5.0)
    ball_y_limit: float = 3.0
    recovery_duration_s: float = 2.0
    fixed_motion_id: int | None = None
    # Baseline A: step categories remain rolling-only. Low ballistic shots need
    # their own feasible launch-height/time ranges before enabling a mixture.
    rolling_probability_by_region: tuple[float, ...] = (
        0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    )

    intercept_region_bounds: tuple[tuple[float, float, float, float], ...] = (
        (0.15, 0.65, 0.35, 0.60),
        (-0.65, -0.15, 0.35, 0.60),
        (0.20, 0.70, 0.70, 0.90),
        (-0.70, -0.20, 0.70, 0.90),
        (0.15, 0.65, 0.11, 0.25),
        (-0.65, -0.15, 0.11, 0.25),
    )

    def __post_init__(self):
        for name in ("fall_penalty", "heading_penalty", "yaw_rate_penalty"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if (
            not math.isfinite(self.heading_deadband_deg)
            or not math.isfinite(self.heading_full_penalty_deg)
            or not 0.0 <= self.heading_deadband_deg < self.heading_full_penalty_deg <= 180.0
        ):
            raise ValueError("Invalid heading penalty range")
        if (
            not math.isfinite(self.yaw_rate_deadband)
            or not math.isfinite(self.yaw_rate_full_penalty)
            or not 0.0 <= self.yaw_rate_deadband < self.yaw_rate_full_penalty
        ):
            raise ValueError("Invalid yaw-rate penalty range")
        for name in (
            "start_x_range",
            "start_y_range",
            "speed_range",
            "start_z_range",
            "flight_time_range",
            "ball_x_bounds",
        ):
            values = tuple(getattr(self, name))
            if (
                len(values) != 2
                or not all(math.isfinite(v) for v in values)
                or values[0] > values[1]
            ):
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, values)
        friction = tuple(self.ball_friction)
        if len(friction) != 3 or any(not math.isfinite(v) or v < 0 for v in friction):
            raise ValueError("Invalid ball_friction")
        object.__setattr__(self, "ball_friction", friction)
        for name in (
            "goal_width",
            "goal_height",
            "post_radius",
            "ball_radius",
            "ball_mass",
            "robot_x_limit",
            "robot_y_limit",
            "stop_speed",
            "stop_hold_s",
            "stop_distance",
            "goal_penalty",
            "ball_y_limit",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.goal_x) or self.goal_x >= -self.robot_x_limit:
            raise ValueError("Goal line must be behind the guard region")
        if (
            not math.isfinite(self.target_margin)
            or self.target_margin < 0
            or self.target_half_width <= 0
        ):
            raise ValueError("Goal must accommodate the ball and target margin")
        if not math.isfinite(self.root_reset_z_offset):
            raise ValueError("Invalid reset height offset")
        if self.start_x_range[0] <= self.robot_x_limit or self.speed_range[0] <= 0:
            raise ValueError("Shots must start in front with positive speed")
        if self.ball_x_bounds[0] >= self.ball_x_bounds[1]:
            raise ValueError("ball_x_bounds must be strictly increasing")
        if self.fixed_motion_id is not None and (
            isinstance(self.fixed_motion_id, bool)
            or not isinstance(self.fixed_motion_id, int)
            or self.fixed_motion_id not in range(6)
        ):
            raise ValueError("fixed_motion_id must be None or an integer in [0, 5]")
        bounds = tuple(tuple(region) for region in self.intercept_region_bounds)
        if len(bounds) != 6 or any(
            len(region) != 4
            or not all(math.isfinite(value) for value in region)
            or region[0] > region[1]
            or region[2] > region[3]
            for region in bounds
        ):
            raise ValueError("intercept_region_bounds must contain six valid bounds")
        object.__setattr__(self, "intercept_region_bounds", bounds)

    @property
    def target_half_width(self):
        # Preserve the baseline's nominal-width convention. Posts are visual.
        return self.goal_width / 2 - self.ball_radius - self.target_margin

    @property
    def six_region_bounds(self):
        """Deprecated compatibility alias; regions are defined at intercept_x."""
        return self.intercept_region_bounds

    @property
    def episode_length_s(self):
        lateral = max(abs(v) for v in self.start_y_range) + self.target_half_width
        distance = math.hypot(self.start_x_range[1] - self.goal_x, lateral)
        return math.ceil(distance / self.speed_range[0] + 3)

    @classmethod
    def from_json(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text()))

    def to_dict(self):
        return asdict(self)
