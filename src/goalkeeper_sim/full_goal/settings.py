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
    speed_range: tuple[float, float] = (0.35, 0.50)
    target_margin: float = 0.02
    robot_x_limit: float = 0.30
    robot_y_limit: float = 1.35
    root_reset_z_offset: float = 0.005
    stop_speed: float = 0.12
    stop_hold_s: float = 0.4
    stop_distance: float = 0.75

    def __post_init__(self):
        for name in ("start_x_range", "start_y_range", "speed_range"):
            values = tuple(getattr(self, name))
            if len(values) != 2 or not all(math.isfinite(v) for v in values) or values[0] > values[1]:
                raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, values)
        friction = tuple(self.ball_friction)
        if len(friction) != 3 or any(not math.isfinite(v) or v < 0 for v in friction):
            raise ValueError("Invalid ball_friction")
        object.__setattr__(self, "ball_friction", friction)
        for name in ("goal_width", "goal_height", "post_radius", "ball_radius", "ball_mass",
                     "robot_x_limit", "robot_y_limit", "stop_speed", "stop_hold_s", "stop_distance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.goal_x) or self.goal_x >= -self.robot_x_limit:
            raise ValueError("Goal line must be behind the guard region")
        if not math.isfinite(self.target_margin) or self.target_margin < 0 or self.target_half_width <= 0:
            raise ValueError("Goal must accommodate the ball and target margin")
        if not math.isfinite(self.root_reset_z_offset):
            raise ValueError("Invalid reset height offset")
        if self.start_x_range[0] <= self.robot_x_limit or self.speed_range[0] <= 0:
            raise ValueError("Shots must start in front with positive speed")

    @property
    def target_half_width(self):
        # Preserve the baseline's nominal-width convention. Posts are visual.
        return self.goal_width / 2 - self.ball_radius - self.target_margin

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
