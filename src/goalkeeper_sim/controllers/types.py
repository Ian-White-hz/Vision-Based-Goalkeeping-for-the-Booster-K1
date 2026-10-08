"""High-level action identifiers independent of joint trajectories."""

from enum import Enum


class ActionId(str, Enum):
    HOLD = "hold"
    REACH_LEFT = "reach_left"
    REACH_RIGHT = "reach_right"
    FOOT_STOP = "foot_stop"
    ADJUST_LEFT = "adjust_left"
    ADJUST_RIGHT = "adjust_right"
