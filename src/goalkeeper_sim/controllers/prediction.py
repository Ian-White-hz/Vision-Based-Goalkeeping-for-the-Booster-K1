"""Constant-velocity arrival prediction for the first controller baseline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BallState:
    """Robot-local ball state; +x is forward and +y is left."""

    position: tuple[float, float, float]
    velocity: tuple[float, float, float]


@dataclass(frozen=True)
class CrossingPrediction:
    time_s: float
    position: tuple[float, float, float]


def predict_crossing(
    state: BallState,
    defensive_line_x: float = 0.25,
    max_time_s: float = 5.0,
) -> CrossingPrediction | None:
    """Predict where an incoming ball crosses a robot-local defensive plane."""
    x, y, z = state.position
    vx, vy, vz = state.velocity
    if abs(vx) < 1e-6:
        return None
    time_s = (defensive_line_x - x) / vx
    if time_s < 0.0 or time_s > max_time_s:
        return None
    return CrossingPrediction(
        time_s=time_s,
        position=(defensive_line_x, y + vy * time_s, z + vz * time_s),
    )
