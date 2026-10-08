"""Dependency-free checks for the first arrival predictor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from goalkeeper_sim.controllers import BallState, predict_crossing


def main() -> None:
    incoming = BallState(position=(2.0, 0.3, 0.09), velocity=(-0.5, -0.1, 0.0))
    prediction = predict_crossing(incoming, defensive_line_x=0.25)
    assert prediction is not None
    assert abs(prediction.time_s - 3.5) < 1e-9
    assert prediction.position[0] == 0.25
    assert abs(prediction.position[1] + 0.05) < 1e-9
    assert prediction.position[2] == 0.09

    stationary = BallState(position=(2.0, 0.0, 0.09), velocity=(0.0, 0.0, 0.0))
    assert predict_crossing(stationary) is None
    print("PASS: constant-velocity defensive-line prediction")


if __name__ == "__main__":
    main()
