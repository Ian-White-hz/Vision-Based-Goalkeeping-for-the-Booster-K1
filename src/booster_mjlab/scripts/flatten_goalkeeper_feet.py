"""Write a copy of the K1 goalkeeper clips with level stance feet.

Usage: python -m booster_mjlab.scripts.flatten_goalkeeper_feet
Reads artifacts/goalkeeper_k1/serial, writes artifacts/goalkeeper_k1/serial_flatfoot.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from booster_mjlab.motion.flat_foot import flatten_stance_feet, k1_model, sole_x_range

CLIPS = ("lefthand", "righthand", "leftjump", "rightjump", "leftstep", "rightstep")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path("artifacts/goalkeeper_k1/serial")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/goalkeeper_k1/serial_flatfoot")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = k1_model()
    reports = {"sole_x_range_m": sole_x_range()}
    for name in CLIPS:
        with (args.source / f"{name}.pkl").open("rb") as file:
            motion = pickle.load(file)
        (
            motion["dof_pos"],
            motion["root_pos"],
            motion["root_rot"],
            reports[name],
        ) = flatten_stance_feet(
            motion["root_pos"], motion["root_rot"], motion["dof_pos"], model
        )
        with (args.output / f"{name}.pkl").open("wb") as file:
            pickle.dump(motion, file, protocol=pickle.HIGHEST_PROTOCOL)
        before, after = reports[name]["before"], reports[name]["after"]
        change = reports[name]["ankle_pitch_change_rad"]
        print(
            f"{name:10s} stance {before['stance_frames']:3d} | foot pitch p50 "
            f"{before['stance_foot_pitch_deg_p10_p50_p90'][1]:+5.1f} -> "
            f"{after['stance_foot_pitch_deg_p10_p50_p90'][1]:+5.1f} deg | COM ahead of "
            f"rear support {before['com_ahead_of_rear_support_m_p10_p50_p90']} -> "
            f"{after['com_ahead_of_rear_support_m_p10_p50_p90']} | behind support "
            f"{before['com_behind_support_fraction']:.2f} -> "
            f"{after['com_behind_support_fraction']:.2f} | level max "
            f"{change['level_max_abs']:.2f}, lean max {change['lean_max_abs']:.2f} rad "
            f"({change['lean_frames']} frames)"
        )
    (args.output / "flat_foot_report.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
