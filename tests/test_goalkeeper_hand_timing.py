"""Independent timing examples distinguish early departure, lateness and misses."""

import mjlab  # noqa: F401
import json
import numpy as np

from booster_mjlab.scripts.goalkeeper_hand_timing import (
    sample_at_time,
    summarize_hand_timing,
)


def make_rows():
    rows = []
    for i, t in enumerate((0., .1, .2, .3)):
        row = {"time": t, "ball_x": 1. - 5*t, "ball_y": 0., "ball_z": 0.,
               "ball_contact": False}
        for side, distances in (("left", [.4, .05, .4, .4]), ("right", [.4, .4, .4, .05])):
            prefix = f"{side}_hand_"
            row.update({prefix + "x": 0., prefix + "y": distances[i], prefix + "z": 0.,
                        prefix + "target_distance": distances[i], prefix + "contact": False})
            for axis in "xyz":
                row[prefix + "v" + axis] = 1. if axis == "z" else 0.
                row[prefix + "local_v" + axis] = .5 if axis == "z" else 0.
        rows.append(row)
    return rows


rows = make_rows()
r = summarize_hand_timing(rows, [0., 0., 0.], .2)
assert r["reference_kind"] == "ball_plane_crossing"
assert r["actual_plane_crossing_s"] == .2
assert r["hands"]["left"]["close_early_then_outside_at_reference"]
assert not r["hands"]["left"]["entered_after_reference"]
assert r["hands"]["right"]["entered_after_reference"]
np.testing.assert_allclose(r["hands"]["left"]["target_error_xyz_m"], [0., .4, 0.])
assert r["hands"]["left"]["speed_at_reference_m_s"] == 1.
assert r["hands"]["left"]["local_speed_at_reference_m_s"] == .5
np.testing.assert_allclose(sample_at_time(rows, .15)["ball_x"], .25)
assert sample_at_time(rows, .4) is None
assert sample_at_time(rows, -.1) is None
assert "ball_contact" not in sample_at_time(rows, .15)
json.dumps(summarize_hand_timing(rows, np.array([0., 0., 0.], dtype=np.float32), .2))

# An actual earlier collision changes the interpretation of the nominal plane.
rows[1]["ball_contact"] = True
rows[1]["left_hand_contact"] = True
r = summarize_hand_timing(rows, [0., 0., 0.], .2)
assert r["reference_kind"] == "first_robot_contact" and r["reference_s"] == .1
assert not r["hands"]["left"]["close_early_then_outside_at_reference"]
assert r["hands"]["left"]["first_hand_contact_s"] == .1
rows[1]["ball_contact"] = False
r = summarize_hand_timing(rows, [0., 0., 0.], .2)
assert r["first_robot_contact_s"] == .1 and r["first_whole_robot_sensor_contact_s"] is None

# Never arriving is different from arriving late. If the trial ended before
# the nominal event, leave the reference sample absent instead of extrapolating.
rows = make_rows()
for row in rows:
    row["ball_x"] = 2.
    row["right_hand_target_distance"] = .4
r = summarize_hand_timing(rows, [0., 0., 0.], .5)
assert r["reference_kind"] == "nominal_only" and r["reference_sample"] is None
assert r["hands"]["right"]["never_entered_target_radius"]
assert not r["hands"]["right"]["entered_after_reference"]
print("Hand timing: early departure, late arrival, no arrival, interpolation and early contact passed")
