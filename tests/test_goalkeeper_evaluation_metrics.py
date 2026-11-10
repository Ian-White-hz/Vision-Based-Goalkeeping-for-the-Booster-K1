"""Independent hand calculations, fixed mirrored shots and outcome ranking."""

import mjlab  # noqa: F401
from dataclasses import replace
import torch

from booster_mjlab.tasks.goalkeeper.evaluation import (
    ArrivalTracker, fixed_suite, selection_score, summarize_cases,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings


settings = FullGoalSettings()
shots = fixed_suite(settings, [101, 102], gravity=9.81)
again = fixed_suite(settings, [101, 102], gravity=9.81)
for key in shots:
    torch.testing.assert_close(shots[key], again[key], rtol=0, atol=0)
assert torch.bincount(shots["region"], minlength=6).tolist() == [2] * 6
for key in ("start_position_e", "velocity", "target_position_e", "goal_target_position_e"):
    mirror = shots[key][:3].clone()
    mirror[:, 1] *= -1
    torch.testing.assert_close(mirror, shots[key][3:6], rtol=0, atol=0)
default = fixed_suite(settings, range(100, 120), gravity=9.81)
assert torch.bincount(default["region"], minlength=6).tolist() == [20] * 6
bounds = list(settings.intercept_region_bounds)
bounds[1] = (-.4, -.3, .4, .5)
asymmetric = fixed_suite(replace(settings, intercept_region_bounds=tuple(bounds)), [101], gravity=9.81)
right = asymmetric["target_position_e"][asymmetric["region"] == 1]
assert ((right[:, 1] >= -.4) & (right[:, 1] <= -.3)).all()

previous = torch.tensor([[1., 0, 0], [1., 0, 0], [1., 0, 0]])
points = torch.tensor([[[0., .1, 0], [0., .5, 0], [0., 0, 0]]]).expand(3, -1, -1).clone()
tracker = ArrivalTracker(previous, points, torch.tensor([0, 1, 4]), 0.)
current = torch.tensor([[-1., 0, 0], [.8, 0, 0], [-1., 0, 0]])
tracker.update(current, points, torch.tensor([False, True, True]),
               torch.tensor([True, True, False]), time_s=.02, dt=.02)
a, b, c = tracker.records
assert abs(a["time_s"] - .01) < 1e-8 and abs(a["selected_hand_error_m"] - .1) < 1e-7
assert a["reference"] == "plane_crossing" and a["nearest_blocker_error_m"] == 0
assert b["reference"] == "first_robot_contact" and b["time_s"] == .02
assert c is None
tracker.update(current, points + 2, torch.ones(3, dtype=torch.bool),
               torch.zeros(3, dtype=torch.bool), time_s=.04, dt=.02)
assert tracker.records == [a, b, None]  # no completed-lane contamination

cases = [{"region": i, "save": False, "safe_save": False, "goal": True,
          "fall": False, "contact": True, "miss": True, "unresolved": False,
          "cutoff": False, "arrival": None} for i in range(6)]
metrics = summarize_cases(cases)
assert metrics["macro"]["contact_rate"] == 1 and metrics["macro"]["safe_save_rate"] == 0
assert metrics["macro"]["arrival_hand_error_m"] == 2
baseline = selection_score(metrics)
cases[0].update(save=True, safe_save=True, goal=False, miss=False)
assert selection_score(summarize_cases(cases)) > baseline
cases[0].update(safe_save=False, fall=True)
assert selection_score(summarize_cases(cases)) < baseline  # save + fall is not safe save
cases[0].update(save=False, fall=False, goal=True, miss=True, arrival=a)
assert selection_score(summarize_cases(cases)) > baseline  # geometry only breaks outcome ties
print("Fixed mirror/seed balance, arrival interpolation, contact timing, terminal masks, missing-error penalty and physical outcome ranking passed")
