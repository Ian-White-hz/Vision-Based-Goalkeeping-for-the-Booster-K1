"""Check a strictly mirrored pair of hand-height goalkeeper shots."""

import torch

from booster_mjlab.scripts.diagnose_goalkeeper_policy import make_mirrored_shots
from booster_mjlab.tasks.goalkeeper.scene.events import compute_ballistic_intercept
from booster_mjlab.tasks.goalkeeper.scene.settings import (
    FullGoalSettings,
)


settings = FullGoalSettings()
pair = make_mirrored_shots(settings, seed=42, left_region=0)

mirror_fields = (
    "start_position_e",
    "target_position_e",
    "goal_target_position_e",
    "velocity",
)

for name in mirror_fields:
    torch.testing.assert_close(
        pair[name][0, [0, 2]],
        pair[name][1, [0, 2]],
    )
    torch.testing.assert_close(
        pair[name][0, 1],
        -pair[name][1, 1],
    )

for name in ("flight_time", "intercept_time"):
    torch.testing.assert_close(pair[name][0], pair[name][1])

assert pair["region"].tolist() == [0, 1]
assert not pair["is_rolling"].any()
assert pair["target_position_e"][0, 1] > 0.0
assert pair["target_position_e"][1, 1] < 0.0

print("起点：", pair["start_position_e"])
print("拦截目标：", pair["target_position_e"])
print("初速度：", pair["velocity"])
print("左右镜像来球检查通过")

# Verify all three mirrored pairs and their actual nominal intercepts.
for left_region in (0, 2, 4):
    pair = make_mirrored_shots(settings, seed=42, left_region=left_region)
    assert pair["region"].tolist() == [left_region, left_region + 1]
    for name in mirror_fields:
        torch.testing.assert_close(pair[name][0, [0, 2]], pair[name][1, [0, 2]])
        torch.testing.assert_close(pair[name][0, 1], -pair[name][1, 1])
    torch.testing.assert_close(pair["intercept_time"][0], pair["intercept_time"][1])
    if left_region == 4:
        assert pair["is_rolling"].all()
        intercept = (
            pair["start_position_e"]
            + pair["velocity"] * pair["intercept_time"][:, None]
        )
    else:
        assert not pair["is_rolling"].any()
        intercept, time = compute_ballistic_intercept(
            pair["start_position_e"], pair["velocity"], intercept_x=settings.intercept_x
        )
        torch.testing.assert_close(time, pair["intercept_time"])
    torch.testing.assert_close(
        intercept, pair["target_position_e"], atol=2e-6, rtol=1e-5
    )

print("左右手、左右跳跃、左右滚地球的镜像弹道检查通过")
