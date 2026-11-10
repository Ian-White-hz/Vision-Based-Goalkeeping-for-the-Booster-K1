"""Heading deadband, angle wrap, symmetry, capped rate and density scaling."""

from types import SimpleNamespace as NS

import mjlab  # noqa: F401
import torch

from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    excessive_yaw_rate_cost,
    excessive_yaw_rate_from_env,
    heading_deviation_cost,
    heading_deviation_from_env,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings


def yaw_quaternions(degrees):
    angle = torch.deg2rad(torch.tensor(degrees, dtype=torch.float32)) / 2
    return torch.stack((angle.cos(), angle * 0, angle * 0, angle.sin()), dim=-1)


q = yaw_quaternions([0, 20, 55, 90, 135, -55, 179, -179])
reference = yaw_quaternions([0, 0, 0, 0, 0, 0, -179, 179])
cost = heading_deviation_cost(q, reference)
torch.testing.assert_close(
    cost, torch.tensor([0.0, 0.0, 0.25, 1.0, 1.0, 0.25, 0.0, 0.0]), atol=1e-6, rtol=1e-5
)
torch.testing.assert_close(heading_deviation_cost(-q, -reference), cost)
torch.testing.assert_close(
    heading_deviation_cost(yaw_quaternions([95, -15]), yaw_quaternions([40, 40])),
    torch.tensor([0.25, 0.25]),
)
assert ((0 <= cost) & (cost <= 1)).all()

rate = torch.tensor([0.0, 0.5, 1.75, 3.0, 10.0, -1.75])
rate_cost = excessive_yaw_rate_cost(rate)
torch.testing.assert_close(rate_cost, torch.tensor([0.0, 0.0, 0.25, 1.0, 1.0, 0.25]))
torch.testing.assert_close(excessive_yaw_rate_cost(-rate), rate_cost)

# The heading wrapper references the configured default root orientation, never
# the last physical episode's terminal quaternion; rate uses world vertical z.
root_state = torch.zeros((2, 13))
root_state[:, 3:7] = yaw_quaternions([40, 40])
env = NS(
    scene={
        "robot": NS(
            data=NS(
                root_link_quat_w=yaw_quaternions([95, -15]),
                default_root_state=root_state,
                root_link_ang_vel_w=torch.tensor(
                    [[8.0, -4.0, 1.75], [-8.0, 4.0, -1.75]]
                ),
            )
        )
    }
)
torch.testing.assert_close(heading_deviation_from_env(env), torch.tensor([0.25, 0.25]))
torch.testing.assert_close(excessive_yaw_rate_from_env(env), torch.tensor([0.25, 0.25]))

# These are running costs and are multiplied by dt once, unlike goal events.
settings = FullGoalSettings()
for dt in (0.01, 0.02, 0.04):
    steps = round(1 / dt)
    task = (
        (-settings.heading_penalty * 0.25 - settings.yaw_rate_penalty * 0.25)
        * dt
        * steps
    )
    assert abs(task - (-1.5)) < 1e-6
    assert abs(0.6 * task - (-0.9)) < 1e-6
assert (
    FullGoalSettings(heading_penalty=0.0, yaw_rate_penalty=0.0).heading_penalty == 0.0
)

invalid = [
    {"heading_penalty": -1},
    {"yaw_rate_penalty": float("nan")},
    {"heading_deadband_deg": 90},
    {"heading_full_penalty_deg": 181},
    {"heading_deadband_deg": -1},
    {"yaw_rate_deadband": 3},
    {"yaw_rate_full_penalty": 0},
    {"yaw_rate_deadband": float("nan")},
]
for params in invalid:
    try:
        FullGoalSettings(**params)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Invalid turning settings accepted: {params}")

for call in (
    lambda: heading_deviation_cost(q[:, :3], reference),
    lambda: heading_deviation_cost(q, reference, deadband_deg=90),
    lambda: excessive_yaw_rate_cost(rate[:, None]),
    lambda: excessive_yaw_rate_cost(rate, deadband=3),
):
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid turning cost input accepted")

print("朝向代价:", cost)
print("转速代价:", rate_cost)
print("20 度死区、最短角差、左右对称、限幅、默认方向与 dt 缩放检查通过")
