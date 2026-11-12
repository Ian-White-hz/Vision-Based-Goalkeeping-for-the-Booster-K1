"""Tilt deadband, symmetry, saturation and invalid configuration checks."""

import torch

from booster_mjlab.tasks.goalkeeper.mdp.rewards import body_tilt_cost


angles = torch.deg2rad(torch.tensor([0., 20., 30., 40., 50., 60., 90., 180.]))
gravity = torch.stack((angles.sin(), torch.zeros_like(angles), -angles.cos()), -1)
expected = torch.tensor([0., 0., .0625, .25, .5625, 1., 1., 1.])
torch.testing.assert_close(body_tilt_cost(gravity), expected, atol=1e-6, rtol=1e-5)
for sign in (-1, 1):
    mirrored = gravity.clone()
    mirrored[:, 0] *= sign
    torch.testing.assert_close(body_tilt_cost(mirrored), expected, atol=1e-6, rtol=1e-5)
    # Roll and pitch have the same cost; vertical-axis heading does not enter it.
    torch.testing.assert_close(body_tilt_cost(mirrored[:, [1, 0, 2]]), expected,
                               atol=1e-6, rtol=1e-5)
assert body_tilt_cost(torch.empty(0, 3)).shape == (0,)
assert torch.isfinite(body_tilt_cost(torch.tensor([[0., 0., -1.000001]]))).all()
for params in ({"deadband_deg": -1.}, {"deadband_deg": 60.},
               {"full_penalty_deg": 181.}, {"deadband_deg": float("nan")}):
    try:
        body_tilt_cost(gravity, **params)
    except ValueError:
        pass
    else:
        raise AssertionError(params)
print("Tilt cost: deadband, roll/pitch symmetry, saturation and validation passed")
