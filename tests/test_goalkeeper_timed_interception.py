"""Hand-calculated temporal coverage, prediction and root-only progress."""

from dataclasses import replace

import mjlab  # noqa: F401
import torch

from booster_mjlab.tasks.goalkeeper.mdp.interception import (
    InterceptionCfg, body_lateral_progress, coverage_factors, intercept_prediction,
)


cfg = InterceptionCfg()
distance = torch.tensor([.1, .1, .1, .1])
speed = torch.tensor([0., 0., 0., 3.])
tau = torch.tensor([.4, .1, 0., 0.])
w, hold = coverage_factors(distance, speed, tau, cfg)
torch.testing.assert_close(w, torch.tensor([.2, .4, 1., 1.]))
torch.testing.assert_close(hold, torch.tensor([1., 1., 1., .775]))
assert (w * hold)[2] > (w * hold)[3] > (w * hold)[0]
# A point that was close early but is distant at arrival cannot retain a
# memorized attainment bonus. Constant coverage earns more integrated score.
t = torch.linspace(.4, 0., 21)
w, _ = coverage_factors(torch.full_like(t, .1), torch.zeros_like(t), t, cfg)
held = (w * torch.sigmoid(torch.full_like(t, .5))).sum()
sweep_distance = torch.where(t > .25, .1, .5)
swept = (w * torch.sigmoid(5 * (.2 - sweep_distance))).sum()
assert held > swept

nominal = torch.tensor([[.1, .3, .5]]).repeat(4, 1)
pos = torch.tensor([[.5, .1, .4], [.5, .1, .4], [2.1, .1, .4], [.5, .1, .4]])
vel = torch.tensor([[-4., 4., 2.]]).repeat(4, 1)
rolling = torch.tensor([False, True, False, False])
target, times, valid = intercept_prediction(pos, vel, nominal, rolling, torch.tensor([0., 0., -10.]), cfg)
# First row: tau=.1, predicted y=.5/z=.55; blend=.5 -> y=.4,z=.525.
torch.testing.assert_close(target[0], torch.tensor([.1, .4, .525]))
torch.testing.assert_close(target[1], torch.tensor([.1, .4, .5]))
torch.testing.assert_close(target[2], nominal[2])  # far prediction cannot drift target
torch.testing.assert_close(times, torch.tensor([.1, .1, .5, .1]))
assert valid.all() and torch.equal(nominal, torch.tensor([[.1, .3, .5]]).repeat(4, 1))
assert (target - nominal).abs().max() <= cfg.prediction_limit_m
# Mirror y and vy, keeping x/z/time fixed.
pos[:, 1] *= -1
vel[:, 1] *= -1
nominal[:, 1] *= -1
mirrored, _, _ = intercept_prediction(pos, vel, nominal, rolling, torch.tensor([0., 0., -10.]), cfg)
target[:, 1] *= -1
torch.testing.assert_close(mirrored, target)
# Stopped/outgoing/past-plane/NaN inputs have no active shaping.
pos[:, 0] = torch.tensor([.5, .5, 0., float('nan')])
vel[:, 0] = torch.tensor([0., 1., -4., -4.])
p, _, valid = intercept_prediction(pos, vel, nominal, rolling, torch.tensor([0., 0., -10.]), cfg)
assert not valid.any() and torch.isfinite(p).all()

# +/- target direction, wrong-way root motion, overshoot, in-reach targets,
# and a near blocker that still receives guidance outside 5 cm. No arm
# velocity occurs in this interface.
target_y = torch.tensor([.8, -.8, .8, .8, .2, .8])
root_y = torch.tensor([0., 0., 0., 1.6, 0., 0.])
vy = torch.full((6,), .5)
d = torch.tensor([.6, .6, .6, .6, .6, .1])
vy[2] = -.5
r = body_lateral_progress(target_y, root_y, vy, d)
torch.testing.assert_close(r, torch.tensor([.25, -.25, -.25, -.25, 0., 1 / 24]))
assert body_lateral_progress(target_y, root_y, vy * 0, d).eq(0).all()
assert body_lateral_progress(target_y, root_y, vy, d, movement_weight=0).eq(0).all()
for kwargs in ({'arrival_window_s':0.}, {'prediction_limit_m':-1.}, {'settling_weight':1.1},
               {'preparation_weight':float('nan')}, {'settling_speed_m_s':0.}):
    try:
        replace(cfg, **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(kwargs)
print("Temporal hand calculations, early sweep vs hold, rolling/ballistic prediction, fixed plane, mirror, invalid states and root-only direction passed")
