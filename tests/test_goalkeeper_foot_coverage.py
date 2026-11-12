"""Analytical geometry checks independent of MuJoCo reward wiring."""
import math

import torch

from booster_mjlab.tasks.goalkeeper.mdp.foot_coverage import rolling_foot_gap, segment_box_distance

# Segment interior crosses the box even though both endpoints are outside.
half = torch.tensor([.1, .04, .03])
start = torch.tensor([[-.4, 0., 0.], [-.4, .14, 0.], [-.4, .14, .13]])
end = start.clone()
end[:, 0] = .4
torch.testing.assert_close(segment_box_distance(start, end, torch.zeros(3), half),
                           torch.tensor([0., .1, math.sqrt(.02)]), atol=1e-5, rtol=1e-5)
torch.testing.assert_close(segment_box_distance(end, start, torch.zeros(3), half),
                           segment_box_distance(start, end, torch.zeros(3), half))

feet = torch.tensor([[[0., .25, .03], [0., -.25, .03]]])
quat = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])
centers = torch.zeros(2, 3)
halves = half.repeat(2, 1)
ball = torch.tensor([[1., .25, .1]])
vel = torch.tensor([[-1., 0., 0.]])

def gap(p=ball, v=vel, f=feet, q=quat):
    return rolling_foot_gap(p, v, f, q, centers, halves, ball_radius=.1)

assert gap()[0, 0] < 0 and gap()[0, 1] > 0
middle = ball.clone()
middle[:, 1] = 0
assert (gap(p=middle) > 0).all(), 'Empty space between feet must remain uncovered'
raised = feet.clone()
raised[:, :, 2] = .6
assert (gap(f=raised) > 0).all()
# Foot ahead of the nominal intercept plane still blocks the approaching ball.
ahead = feet.clone()
ahead[:, :, 0] = .3
assert gap(f=ahead)[0, 0] < 0
# No use of trajectories behind the ball, outgoing balls or distant future feet.
past = feet.clone()
past[:, :, 0] = 2
assert torch.isinf(gap(f=past)).all()
assert torch.isinf(gap(v=-vel)).all()
far = feet.clone()
far[:, :, 0] = -5
assert torch.isinf(gap(f=far)).all()
# Translation invariance, including nonzero environment origins.
offset = torch.tensor([4., -3., 1.])
torch.testing.assert_close(gap(p=ball+offset, f=feet+offset), gap())
# A 90-degree yaw expands lateral reach for the long foot axis.
edge = ball.clone()
edge[:, 1] += .18
turned = quat.clone()
turned[:] = torch.tensor([math.sqrt(.5), 0., 0., math.sqrt(.5)])
assert gap(p=edge)[0, 0] > 0 and gap(p=edge, q=turned)[0, 0] < 0
# Pitch rotates the long axis into height; world-axis bounds would get this wrong.
high = ball.clone()
high[:, 2] = .21
pitched = quat.clone()
pitched[:] = torch.tensor([math.sqrt(.5), 0., math.sqrt(.5), 0.])
assert gap(p=high)[0, 0] > 0 and gap(p=high, q=pitched)[0, 0] < 0
# Mirror positions and velocity in y; symmetric boxes swap covered feet.
mirrored = ball.clone()
mirrored[:, 1] *= -1
torch.testing.assert_close(gap(p=mirrored), gap().flip(-1))
bad = ball.clone()
bad[:, 1] = float('nan')
assert torch.isinf(gap(p=bad)).all()
print('PASS: segment interiors, sphere/box gap, foot gap, yaw, raised feet, forward feet, horizon and mirrors')
