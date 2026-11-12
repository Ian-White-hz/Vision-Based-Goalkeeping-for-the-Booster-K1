"""Independent hand calculations for command tracking and stopping behavior."""
import math

import torch

from booster_mjlab.tasks.goalkeeper.mdp.interception import (
    lateral_velocity_command,
    lateral_velocity_tracking_reward,
)

target = torch.tensor([.6, .4, .2, -.6, .6, .6])
root = torch.zeros(6)
distance = torch.tensor([1., 1., 1., 1., .2, .05])
command = lateral_velocity_command(target, root, distance)
torch.testing.assert_close(command, torch.tensor([.3, .1, 0., -.3, .15, 0.]))
torch.testing.assert_close(lateral_velocity_command(-target, -root, distance), -command)
assert lateral_velocity_command(torch.tensor([10.]), torch.zeros(1), torch.ones(1)).item() <= .300001

reward = lateral_velocity_tracking_reward(target, root, command, distance)
expected = torch.tensor([2*(1-math.exp(-2.25)), 2*(1-math.exp(-.25)), 0.,
                         2*(1-math.exp(-2.25)), 2*(1-math.exp(-.5625)), 0.])
torch.testing.assert_close(reward, expected)
torch.testing.assert_close(lateral_velocity_tracking_reward(-target, -root, -command, distance), reward)
assert lateral_velocity_tracking_reward(target, root, torch.zeros(6), distance).eq(0).all()
assert lateral_velocity_tracking_reward(target, root, command, distance, movement_weight=0).eq(0).all()

# Correct speed wins; overspeed is not an unlimited source of progress reward.
t = torch.full((5,), .6)
v = torch.tensor([-.3, 0., .3, .6, .9])
r = lateral_velocity_tracking_reward(t, torch.zeros(5), v, torch.ones(5))
assert r[0] < 0 and r[1] == 0 and r[2] > 0 and r[4] < 0
torch.testing.assert_close(r[3], torch.tensor(0.), atol=1e-6, rtol=0)
# Both dead corridor and blocker coverage request stopping, not disabled reward.
stop = lateral_velocity_tracking_reward(torch.tensor([.2, .6]), torch.zeros(2),
                                        torch.full((2,), .2), torch.tensor([1., .05]))
torch.testing.assert_close(stop, torch.full((2,), 2*(math.exp(-1)-1)))
assert lateral_velocity_command(torch.tensor([.6]), torch.tensor([1.]), torch.ones(1)) < 0

# Invalid samples are masked; all scalar configuration errors fail explicitly.
invalid = lateral_velocity_tracking_reward(torch.tensor([float('nan'), .6, .6]),
    torch.zeros(3), torch.tensor([.2, float('inf'), .2]), torch.tensor([1., 1., -1.]))
assert invalid.eq(0).all()
for name, value in [('position_gain', 0.), ('max_speed', 0.), ('velocity_std', 0.),
                    ('fade_distance', -1.), ('lateral_reach', -1.),
                    ('reach_threshold', -1.), ('movement_weight', float('nan'))]:
    try:
        lateral_velocity_tracking_reward(target, root, command, distance, **{name: value})
    except ValueError:
        pass
    else:
        raise AssertionError(name)
print('PASS: hand-calculated commands/rewards, symmetry, stationary baseline, braking, overspeed, masks')
