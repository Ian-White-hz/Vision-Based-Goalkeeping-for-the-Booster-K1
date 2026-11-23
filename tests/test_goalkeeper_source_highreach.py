import math

import torch

from booster_mjlab.tasks.goalkeeper.source.k1_rewards import K1Rewards
from booster_mjlab.tasks.goalkeeper.source.settings import (
    K1_ONLY_REWARDS,
    REWARD_SCALES,
)

# One env per case: (region, ball x, hand0 offset (dy, dz), hand1 offset,
# ball x velocity now, tilt (projected gravity x), expected reward).
FAR = (2.0, -2.0)


def shape(q):
    # q = sum of squared sigma-normalized offsets; tight + 3x wide Gaussian.
    return 0.5 * math.exp(-q) + 0.5 * math.exp(-q / 9.0)


E = shape(1.0)
CASES = [
    (2, 0.8, (0.0, 0.0), FAR, -3.0, 0.0, 1.0),  # at the crossing point
    (2, 0.8, (0.0, -0.05), FAR, -3.0, 0.0, shape(0.25)),  # 5 cm low
    (2, 0.8, (0.0, -0.10), FAR, -3.0, 0.0, E),  # 10 cm low
    (2, 0.8, (0.0, 0.10), FAR, -3.0, 0.0, 1.0),  # up to 10 cm high is free
    (2, 0.8, (0.0, 0.25), FAR, -3.0, 0.0, E),  # 15 cm past the free band
    (3, 0.8, (0.15, 0.0), FAR, -3.0, 0.0, E),  # 15 cm aside
    (2, 0.8, FAR, (0.0, 0.0), -3.0, 0.0, 1.0),  # either hand counts
    (2, 0.8, (0.0, 0.0), FAR, -3.0, 0.6, 1 - 0.36),  # tilted trunk
    (2, 0.8, (0.3, -0.2), FAR, -3.0, 0.0, shape(8.0)),  # far: wide part only
    (0, 0.8, (0.0, 0.0), FAR, -3.0, 0.0, 0.0),  # mid region: off
    (4, 0.8, (0.0, 0.0), FAR, -3.0, 0.0, 0.0),  # low region: off
    (3, 2.0, (0.0, 0.0), FAR, -3.0, 0.0, 0.0),  # far phase (x > PHASE1_X)
    (3, -0.2, (0.0, 0.0), FAR, -3.0, 0.0, 0.0),  # ball behind the robot
    (2, 0.8, (0.0, 0.0), FAR, 1.0, 0.0, 0.0),  # deflected
]

n = len(CASES)
s = object.__new__(K1Rewards)
s.end_regions = torch.tensor([c[0] for c in CASES])
s.env_origins = torch.zeros(n, 3)
s.end_target = torch.tensor([[0.0, 0.4, 1.0]]).repeat(n, 1)
s.ball_states = torch.zeros(n, 13)
s.ball_states[:, 0] = torch.tensor([c[1] for c in CASES])
s.ball_states[:, 7] = torch.tensor([c[4] for c in CASES])
s.ball_vel = torch.full((n,), -3.0)
s.hand_indices = [0, 1]
s.rigid_body_states = torch.zeros(n, 2, 13)
for i, c in enumerate(CASES):
    for k, (dy, dz) in enumerate((c[2], c[3])):
        s.rigid_body_states[i, k, :3] = torch.tensor([0.0, 0.4 + dy, 1.0 + dz])
s.projected_gravity = torch.zeros(n, 3)
s.projected_gravity[:, 0] = torch.tensor([c[5] for c in CASES])
s.projected_gravity[:, 2] = -1.0

reward = s._reward_highreach()
torch.testing.assert_close(reward, torch.tensor([c[6] for c in CASES]))
assert REWARD_SCALES["highreach"] == 0.0  # opt-in
assert set(K1_ONLY_REWARDS) <= set(REWARD_SCALES)
print("highreach: shape, either hand, tilt, region and phase gating passed")
