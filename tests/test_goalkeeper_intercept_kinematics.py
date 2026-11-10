"""Closing speed is actual nearest-blocker progress, independent of labels."""

import mjlab  # noqa: F401
import torch

from booster_mjlab.tasks.goalkeeper.mdp.shot_state import nearest_intercept_kinematics

torch.set_default_dtype(torch.float64)
target = torch.tensor([[0., .8, .5], [0., -.8, .5], [0., 0., 1.2]])
points = torch.tensor([[[0., 0., .5], [2., 2., 2.]]]).repeat(3, 1, 1)
velocity = torch.tensor([[[0., .5, 0.], [0., 0., 0.]]]).repeat(3, 1, 1)
velocity[2, 0] = torch.tensor([0., 0., .5])
distance, closing = nearest_intercept_kinematics(points, velocity, target)
torch.testing.assert_close(distance, torch.tensor([.8, .8, .7]))
torch.testing.assert_close(closing, torch.tensor([.5, -.5, .5]))

# Verify against the independently measured change in distance, not a second
# implementation of the dot product. The target is fixed during each shot.
eps = 1e-7
next_distance = torch.linalg.vector_norm(points + eps * velocity - target[:, None], dim=-1).amin(dim=1)
torch.testing.assert_close((distance - next_distance) / eps, closing, atol=1e-7, rtol=1e-6)

# Overshoot automatically reverses the requested direction, with the same v.
points[0, 0, 1] = 1.6
torch.testing.assert_close(nearest_intercept_kinematics(points, velocity, target)[1][0], torch.tensor(-.5))
# Tangential movement gives zero first-order progress.
velocity[0, 0] = torch.tensor([.5, 0., 0.])
assert nearest_intercept_kinematics(points, velocity, target)[1][0] == 0

# Candidate identity, order and a world translation do not change the result.
for selected in range(5):
    p = torch.full((3, 5, 3), 9.)
    v = torch.zeros_like(p)
    p[:, selected] = points[:, 0]
    v[:, selected] = velocity[:, 0]
    d, c = nearest_intercept_kinematics(p, v, target)
    torch.testing.assert_close(d, nearest_intercept_kinematics(points, velocity, target)[0])
    torch.testing.assert_close(c, nearest_intercept_kinematics(points, velocity, target)[1])
    torch.testing.assert_close(nearest_intercept_kinematics(p.flip(1), v.flip(1), target)[1], c)
    offset = torch.tensor([[30., -20., 0.]])
    torch.testing.assert_close(nearest_intercept_kinematics(p + offset[:, None], v, target + offset)[1], c)
    p[..., 1] *= -1
    v[..., 1] *= -1
    mirrored_target = target.clone()
    mirrored_target[:, 1] *= -1
    torch.testing.assert_close(nearest_intercept_kinematics(p, v, mirrored_target)[1], c)

# Exact ties use the right derivative of min(distance), regardless of order.
p = torch.tensor([[[0., -.5, 0.], [0., .5, 0.]]])
v = torch.tensor([[[0., .2, 0.], [0., -.7, 0.]]])
t = torch.zeros(1, 3)
torch.testing.assert_close(nearest_intercept_kinematics(p, v, t)[1], torch.tensor([.7]))
torch.testing.assert_close(nearest_intercept_kinematics(p.flip(1), v.flip(1), t)[1], torch.tensor([.7]))
assert nearest_intercept_kinematics(t[:, None], torch.ones(1, 1, 3), t)[1].eq(0).all()
assert all(torch.isfinite(x).all() for x in nearest_intercept_kinematics(t[:, None], torch.ones(1, 1, 3), t))
torch.set_default_dtype(torch.float32)
print("Finite-difference, left/right/up, overshoot, tangent, all blockers, ties, offsets and mirror passed")
