import mjlab  # noqa: F401 -- initialize mjlab before task imports
import torch

from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    intercept_proximity_reward,
    intercept_movement_reward,
    movement_guided_intercept_reward,
)


distance = torch.tensor([0.0, 0.1, 0.2, 0.4, 0.2])
active = torch.tensor([True, True, True, True, False])

reward = intercept_proximity_reward(distance, active)

assert reward.shape == (5,)
assert torch.isfinite(reward).all()

torch.testing.assert_close(
    reward,
    torch.tensor([
        0.7310586,
        0.6224593,
        0.5,
        0.2689414,
        0.0,
    ]),
)

assert (reward[:3] > reward[1:4]).all()

# 所有环境都退出拦截阶段后，奖励全部为零。
disabled = intercept_proximity_reward(
    distance,
    torch.zeros_like(active),
)
torch.testing.assert_close(disabled, torch.zeros_like(distance))

print("统一距离奖励:", reward)
print("距离单调性和阶段开关检查通过")

# Independent hand calculations: d=.5 gives gate=1, p=sigmoid(-1.5).
# Signed progress distinguishes approaching, stationary and receding motion.
distance = torch.tensor([.5, .5, .5, .35, .2, .1, .5, .5])
closing = torch.tensor([.5, 0., -.5, .5, .5, -.5, 10., -10.])
movement = intercept_movement_reward(distance, closing)
torch.testing.assert_close(movement, torch.tensor([1.5, 0., -1.5, .75, 0., 0., 3., -3.]))
gravity = torch.tensor([[0., 0., -1.]]).repeat(8, 1)
actual = movement_guided_intercept_reward(distance, torch.ones(8, dtype=torch.bool), closing, gravity)
torch.testing.assert_close(actual, torch.sigmoid(5 * (.2 - distance)) + movement)
assert actual[0] > actual[1] > actual[2] and actual[2] < 0

# Attained points retain proximity but cannot earn a motion bonus by rushing
# past the target; after leaving the radius, receding motion is penalized.
assert intercept_movement_reward(torch.tensor([.2, .1, .0]), torch.ones(3)).eq(0).all()
assert intercept_movement_reward(torch.tensor([.3]), torch.tensor([-.5])).item() < 0
disabled = movement_guided_intercept_reward(distance, torch.zeros(8, dtype=torch.bool), closing, gravity)
assert disabled.eq(0).all()
gravity[:, 0] = 1
assert movement_guided_intercept_reward(distance, torch.ones(8, dtype=torch.bool), closing, gravity).eq(0).all()

# Configuration accepts zero movement weight for an explicit ablation.
assert intercept_movement_reward(distance, closing, movement_weight=0).eq(0).all()
for name, value in (("movement_weight", -1.), ("movement_weight", float("nan")),
                    ("movement_fade_distance", 0.), ("movement_speed_limit", float("inf")),
                    ("movement_speed_limit", -1.), ("reach_threshold", -1.)):
    try:
        intercept_movement_reward(distance, closing, **{name: value})
    except ValueError:
        pass
    else:
        raise AssertionError((name, value))
print("Signed progress, approach/recede, attainment fade, speed cap, gating and config checks passed")
