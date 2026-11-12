"""Tilt-only event costs and a 5 cm movement cutoff, with hand calculations."""

from types import SimpleNamespace as NS

import mjlab  # noqa: F401
import torch

from booster_mjlab.amp.runners.reward_mixing import mix_amp_rewards
from booster_mjlab.tasks.goalkeeper.mdp.interception import body_lateral_progress
from booster_mjlab.tasks.goalkeeper.mdp.rewards import fall_termination_cost
from booster_mjlab.tasks.goalkeeper.mdp.terminations import excessive_body_tilt
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings


settings = FullGoalSettings()
assert settings.fall_penalty == 5.0
tilts = torch.deg2rad(torch.tensor([0., 55., 65., -65.]))
robot = NS(data=NS(projected_gravity_b=torch.stack(
    (tilts.sin(), torch.zeros_like(tilts), -tilts.cos()), dim=-1,
)))
termination = NS(func=excessive_body_tilt, params={"max_tilt_deg": 60.})


def get_term_cfg(name):
    assert name == "excessive_body_tilt"
    return termination


env = NS(scene={"robot": robot}, cfg=NS(scale_rewards_by_dt=True), step_dt=.02,
         termination_manager=NS(get_term_cfg=get_term_cfg),
         # These deliberately disagree with tilt: generic done flags must not
         # decide this reward, and a concurrent timeout must not hide a fall.
         reset_terminated=torch.tensor([True, True, True, False]),
         reset_time_outs=torch.tensor([True, False, False, True]))
expected = torch.tensor([0., 0., -settings.fall_penalty, -settings.fall_penalty])
for scaled in (False, True):
    env.cfg.scale_rewards_by_dt = scaled
    for dt in (.01, .02, .04):
        env.step_dt = dt
        raw = fall_termination_cost(env)
        torch.testing.assert_close(fall_termination_cost(env), raw)
        task = -settings.fall_penalty * raw * (dt if scaled else 1.)
        torch.testing.assert_close(task, expected)
        mixed, _, _ = mix_amp_rewards(task, torch.zeros(4), style_weight=.4,
                                      step_dt=dt, scale_rewards_by_dt=scaled)
        torch.testing.assert_close(mixed, .6 * expected)

# Reuse the configured threshold rather than hardcoding another angle in reward.
termination.params["max_tilt_deg"] = 75.
assert not fall_termination_cost(env).any()
assert FullGoalSettings(fall_penalty=0.).fall_penalty == 0.
for value in (-1., float("nan"), float("inf")):
    try:
        FullGoalSettings(fall_penalty=value)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Invalid fall penalty accepted: {value}")

d = torch.tensor([0., .05, .051, .10, .20, .35, .50])
target, root, speed = torch.full_like(d, .8), torch.zeros_like(d), torch.full_like(d, .5)
new = body_lateral_progress(target, root, speed, d)
old = body_lateral_progress(target, root, speed, d, reach_threshold=.2)
torch.testing.assert_close(new, .25 * torch.tensor([0., 0., 1/300, 1/6, .5, 1., 1.]), atol=1e-7, rtol=1e-5)
torch.testing.assert_close(old, .25 * torch.tensor([0., 0., 0., 0., 0., .5, 1.]))
assert new[2] > 0 and new[3] > 0 and new[4] > 0
torch.testing.assert_close(body_lateral_progress(-target, -root, -speed, d), new)
torch.testing.assert_close(body_lateral_progress(target, root, -speed, d), -new)
# The separate 30 cm lateral corridor is unchanged.
assert not body_lateral_progress(torch.full_like(d, .3), root, speed, d).any()
print("Fall task/mixed: -5/-3, tilt-only, repeated reads, dt on/off and configured angle passed")
print("5 cm movement cutoff:", new.tolist(), "old:", old.tolist())
