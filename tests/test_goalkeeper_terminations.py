from types import SimpleNamespace

import torch
from mjlab.envs import mdp as common_mdp
from mjlab.scene import SceneCfg

from booster_mjlab.tasks.goalkeeper.goalkeeper_env_cfg import (
    make_goalkeeper_env_cfg,
)

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_env_cfg,
)


cfg = make_goalkeeper_env_cfg(
    scene=SceneCfg(num_envs=2),
    decimation=4,
    episode_length_s=3.0,
)

term = cfg.terminations["time_out"]
assert term.func is common_mdp.time_out
assert term.time_out is True

env = SimpleNamespace(
    episode_length_buf=torch.tensor([149, 150]),
    max_episode_length=150,
)

result = term.func(env, **term.params)

assert result.shape == (2,)
assert result.dtype == torch.bool
assert torch.equal(result, torch.tensor([False, True]))

print("time_out:", result)
print("超时终止检查通过")

angles = torch.deg2rad(
    torch.tensor([0.0, 45.0, 75.0, 180.0])
)

gravity_b = torch.stack(
    (
        torch.zeros_like(angles),
        -torch.sin(angles),
        -torch.cos(angles),
    ),
    dim=-1,
)

tilt_env = SimpleNamespace(
    scene={
        "robot": SimpleNamespace(
            data=SimpleNamespace(
                projected_gravity_b=gravity_b,
            ),
        ),
    },
)

k1_cfg = booster_k1_goalkeeper_env_cfg(num_envs=4)
tilt_term = k1_cfg.terminations["excessive_body_tilt"]

assert tilt_term.time_out is False

result = tilt_term.func(
    tilt_env,
    **tilt_term.params,
)

assert result.shape == (4,)
assert result.dtype == torch.bool

torch.testing.assert_close(
    result,
    torch.tensor([False, False, True, True]),
)

print("身体倾斜终止:", result)
print("身体倾斜判定和配置连接检查通过")