from types import SimpleNamespace

import torch
from mjlab.envs import mdp as common_mdp
from mjlab.scene import SceneCfg

from booster_mjlab.tasks.goalkeeper.goalkeeper_env_cfg import (
    make_goalkeeper_env_cfg,
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