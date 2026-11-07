import mjlab  # noqa: F401 -- 先初始化 mjlab。

from booster_mjlab.tasks.goalkeeper.amp_interfaces import amp_joint_positions
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_amp_env_cfg,
    booster_k1_goalkeeper_env_cfg,
)

cfg = booster_k1_goalkeeper_amp_env_cfg(
    num_envs=2,
)

assert set(cfg.observations) == {"actor", "critic", "amp"}

assert cfg.observations["amp"].terms["joint_pos"].func is amp_joint_positions
assert cfg.observations["amp"].enable_corruption is False

# 普通任务仍然没有 AMP 组。
ordinary = booster_k1_goalkeeper_env_cfg(num_envs=2)
assert "amp" not in ordinary.observations

# 未增加第二份 AMP 奖励或改变已有任务项。
assert cfg.rewards.keys() == ordinary.rewards.keys()
assert cfg.actions.keys() == ordinary.actions.keys()
assert cfg.terminations.keys() == ordinary.terminations.keys()

print("AMP 环境配置入口检查通过")
