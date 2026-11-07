import mjlab  # 先初始化 mjlab。

from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)

from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import (
    booster_k1_goalkeeper_amp_env_cfg,
    booster_k1_goalkeeper_env_cfg,
)


def test_joint_positions(env):
    """测试占位函数，不用于真实任务注册。"""
    return env.q


group = ObservationGroupCfg(
    terms={
        "joint_pos": ObservationTermCfg(
            func=test_joint_positions,
        ),
    },
    concatenate_terms=True,
    enable_corruption=False,
)

cfg = booster_k1_goalkeeper_amp_env_cfg(
    amp_observation=group,
    num_envs=2,
)

assert set(cfg.observations) == {"actor", "critic", "amp"}

# 接入的是副本，读取函数保持不变。
assert cfg.observations["amp"] is not group
assert (
    cfg.observations["amp"].terms["joint_pos"].func
    is test_joint_positions
)

# 普通任务仍然没有 AMP 组。
ordinary = booster_k1_goalkeeper_env_cfg(num_envs=2)
assert "amp" not in ordinary.observations

# 未增加第二份 AMP 奖励或改变已有任务项。
assert cfg.rewards.keys() == ordinary.rewards.keys()
assert cfg.actions.keys() == ordinary.actions.keys()
assert cfg.terminations.keys() == ordinary.terminations.keys()

print("AMP 环境配置入口检查通过")