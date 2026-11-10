import mjlab  # noqa: F401 -- 先初始化 mjlab，再加载项目任务配置。

from dataclasses import asdict

from booster_mjlab.amp.runners.categorical_runner_cfg import (
    CategoricalAmpPpoAlgorithmCfg,
    CategoricalAmpRunnerCfg,
)
from booster_mjlab.tasks.goalkeeper.config.k1.rl_cfg import (
    booster_k1_goalkeeper_amp_runner_cfg,
    booster_k1_goalkeeper_ppo_runner_cfg,
)


base = booster_k1_goalkeeper_ppo_runner_cfg()
cfg = booster_k1_goalkeeper_amp_runner_cfg()

assert isinstance(cfg, CategoricalAmpRunnerCfg)
assert isinstance(cfg.algorithm, CategoricalAmpPpoAlgorithmCfg)
assert cfg.class_name == "CategoricalAmpOnPolicyRunner"
assert cfg.algorithm.class_name == (
    "booster_mjlab.amp.algorithms.categorical_ppo:CategoricalAmpPPO"
)

# 除算法入口之外，PPO 参数保持一致。
base_algorithm = asdict(base.algorithm)
amp_algorithm = asdict(cfg.algorithm)
base_algorithm.pop("class_name")
amp_algorithm.pop("class_name")
assert amp_algorithm == base_algorithm

assert asdict(cfg.actor) == asdict(base.actor)
assert asdict(cfg.critic) == asdict(base.critic)
assert cfg.obs_groups == base.obs_groups
assert "amp" not in cfg.obs_groups["actor"]
assert "amp" not in cfg.obs_groups["critic"]

assert cfg.experiment_name == "k1_goalkeeper_amp"
assert cfg.amp.style_weight == 0.4
assert cfg.amp.reward_mode == "direct"
assert cfg.amp.hidden_layer_sizes == (512, 256)
assert cfg.amp.expert_sampler_factory.endswith(":create_goalkeeper_expert_sampler")
assert cfg.amp.motion_id_reader.endswith(":get_amp_motion_ids")

# 不同配置实例之间不能相互影响。
another = booster_k1_goalkeeper_amp_runner_cfg()
cfg.amp.style_weight = 0.1
assert another.amp.style_weight == 0.4

assert base.experiment_name == "k1_goalkeeper"

print("Goalkeeper AMP 专用 RL 配置检查通过")
