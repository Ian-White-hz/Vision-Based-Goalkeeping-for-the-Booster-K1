from dataclasses import asdict

import mjlab  # 先初始化 mjlab。
import torch

from goalkeeper_amp_test_utils import FakeEnv, FakeExpertSampler
from booster_mjlab.amp.algorithms.categorical_ppo import CategoricalAmpPPO
from booster_mjlab.amp.runners.categorical_on_policy_runner import (
    CategoricalAmpOnPolicyRunner,
)
from booster_mjlab.tasks.goalkeeper.config.k1.rl_cfg import (
    booster_k1_goalkeeper_amp_runner_cfg,
)


cfg = booster_k1_goalkeeper_amp_runner_cfg()

# 只缩短测试，不改变正式配置函数和网络结构。
cfg.num_steps_per_env = 4
cfg.algorithm.num_learning_epochs = 1
cfg.algorithm.num_mini_batches = 1

# 模拟入口只用于本测试。
cfg.amp.expert_sampler_factory = (
    "goalkeeper_amp_test_utils:create_fake_expert_sampler"
)
cfg.amp.motion_id_reader = (
    "goalkeeper_amp_test_utils:read_motion_ids"
)

env = FakeEnv()

runner = CategoricalAmpOnPolicyRunner(
    env,
    asdict(cfg),
    log_dir=None,
    device="cpu",
)

assert isinstance(runner.alg, CategoricalAmpPPO)
assert runner.amp_updater.expert_sampler is None

runner.learn(1)

# 两个模拟环境，各采集四步。
assert env.step_count == 4
assert runner.total_timesteps == 8
assert runner.current_learning_iteration == 1

# 确认真正请求过 expert 数据。
sampler = runner.amp_updater.expert_sampler
assert isinstance(sampler, FakeExpertSampler)
assert sampler.requests

# 更新后，三个学习模块的参数不能出现 NaN 或 Inf。
for module in (
    runner.alg.actor,
    runner.alg.critic,
    runner.amp_model,
):
    for parameter in module.parameters():
        assert torch.isfinite(parameter).all()

print("Goalkeeper AMP 正式配置的一轮更新检查通过")