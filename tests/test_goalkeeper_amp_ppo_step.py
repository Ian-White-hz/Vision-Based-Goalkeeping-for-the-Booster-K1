from types import SimpleNamespace

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from booster_mjlab.amp.modules.categorical_collector import (
    CategoricalAmpCollector,
)
from booster_mjlab.amp.modules.categorical_discriminator import (
    CategoricalAmpDiscriminator,
)
from booster_mjlab.amp.modules.categorical_replay import (
    CategoricalAmpReplayBuffer,
)
from booster_mjlab.amp.runners.categorical_step import (
    collect_categorical_amp_step,
)

import copy

from booster_mjlab.amp.algorithms.categorical_amp import (
    CategoricalAmpUpdater,
)
from booster_mjlab.amp.runners.categorical_iteration import (
    run_categorical_amp_iteration,
)

class FakeEnv:
    def __init__(self):
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.num_actions = 22
        self.step_dt = 0.02
        self.cfg = SimpleNamespace(scale_rewards_by_dt=True)
        self.motion_ids = torch.tensor([0, 3], dtype=torch.int64)
        self.q = torch.zeros(2, 22)

    @property
    def unwrapped(self):
        return self

    def get_observations(self):
        return TensorDict(
            {
                "actor": torch.zeros(2, 92),
                "critic": torch.zeros(2, 95),
                "amp": self.q.clone(),
            },
            batch_size=[2],
        )

    def step(self, actions):
        assert actions.shape == (2, 22)

        self.q = torch.stack([
            torch.full((22,), 0.1),
            torch.full((22,), -0.2),
        ])
        self.motion_ids[1] = 5

        return (
            self.get_observations(),
            torch.tensor([0.2, -0.4]),
            torch.tensor([0, 1], dtype=torch.int64),
            {"time_outs": torch.tensor([False, False])},
        )


torch.manual_seed(42)

env = FakeEnv()
obs = env.get_observations()
groups = {"actor": ["actor"], "critic": ["critic"]}

actor = MLPModel(
    obs,
    groups,
    "actor",
    output_dim=22,
    hidden_dims=(32, 16),
    activation="relu",
    distribution_cfg={
        "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
        "init_std": 0.1,
    },
)
critic = MLPModel(
    obs,
    groups,
    "critic",
    output_dim=1,
    hidden_dims=(32, 16),
    activation="relu",
)

storage = RolloutStorage("rl", 2, 1, obs, [22], "cpu")

alg = PPO(
    actor,
    critic,
    storage,
    num_learning_epochs=1,
    num_mini_batches=1,
    device="cpu",
)

amp_model = CategoricalAmpDiscriminator()
replay = CategoricalAmpReplayBuffer(capacity_per_class=8)

with torch.no_grad():
    # 初始化判别器的线性层，使其输出在初始阶段对每个类别的专家和策略特征有不同的偏置。
    for motion_id, discriminator in enumerate(amp_model.discriminators):
        discriminator.linear.weight.zero_()
        discriminator.linear.bias.fill_(float(motion_id))

# 持久组件在采集的 inference_mode 外创建。
collector = CategoricalAmpCollector(amp_model, replay, obs["amp"])

with torch.inference_mode():
    next_obs, rewards, dones, extras, metrics = collect_categorical_amp_step(
        env,
        alg,
        collector,
        obs,
        read_motion_ids=lambda raw_env: raw_env.motion_ids,
        style_weight=0.4,
    )

    expected_amp = torch.stack([
        F.softplus(torch.tensor(0.0)),
        torch.tensor(0.0),
    ])
    expected_total = (
        0.6 * torch.tensor([0.2, -0.4])
        + 0.4 * 0.02 * expected_amp
    )

    torch.testing.assert_close(metrics["amp_raw"], expected_amp)
    torch.testing.assert_close(rewards, expected_total)

    # 本测试没有 timeout bootstrap，Storage 应保存混合奖励。
    assert storage.step == 1
    torch.testing.assert_close(
        storage.rewards[0, :, 0],
        expected_total,
    )
    assert replay.counts == (1, 0, 0, 0, 0, 0)

    alg.compute_returns(next_obs)

# 参数更新必须在 inference_mode 外。
losses = alg.update()

assert torch.isfinite(torch.tensor(list(losses.values()))).all().item()
assert storage.step == 0

print("混合奖励:", rewards)
print("PPO loss:", losses)
print("AMP 奖励进入真实 PPO 检查通过")


class MultiStepFakeEnv(FakeEnv):
    def __init__(self):
        super().__init__()
        self.step_count = 0

    def step(self, actions):
        assert actions.shape == (2, 22)
        self.step_count += 1
        self.q = self.q + 0.01

        dones = torch.zeros(2, dtype=torch.int64)

        if self.step_count == 1:
            self.motion_ids[1] = 5
            self.q[1] = -0.2
            dones[1] = 1

        return (
            self.get_observations(),
            torch.tensor([0.2, -0.4]),
            dones,
            {"time_outs": torch.tensor([False, False])},
        )


class FakeExpertSampler:
    def sample(self, motion_ids, *, frame_dt):
        assert frame_dt == 0.02
        return torch.full(
            (motion_ids.shape[0], 2, 22),
            0.2,
            device=motion_ids.device,
        )

rollout_steps = 4

env_multi = MultiStepFakeEnv()
obs_multi = env_multi.get_observations()

storage_multi = RolloutStorage(
    "rl", 2, rollout_steps, obs_multi, [22], "cpu",
)

actor_multi = MLPModel(
    obs_multi,
    groups,
    "actor",
    output_dim=22,
    hidden_dims=(32, 16),
    activation="relu",
    distribution_cfg={
        "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
        "init_std": 0.1,
    },
)

critic_multi = MLPModel(
    obs_multi,
    groups,
    "critic",
    output_dim=1,
    hidden_dims=(32, 16),
    activation="relu",
)

# 复制网络参数，不复制旧网络缓存的计算结果。
actor_multi.load_state_dict(actor.state_dict())
critic_multi.load_state_dict(critic.state_dict())

alg_multi = PPO(
    actor_multi,
    critic_multi,
    storage_multi,
    num_learning_epochs=1,
    num_mini_batches=1,
    device="cpu",
)

amp_model_multi = CategoricalAmpDiscriminator(
    empirical_normalization=True,
)
replay_multi = CategoricalAmpReplayBuffer(capacity_per_class=16)

collector_multi = CategoricalAmpCollector(
    amp_model_multi,
    replay_multi,
    obs_multi["amp"],
)

amp_optimizer = torch.optim.Adam(
    amp_model_multi.parameters(),
    lr=1.0e-3,
)

updater_multi = CategoricalAmpUpdater(
    amp_model_multi,
    replay_multi,
    FakeExpertSampler(),
    amp_optimizer,
    frame_dt=0.02,
)

# 连续运行两轮，所有组件都保持同一实例。
for iteration in range(2):
    obs_multi, summary = run_categorical_amp_iteration(
        env_multi,
        alg_multi,
        collector_multi,
        updater_multi,
        obs_multi,
        read_motion_ids=lambda raw_env: raw_env.motion_ids,
        num_steps=rollout_steps,
        amp_batch_size=8,
        style_weight=0.4,
    )

    assert storage_multi.step == 0

    # 类别 3 的结束步被排除，之后环境 1 使用类别 5。
    expected_count = rollout_steps * (iteration + 1)
    assert replay_multi.counts == (
        expected_count, 0, 0, 0, 0, expected_count - 1,
    )

    assert set(summary["discriminator"]) == {0, 5}

    for network_id, network in enumerate(amp_model_multi.discriminators):
        normalizer = network.amp_normalizer
        assert normalizer is not None

        # 每轮更新：8 个 expert + 8 个 policy。
        expected_norm_count = (
            16 * (iteration + 1)
            if network_id in (0, 5)
            else 0
        )
        assert normalizer.count.item() == expected_norm_count

    assert torch.isfinite(
        torch.tensor(list(summary["ppo"].values()))
    ).all().item()

    for metrics in summary["discriminator"].values():
        assert torch.isfinite(
            torch.tensor(list(metrics.values()))
        ).all().item()

    print(f"Iteration {iteration} PPO:", summary["ppo"])
    print("Replay 数量:", replay_multi.counts)

print("完整 PPO + AMP iteration 检查通过")


from pathlib import Path
from tempfile import TemporaryDirectory


# 只在临时目录保存测试文件。
with TemporaryDirectory(prefix="goalkeeper_amp_") as directory:
    checkpoint_path = Path(directory) / "amp_state.pt"

    torch.save(updater_multi.state_dict(), checkpoint_path)

    saved_state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

restored_model = CategoricalAmpDiscriminator(
    empirical_normalization=True,
)
restored_replay = CategoricalAmpReplayBuffer(capacity_per_class=16)

# 故意使用不同学习率，检查加载后是否恢复保存值。
restored_optimizer = torch.optim.Adam(
    restored_model.parameters(),
    lr=0.123,
)

restored_updater = CategoricalAmpUpdater(
    restored_model,
    restored_replay,
    FakeExpertSampler(),
    restored_optimizer,
    frame_dt=0.02,
)

restored_updater.load_state_dict(saved_state)

# 包含网络参数，以及六个 normalizer 的统计量。
restored_model_state = restored_model.state_dict()

assert restored_model_state.keys() == saved_state["model"].keys()

for name, expected in saved_state["model"].items():
    torch.testing.assert_close(
        restored_model_state[name],
        expected,
        rtol=0,
        atol=0,
    )

# 检查优化器参数组，以及 Adam 的动量和步数。
actual_optimizer_state = restored_optimizer.state_dict()
expected_optimizer_state = saved_state["optimizer"]

assert (
    actual_optimizer_state["param_groups"]
    == expected_optimizer_state["param_groups"]
)
assert (
    actual_optimizer_state["state"].keys()
    == expected_optimizer_state["state"].keys()
)

for parameter_id, expected_buffers in expected_optimizer_state["state"].items():
    actual_buffers = actual_optimizer_state["state"][parameter_id]

    for name, expected in expected_buffers.items():
        actual = actual_buffers[name]

        if torch.is_tensor(expected):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual == expected

# Replay 不恢复，必须从新环境重新采集。
assert restored_replay.counts == (0, 0, 0, 0, 0, 0)

# 同样的输入，恢复后的奖励应一致。
test_features = torch.randn(4, 44)
test_ids = torch.tensor([0, 3, 5, 0], dtype=torch.int64)

torch.testing.assert_close(
    restored_model.predict_reward(test_features, test_ids),
    amp_model_multi.predict_reward(test_features, test_ids),
)

# 补充新数据后，验证恢复的优化器仍能继续更新。
restored_replay.insert(
    torch.full((8, 44), -0.2),
    torch.zeros(8, dtype=torch.int64),
)

resume_metrics = restored_updater.update_class(
    motion_id=0,
    batch_size=8,
)

assert resume_metrics is not None
assert torch.isfinite(
    torch.tensor(list(resume_metrics.values()))
).all().item()

print("AMP 参数、归一化、优化器恢复及继续更新检查通过")


## Step 3: 检查 CategoricalAmpOnPolicyRunner 的初始化和组件引用
from dataclasses import asdict

from booster_mjlab.amp.runners.categorical_runner_cfg import (
    CategoricalAmpRunnerCfg,
)
from booster_mjlab.amp.runners.categorical_on_policy_runner import (
    CategoricalAmpOnPolicyRunner,
)


runner_cfg = CategoricalAmpRunnerCfg()

# 独立测试使用小网络与小缓冲。
runner_cfg.actor.hidden_dims = (32, 16)
runner_cfg.critic.hidden_dims = (32, 16)
runner_cfg.num_steps_per_env = 4
runner_cfg.algorithm.num_learning_epochs = 1
runner_cfg.algorithm.num_mini_batches = 1
runner_cfg.amp.capacity_per_class = 16
runner_cfg.amp.batch_size = 8
runner_cfg.logger = "tensorboard"
runner_cfg.upload_model = False

runner = CategoricalAmpOnPolicyRunner(
    MultiStepFakeEnv(),
    asdict(runner_cfg),
    log_dir=None,
    device="cpu",
)

assert isinstance(runner.alg, PPO)
assert runner.alg.actor.obs_dim == 92
assert runner.alg.critic.obs_dim == 95
assert runner.alg.storage.num_transitions_per_env == 4

assert len(runner.amp_model.discriminators) == 6
assert runner.amp_replay.counts == (0, 0, 0, 0, 0, 0)

# 所有 AMP 组件必须引用同一组网络和 Replay。
assert runner.amp_collector.model is runner.amp_model
assert runner.amp_updater.model is runner.amp_model
assert runner.amp_collector.replay is runner.amp_replay
assert runner.amp_updater.replay is runner.amp_replay
assert runner.amp_updater.optimizer is runner.amp_optimizer

assert runner.amp_updater.expert_sampler is None

print("正式 Runner 配置与组件初始化检查通过")