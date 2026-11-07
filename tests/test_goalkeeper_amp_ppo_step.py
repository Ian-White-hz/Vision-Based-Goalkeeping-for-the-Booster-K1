import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from booster_mjlab.amp.algorithms.categorical_ppo import (
    AMP_FEATURES,
    AMP_IDS,
    AMP_VALID,
    CategoricalAmpPPO,
)
from booster_mjlab.amp.runners.categorical_step import collect_categorical_amp_step
from goalkeeper_amp_test_utils import FakeEnv, make_runner, read_motion_ids

torch.manual_seed(42)
runner, sampler = make_runner(env=FakeEnv(timeouts=True))
alg = runner.alg
assert isinstance(alg, CategoricalAmpPPO)
assert alg.actor.obs_dim == 92 and alg.critic.obs_dim == 95
assert not hasattr(runner, "amp_replay")
assert not hasattr(runner.amp_updater, "optimizer")
assert runner.amp_updater is alg.amp_updater
assert runner.amp_collector.model is alg.amp_model
assert len(alg.optimizer.param_groups) == 13
assert alg.optimizer.param_groups[0]["name"] == "actor_critic"
assert alg.optimizer.param_groups[0]["weight_decay"] == 0
for group in alg.optimizer.param_groups[1:]:
    expected_decay = 1.0e-3 if group["name"].startswith("amp_trunk") else 0.1
    assert group["weight_decay"] == expected_decay
parameters = [p for group in alg.optimizer.param_groups for p in group["params"]]
assert len({id(p) for p in parameters}) == len(parameters)
assert {id(p) for p in parameters} == {
    id(p)
    for network in (alg.actor, alg.critic, alg.amp_model)
    for p in network.parameters()
}

with torch.no_grad():
    for g, discriminator in enumerate(runner.amp_model.discriminators):
        discriminator.linear.weight.zero_()
        discriminator.linear.bias.fill_(float(g))
obs = runner.env.get_observations()
with torch.inference_mode():
    obs, rewards, dones, extras, metrics = collect_categorical_amp_step(
        runner.env, alg, runner.amp_collector, obs, read_motion_ids, style_weight=0.4
    )
torch.testing.assert_close(metrics["amp_raw"], torch.tensor([0.75, 0.0]))
torch.testing.assert_close(rewards, torch.tensor([0.42, -0.24]))
assert alg.storage.observations[AMP_IDS][0].tolist() == [0, 3]
assert alg.storage.observations[AMP_VALID][0].tolist() == [True, False]
torch.testing.assert_close(
    alg.storage.rewards[0, :, 0],
    rewards + alg.gamma * alg.storage.values[0, :, 0] * extras["time_outs"],
)
# Clearing this isolated diagnostic rollout must not train or update moments.
alg.storage.clear()
print("共享 Adam、AMP 元数据同步存储、原版混合奖励和 timeout bootstrap 检查通过")

runner, sampler = make_runner(epochs=2, minibatches=2)
alg = runner.alg
actor_before = {k: p.detach().clone() for k, p in alg.actor.named_parameters()}
d_before = {k: p.detach().clone() for k, p in alg.amp_model.named_parameters()}
steps = []
hook = alg.optimizer.register_step_post_hook(lambda *_: steps.append(1))
runner.learn(2)
hook.remove()
assert runner.current_learning_iteration == 2
assert alg.storage.step == 0
# 2 iterations * 2 epochs * 2 minibatches: not separate PPO / D steps.
assert len(steps) == 8
# Only the first rollout's first step has a done: (7 + 8) * 2 epochs * 2 populations.
assert math.isclose(alg.amp_model.amp_normalizer.count.item(), 60.0001)
assert sum(len(ids) for ids in sampler.requests) == 30
assert any(
    not torch.equal(actor_before[k], p.detach())
    for k, p in alg.actor.named_parameters()
)
assert any(
    not torch.equal(d_before[k], p.detach())
    for k, p in alg.amp_model.named_parameters()
)
assert set(alg.last_amp_metrics) == {0, 5}
assert torch.isfinite(
    torch.tensor(list(runner._last_iteration_summary["ppo"].values()))
).all()

# No stale old-rollout samples: only the current rollout is used on next learn().
requests_before = len(sampler.requests)
runner.learn(1)
assert runner.current_learning_iteration == 3
assert math.isclose(alg.amp_model.amp_normalizer.count.item(), 92.0001)
assert sum(len(ids) for ids in sampler.requests[requests_before:]) == 16
assert alg.storage.step == 0
print("多轮真实 PPO、同 minibatch 联合更新、expert 数量、无跨轮 Replay 检查通过")

# The same PPO permutation must apply to actor data and AMP metadata.
obs = runner.env.get_observations()
with torch.inference_mode():
    for _ in range(4):
        obs, *_ = collect_categorical_amp_step(
            runner.env,
            alg,
            runner.amp_collector,
            obs,
            read_motion_ids,
            style_weight=0.4,
        )
markers = torch.arange(8, dtype=torch.float32).reshape(4, 2)
alg.storage.observations["actor"][:, :, 0] = markers
alg.storage.observations[AMP_FEATURES][:, :, 0] = markers
for batch in alg.storage.mini_batch_generator(2, 2):
    torch.testing.assert_close(
        batch.observations["actor"][:, 0], batch.observations[AMP_FEATURES][:, 0]
    )
alg.storage.clear()
print("PPO 与 AMP minibatch 同排列检查通过")

# Algorithm checkpoints include one normalizer, six Ds and the joint optimizer.
with TemporaryDirectory(prefix="goalkeeper_amp_") as directory:
    path = Path(directory) / "state.pt"
    torch.save(alg.save(), path)
    state = torch.load(path, weights_only=True, map_location="cpu")
restored, _ = make_runner(epochs=2, minibatches=2)
assert restored.alg.load(state, None, True)
for key, value in alg.amp_model.state_dict().items():
    torch.testing.assert_close(
        restored.amp_model.state_dict()[key], value, rtol=0, atol=0
    )
for network_name in ("actor", "critic"):
    for key, value in getattr(alg, network_name).state_dict().items():
        torch.testing.assert_close(
            getattr(restored.alg, network_name).state_dict()[key], value, rtol=0, atol=0
        )
expected_optimizer = state["optimizer_state_dict"]
actual_optimizer = restored.alg.optimizer.state_dict()
assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
for parameter_id, buffers in expected_optimizer["state"].items():
    for key, value in buffers.items():
        torch.testing.assert_close(actual_optimizer["state"][parameter_id][key], value)

features = torch.randn(4, 44)
ids = torch.tensor([0, 3, 5, 0])
torch.manual_seed(123)
expected = alg.amp_model.predict_reward(features, ids)
torch.manual_seed(123)
actual = restored.amp_model.predict_reward(features, ids)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
restored.learn(1)
assert restored.alg.storage.step == 0
# Actor-only loading does not need a sampler and does not overwrite AMP.
actor_only, _ = make_runner()
actor_only.amp_updater.expert_sampler = None
assert not actor_only.alg.load(state, {"actor": True}, True)
bad = dict(state)
bad.pop("categorical_amp")
try:
    actor_only.alg.load(bad, None, True)
except ValueError:
    pass
else:
    raise AssertionError("Old/missing AMP state was silently accepted")
print("联合 checkpoint 保存、恢复、继续更新及 actor-only 加载检查通过")

invalid, empty_sampler = make_runner(env=FakeEnv(all_done=True))
normalizer_before = {
    k: v.clone() for k, v in invalid.amp_model.amp_normalizer.state_dict().items()
}
invalid.learn(1)
assert invalid.alg.last_amp_metrics == {}
assert not empty_sampler.requests
for key, value in invalid.amp_model.amp_normalizer.state_dict().items():
    torch.testing.assert_close(value, normalizer_before[key], rtol=0, atol=0)
print("全 done 时保留 PPO、跳过 AMP 及统计更新检查通过")


# Part B binding still validates names before replacing valid bindings.
class WrongSampler:
    joint_names = ("wrong",) * 22

    def sample(self, ids, *, frame_dt):
        raise AssertionError("must not sample")


previous = restored.amp_updater.expert_sampler
try:
    restored.bind_amp_interfaces(WrongSampler(), read_motion_ids)
except ValueError:
    pass
else:
    raise AssertionError("Wrong joint order accepted")
assert restored.amp_updater.expert_sampler is previous
print("Part B 接口检查通过")

# Also exercise the production discriminator widths, not only tiny test MLPs.
full_size, _ = make_runner(amp_hidden_dims=(512, 256), minibatches=2)
full_size.alg.schedule = "adaptive"
full_size.learn(1)
assert full_size.amp_model.discriminators[0].trunk[0].out_features == 512
assert full_size.amp_model.discriminators[0].trunk[2].out_features == 256
assert all(
    group["lr"] == full_size.alg.learning_rate
    for group in full_size.alg.optimizer.param_groups
)
assert torch.isfinite(
    torch.tensor(list(full_size._last_iteration_summary["ppo"].values()))
).all()
print("正式 512/256 判别器和联合自适应学习率 smoke 检查通过")
