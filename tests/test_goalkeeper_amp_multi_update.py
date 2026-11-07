import torch

from booster_mjlab.amp.algorithms.categorical_amp import CategoricalAmpUpdater
from booster_mjlab.amp.modules.categorical_discriminator import (
    CategoricalAmpDiscriminator,
)
from booster_mjlab.amp.modules.goalkeeper_normalizer import GoalkeeperAmpNormalizer
from goalkeeper_amp_test_utils import FakeExpertSampler

torch.manual_seed(42)
model = CategoricalAmpDiscriminator(hidden_layer_sizes=(16, 8))
sampler = FakeExpertSampler()
updater = CategoricalAmpUpdater(model, sampler, frame_dt=0.02)
assert all(d.amp_normalizer is None for d in model.discriminators)
assert sum(isinstance(m, GoalkeeperAmpNormalizer) for m in model.modules()) == 1
assert set(k for k in model.state_dict() if "amp_normalizer" in k) == {
    "amp_normalizer.mean",
    "amp_normalizer.var",
    "amp_normalizer.count",
}

# Unequal class counts detect accidental class-balanced resampling.
ids = torch.tensor([0, 0, 0, 3, 5, 5], dtype=torch.int64)
features = torch.arange(6, dtype=torch.float32)[:, None].expand(-1, 44).clone() * 0.1
valid = torch.tensor([True, True, True, True, False, True])
reference = GoalkeeperAmpNormalizer(44)
normalized_policy = reference(features[valid])
expert_values = ids[valid].float() * 0.05 + 0.2
normalized_expert = reference(expert_values[:, None].expand(-1, 44).clone())
reference.update(normalized_policy)
reference.update(normalized_expert)

loss, metrics = updater.compute_loss(features, ids, valid)
assert set(metrics) == {0, 3, 5}
assert [request.tolist() for request in sampler.requests] == [[0, 0, 0], [3], [5]]
assert [metrics[g]["sample_count"] for g in (0, 3, 5)] == [3, 1, 1]
torch.testing.assert_close(
    loss.detach(), torch.tensor(sum(m["total_loss"] for m in metrics.values()))
)
for key, value in model.amp_normalizer.state_dict().items():
    torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
assert model.amp_normalizer.count.item() == 10.0001
loss.backward()
assert all(
    (any(p.grad is not None for p in d.parameters())) == (g in (0, 3, 5))
    for g, d in enumerate(model.discriminators)
)
print("共享归一化、同快照归一化、normalized 更新顺序、类别样本比例和 loss 求和检查通过")
