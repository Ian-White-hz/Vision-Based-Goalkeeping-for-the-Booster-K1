import torch

from booster_mjlab.amp.algorithms.categorical_amp import CategoricalAmpUpdater
from booster_mjlab.amp.modules.categorical_discriminator import (
    CategoricalAmpDiscriminator,
)
from goalkeeper_amp_test_utils import FakeExpertSampler

torch.manual_seed(42)
model = CategoricalAmpDiscriminator(hidden_layer_sizes=(16, 8))
sampler = FakeExpertSampler()
updater = CategoricalAmpUpdater(model, sampler, frame_dt=0.02)
features = torch.full((8, 44), -0.2)
ids = torch.full((8,), 3, dtype=torch.int64)
valid = torch.ones(8, dtype=torch.bool)
before = [p.detach().clone() for p in model.parameters()]
loss, metrics = updater.compute_loss(features, ids, valid)
assert set(metrics) == {3}
assert sampler.requests[0].tolist() == [3] * 8
assert model.amp_normalizer.count.item() == 16.0001
# Loss construction must not take its own optimizer step.
assert all(torch.equal(p, old) for p, old in zip(model.parameters(), before))
loss.backward()
for motion_id, discriminator in enumerate(model.discriminators):
    gradients = [p.grad for p in discriminator.parameters() if p.grad is not None]
    if motion_id == 3:
        assert gradients
        assert all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.square().sum().item() for g in gradients) > 0
    else:
        assert not gradients

state = {key: value.clone() for key, value in model.state_dict().items()}
requests = len(sampler.requests)
zero, empty = updater.compute_loss(features, ids, torch.zeros(8, dtype=torch.bool))
assert zero.item() == 0 and empty == {}
assert len(sampler.requests) == requests
for key, value in model.state_dict().items():
    torch.testing.assert_close(value, state[key], rtol=0, atol=0)
print("按当前 minibatch 取 expert、梯度隔离、无独立更新及空 mask 检查通过")
