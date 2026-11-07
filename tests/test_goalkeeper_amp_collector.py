import torch

from booster_mjlab.amp.modules.categorical_collector import CategoricalAmpCollector
from booster_mjlab.amp.modules.categorical_discriminator import (
    CategoricalAmpDiscriminator,
)

torch.set_num_threads(1)
model = CategoricalAmpDiscriminator(hidden_layer_sizes=(16, 8))
with torch.no_grad():
    for motion_id, discriminator in enumerate(model.discriminators):
        discriminator.linear.weight.zero_()
        discriminator.linear.bias.fill_(float(motion_id))

q0 = torch.stack([torch.zeros(22), torch.full((22,), 0.5)])
collector = CategoricalAmpCollector(model, q0)
ids = torch.tensor([0, 3])
collector.begin_step(ids)
ids[1] = 5
q1 = torch.stack([torch.full((22,), 0.1), torch.full((22,), -0.2)])
rewards = collector.finish_step(q1, torch.tensor([False, True]))
torch.testing.assert_close(rewards, torch.tensor([0.75, 0.0]))
assert collector.last_motion_ids.tolist() == [0, 3]
assert collector.last_valid_mask.tolist() == [True, False]
torch.testing.assert_close(collector.last_features[0, :22], q0[0])
torch.testing.assert_close(collector.last_features[0, 22:], q1[0])

collector.begin_step(ids)
q2 = torch.stack([torch.full((22,), 0.2), torch.full((22,), -0.1)])
rewards = collector.finish_step(q2, torch.tensor([False, False]))
torch.testing.assert_close(rewards, torch.tensor([0.75, 0.0]))
assert collector.last_motion_ids.tolist() == [0, 5]
torch.testing.assert_close(collector.last_features[1, :22], q1[1])
torch.testing.assert_close(collector.last_features[1, 22:], q2[1])
assert not hasattr(collector, "replay")
assert not rewards.requires_grad
print("History、旧类别保存、reset mask 和无 Replay 收集检查通过")
