import torch

from booster_mjlab.amp.algorithms.categorical_amp import (
    CategoricalAmpUpdater,
)
from booster_mjlab.amp.modules.categorical_discriminator import (
    CategoricalAmpDiscriminator,
)
from booster_mjlab.amp.modules.categorical_replay import (
    CategoricalAmpReplayBuffer,
)


class FakeExpertSampler:
    def __init__(self):
        self.requests = []

    def sample(self, motion_ids, *, frame_dt):
        self.requests.append((motion_ids.clone(), frame_dt))
        return torch.full(
            (motion_ids.shape[0], 2, 22),
            0.2,
            device=motion_ids.device,
        )


torch.manual_seed(42)

model = CategoricalAmpDiscriminator()
replay = CategoricalAmpReplayBuffer(capacity_per_class=16)
expert_sampler = FakeExpertSampler()

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

updater = CategoricalAmpUpdater(
    model,
    replay,
    expert_sampler,
    optimizer,
    frame_dt=0.02,
)

# 只插入类别 3 的 policy 数据。
replay.insert(
    torch.full((8, 44), -0.2),
    torch.full((8,), 3, dtype=torch.int64),
)

before = [
    [parameter.detach().clone() for parameter in network.parameters()]
    for network in model.discriminators
]

metrics = updater.update_class(motion_id=3, batch_size=8)

assert metrics is not None
assert torch.isfinite(torch.tensor(list(metrics.values()))).all().item()

requested_ids, requested_dt = expert_sampler.requests[0]
assert requested_ids.tolist() == [3] * 8
assert requested_dt == 0.02

changed = []
for snapshot, network in zip(before, model.discriminators):
    changed.append(any(
        not torch.equal(old, parameter.detach())
        for old, parameter in zip(snapshot, network.parameters())
    ))

assert changed == [False, False, False, True, False, False]

# 类别 0 没有 policy 数据，应跳过，也不请求 expert。
assert updater.update_class(motion_id=0, batch_size=8) is None
assert len(expert_sampler.requests) == 1

print("训练指标:", metrics)
print("参数变化:", changed)
print("分类判别器更新入口检查通过")