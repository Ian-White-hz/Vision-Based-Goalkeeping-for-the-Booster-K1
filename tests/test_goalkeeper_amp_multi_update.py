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
    def sample(self, motion_ids, *, frame_dt):
        assert frame_dt == 0.02

        values = motion_ids.to(torch.float32) + 0.2

        return values[:, None, None].expand(-1, 2, 22).clone()


torch.manual_seed(42)

model = CategoricalAmpDiscriminator(
    empirical_normalization=True,
)
replay = CategoricalAmpReplayBuffer(capacity_per_class=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)

updater = CategoricalAmpUpdater(
    model,
    replay,
    FakeExpertSampler(),
    optimizer,
    frame_dt=0.02,
)


def insert_class(motion_id):
    replay.insert(
        torch.full((8, 44), motion_id - 0.2),
        torch.full((8,), motion_id, dtype=torch.int64),
    )


# 第一轮：只有类别 0、3。
insert_class(0)
insert_class(3)

results1 = updater.update_all(batch_size=8)
assert set(results1) == {0, 3}

for motion_id, network in enumerate(model.discriminators):
    normalizer = network.amp_normalizer
    assert normalizer is not None

    if motion_id in (0, 3):
        # 8 个 expert + 8 个 policy。
        assert normalizer.count.item() == 16

        torch.testing.assert_close(
            normalizer.mean,
            torch.full((44,), float(motion_id)),
        )
        torch.testing.assert_close(
            normalizer.std,
            torch.full((44,), 0.2),
        )
    else:
        assert normalizer.count.item() == 0

# 第二轮：补齐剩下的类别。
for motion_id in (1, 2, 4, 5):
    insert_class(motion_id)

results2 = updater.update_all(batch_size=8)
assert set(results2) == set(range(6))

for motion_id, network in enumerate(model.discriminators):
    normalizer = network.amp_normalizer
    assert normalizer is not None

    expected_count = 32 if motion_id in (0, 3) else 16
    assert normalizer.count.item() == expected_count

    torch.testing.assert_close(
        normalizer.mean,
        torch.full((44,), float(motion_id)),
    )
    torch.testing.assert_close(
        normalizer.std,
        torch.full((44,), 0.2),
    )

for metrics in results2.values():
    assert torch.isfinite(
        torch.tensor(list(metrics.values()))
    ).all().item()

print("第一轮更新类别:", sorted(results1))
print("第二轮更新类别:", sorted(results2))
print("六类调度与归一化隔离检查通过")