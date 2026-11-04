import torch

from booster_mjlab.amp.modules.categorical_replay import (
    CategoricalAmpReplayBuffer,
)


replay = CategoricalAmpReplayBuffer(
    capacity_per_class=2,
)

features = torch.stack([
    torch.full((44,), 0.1),
    torch.full((44,), 1.1),
    torch.full((44,), 0.2),
])
motion_ids = torch.tensor([0, 1, 0], dtype=torch.int64)

replay.insert(features, motion_ids)
assert replay.counts == (2, 1, 0, 0, 0, 0)

# 修改外部数据，不应该修改已经存入的样本。
features.fill_(9.0)

samples1 = replay.sample(motion_id=1, batch_size=8)
torch.testing.assert_close(
    samples1,
    torch.full((8, 44), 1.1),
)

# 类别 0 已满；新样本 0.3 应覆盖最旧的 0.1。
replay.insert(
    torch.full((1, 44), 0.3),
    torch.tensor([0], dtype=torch.int64),
)
assert replay.counts == (2, 1, 0, 0, 0, 0)

samples0 = replay.sample(motion_id=0, batch_size=16)
assert samples0.shape == (16, 44)

# 每行必须是 0.2 或 0.3，不能包含其他类别或被覆盖的数据。
allowed = torch.stack([
    torch.full((44,), 0.2),
    torch.full((44,), 0.3),
])

matches = (
    samples0[:, None, :] == allowed[None, :, :]
).all(dim=-1)
assert matches.any(dim=1).all().item()

# 空输入不改变存储。
replay.insert(
    torch.empty((0, 44)),
    torch.empty((0,), dtype=torch.int64),
)
assert replay.counts == (2, 1, 0, 0, 0, 0)

# 空类别不能偷偷使用其他类别的数据。
try:
    replay.sample(motion_id=2, batch_size=1)
except ValueError:
    pass
else:
    raise AssertionError("Sampling an empty class should fail")

print("各类别样本数:", replay.counts)
print("分类 Replay Buffer 检查通过")