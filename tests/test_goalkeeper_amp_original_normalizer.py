import numpy as np
import torch

from booster_mjlab.amp.modules.goalkeeper_normalizer import GoalkeeperAmpNormalizer


normalizer = GoalkeeperAmpNormalizer(2)
mean = np.zeros(2, dtype=np.float64)
var = np.ones(2, dtype=np.float64)
count = 1.0e-4
for raw in (
    torch.tensor([[2.0, 4.0], [4.0, 8.0]]),
    torch.tensor([[10.0, 20.0], [12.0, 24.0]]),
):
    expected = np.clip(
        (raw.numpy() - mean.astype(np.float32))
        / np.sqrt((var + 1.0e-4).astype(np.float32)),
        -10,
        10,
    )
    normalized = normalizer(raw)
    torch.testing.assert_close(normalized, torch.from_numpy(expected))
    normalizer.update(normalized)
    batch_mean = expected.mean(axis=0)
    batch_var = expected.var(axis=0)
    n = len(expected)
    delta = batch_mean - mean
    total = count + n
    mean = mean + delta * n / total
    var = (var * count + batch_var * n + delta**2 * count * n / total) / total
    count = total
    torch.testing.assert_close(normalizer.mean, torch.from_numpy(mean))
    torch.testing.assert_close(normalizer.var, torch.from_numpy(var))
    assert normalizer.count.item() == count

before = {name: value.clone() for name, value in normalizer.state_dict().items()}
torch.testing.assert_close(
    normalizer(torch.tensor([[1.0e6, -1.0e6]])), torch.tensor([[10.0, -10.0]])
)
normalizer.update(torch.empty(0, 2))
for name, value in normalizer.state_dict().items():
    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
restored = GoalkeeperAmpNormalizer(2)
restored.load_state_dict(before)
torch.testing.assert_close(restored(raw), normalizer(raw), rtol=0, atol=0)
assert not list(normalizer.parameters())
print("原版 normalizer 公式、更新、裁剪、只读读取及恢复检查通过")
