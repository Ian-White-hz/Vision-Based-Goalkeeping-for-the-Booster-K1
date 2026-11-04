import torch

from booster_mjlab.amp.modules.history import (
    TwoFrameAmpHistory,
    select_valid_amp_samples,
)


q0 = torch.stack([
    torch.full((22,), 0.0),
    torch.full((22,), 0.5),
])

history = TwoFrameAmpHistory(q0)

# 第一控制步：环境 0 正常，环境 1 返回 reset 后状态。
q1 = torch.stack([
    torch.full((22,), 0.1),
    torch.full((22,), -0.2),
])

windows1, valid1 = history.push(
    q1,
    dones=torch.tensor([False, True]),
)

assert windows1.shape == (2, 2, 22)
assert valid1.tolist() == [True, False]

# 正常环境保留真实两帧。
torch.testing.assert_close(windows1[0, 0], q0[0])
torch.testing.assert_close(windows1[0, 1], q1[0])

# reset 环境不包含旧回合的 0.5。
torch.testing.assert_close(windows1[1, 0], q1[1])
torch.testing.assert_close(windows1[1, 1], q1[1])

windows1_snapshot = windows1.clone()

# 第二控制步：两个环境都正常。
q2 = torch.stack([
    torch.full((22,), 0.2),
    torch.full((22,), -0.1),
])

windows2, valid2 = history.push(
    q2,
    dones=torch.tensor([False, False]),
)

assert valid2.tolist() == [True, True]
torch.testing.assert_close(windows2[:, 0], q1)
torch.testing.assert_close(windows2[:, 1], q2)

# 后续更新不能修改之前已经返回的窗口。
torch.testing.assert_close(windows1, windows1_snapshot)

print("第一步有效窗口:", valid1)
print("第二步有效窗口:", valid2)
print("两帧历史与部分环境 reset 检查通过")


## Step 2: 测试 select_valid_amp_samples 函数
test_windows = torch.arange(
    3 * 2 * 22,
    dtype=torch.float32,
).reshape(3, 2, 22)

test_valid = torch.tensor([True, False, True])
test_motion_ids = torch.tensor([0, 3, 5], dtype=torch.int64)

features, motion_ids = select_valid_amp_samples(
    test_windows,
    test_valid,
    test_motion_ids,
)

assert features.shape == (2, 44)
assert motion_ids.tolist() == [0, 5]

torch.testing.assert_close(
    features[0],
    test_windows[0].flatten(),
)
torch.testing.assert_close(
    features[1],
    test_windows[2].flatten(),
)

# 所有窗口无效时，返回形状明确的空样本。
empty_features, empty_ids = select_valid_amp_samples(
    test_windows,
    torch.zeros(3, dtype=torch.bool),
    test_motion_ids,
)

assert empty_features.shape == (0, 44)
assert empty_ids.shape == (0,)

print("有效样本类别:", motion_ids)
print("窗口筛选与类别绑定检查通过")