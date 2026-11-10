import mjlab  # 先初始化 mjlab。
import torch

from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    nearest_intercept_distance,
)


# 这里只用人工坐标验证数学计算，不连接真实机器人。
points = torch.tensor([
    [0.1,  0.2, 0.5],
    [0.1, -0.2, 0.5],
    [0.1,  0.0, 0.11],
])

blockers = points.unsqueeze(0).repeat(4, 1, 1)

targets = torch.tensor([
    [0.1,  0.2, 0.5],
    [0.1, -0.2, 0.5],
    [0.1,  0.0, 0.11],
    [0.1,  0.6, 0.5],
])

distance = nearest_intercept_distance(blockers, targets)

assert distance.shape == (4,)
torch.testing.assert_close(
    distance,
    torch.tensor([0.0, 0.0, 0.0, 0.4]),
)

# 交换候选点顺序，不应该改变结果。
reordered = blockers[:, [2, 0, 1], :]
torch.testing.assert_close(
    nearest_intercept_distance(reordered, targets),
    distance,
)

print("最小拦截距离:", distance)
print("统一候选点距离计算检查通过")