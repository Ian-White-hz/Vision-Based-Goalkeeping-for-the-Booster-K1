from types import SimpleNamespace as NS
import torch

from booster_mjlab.tasks.goalkeeper.mdp.observations import ball_position_b

s = 2 ** -0.5

env = NS(scene={
    "robot": NS(data=NS(
        root_link_pos_w=torch.tensor([
            [0., 0., 1.],
            [10., 0., 1.],
        ]),
        root_link_quat_w=torch.tensor([
            [1., 0., 0., 0.],  # 不旋转
            [s, 0., 0., s],    # 绕世界 Z 轴旋转 +90°
        ]),
    )),
    "ball": NS(data=NS(
        root_link_pos_w=torch.tensor([
            [1., 0., 1.],
            [11., 0., 1.],
        ]),
    )),
})

result = ball_position_b(env)
print(result)

expected = torch.tensor([
    [1., 0., 0.],
    [0., -1., 0.],
])

assert result.shape == (2, 3)
assert torch.allclose(result, expected, atol=1e-6)
print("test ball_position_bPassed")