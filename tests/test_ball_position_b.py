from types import SimpleNamespace as NS
import torch

from booster_mjlab.tasks.goalkeeper.mdp.observations import (
    ball_position_b,
    simulated_ball_measurement_b
)

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


measurement = simulated_ball_measurement_b(
    env,
    max_range=5.0,
)

torch.testing.assert_close(
    measurement,
    torch.tensor([
        [1.0,  0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0, 1.0],
    ]),
    atol=1e-6,
    rtol=1e-6,
)

out_of_range = simulated_ball_measurement_b(
    env,
    max_range=0.5,
)

torch.testing.assert_close(
    out_of_range,
    torch.zeros(2, 4),
)

print("模拟来球测量检查通过")