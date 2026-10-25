from types import SimpleNamespace as NS

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.scene import SceneCfg

from booster_mjlab.tasks.goalkeeper.goalkeeper_env_cfg import (
    make_goalkeeper_env_cfg,
)
from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    MotorLimitPenalty,
)


cfg = make_goalkeeper_env_cfg(
    scene=SceneCfg(num_envs=3),
    decimation=4,
    episode_length_s=3.0,
)


def mock_actuator(joint_id, speed_limit, torque_limit):
    return NS(
        target_ids=torch.tensor([joint_id]),
        cfg=NS(
            transmission_type=TransmissionType.JOINT,
            motor=NS(velocity_limit=speed_limit),
            effort_limit=torque_limit,
        ),
    )


robot = NS(
    actuators=[
        mock_actuator(0, 10.0, 100.0),
        mock_actuator(1, 20.0, 50.0),
    ],
    data=NS(
        joint_vel=torch.tensor([
            [9.0, 18.0, 1000.0],
            [9.5, -18.25, 1000.0],
            [12.0, -25.0, 1000.0],
        ]),
        qfrc_actuator=torch.tensor([
            [95.0, -47.5, 1000.0],
            [96.0, -49.5, 1000.0],
            [-100.0, 50.0, 1000.0],
        ]),
    ),
)

env = NS(scene={"robot": robot})

cases = [
    (
        "dof_vel_limits",
        -2.0,
        torch.tensor([0.0, 0.75, 2.0]),
    ),
    (
        "torque_limits",
        -3.0,
        torch.tensor([0.0, 3.0, 7.5]),
    ),
]

for name, expected_weight, expected_cost in cases:
    term = cfg.rewards[name]

    assert term.func is MotorLimitPenalty
    assert term.weight == expected_weight

    # 模拟 RewardManager 对 class-based term 的实例化。
    calculator = term.func(cfg=term, env=env)
    cost = calculator(env, **term.params)

    assert cost.shape == (3,)
    assert torch.isfinite(cost).all()

    torch.testing.assert_close(cost, expected_cost)
    torch.testing.assert_close(
        cost * term.weight,
        expected_cost * expected_weight,
    )

    print(name, cost)

print("速度与力矩软限值奖励检查通过")