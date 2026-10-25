from copy import deepcopy
from types import SimpleNamespace as NS

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.scene import SceneCfg

from booster_mjlab.tasks.goalkeeper.goalkeeper_env_cfg import (
    make_goalkeeper_env_cfg,
)
from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    ActuatedJointAccelerationPenalty,
    NormalizedJointTorquePenalty,
    feet_orientation_exp,
)


cfg = make_goalkeeper_env_cfg(
    scene=SceneCfg(num_envs=4),
    decimation=4,
    episode_length_s=3.0,
)


def mock_actuator(joint_id, kp):
    return NS(
        target_ids=torch.tensor([joint_id]),
        cfg=NS(
            transmission_type=TransmissionType.JOINT,
            stiffness=kp,
        ),
    )


robot = NS(
    actuators=[
        mock_actuator(1, 20.0),
        mock_actuator(0, 10.0),
    ],
    data=NS(
        joint_vel=torch.zeros(4, 3),
        qfrc_actuator=torch.tensor([
            [0.0, 0.0, 1000.0],
            [10.0, -20.0, 1000.0],
            [-10.0, 20.0, 1000.0],
            [20.0, 20.0, 1000.0],
        ]),
    ),
)

env = NS(
    scene={"robot": robot},
    step_dt=0.1,
    common_step_counter=0,
)

acc_term = cfg.rewards["dof_acc"]
assert acc_term.func is ActuatedJointAccelerationPenalty
assert acc_term.weight == -2.5e-7

acc_calculator = acc_term.func(cfg=acc_term, env=env)

robot.data.joint_vel[:] = torch.tensor([
    [0.0, 0.0, 1000.0],
    [0.1, -0.2, 1000.0],
    [-0.1, 0.2, 1000.0],
    [0.3, 0.4, 1000.0],
])
env.common_step_counter = 1

expected_acc = torch.tensor([0.0, 5.0, 5.0, 25.0])

acc_cost = acc_calculator(env, **acc_term.params)
assert acc_cost.shape == (4,)
torch.testing.assert_close(acc_cost, expected_acc)

# 同一步再次调用，不应重新更新历史。
torch.testing.assert_close(
    acc_calculator(env, **acc_term.params),
    expected_acc,
)

# 下一步速度不变，加速度为零。
env.common_step_counter = 2
torch.testing.assert_close(
    acc_calculator(env, **acc_term.params),
    torch.zeros(4),
)

# 模拟仅环境 1 重置，其他环境继续运行。
robot.data.joint_vel[1, :2] = 0.0
acc_calculator.reset(torch.tensor([1]))

robot.data.joint_vel[1, 0] = 0.2
env.common_step_counter = 3

torch.testing.assert_close(
    acc_calculator(env, **acc_term.params),
    torch.tensor([0.0, 4.0, 0.0, 0.0]),
)

print("控制周期加速度及历史重置检查通过")

torque_term = cfg.rewards["torques"]
assert torque_term.func is NormalizedJointTorquePenalty
assert torque_term.weight == -1e-5

torque_calculator = torque_term.func(cfg=torque_term, env=env)
torque_cost = torque_calculator(env, **torque_term.params)

assert torque_cost.shape == (4,)
assert torch.isfinite(torque_cost).all()
torch.testing.assert_close(
    torque_cost,
    torch.tensor([0.0, 2.0, 2.0, 5.0]),
)

print("按配置 Kp 归一化的力矩检查通过")

feet_term = cfg.rewards["feetorientaion"]
assert feet_term.func is feet_orientation_exp
assert feet_term.weight == 3.0

# 三个 body：非脚部、左脚、右脚。
quaternion = torch.zeros(4, 3, 4)
quaternion[..., 0] = 1.0

s = 2.0 ** -0.5
roll_90 = torch.tensor([s, s, 0.0, 0.0])
yaw_90 = torch.tensor([s, 0.0, 0.0, s])

# 非脚部也倾斜，验证它不会参与奖励。
quaternion[:, 0] = roll_90
quaternion[1, 1] = roll_90
quaternion[2, 1:3] = roll_90
quaternion[3, 1:3] = yaw_90

robot.data.body_link_quat_w = quaternion
robot.data.gravity_vec_w = torch.tensor(
    [[0.0, 0.0, -1.0]]
).repeat(4, 1)

feet_params = deepcopy(feet_term.params)
assert feet_params["asset_cfg"].body_names == (
    "left_foot_link", "right_foot_link"
)

# 模拟 manager 初始化时的名称解析。
feet_params["asset_cfg"].body_ids = [1, 2]

feet_reward = feet_term.func(env, **feet_params)
expected_feet = torch.exp(
    torch.tensor([0.0, -5.0, -10.0, 0.0])
)

assert feet_reward.shape == (4,)
assert torch.isfinite(feet_reward).all()
torch.testing.assert_close(feet_reward, expected_feet)

print("脚部姿态奖励:", feet_reward)
print("本组三项奖励检查通过")