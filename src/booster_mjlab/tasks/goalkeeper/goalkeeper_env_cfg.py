from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as common_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg
)
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from mjlab.managers import SceneEntityCfg

from mjlab.scene import SceneCfg
from .mdp.observations import (
    ball_position_b,
    simulated_ball_measurement_b,
)
from .mdp.rewards import (
    base_ang_vel_xy_l2,
    base_backward_velocity_penalty,
    base_line_deviation,
    knee_height_below_minimum,
    feet_orientation_exp,
    ActuatedJointAccelerationPenalty,
    NormalizedJointTorquePenalty,
    MotorLimitPenalty
)

def make_goalkeeper_env_cfg(
    scene: SceneCfg,
    *,
    decimation: int,
    episode_length_s: float,
) -> ManagerBasedRlEnvCfg:
    
    actor_terms = {
        "ball_measurement": ObservationTermCfg(
            func=simulated_ball_measurement_b,
            params={
                "robot_name": "robot",
                "ball_name": "ball",
                "max_range": 5.0,
            },
            # 历史数据长度, 提过过去5帧的观测数据
            history_length=5,
            # 是否将历史数据展平为一维向量, 如果为False, 则返回一个形状为 (B, history_length, feature_dim) 的张量
            flatten_history_dim=True
        ),
        # 机器人基座朝向重力方向的投影3维
        "projected_gravity": ObservationTermCfg(
            func=common_mdp.projected_gravity,
        ),
        # 机器人基座角速度3维
        "base_ang_vel": ObservationTermCfg(
            func=common_mdp.base_ang_vel,
        ),
        # 相对关节角度22维
        "joint_pos": ObservationTermCfg(
            func=common_mdp.joint_pos_rel,
        ),
        # 相对关节角速度22维
        "joint_vel": ObservationTermCfg(
            func=common_mdp.joint_vel_rel,
        ),
        # 上一时刻动作22维
        "actions": ObservationTermCfg(
            func=common_mdp.last_action,
        )
    }

    critic_terms = deepcopy(actor_terms)
    critic_terms["ball_position"] = ObservationTermCfg(
        func=ball_position_b,
        params={
            "robot_name": "robot",
            "ball_name": "ball"
        }
    )
    # critic_terms = {
    #     "ball_position": ObservationTermCfg(
    #         func=ball_position_b,
    #         params={
    #             "robot_name": "robot",
    #             "ball_name": "ball"
    #         }
    #     ),
    #     "projected_gravity": ObservationTermCfg(
    #         func=common_mdp.projected_gravity,
    #     ),
    #     "base_ang_vel": ObservationTermCfg(
    #         func=common_mdp.base_ang_vel,
    #     )
    # }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False
        )
    }

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        )
    }

    rewards: dict[str, RewardTermCfg] = {
        # 平滑性惩罚，鼓励机器人动作平滑
        "smoothness": RewardTermCfg(
            func=common_mdp.action_acc_l2,
            weight=-0.1,
        ),
        # 关节速度惩罚，防止机器人关节运动过快
        "dof_vel": RewardTermCfg(
            func=common_mdp.joint_vel_l2,
            weight=-5e-4,
        ),
        # 电机扭矩惩罚，防止机器人关节扭矩过大
        "ang_vel_xy": RewardTermCfg(
            func=base_ang_vel_xy_l2,
            weight=-0.1,
            params={"robot_name": "robot"}
        ),
        # 关节位置惩罚，防止机器人关节位置超出限制
        "dof_pos_limits": RewardTermCfg(  
            func=common_mdp.joint_pos_limits,
            weight=-3.0,
        ),
        # 向后运动惩罚，防止机器人后退
        "noretreat": RewardTermCfg(
            func=base_backward_velocity_penalty,
            weight=-2.0,
            params={"robot_name": "robot"}
        ),
        # 基于基座位置的偏离惩罚，防止机器人离开球门
        "stayonline": RewardTermCfg(
            func=base_line_deviation,
            weight=-2.0,
            params={
                "robot_name": "robot",
                "reference_x": 0.0,
                "deadband": 0.2,
                "max_distance": 1.2,
            }
        ),
        # 膝关节高度惩罚，防止机器人下蹲过低
        "penalize_kneeheight": RewardTermCfg(
            func=knee_height_below_minimum,
            weight=-100.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("Left_Shank", "Right_Shank"),
                ),
                "minimum_height": 0.15,
            }
        ),
        # 关节速度和电机扭矩限制惩罚
        "dof_vel_limits": RewardTermCfg(
            func=MotorLimitPenalty,
            weight=-2.0,
            params={
                "robot_name": "robot",
                "kind": "velocity",
                "soft_ratio": 0.9,
            },
        ),
        "torque_limits": RewardTermCfg(
            func=MotorLimitPenalty,
            weight=-3.0,
            params={
                "robot_name": "robot",
                "kind": "torque",
                "soft_ratio": 0.95,
            },
        ),
        "dof_acc": RewardTermCfg(
            func=ActuatedJointAccelerationPenalty,
            weight=-2.5e-7,
            params={"robot_name": "robot"},
        ),
        "torques": RewardTermCfg(
            func=NormalizedJointTorquePenalty,
            weight=-1e-5,
            params={"robot_name": "robot"},
        ),
        "feetorientaion": RewardTermCfg(
            func=feet_orientation_exp,
            weight=3.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_foot_link", "right_foot_link"),
                ),
                "coefficient": 5.0,
            },
        ),
    }

    terminations: dict[str, TerminationTermCfg] = {
        "time_out": TerminationTermCfg(
            func=common_mdp.time_out,
            time_out=True
        )
    }

    return ManagerBasedRlEnvCfg(
        scene=scene,
        decimation=decimation,
        episode_length_s=episode_length_s,
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminations=terminations,
    )