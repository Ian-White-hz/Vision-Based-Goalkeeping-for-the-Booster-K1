from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    EventTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg
)
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab.managers.observation_manager import ObservationGroupCfg

from booster_mjlab.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from booster_mjlab.tasks.goalkeeper.goalkeeper_env_cfg import(
    make_goalkeeper_env_cfg
)
from booster_mjlab.tasks.goalkeeper.scene.scene_cfg import (
    make_goalkeeper_scene_cfg,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import (
    FullGoalSettings,
)
from booster_mjlab.tasks.goalkeeper.scene.events import (
    sample_shots,
    launch_shots,
    launch_six_region_shots,
)
from booster_mjlab.tasks.goalkeeper.mdp.rewards import (
    FeetSlippageReward,
    JumpLandingReward,
    excessive_foot_contact_from_env,
    post_recovery_from_env,
    post_upper_joint_position_from_env,
    ee_reach_from_env,
    stopball_from_env,
    target_success_from_env,
)
from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    reset_shot_outcome_state,
)
from booster_mjlab.tasks.goalkeeper.mdp.outcomes import (
    reset_goalkeeper_outcome_state,
    update_goalkeeper_outcome_state,
)
from booster_mjlab.tasks.goalkeeper.mdp.terminations import (
    shot_recovery_finished,
    excessive_body_tilt
)

from typing import Literal

def booster_k1_goalkeeper_env_cfg(
    *,
    num_envs: int = 1,
    settings: FullGoalSettings | None = None,
    shot_mode: Literal["rolling", "six_region"] = "six_region",
) ->ManagerBasedRlEnvCfg:
    if settings is None:
        settings = FullGoalSettings()

    scene = make_goalkeeper_scene_cfg(
        robot_cfg=get_k1_robot_cfg(),
        settings=settings,
        num_envs=num_envs
    )

    cfg = make_goalkeeper_env_cfg(
        scene=scene,
        decimation=4,
        episode_length_s=settings.episode_length_s,
    )

    # 物理200Hz, 机器人控制50Hz
    cfg.sim.mujoco.timestep = 0.005

    cfg.sim.mujoco.timestep = 0.005

    # Viser 从地面上方观察场景。
    cfg.viewer.elevation = 25.0
    cfg.viewer.distance = 4.0
    cfg.viewer.azimuth = 90.0

    # 每个环境的物理约束容量。
    cfg.sim.njmax = 300

    action = cfg.actions["joint_pos"]
    assert isinstance(action, JointPositionActionCfg)
    action.scale = dict(K1_ACTION_SCALE)

    if shot_mode == "rolling":
        launch_function = launch_shots
    elif shot_mode == "six_region":
        launch_function = launch_six_region_shots
    else:
        raise ValueError(f"Invalid shot_mode: {shot_mode}")

    cfg.events["full_goal_launch"] = EventTermCfg(
        func=launch_function,
        mode="reset",
        params={"settings": settings},
    )
    cfg.events["reset_shot_outcome"] = EventTermCfg(
        func=reset_shot_outcome_state,
        mode="reset",
        params={
            "ball_name": "ball",
            "velocity_change_threshold": 2.0,
        },
    )
    cfg.events["reset_goalkeeper_outcome"] = EventTermCfg(
        func=reset_goalkeeper_outcome_state,
        mode="reset",
        params={
            "settings": settings,
        },
    )
    cfg.events["update_goalkeeper_outcome"] = EventTermCfg(
        func=update_goalkeeper_outcome_state,
        mode="step",
    )

    cfg.terminations["shot_recovery_finished"] = TerminationTermCfg(
        func=shot_recovery_finished,
        params={
            "recovery_duration_s": settings.recovery_duration_s,
        },
    )
    cfg.terminations["excessive_body_tilt"] = TerminationTermCfg(
        func=excessive_body_tilt,
        params={
            "robot_name": "robot",
            "max_tilt_deg": 60.0,
        },
    )

    feet_ground_contact = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            entity="robot",
            pattern=r"^(left_foot_link|right_foot_link)$",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    cfg.scene.sensors = (
        *cfg.scene.sensors,
        feet_ground_contact,
    )

    cfg.rewards["feet_slippage"] = RewardTermCfg(
        func=FeetSlippageReward,
        weight=3.0,
        params={
            "robot_name": "robot",
            "sensor_name": "feet_ground_contact",
            "contact_threshold": 1.0,
            "coefficient": 10.0
        }
    )
    cfg.rewards["penalize_sharpcontact"] = RewardTermCfg(
        func=excessive_foot_contact_from_env,
        weight=-100.0,
        params={
            "sensor_name": "feet_ground_contact",
            "max_contact_force": 1000.0,
        }
    )

    recovery_terms = (
        ("postorientation", "orientation", 3.0),
        ("postangvel", "angular_velocity", 3.0),
        ("postlinvel", "linear_velocity", 1.0),
    )
    for name, kind, weight in recovery_terms:
        cfg.rewards[name] = RewardTermCfg(
            func=post_recovery_from_env,
            weight=weight,
            params={
                "kind": kind,
                "robot_name": "robot",
                "ball_name": "ball",
                "reference_x": 0.0,
                "velocity_change_threshold": 2.0,
                "coefficient": 3.0,
            }
        )
    cfg.rewards["postupperdofpos"] = RewardTermCfg(
        func=post_upper_joint_position_from_env,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=(
                    "Left_Elbow_Pitch",
                    "Left_Elbow_Yaw",
                    "Right_Elbow_Pitch",
                    "Right_Elbow_Yaw",
                ),
            ),
            "ball_name": "ball",
            "reference_x": 0.0,
            "velocity_change_threshold": 2.0,
            "coefficient": 1.0,
        }
    )
    cfg.rewards["eereach"] = RewardTermCfg(
        func=ee_reach_from_env,
        weight=10.0,
        params={
            "hand_cfg": SceneEntityCfg(
                "robot",
                site_names=("left_hand", "right_hand"),
                preserve_order=True,
            ),
            "ball_name": "ball",
            "curriculum_level": 0.0,
            "distance_sigma": 5.0,
            "reach_threshold": 0.2,
        },
    )
    cfg.rewards["stopball"] = RewardTermCfg(
        func=stopball_from_env,
        weight=100.0,
    )
    cfg.rewards["success"] = RewardTermCfg(
        func=target_success_from_env,
        weight=5.0,
        params={
            "hand_cfg": SceneEntityCfg(
                "robot",
                site_names=("left_hand", "right_hand"),
                preserve_order=True,
            ),
            "strict_threshold": 0.15,
        }
    )
    cfg.rewards["successland"] = RewardTermCfg(
        func=JumpLandingReward,
        weight=4.0,
        params={
            "robot_name": "robot",
            "sensor_name": "feet_ground_contact",
            "minimum_rise": 0.08,
            "force_sign": -1.0,
            "contact_threshold": 1.0,
        },
    )

    return cfg

def booster_k1_goalkeeper_amp_env_cfg(
    *,
    amp_observation: ObservationGroupCfg,
    num_envs: int = 1,
    settings: FullGoalSettings | None = None,
) -> ManagerBasedRlEnvCfg:
    """复用 goalkeeper 环境，接入外部提供的 AMP 观测组。"""
    cfg = booster_k1_goalkeeper_env_cfg(
        num_envs=num_envs,
        settings=settings,
        shot_mode="six_region",
    )

    cfg.observations["amp"] = deepcopy(amp_observation)

    return cfg
