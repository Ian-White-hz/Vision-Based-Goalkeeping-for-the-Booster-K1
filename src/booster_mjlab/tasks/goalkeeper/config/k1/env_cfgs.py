from copy import deepcopy
from dataclasses import asdict
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import (
    EventTermCfg,
    MetricsTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from booster_mjlab.robots import K1_ACTION_SCALE, get_k1_robot_cfg
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER
from booster_mjlab.tasks.goalkeeper.amp_interfaces import (
    MOTION_NAMES,
    amp_joint_positions,
)
from booster_mjlab.tasks.goalkeeper.goalkeeper_env_cfg import make_goalkeeper_env_cfg
from booster_mjlab.tasks.goalkeeper.scene.scene_cfg import (
    make_goalkeeper_scene_cfg,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import (
    FullGoalSettings,
)
from booster_mjlab.tasks.goalkeeper.scene.events import (
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
    body_intercept_from_env,
    stopball_from_env,
    target_success_from_env,
    goal_conceded_cost,
    fall_termination_cost,
    body_tilt_from_env,
    heading_deviation_from_env,
    excessive_yaw_rate_from_env,
)
from booster_mjlab.tasks.goalkeeper.mdp.shot_state import (
    reset_shot_outcome_state,
)
from booster_mjlab.tasks.goalkeeper.mdp.interception import InterceptionCfg
from booster_mjlab.tasks.goalkeeper.mdp.outcomes import (
    reset_goalkeeper_outcome_state,
    update_goalkeeper_outcome_state,
)
from booster_mjlab.tasks.goalkeeper.mdp.terminations import (
    shot_recovery_finished,
    excessive_body_tilt,
)
from booster_mjlab.tasks.goalkeeper.mdp.metrics import (
    category_coverage,
    category_fall,
    category_goal,
    category_save,
    category_contact,
    category_velocity_change,
    category_miss,
    category_unresolved,
)


def booster_k1_goalkeeper_env_cfg(
    *,
    num_envs: int = 1,
    settings: FullGoalSettings | None = None,
    shot_mode: Literal["rolling", "six_region"] = "six_region",
) -> ManagerBasedRlEnvCfg:
    if settings is None:
        settings = FullGoalSettings()

    scene = make_goalkeeper_scene_cfg(
        robot_cfg=get_k1_robot_cfg(), settings=settings, num_envs=num_envs
    )

    cfg = make_goalkeeper_env_cfg(
        scene=scene,
        decimation=4,
        episode_length_s=settings.episode_length_s,
    )

    # 物理200Hz, 机器人控制50Hz
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
            "rolling_velocity_change_fraction": 0.8,
            "contact_sensor_name": "robot_ball_contact",
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

    robot_ball_contact = ContactSensorCfg(
        name="robot_ball_contact",
        primary=ContactMatch(mode="body", entity="ball", pattern="football"),
        secondary=ContactMatch(mode="subtree", entity="robot", pattern="Trunk"),
        secondary_policy="error",
        fields=("found", "force"),
        reduce="maxforce",
        num_slots=1,
        history_length=cfg.decimation,
    )

    cfg.scene.sensors = (
        *cfg.scene.sensors,
        feet_ground_contact,
        robot_ball_contact,
    )

    cfg.rewards["feet_slippage"] = RewardTermCfg(
        func=FeetSlippageReward,
        weight=3.0,
        params={
            "robot_name": "robot",
            "sensor_name": "feet_ground_contact",
            "contact_threshold": 1.0,
            "coefficient": 10.0,
        },
    )
    cfg.rewards["penalize_sharpcontact"] = RewardTermCfg(
        func=excessive_foot_contact_from_env,
        weight=-100.0,
        params={
            "sensor_name": "feet_ground_contact",
            "max_contact_force": 1000.0,
        },
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
            },
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
        },
    )
    # Define allowed blockers once; each manager term resolves its own copy.
    blocker_cfg = SceneEntityCfg(
        "robot",
        site_names=("left_hand", "right_hand"),
        body_names=("Trunk", "left_foot_link", "right_foot_link"),
        preserve_order=True,
    )
    guidance = asdict(InterceptionCfg())
    cfg.rewards["eereach"] = RewardTermCfg(
        func=ee_reach_from_env,
        weight=10.0,
        params={
            "blocker_cfg": deepcopy(blocker_cfg),
            "ball_name": "ball",
            "distance_sigma": 5.0,
            "reach_threshold": 0.2,
            "guidance": deepcopy(guidance),
        },
    )
    cfg.rewards["body_intercept"] = RewardTermCfg(
        func=body_intercept_from_env, weight=10.,
        params={"blocker_cfg": deepcopy(blocker_cfg), "guidance": deepcopy(guidance),
                "lateral_reach": .3, "movement_weight": 2.0, "fade_distance": .3,
                "reach_threshold": .05, "position_gain": 1.,
                "max_speed": .3, "velocity_std": .2},
    )
    cfg.rewards["stopball"] = RewardTermCfg(
        func=stopball_from_env,
        weight=100.0,
    )
    cfg.rewards["goal_conceded"] = RewardTermCfg(
        func=goal_conceded_cost,
        weight=-settings.goal_penalty,
    )
    cfg.rewards["fall"] = RewardTermCfg(
        func=fall_termination_cost,
        weight=-settings.fall_penalty,
    )
    cfg.rewards["body_tilt"] = RewardTermCfg(
        func=body_tilt_from_env,
        weight=-5.0,
        params={
            "robot_name": "robot",
            "deadband_deg": 20.0,
            "full_penalty_deg": 60.0,
        },
    )
    cfg.rewards["heading_deviation"] = RewardTermCfg(
        func=heading_deviation_from_env,
        weight=-settings.heading_penalty,
        params={
            "robot_name": "robot",
            "deadband_deg": settings.heading_deadband_deg,
            "full_penalty_deg": settings.heading_full_penalty_deg,
        },
    )
    cfg.rewards["yaw_rate"] = RewardTermCfg(
        func=excessive_yaw_rate_from_env,
        weight=-settings.yaw_rate_penalty,
        params={
            "robot_name": "robot",
            "deadband": settings.yaw_rate_deadband,
            "full_penalty_rate": settings.yaw_rate_full_penalty,
        },
    )
    cfg.rewards["success"] = RewardTermCfg(
        func=target_success_from_env,
        weight=5.0,
        params={
            "blocker_cfg": deepcopy(blocker_cfg),
            "strict_threshold": 0.15,
            "guidance": deepcopy(guidance),
        },
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

    for motion_id, motion_name in enumerate(MOTION_NAMES):
        metric_params = {"motion_id": motion_id}
        cfg.metrics[f"Goalkeeper/{motion_name}/coverage"] = MetricsTermCfg(
            func=category_coverage,
            params=metric_params,
            reduce="last",
        )
        cfg.metrics[f"Goalkeeper/{motion_name}/save"] = MetricsTermCfg(
            func=category_save,
            params=metric_params,
            reduce="last",
        )
        for suffix, function in (
            ("contact", category_contact),
            ("velocity_change", category_velocity_change),
            ("miss", category_miss),
            ("unresolved", category_unresolved),
        ):
            cfg.metrics[f"Goalkeeper/{motion_name}/{suffix}"] = MetricsTermCfg(
                func=function,
                params=metric_params,
                reduce="last",
            )
        cfg.metrics[f"Goalkeeper/{motion_name}/goal"] = MetricsTermCfg(
            func=category_goal,
            params=metric_params,
            reduce="last",
        )
        cfg.metrics[f"Goalkeeper/{motion_name}/fall"] = MetricsTermCfg(
            func=category_fall,
            params=metric_params,
            reduce="last",
        )

    return cfg


def booster_k1_goalkeeper_amp_env_cfg(
    *,
    num_envs: int = 1,
    settings: FullGoalSettings | None = None,
) -> ManagerBasedRlEnvCfg:
    """Reuse the goalkeeper task with a raw absolute-joint AMP group."""
    cfg = booster_k1_goalkeeper_env_cfg(
        num_envs=num_envs,
        settings=settings,
        shot_mode="six_region",
    )

    cfg.observations["amp"] = ObservationGroupCfg(
        terms={
            "joint_pos": ObservationTermCfg(
                func=amp_joint_positions,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot",
                        joint_names=K1_JOINT_ORDER,
                        preserve_order=True,
                    )
                },
            )
        },
        concatenate_terms=True,
        enable_corruption=False,
    )

    return cfg
