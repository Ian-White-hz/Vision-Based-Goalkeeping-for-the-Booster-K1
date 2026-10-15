"""Reusable scene adapter: preserve a new policy's observation/action/reward design."""

import copy
import math
from mjlab.managers import EventTermCfg, TerminationTermCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg
from mjlab.viewer.viewer_config import ViewerConfig

from goalkeeper_sim.env_cfg import goalkeeper_env_cfg
from .assets import get_ball_cfg, get_goal_cfg
from .events import launch_shots
from .outcomes import FullGoalOutcome, ball_stopped, fallen, invalid_ball, out_of_guard_region
from .settings import FullGoalSettings


def head_camera_quaternion():
    angle = math.radians(50) / 2
    c, s = math.cos(angle), math.sin(angle)
    return (0.5 * (c + s), 0.5 * (c - s), -0.5 * (c - s), -0.5 * (c + s))


def full_goal_env_cfg(play=False, *, num_envs=None, settings=None, policy_env_cfg=None, enable_camera=False):
    """Add this world to an explicit new task, or use the existing truth baseline.

    No pretrained actor, detector, AMP runner or high-level interception controller
    is imported. A camera adds a sensor, not an RGB observation or a detector.
    """
    settings = settings or FullGoalSettings()
    cfg = copy.deepcopy(policy_env_cfg) if policy_env_cfg is not None else goalkeeper_env_cfg(play=False)
    if num_envs is not None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        cfg.scene.num_envs = num_envs
    cfg.scene.entities["ball"] = get_ball_cfg(settings)
    cfg.scene.entities["goal"] = get_goal_cfg(settings)
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    # Fixed robot resets and no inherited startup randomization match the saved scene.
    cfg.events = {k: v for k, v in cfg.events.items() if k in ("reset_base", "reset_robot_joints")}
    cfg.events["reset_base"].params["pose_range"] = {"z": (settings.root_reset_z_offset,) * 2}
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.events["reset_robot_joints"].params["position_range"] = (0.0, 0.0)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (0.0, 0.0)
    cfg.events["full_goal_launch"] = EventTermCfg(func=launch_shots, mode="reset", params={"settings": settings})
    if policy_env_cfg is None:
        cfg.curriculum = {}
    if play:
        cfg.observations["actor"].enable_corruption = False
    sensors = {s.name: s for s in cfg.scene.sensors}
    for required in ("feet_ground_contact", "non_foot_ground_contact"):
        if required not in sensors:
            raise ValueError(f"K1 task must define {required}")
    sensors["body_ball_contact"] = ContactSensorCfg(
        name="body_ball_contact", primary=ContactMatch(mode="body", entity="robot", pattern=".*",
        exclude=("left_hand_link", "right_hand_link", "left_hand_end_ball", "right_hand_end_ball")),
        secondary=ContactMatch(mode="geom", entity="ball", pattern="football_geom"),
        fields=("found",), reduce="none", num_slots=1)
    sensors["ball_ground_contact"] = ContactSensorCfg(
        name="ball_ground_contact", primary=ContactMatch(mode="geom", entity="ball", pattern="football_geom"),
        secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found",), reduce="none", num_slots=1)
    if enable_camera:
        sensors["keeper_rgb"] = CameraSensorCfg(name="keeper_rgb", parent_body="robot/Head_2", pos=(.12, 0, .07),
            quat=head_camera_quaternion(), fovy=100., width=128, height=96,
            data_types=("rgb",), use_textures=True, use_shadows=False)
    cfg.scene.sensors = tuple(sensors.values())
    cfg.sim.nconmax = cfg.sim.contact_sensor_maxmatch = 128
    cfg.sim.mujoco.timestep, cfg.decimation = .005, 4
    cfg.episode_length_s = settings.episode_length_s
    terms = dict(cfg.terminations)
    terms.pop("goal_conceded", None)
    terms["fell_over"] = TerminationTermCfg(func=fallen)
    terms["out_of_guard_region"] = TerminationTermCfg(func=out_of_guard_region, params={"settings": settings})
    terms["invalid_ball"] = TerminationTermCfg(func=invalid_ball)
    terms["ball_stopped"] = TerminationTermCfg(func=ball_stopped)
    cfg.terminations = {"goal_conceded": TerminationTermCfg(func=FullGoalOutcome, params={"settings": settings}), **terms}
    cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    cfg.viewer.lookat, cfg.viewer.distance = (.4, 0, .4), 3.
    cfg.viewer.elevation, cfg.viewer.azimuth = 20., 150.
    return cfg
