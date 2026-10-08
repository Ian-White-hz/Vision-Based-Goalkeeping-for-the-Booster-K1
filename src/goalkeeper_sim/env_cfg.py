"""K1 goalkeeping environment built on the course Booster baseline."""

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from booster_mjlab.tasks.velocity import mdp as velocity_mdp
from booster_mjlab.tasks.velocity.config.k1.env_cfgs import booster_k1_flat_env_cfg

from . import mdp
from .assets import BALL_RADIUS_M, get_ball_cfg


def goalkeeper_env_cfg(play: bool = False):
    """Add a controllable rolling ball to the validated flat-ground K1 task."""
    cfg = booster_k1_flat_env_cfg(play=play)
    cfg.scene.entities["ball"] = get_ball_cfg()

    ball_observation = ObservationTermCfg(
        func=mdp.ball_state_b,
        params={"robot_name": "robot", "ball_name": "ball"},
    )
    cfg.observations["actor"].terms["ball_state"] = ball_observation
    cfg.observations["critic"].terms["ball_state"] = ball_observation

    # Start from a repeatable centered robot. Ball launch state remains randomized.
    cfg.events["reset_base"].params["pose_range"] = {}
    cfg.events["reset_base"].params["velocity_range"] = {}
    cfg.events.pop("push_robot", None)
    cfg.events["reset_ball"] = EventTermCfg(
        func=mdp.reset_ball_state,
        mode="reset",
        params={
            "position_range": {
                "x": (1.5, 2.5),
                "y": (-0.6, 0.6),
                "z": (BALL_RADIUS_M, BALL_RADIUS_M),
            },
            "velocity_range": {
                "x": (-0.6, -0.2),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
            },
        },
    )

    # The migrated locomotion rewards expect a twist command. Keep it at zero
    # until goalkeeper-specific actions and rewards are introduced.
    twist = cfg.commands["twist"]
    assert isinstance(twist, velocity_mdp.UniformVelocityCommandCfg)
    twist.rel_standing_envs = 1.0
    twist.rel_heading_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.ranges.lin_vel_x = (0.0, 0.0)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    twist.ranges.heading = (0.0, 0.0)
    cfg.curriculum.pop("command_vel", None)

    cfg.viewer.distance = 4.0
    cfg.viewer.azimuth = 180.0
    return cfg
