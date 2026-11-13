"""Independent source-flow task: do not inherit tuned goalkeeper rewards."""

from dataclasses import dataclass, field, replace
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    EventTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    SceneEntityCfg,
)
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from booster_mjlab.robots import get_k1_robot_cfg
from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER
from booster_mjlab.tasks.goalkeeper.scene.scene_cfg import make_goalkeeper_scene_cfg
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings
from booster_mjlab.tasks.goalkeeper.config.k1.rl_cfg import (
    booster_k1_goalkeeper_ppo_runner_cfg,
)
from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg
from .settings import SourceSettings, REWARD_SCALES, GOAL_WIDTH, GOAL_HEIGHT, GOAL_X
from .state import observation, reward, termination
from .events import reset, perturb
from .actions import SourceActionCfg, SourceMotorCfg
from .physics import ball_cfg, restitution, body_mass, body_com_offset, geom_friction


@dataclass(kw_only=True)
class SourceEnvCfg(ManagerBasedRlEnvCfg):
    source_settings: SourceSettings = field(default_factory=SourceSettings)


@dataclass
class SourceRunnerCfg(RslRlOnPolicyRunnerCfg):
    dataset_root: str = "artifacts/goalkeeper_k1/serial"
    value_smoothness_coef: float = 0.1
    smoothness_upper_bound: float = 1.0
    smoothness_lower_bound: float = 0.1


def env_cfg(num_envs=6144, play=False, randomize=True):
    robot = get_k1_robot_cfg()
    robot.articulation = replace(
        robot.articulation,
        actuators=tuple(
            SourceMotorCfg(
                target_names_expr=a.target_names_expr,
                effort_limit=a.effort_limit,
                stiffness=a.stiffness,
                damping=a.damping,
                motor=a.motor,
                armature=a.armature,
            )
            for a in robot.articulation.actuators
        ),
    )
    scene = make_goalkeeper_scene_cfg(
        robot,
        settings=FullGoalSettings(
            ball_radius=0.1,
            ball_mass=0.1,
            goal_height=GOAL_HEIGHT,
            goal_width=GOAL_WIDTH,
            goal_x=GOAL_X,
        ),
        num_envs=num_envs,
    )
    # Visual goal only: all goal geoms have collision masks set to zero.
    scene.env_spacing = 5.0
    scene.entities["ball"] = ball_cfg()
    scene.sensors = (
        ContactSensorCfg(
            name="source_feet_contact",
            primary=ContactMatch(
                mode="subtree",
                entity="robot",
                pattern=r"^(left_foot_link|right_foot_link)$",
            ),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
        ),
    )
    cfg = SourceEnvCfg(
        scene=scene,
        decimation=4,
        episode_length_s=3.0,
        source_settings=SourceSettings(play=play, randomize=randomize),
        observations={
            k: ObservationGroupCfg(
                terms={
                    "state": ObservationTermCfg(func=observation, params={"kind": k})
                },
                concatenate_terms=True,
                enable_corruption=False,
            )
            for k in ("actor", "critic", "amp")
        },
        actions={
            "joint_pos": SourceActionCfg(
                entity_name="robot",
                actuator_names=K1_JOINT_ORDER,
                preserve_order=True,
                scale=0.25,
            )
        },
        rewards={
            name: RewardTermCfg(func=reward, weight=weight, params={"name": name})
            for name, weight in REWARD_SCALES.items()
        },
        terminations={
            k: TerminationTermCfg(
                func=termination, time_out=(k == "timeout"), params={"kind": k}
            )
            for k in ("timeout", "knees", "gravity", "force")
        },
        events={
            "source_reset": EventTermCfg(func=reset, mode="reset"),
            "source_perturb": EventTermCfg(func=perturb, mode="step"),
        },
    )
    if randomize:
        cfg.events.update(
            {
                "source_payload": EventTermCfg(
                    func=body_mass,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
                        "ranges": (-5.0, 10.0),
                        "operation": "add",
                    },
                ),
                "source_link_mass": EventTermCfg(
                    func=body_mass,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot", body_names=("(?!Trunk$).*",)
                        ),
                        "ranges": (0.8, 1.2),
                        "operation": "scale",
                    },
                ),
                "source_com": EventTermCfg(
                    func=body_com_offset,
                    mode="startup",
                    params={
                        "asset_cfg": SceneEntityCfg("robot", body_names=("Trunk",)),
                        "ranges": (-0.1, 0.1),
                    },
                ),
                "source_friction": EventTermCfg(
                    func=geom_friction,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot", geom_names=(".*_collision",)
                        ),
                        "ranges": (0.1, 2.0),
                        "operation": "abs",
                        "axes": [0],
                        "shared_random": True,
                    },
                ),
                "source_restitution": EventTermCfg(
                    func=restitution,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(
                            "robot", geom_names=(".*_collision",)
                        )
                    },
                ),
            }
        )
    cfg.sim.mujoco.timestep = 0.005
    cfg.sim.njmax = 1024
    cfg.viewer.distance = 4.0
    cfg.viewer.elevation = 25.0
    return cfg


def runner_cfg():
    from dataclasses import fields

    base = booster_k1_goalkeeper_ppo_runner_cfg()
    cfg = SourceRunnerCfg(**{f.name: getattr(base, f.name) for f in fields(base)})
    cfg.experiment_name = "k1_goalkeeper_source"
    cfg.num_steps_per_env = 100
    cfg.max_iterations = 200000
    cfg.save_interval = 200
    cfg.seed = 1
    cfg.actor.hidden_dims = (512, 256, 256)
    cfg.critic.hidden_dims = (512, 256, 256)
    cfg.actor.activation = cfg.critic.activation = "elu"
    cfg.actor.obs_normalization = cfg.critic.obs_normalization = False
    return cfg
