from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg
)
from mjlab.scene import SceneCfg

from .mdp.observations import ball_position_b

def make_goalkeeper_env_cfg(
    scene: SceneCfg,
    *,
    decimation: int,
    episode_length_s: float,
) -> ManagerBasedRlEnvCfg:
    
    actor_terms = {
        "ball_position": ObservationTermCfg(
            func=ball_position_b,
            params={
                "robot_name": "robot",
                "ball_name": "ball"
            }
        )
    }

    critic_terms = deepcopy(actor_terms)

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

    return ManagerBasedRlEnvCfg(
        scene=scene,
        decimation=decimation,
        episode_length_s=episode_length_s,
        observations=observations
    )