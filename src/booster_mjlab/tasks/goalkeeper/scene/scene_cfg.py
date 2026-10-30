from mjlab.entity import EntityCfg
from mjlab.scene import SceneCfg
from mjlab.terrains import TerrainEntityCfg

from booster_mjlab.terrains.contact import terrain_collisions

from .assets import get_ball_cfg, get_goal_cfg
from .settings import FullGoalSettings

def make_goalkeeper_scene_cfg(
    robot_cfg: EntityCfg,
    *,
    settings: FullGoalSettings | None = None,
    num_envs: int = 1,
) -> SceneCfg:
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}")
    
    if settings is None:
        settings = FullGoalSettings()

    return SceneCfg(
        num_envs=num_envs,
        terrain=TerrainEntityCfg(
            terrain_type="plane",
            collisions=terrain_collisions(),
        ),
        entities={
            "robot": robot_cfg,
            "ball": get_ball_cfg(settings),
            "goal": get_goal_cfg(settings),
        }
    )