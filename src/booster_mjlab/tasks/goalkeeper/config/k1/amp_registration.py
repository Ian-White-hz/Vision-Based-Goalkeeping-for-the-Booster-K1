from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.amp.runners.categorical_on_policy_runner import (
    CategoricalAmpOnPolicyRunner,
)
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings

from .env_cfgs import booster_k1_goalkeeper_amp_env_cfg
from .rl_cfg import booster_k1_goalkeeper_amp_runner_cfg


AMP_TASK_ID = "Mjlab-Goalkeeper-AMP-Flat-Booster-K1"


def register_booster_k1_goalkeeper_amp_task(
    *,
    num_envs: int = 32,
    settings: FullGoalSettings | None = None,
) -> None:
    """Register the concrete categorical AMP goalkeeper task."""
    rl_cfg = booster_k1_goalkeeper_amp_runner_cfg()

    env_cfg = booster_k1_goalkeeper_amp_env_cfg(
        num_envs=num_envs,
        settings=settings,
    )
    play_env_cfg = booster_k1_goalkeeper_amp_env_cfg(
        num_envs=1,
        settings=settings,
    )

    register_mjlab_task(
        task_id=AMP_TASK_ID,
        env_cfg=env_cfg,
        play_env_cfg=play_env_cfg,
        rl_cfg=rl_cfg,
        runner_cls=CategoricalAmpOnPolicyRunner,
    )
