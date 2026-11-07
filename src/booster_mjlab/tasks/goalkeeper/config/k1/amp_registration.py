from mjlab.managers.observation_manager import ObservationGroupCfg
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
    amp_observation: ObservationGroupCfg,
    expert_sampler_factory: str,
    motion_id_reader: str,
    num_envs: int = 32,
    settings: FullGoalSettings | None = None,
) -> None:
    """在真实 AMP 接口到位后，显式注册 AMP 任务。"""
    if not expert_sampler_factory or not motion_id_reader:
        raise ValueError("Both AMP interface paths must be provided")

    rl_cfg = booster_k1_goalkeeper_amp_runner_cfg()
    rl_cfg.amp.expert_sampler_factory = expert_sampler_factory
    rl_cfg.amp.motion_id_reader = motion_id_reader

    env_cfg = booster_k1_goalkeeper_amp_env_cfg(
        amp_observation=amp_observation,
        num_envs=num_envs,
        settings=settings,
    )
    play_env_cfg = booster_k1_goalkeeper_amp_env_cfg(
        amp_observation=amp_observation,
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