from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.rl.runner import BoosterOnPolicyRunner

from .env_cfgs import booster_k1_goalkeeper_env_cfg
from .rl_cfg import booster_k1_goalkeeper_ppo_runner_cfg
from .amp_registration import register_booster_k1_goalkeeper_amp_task

register_mjlab_task(
    task_id="Mjlab-Goalkeeper-Flat-Booster-K1",
    env_cfg=booster_k1_goalkeeper_env_cfg(
        num_envs=32,
        shot_mode="six_region",
    ),
    play_env_cfg=booster_k1_goalkeeper_env_cfg(
        num_envs=1,
        shot_mode="six_region",
    ),
    rl_cfg=booster_k1_goalkeeper_ppo_runner_cfg(),
    runner_cls=BoosterOnPolicyRunner,
)

register_booster_k1_goalkeeper_amp_task()
