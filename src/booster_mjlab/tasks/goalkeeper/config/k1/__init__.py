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

# Isolated reproduction of the upstream G1 task flow on the K1 plant.
from ...source.config import env_cfg as source_env_cfg, runner_cfg as source_runner_cfg
from ...source.runner import SourceRunner

register_mjlab_task(
    task_id="Mjlab-Goalkeeper-Source-Flat-Booster-K1",
    env_cfg=source_env_cfg(),
    play_env_cfg=source_env_cfg(num_envs=6, play=True),
    rl_cfg=source_runner_cfg(),
    runner_cls=SourceRunner,
)

# Same task with the actor's ball input from the geometric head-camera model
# and tracker (vision/ball_input.py); checkpoints of the truth task load as is.
register_mjlab_task(
    task_id="Mjlab-Goalkeeper-Source-Vision-Flat-Booster-K1",
    env_cfg=source_env_cfg(ball_input="synthetic"),
    play_env_cfg=source_env_cfg(num_envs=6, play=True, ball_input="synthetic"),
    rl_cfg=source_runner_cfg(),
    runner_cls=SourceRunner,
)
