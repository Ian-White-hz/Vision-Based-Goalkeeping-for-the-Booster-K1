"""MJLab task registration for the K1 goalkeeping simulation."""

from mjlab.tasks.registry import register_mjlab_task

from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from booster_mjlab.tasks.velocity.config.k1.rl_cfg import booster_k1_ppo_runner_cfg

from . import TASK_ID
from .env_cfg import goalkeeper_env_cfg

agent_cfg = booster_k1_ppo_runner_cfg()
agent_cfg.experiment_name = "goalkeeper_k1"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=goalkeeper_env_cfg(),
    play_env_cfg=goalkeeper_env_cfg(play=True),
    rl_cfg=agent_cfg,
    runner_cls=BoosterOnPolicyRunner,
)
