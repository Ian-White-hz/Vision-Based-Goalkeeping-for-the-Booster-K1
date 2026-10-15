"""Opt-in registrations; importing the original task does not import this module."""

from mjlab.tasks.registry import register_mjlab_task
from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from booster_mjlab.tasks.velocity.config.k1.rl_cfg import booster_k1_ppo_runner_cfg

from . import RGB_TASK_ID, TASK_ID
from .env_cfg import full_goal_env_cfg

for task_id, camera in ((TASK_ID, False), (RGB_TASK_ID, True)):
    agent = booster_k1_ppo_runner_cfg()
    agent.experiment_name = "goalkeeper_full_goal_rgb" if camera else "goalkeeper_full_goal"
    register_mjlab_task(task_id=task_id, env_cfg=full_goal_env_cfg(enable_camera=camera),
        play_env_cfg=full_goal_env_cfg(play=True, enable_camera=camera),
        rl_cfg=agent, runner_cls=BoosterOnPolicyRunner)
