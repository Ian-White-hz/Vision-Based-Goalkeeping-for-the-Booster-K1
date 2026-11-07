from unittest.mock import patch

import mjlab

from mjlab.managers.observation_manager import (
    ObservationGroupCfg,
    ObservationTermCfg,
)
from mjlab.tasks.registry import list_tasks

from booster_mjlab.tasks.goalkeeper.config.k1 import (
    amp_registration as registration,
)


def fake_joint_positions(env):
    return env.q


group = ObservationGroupCfg(
    terms={
        "joint_pos": ObservationTermCfg(
            func=fake_joint_positions,
        ),
    },
    concatenate_terms=True,
    enable_corruption=False,
)

before = set(list_tasks())

with patch.object(registration, "register_mjlab_task") as register:
    registration.register_booster_k1_goalkeeper_amp_task(
        amp_observation=group,
        expert_sampler_factory="test_only:factory",
        motion_id_reader="test_only:reader",
        num_envs=2,
    )

    register.assert_called_once()
    arguments = register.call_args.kwargs

assert arguments["task_id"] == registration.AMP_TASK_ID
assert arguments["runner_cls"] is registration.CategoricalAmpOnPolicyRunner

assert "amp" in arguments["env_cfg"].observations
assert "amp" in arguments["play_env_cfg"].observations
assert arguments["env_cfg"].scene.num_envs == 2
assert arguments["play_env_cfg"].scene.num_envs == 1

rl_cfg = arguments["rl_cfg"]
assert rl_cfg.amp.expert_sampler_factory == "test_only:factory"
assert rl_cfg.amp.motion_id_reader == "test_only:reader"
assert rl_cfg.algorithm.class_name.endswith(":CategoricalAmpPPO")

# 没有向真实注册表写入占位任务。
assert set(list_tasks()) == before

print("AMP task 注册入口组装检查通过")