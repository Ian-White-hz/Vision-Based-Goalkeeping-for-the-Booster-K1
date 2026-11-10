from copy import deepcopy
from dataclasses import fields

from mjlab.rl import RslRlModelCfg

from booster_mjlab.rl.config import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from booster_mjlab.amp.runners.categorical_runner_cfg import (
    CategoricalAmpCfg,
    CategoricalAmpPpoAlgorithmCfg,
    CategoricalAmpRunnerCfg,
    PeriodicEvaluationCfg,
)


def booster_k1_goalkeeper_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        obs_groups={
            "actor": ("actor",),
            "critic": ("critic",),
        },
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="relu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": ("rsl_rl.modules.distribution:GaussianDistribution"),
                "init_std": 1.0,
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="relu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="k1_goalkeeper",
        num_steps_per_env=24,
        max_iterations=300,
        save_interval=50,
        logger="tensorboard",
        upload_model=False,
    )


def booster_k1_goalkeeper_amp_runner_cfg() -> CategoricalAmpRunnerCfg:
    base = booster_k1_goalkeeper_ppo_runner_cfg()

    # 复用 PPO 超参数，但保留 AMP 算法自己的 class_name。
    algorithm_params = {
        item.name: deepcopy(getattr(base.algorithm, item.name))
        for item in fields(base.algorithm)
        if item.name != "class_name"
    }

    return CategoricalAmpRunnerCfg(
        obs_groups=deepcopy(base.obs_groups),
        actor=deepcopy(base.actor),
        critic=deepcopy(base.critic),
        algorithm=CategoricalAmpPpoAlgorithmCfg(**algorithm_params),
        amp=CategoricalAmpCfg(
            style_weight=0.4,
            reward_mode="direct",
            hidden_layer_sizes=(512, 256),
            dataset_root="artifacts/goalkeeper_k1/serial",
            expert_sampler_factory=(
                "booster_mjlab.tasks.goalkeeper.amp_interfaces:"
                "create_goalkeeper_expert_sampler"
            ),
            motion_id_reader=(
                "booster_mjlab.tasks.goalkeeper.amp_interfaces:get_amp_motion_ids"
            ),
        ),
        experiment_name="k1_goalkeeper_amp",
        evaluation=PeriodicEvaluationCfg(
            interval=500,
            evaluator_factory=(
                "booster_mjlab.tasks.goalkeeper.evaluation:GoalkeeperEvaluator"
            ),
        ),
        num_steps_per_env=base.num_steps_per_env,
        max_iterations=base.max_iterations,
        save_interval=base.save_interval,
        logger=base.logger,
        upload_model=base.upload_model,
        warm_start=False,
    )
