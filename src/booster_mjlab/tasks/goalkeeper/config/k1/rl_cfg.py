from mjlab.rl import RslRlModelCfg

from booster_mjlab.rl.config import(
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg
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
                "class_name": (
                    "rsl_rl.modules.distribution:GaussianDistribution"
                ),
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