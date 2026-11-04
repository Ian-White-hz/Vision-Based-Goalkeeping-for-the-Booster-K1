import math

import torch

from booster_mjlab.amp.runners.categorical_step import (
    collect_categorical_amp_step
)


def run_categorical_amp_iteration(
    env,
    alg,
    collector,
    amp_updater,
    obs,
    read_motion_ids,
    *,
    num_steps: int,
    amp_batch_size: int,
    style_weight: float,
    on_env_step=None,
):
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise RuntimeError("Call iteration outside no_grad/inference_mode")

    if num_steps <= 0 or amp_batch_size <= 0:
        raise ValueError("Step count and batch size must be positive")

    if alg.storage.step != 0:
        raise RuntimeError("PPO storage must be empty before iteration")

    if num_steps != alg.storage.num_transitions_per_env:
        raise ValueError("num_steps must match PPO rollout storage")

    if (
        amp_updater.model is not collector.model
        or amp_updater.replay is not collector.replay
    ):
        raise ValueError("Collector and updater must share model and replay")

    if not math.isclose(
        amp_updater.frame_dt,
        env.unwrapped.step_dt,
        rel_tol=1e-7,
    ):
        raise ValueError("Expert frame_dt must match environment step_dt")

    if alg.is_multi_gpu:
        raise NotImplementedError("Categorical AMP currently supports one device")

    alg.train_mode()

    reward_sums = {
        "total": 0.0,
        "amp_raw": 0.0,
        "task_component": 0.0,
        "style_component": 0.0,
    }

    # 采集，在 inference_mode 下运行环境步长，收集奖励和指标
    with torch.inference_mode():
        for _ in range(num_steps):
            obs, rewards, dones, extras, metrics = (
                collect_categorical_amp_step(
                    env,
                    alg,
                    collector,
                    obs,
                    read_motion_ids,
                    style_weight=style_weight,
                )
            )

            reward_sums["total"] += rewards.mean().item()
            for name, value in metrics.items():
                reward_sums[name] += value.mean().item()

            if on_env_step is not None:
                on_env_step(rewards, dones, extras)

        alg.compute_returns(obs)  

    ppo_losses = alg.update()
    discriminator_losses = amp_updater.update_all(
        batch_size=amp_batch_size
    )

    summary = {
        "ppo": ppo_losses,
        "discriminator": discriminator_losses,
        "reward_means": {
            name: value / num_steps
            for name, value in reward_sums.items()
        },
    }

    return obs, summary
    