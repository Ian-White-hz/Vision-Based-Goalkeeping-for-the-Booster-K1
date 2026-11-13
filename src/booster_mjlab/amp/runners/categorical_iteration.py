import math
import time

import torch

from booster_mjlab.amp.runners.categorical_step import collect_categorical_amp_step


def run_categorical_amp_iteration(
    env,
    alg,
    collector,
    obs,
    read_motion_ids,
    *,
    num_steps: int,
    style_weight: float,
    reward_scale: float = 1.0,
    on_env_step=None,
):
    if not torch.is_grad_enabled() or torch.is_inference_mode_enabled():
        raise RuntimeError("Call iteration outside no_grad/inference_mode")
    if num_steps <= 0 or num_steps != alg.storage.num_transitions_per_env:
        raise ValueError("num_steps must match PPO rollout storage")
    if alg.storage.step != 0:
        raise RuntimeError("PPO storage must be empty before iteration")
    if collector.model is not alg.amp_model:
        raise ValueError("Collector and PPO must share the AMP model")
    if not math.isclose(alg.amp_updater.frame_dt, env.unwrapped.step_dt, rel_tol=1e-7):
        raise ValueError("Expert frame_dt must match environment step_dt")

    alg.train_mode()
    sums = dict(total=0.0, amp_raw=0.0, task_component=0.0, style_component=0.0)
    valid_count = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(num_steps):
            obs, rewards, dones, extras, metrics = collect_categorical_amp_step(
                env,
                alg,
                collector,
                obs,
                read_motion_ids,
                style_weight=style_weight,
                reward_scale=reward_scale,
            )
            sums["total"] += rewards.mean().item()
            for name, value in metrics.items():
                sums[name] += value.mean().item()
            valid_count += collector.last_valid_mask.sum().item()
            if on_env_step is not None:
                on_env_step(rewards, dones, extras)
        collect_time = time.perf_counter() - start
        learn_start = time.perf_counter()
        alg.compute_returns(obs)

    # One update owns both PPO and AMP; do not update a second time afterward.
    losses = alg.update()
    learn_time = time.perf_counter() - learn_start
    return obs, {
        "ppo": losses,
        "discriminator": alg.last_amp_metrics,
        "reward_means": {name: value / num_steps for name, value in sums.items()},
        "amp_valid_fraction": valid_count / (num_steps * env.num_envs),
        "timing": {"collection": collect_time, "learning": learn_time},
    }
