import math

import torch


@torch.no_grad()
def mix_amp_rewards(
    task_rewards: torch.Tensor,
    amp_rewards: torch.Tensor,
    *,
    style_weight: float,
    reward_scale: float = 1.0,
    step_dt: float,
    scale_rewards_by_dt: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """At reward_scale=1, match the source runner: (1-w)*env_reward + w*(0.5*AMP), without extra dt.

    step_dt and scale_rewards_by_dt describe the environment's task reward.
    They are retained as explicit inputs, but never rescale either component.
    """
    if task_rewards.ndim != 1 or amp_rewards.shape != task_rewards.shape:
        raise ValueError("Both rewards must have the same shape (N,)")

    if (
        task_rewards.dtype != torch.float32
        or amp_rewards.dtype != task_rewards.dtype
        or amp_rewards.device != task_rewards.device
    ):
        raise ValueError("Rewards must use float32 on the same device")

    if not math.isfinite(style_weight) or not 0.0 <= style_weight <= 1.0:
        raise ValueError("style_weight must be finite and in [0, 1]")

    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("step_dt must be finite and positive")

    if (
        not torch.isfinite(task_rewards).all().item()
        or not torch.isfinite(amp_rewards).all().item()
    ):
        raise FloatingPointError("Non-finite reward input")

    if not math.isfinite(reward_scale) or reward_scale < 0.0:
        raise ValueError("reward_scale must be finite and nonnegative")

    task_component = (1.0 - style_weight) * task_rewards
    style_component = reward_scale * style_weight * (0.5 * amp_rewards)

    total_rewards = task_component + style_component

    return total_rewards, task_component, style_component
