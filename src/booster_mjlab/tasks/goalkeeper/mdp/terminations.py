import math

import torch

from .outcomes import update_goalkeeper_outcome_state


def shot_recovery_finished(
    env,
    recovery_duration_s: float = 2.0
) -> torch.Tensor:

    if (
        not math.isfinite(recovery_duration_s)
        or recovery_duration_s <= 0.0
    ):
        raise ValueError("recovery_duration_s must be positive and finite.")

    state = update_goalkeeper_outcome_state(env)

    recovery_steps = math.ceil(
        recovery_duration_s / env.step_dt
    )

    elapsed_steps = (
        env.common_step_counter - state.shot_resolved_step
    )

    return (
        state.shot_resolved
        & (elapsed_steps >= recovery_steps)
    )