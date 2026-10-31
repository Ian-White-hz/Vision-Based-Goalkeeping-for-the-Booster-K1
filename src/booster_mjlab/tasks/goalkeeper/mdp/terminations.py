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

def excessive_body_tilt(
    env,
    robot_name: str = "robot",
    max_tilt_deg: float = 60.0,
) -> torch.Tensor:
    if (
        not math.isfinite(max_tilt_deg)
        or not 0.0 < max_tilt_deg < 180.0
    ):
        raise ValueError("max_tilt_deg must be in (0, 90)")

    robot = env.scene[robot_name]

    gravity_z_b = robot.data.projected_gravity_b[:, 2]

    gravity_z_limit = -math.cos(
        math.radians(max_tilt_deg)
    )

    return gravity_z_b > gravity_z_limit