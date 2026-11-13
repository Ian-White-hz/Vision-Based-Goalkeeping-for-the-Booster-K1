"""Source reward constants with explicit K1 goal-plane region adaptations."""

from dataclasses import dataclass

MOTION_NAMES = (
    "lefthand",
    "righthand",
    "leftjump",
    "rightjump",
    "leftstep",
    "rightstep",
)
# width min/max, height min/max, in the source order.
# K1 baseline: ball-centre bounds at goal x=-0.8. High regions remain
# challenging; these bounds are not a claim that every shot is reachable.
REGIONS = (
    (0.1, 0.6, 0.4, 0.75),
    (-0.6, -0.1, 0.4, 0.75),
    (0.1, 0.6, 0.75, 1.05),
    (-0.6, -0.1, 0.75, 1.05),
    (0.1, 0.6, 0.12, 0.3),
    (-0.6, -0.1, 0.12, 0.3),
)
BOUNDS = REGIONS
EVAL_REGIONS = REGIONS
REWARD_SCALES = dict(
    eereach=10.0,
    success=5.0,
    stopball=100.0,
    stayonline=-2.0,
    noretreat=-2.0,
    successland=4.0,
    feetorientaion=3.0,
    penalize_sharpcontact=-100.0,
    penalize_kneeheight=-100.0,
    feet_slippage=3.0,
    postorientation=3.0,
    postangvel=3.0,
    postupperdofpos=1.0,
    ang_vel_xy=-0.1,
    dof_acc=-2.5e-7,
    smoothness=-0.1,
    torques=-1e-5,
    dof_vel=-5e-4,
    dof_pos_limits=-3.0,
    dof_vel_limits=-2.0,
    torque_limits=-3.0,
)


# No waist on K1: omit postwaistdofpos/deviation_waist_pitch_joint, don't substitute hips.
@dataclass(frozen=True)
class SourceSettings:
    play: bool = False
    randomize: bool = True

    # Ball perturbations are opt-in; robot randomization remains independent.
    ball_drag: bool = False
    ball_force_noise: bool = False
    ball_velocity_noise: bool = False
