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
# Compact K1 training goal; dimensions are a task choice, not a measured
# dynamic reach guarantee. Ball-centre bounds leave radius clearance.
GOAL_WIDTH = 1.5
GOAL_HEIGHT = 1.1
GOAL_X = -0.8
REGIONS = (
    (0.05, 0.55, 0.3, 0.65),
    (-0.55, -0.05, 0.3, 0.65),
    (0.05, 0.55, 0.65, 0.9),
    (-0.55, -0.05, 0.65, 0.9),
    (0.05, 0.55, 0.12, 0.3),
    (-0.55, -0.05, 0.12, 0.3),
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
