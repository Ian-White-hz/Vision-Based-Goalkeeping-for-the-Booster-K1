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
GOAL_X = -0.1
INTERCEPT_X = 0.0
# Shot launch: origin x range and time to reach the robot plane. 3..4.5 m with
# .6...8 s gives |vx| ~4.4..6.4 m/s and 25..40 deg descent (10-90%); the
# previous 1.5..2.5 m / .7...9 s produced 47..62 deg lobs.
SHOT_START_X = (3.0, 4.5)
SHOT_INTERCEPT_TIME = (0.6, 0.8)
# Actor sees the ball only inside this robot-frame x range (upstream 3.4 for
# 3..5 m launches); must exceed SHOT_START_X[1] or early flight is hidden.
BALL_VISIBLE_X = 4.8
# Ball-phase thresholds rescaled for K1 shots (see k1_rewards.py).
# Upstream: far phase x>1.5, deflection dvx>2 (~36% of median |vx|), behind
# x<0. With K1 shots at |vx|~5.3 m/s, x=1.5 again leaves ~.28 s of reach phase.
PHASE1_X = 1.5
DEFLECT_RATIO = 0.4
DEFLECT_MIN = 0.5
BEHIND_X = GOAL_X
# EMA weight per finished episode for the episode-length curriculum signal.
CURRICULUM_EMA = 0.01
INTERCEPT_HEIGHTS = (
    (0.4, 0.75),
    (0.4, 0.75),
    (0.8, 1.05),
    (0.8, 1.05),
    (0.15, 0.35),
    (0.15, 0.35),
)
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
    postlinvel=1.0,
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
