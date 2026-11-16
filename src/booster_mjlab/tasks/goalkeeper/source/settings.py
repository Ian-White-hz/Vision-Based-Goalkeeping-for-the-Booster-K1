"""Source reward constants with explicit K1 robot-plane region adaptations."""

from dataclasses import dataclass

MOTION_NAMES = (
    "lefthand",
    "righthand",
    "leftjump",
    "rightjump",
    "leftstep",
    "rightstep",
)
# K1 training goal; dimensions are a task choice. Standing arm-only hand reach
# is z<=1.08 m and |y|<=.50 m (mid/high), |y|<=.30 m (low band). The 1.3 m bar
# admits high shots up to 1.16 m at the robot plane (needs a jump); width 2.0 m
# fits mid/low shots out to |y|=.85 m (posts at 1.0 m, ball radius and margin),
# which need a step.
GOAL_WIDTH = 2.0
GOAL_HEIGHT = 1.3
# Goal line sits behind the robot (root at x=0, heels at about -0.1) so the
# keeper stands in front of the goal. Visual only; no collision.
GOAL_X = -0.4
# Robot plane where regions and reward targets are defined.
INTERCEPT_X = 0.0
BALL_RADIUS = 0.1
# Physics step; the sampler cancels semi-implicit Euler gravity drift for it.
SIM_DT = 0.005
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
# Behind line is just past the robot body, independent of the goal line.
PHASE1_X = 1.5
DEFLECT_RATIO = 0.4
DEFLECT_MIN = 0.5
BEHIND_X = -0.1
# True: save credit (stopball / success flag) only when an arm subtree touched
# the ball. False is upstream: a deflection by any body part counts; regions
# start outside the body, so standing still earns little.
ARM_SAVE_CREDIT = False
# eereach factor after the ball passed or was deflected. Upstream 2.0 rewarded
# holding hands at a frozen target for the remaining ~2 s (15..20 per episode,
# about a third of positive reward) regardless of the save.
BEHIND_REACH_SCALE = 0.0
# Minimum arm-ball contact force (N) in any physics substep to count a touch.
ARM_TOUCH_FORCE = 0.1
# EMA weight per finished episode for the episode-length curriculum signal.
CURRICULUM_EMA = 0.01
# Elbow pitch rotates about the forearm axis (I ~ .0015 kg m^2 incl. armature).
# The source PD is explicit at SIM_DT, stable only for kd*dt/I < 2; booster-lab
# kd=1 gives ~3.3 and the torque chatters between +/-14 Nm. 0.3 is about
# critical damping (2*sqrt(kp*I) ~ .25) for kp=10.
ELBOW_PITCH_DAMPING = 0.3
# Startup randomization scaled to K1 (19.8 kg, trunk ~6.5 kg). Upstream G1
# values (-5..+10 kg, +/-.1 m COM) gave trunk 1.5..16.5 kg and 59% falls for a
# deterministic policy (7% without randomization).
TRUNK_PAYLOAD_KG = (-1.0, 3.0)
TRUNK_COM_OFFSET_M = 0.03
# Robot base push (uniform xy velocity, m/s, every 750 control steps). Upstream
# G1 used 1.5; on K1 it raised deterministic falls from 12% to 28%.
PUSH_VEL = 0.5
# AMP discriminator inputs: indices into K1_JOINT_ORDER. All 22 joints, as
# upstream. Head+arms only (range(10)) let a standing keeper earn full style
# reward. Legs need the flat-foot clips (AMP_DATASET): the raw retargets put
# the flat-foot static COM behind the heel and AMP pulled the stance backward.
AMP_JOINT_IDS = tuple(range(22))
AMP_DATASET = "artifacts/goalkeeper_k1/serial_flatfoot"
# Ball-centre (y_min, y_max, z_min, z_max) at the ROBOT plane x=INTERCEPT_X.
# Order: mid L/R, high L/R, low L/R. The goal plane only has to be scored on
# (inside posts, under bar); low shots may bounce between robot and goal.
# Upstream G1 regions (mid y .2..1.2 z .4..1.2, high y 0..1 z 1.2..1.6, low
# y .2..1.2 z .1..3) scaled by the K1/G1 height ratio ~.72: mid/low reach
# |y|=.85 (step), high reaches z=1.16 and |y|=.72 (jump). Mid/low start at
# |y|=.15, outside the body; low z is floored by the ball radius.
REGIONS = (
    (0.15, 0.85, 0.3, 0.85),
    (-0.85, -0.15, 0.3, 0.85),
    (0.05, 0.72, 0.87, 1.16),
    (-0.72, -0.05, 0.87, 1.16),
    (0.15, 0.85, 0.12, 0.25),
    (-0.85, -0.15, 0.12, 0.25),
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
