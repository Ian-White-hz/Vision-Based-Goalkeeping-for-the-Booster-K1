# K1 shot shape, ball-phase thresholds and smoothed curriculum

Branch `experiment/k1-source-goalkeeper`, task `Mjlab-Goalkeeper-Source-Flat-Booster-K1`.

## Motivation (diagnosis of v3 model_1000)

Upstream reward thresholds are sized for G1 shots (start x=3..5 m, vx about
-3..-12 m/s). After the K1 compact-goal change shots start at x=1.5..2.5 m with
vx -1.72..-3.55 m/s (median -2.72, 6000 sampled shots).

- Far-phase positioning (`eereach` phase1, ball x>1.5) lasted 0.21 s on average and
  overlapped the 3..10-step invisible/action-hold window, so it was effectively gone.
- Deflection (`vx - vx0 > 2`) needed about 75% of median K1 speed (upstream about 36%).
  Frozen eval: region 0 touched (dvx>.5) 59%, credited 42%.
- `x > 0` cut off contacts with the ball centre just behind the robot plane.
- Curriculum level = mean length of the episodes resetting at the update step.
  With 512 envs it flipped 0/1/2/1/3/2, so reach/success/stopball multipliers
  (1..2.5x) and joint-limit multipliers (1..3x) changed from one update to the next.

Shot shape: every shot was a plunging lob. At the robot plane the descent angle was
47/54/62 deg (10/50/90%), apex 1.1..1.6 m (above the 0.76 m Head_2 origin), and |vz|
exceeded |vx|. Cause: 1.5..2.5 m launch distance with .7...9 s to arrive. Each
control step drops the ball ~7 cm near the robot, and a 10 cm hand x error is
~14 cm of height error. Upstream descent is about 20..50 deg.

Keeper position: the robot stood on the goal line (goal x=-0.1, heels at about -0.1).
Regions were defined at the goal plane AND intercept heights at the robot plane;
with descending shots, that dual constraint made a goal further back infeasible
for low regions (0% acceptance at -0.5).

## Changes

Shots (option B): `SHOT_START_X`=3..4.5 m, `SHOT_INTERCEPT_TIME`=.6...8 s. Analytic,
60k samples: descent 25/33/40 deg, apex median 1.11 m, |vx| 4.4/5.3/6.4 m/s.
Reaction time is ~0.1 s shorter than before. High regions have lower rejection-sampler
acceptance (~51%), which the 512-attempt sampler absorbs. Actor ball visibility
cutoff `BALL_VISIBLE_X` raised from 3.4 to 4.8 m so launches are not hidden.

Regions are now ball-centre rectangles at the ROBOT plane x=0, with the old intercept
heights as z ranges: mid .4...75, high .8..1.05, low .15...35 m, y +/-[.05,.55].
The goal plane only has to be scored on (inside posts and under the bar, with ball
radius and a 3 cm margin), and bounces between robot and goal are allowed. Goal line
moved to `GOAL_X`=-0.4, about 0.3 m behind the heels. The sampler aims directly at
the robot-plane point and adds g*SIM_DT/2 to vz, which exactly cancels MuJoCo's
semi-implicit Euler drop (measured -g*dt*t/2 before compensation, ~1e-15 after),
so the reward target equals the simulated crossing. `BEHIND_X` is decoupled from
the goal and stays at -0.1, just past the robot body.

Goal size kept at 1.5 x 1.1 m. Standing arm-only hand reach (kinematic sweep of
4 arm joints): z<=1.08 m; |y|<=.50 m in the mid/high bands and <=.30 m in the low
band. With a 0.1 m ball radius, the bar is reachable without jumping. The mid/high
corners (ball centre |y|<=.55) are at the edge of reach; low corners need a step or
lean. Goal width/robot height is about 1.6, compared with about 2.7 for the upstream G1 task, so
this is a compact, easier goal. Widen it later as a curriculum.

`source/k1_rewards.py` overrides eereach, stopball and the four post* terms with
threshold attributes. Formulas are unchanged. `vendor/` is untouched. With the new shot
speed, upstream far-phase x=1.5 again leaves ~.28 s of reach phase, so `PHASE1_X`
returns to 1.5. Deflection and behind thresholds are still needed.

| threshold | upstream | K1 |
| --- | --- | --- |
| far phase | x > 1.5 | `PHASE1_X`=1.5 (attribute; same value under new shots) |
| deflection | dvx > 2 | dvx > max(0.4 * abs(vx0), 0.5) |
| behind / deflection line | x < 0 / x > 0 | `BEHIND_X` = -0.1 (goal line now -0.4) |
| curriculum signal | batch mean length | EMA over finished episodes (weight 0.01 per episode; first batch initialises) |

The EMA is saved in checkpoints and logged as `Train/curriculum_episode_length_ema`.
Older checkpoints resume with the sentinel, so the first finished batch initialises it.

## Validation

- `test_goalkeeper_source_flow.py`: with the upstream thresholds set on the state,
  every applicable reward still matches upstream exactly on real K1 state.
  The jump-bonus check moved its ball to x=0.5, inside the reach phase.
- New `test_goalkeeper_source_k1_thresholds.py`: relative and floor deflection,
  the one-shot reward, the behind line, the far/near phase switch, and EMA smoothing.
- Randomization and small-jump tests pass. Ruff passes.

- Region arrival (rewritten): 600 shots in MuJoCo. Every robot-plane crossing is
  inside its region and within `SHOT_INTERCEPT_TIME`, with no ground contact before
  the robot plane. Target error < 1e-4 m. Every goal-plane crossing is inside the
  goal mouth. Bounce-before-goal fraction: low regions ~1.0, region 1 .06, others 0.
- Threshold test also asserts median descent < 40 deg, 90th percentile < 48 deg,
  median |vx| > 4.5, and visibility above the launch range.

Detection check before the shot change (v3 model_1000, old lob shots, frozen,
100 per region):

| region | proxy save | ball kept out | touched dvx>.5 |
| --- | --- | --- | --- |
| 0 / 1 | .62 / .54 | .48 / .46 | .65 / .62 |
| 2 / 3 | .19 / .15 | .09 / .11 | .16 / .15 |
| 4 / 5 | .82 / .78 | .79 / .69 | .85 / .81 |

The proxy now tracks touches but over-credits by about 10 pp: some deflections still
enter the goal. Report "ball kept out" as the outcome metric, not the proxy.
Old checkpoints load but were trained on lobs; retrain from scratch. No training
result yet; this is not a learned-policy claim.

## Exploration noise

Evidence from shotB (512 envs, upstream noise settings, 1000 iterations, old goal at -0.1):
- Arm action std rose 1.06 -> 1.23 and leg std stayed .82..87. Upstream's converged
  G1 checkpoint (iteration 25,600, 6144 envs) has shoulder/elbow std ~.4.
- Training episode-length EMA was 117/150 steps (frequent falls under noise).
- Frozen eval was flat from 400 to 1000 iterations: ball kept out 34% -> 34%, falls 7% -> 15%.
  Hands moved ~0.5 m toward targets, but root lateral motion was 8 cm, with no steps
  or jumps. High-band balls missed by .25..30 m although standing arm-only reach covers them.

`runner_cfg()` for the source task now sets `init_std` 0.5 (was 1.0) and `entropy_coef`
.003 (was .01). The shared K1 goalkeeper `rl_cfg.py` is unchanged. A threshold
test asserts both values. A 12-env CLI smoke run wrote them to `params/agent.yaml`,
and the checkpoint std was 0.500.

## Elbow PD stability and K1-scaled randomization

Evidence from `source_k1_goal04_std05` model_2000 (512 envs, 2000 iterations), training
conditions (randomized, stochastic actions), per 150 steps, task x0.6 / AMP x0.4:
- Positive 48, negative -21. eereach 24.2 (15.3 after the ball passed, where the 2x
  behind-phase factor rewards hands near a frozen target), AMP 8.7, stopball 1.4,
  success 2.6. Zero action still collects eereach 19.
- torque_limits -11.7 (zero action -13). Elbow pitch was at 97% of its 14 Nm limit
  even at the default pose. Elbow pitch turns about the forearm axis
  (I ~ .0015 kg m^2 with armature). The source PD is explicit at 5 ms, and kd=1
  gives kd*dt/I ~ 3.3, above the stability bound of 2. The torque chattered between the limits.
- Episodes: 64% ended in a fall (deterministic, no pushes: 59%; randomization off: 7%).
  92% of falls happened after the ball had passed; almost all were pitch falls.
  Fall rate rose with trunk mass by quartile: .23/.36/.47/.60. Upstream payload -5..+10 kg
  on a ~6.5 kg K1 trunk gave 1.5..16.5 kg.
- Mean episode return: kept out & stood 39, goal & stood 29, fell ~9. Not falling
  was worth 20..30 and a save about 10, so standing still was the safe policy.

Changes: `ELBOW_PITCH_DAMPING`=0.3 (about critical damping for kp=10; elbow yaw and
shoulders keep kd=1). Peak elbow |tau|/limit at default pose is now .005 instead of 1.0.
`TRUNK_PAYLOAD_KG`=(-1, 3) and `TRUNK_COM_OFFSET_M`=.03. Link-mass scale, friction,
restitution and pushes are unchanged. `test_goalkeeper_source_k1_stability.py` checks
the gains, the absence of elbow saturation and the randomization ranges.
Deployment note: this is a fix for simulation discretization. The hardware motor driver runs its PD
loop much faster, so it does not imply kd=0.3 on the robot.
Old checkpoints were trained with the chattering elbows; retrain from scratch.
Reward changes (behind-phase eereach, stopball scale) are deferred to a separate ablation.

## AMP without legs and smaller pushes (quick ablation)

Evidence from `source_k1_elbowfix_dr` model_1000. Training-condition falls fell to 26%, and torque_limits to -0.5.
- Ball hits do not cause the falls. Deterministic fall rate was 27% with a touch and 30% without.
  The median ball impulse was 0.26 N s, about 1 cm/s of robot velocity, and the median fall came 43 steps after the touch.
- Falls are slow backward pitch topples after the ball has passed, which start about 60 steps
  before termination. Fall rate: 28% in training conditions, 12% without pushes, 1% without randomization.
  By trunk-COM-x quartile (back to front) it was .23/.14/.06/.02; by foot-friction quartile, .04 to .17.
- Static COM in the foot frame (sole -.066..+.109 m): default pose +.038, policy stance
  -.018, expert clip medians -.046..-.082 (step clips behind the heel), and jump-clip
  10th percentiles -.18..-.26. The retargeted K1 clips lean back, and the joint-only
  AMP discriminator rewarded that stance (AMP was 24% of positive reward).

Changes: `AMP_JOINT_IDS` = head and arms (10 joints, 20-dim window). The policy obs and
the expert windows use the same subset, and the legs are shaped by task rewards only. `PUSH_VEL` is 0.5 m/s
(was 1.5). Old checkpoints cannot fully resume, because the discriminator input changed from 44 to 20, but their
policy weights still load with `load_cfg`. Proper fix if this helps: re-retarget the clips with a
support-polygon COM constraint and restore leg AMP.

## Free saves: regions off the body, arm-only credit, no behind-phase reach

Evidence from `source_k1_amparms_push05` model_1000 (512 envs, 1000 iterations):
- Training-condition falls are 5%, and the episode-length EMA is 146/150. Balance is no longer the bottleneck.
- On the same shots, with no randomization, a frozen default-pose keeper (state rewritten every step)
  kept out 26% of balls (mid .38/.28, high .01/.00, low .41/.48). The policy kept out 36%
  (mid .63/.56, high .02/.00, low .43/.51). Over checkpoints 200..1000 the policy rose from 28% to 36%, flat after
  600. It only moves its arms (hands ~.4 m, root lateral 4 cm, no jumps). High-ball hands
  stayed .33 m from the target.
- Mid/low regions started at |y|=.05, inside the body (upstream G1: .2..1.2 m), and any
  deflection, including by trunk or legs, gave stopball/success. The frozen keeper touched 47% of balls.
- Reward per 150 steps: +61, of which ~45 came after the ball had passed (eereach x2
  19.1, AMP 13, posture terms). stopball+success was 4.5. A save was worth ~+18 return and a fall ~-45.
- Losses look healthy: value 5..10, |surrogate| ~.004, expert .01..3. Reward/total
  has been flat at ~.40 since iteration ~250. Adaptive LR fell to 1e-4..2e-4.
  Budget: 5.1e7 samples, versus ~1.6e10 for the upstream converged G1 checkpoint.

Changes:
- `REGIONS` mid/low |y| .15..85 (high unchanged, .05..55; it had no free saves). `GOAL_WIDTH`
  2.0 so that robot-plane shots out to .85 m score inside the posts.
- `ARM_SAVE_CREDIT`: stopball and the success flag need a ball contact with either arm
  subtree (`Left_Arm_1`/`Right_Arm_1` to the hand) in any physics substep. The new contact
  sensor `source_arm_ball_contact` has `history_length`=4, and the latch is `arm_touch`, cleared on reset.
  A trunk, head or leg deflection still closes the save window, but without credit.
- `BEHIND_REACH_SCALE` 0 (upstream 2): no eereach after the ball has passed or been deflected.
- The parity test sets `arm_save_credit=False` and `behind_reach_scale=2`, and still matches upstream.
  The new `test_goalkeeper_source_save_credit.py` uses real MuJoCo contacts: a shot at the hand is credited,
  a shot at the trunk is deflected without credit, behind-phase eereach is 0, and reset clears the latch.
  It also checks the region and goal geometry. The threshold test sets the latch explicitly.

Measured on the new setup (same shots, frozen keeper vs. old model_1000 policy):

| | kept out | save credit | arm touch |
| --- | --- | --- | --- |
| frozen default pose | .059 (mid .04/.09, low .11/.11) | .079 | .136 |
| old policy (trained on old setup) | .123 | .205 | .303 |

Reward composition with the old policy (stochastic, per 150 steps): behind-phase
eereach .03 (was 19.1). AMP 17.0 is now the largest term, and stopball+success is 3.5.
Remaining free terms: `success` still pays `(flag+1)` while the hand is within .15 m of the
frozen target after the ball has passed (2.5 per episode), and the post* posture terms
pay for standing still for ~2 s. These are candidates for the next ablation, together with AMP weight
and episode length. Retrain from scratch.
