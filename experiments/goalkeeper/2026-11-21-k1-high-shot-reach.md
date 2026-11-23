# K1 high shots: reach diagnosis and foot contact sign

Checkpoint `2026-11-17_11-28-20_source_k1_upstream_like_cont/model_12000.pt`,
truth ball input, 600 shots (100/region), play, deterministic
(`source/diagnose_reach.py`).

## Reach
- Standing kinematics: shoulder .75 m, head top .81 m, straight arm overhead
  1.13 m. AMP clips reach hand 1.23 (leftjump), 1.25 (rightjump), 1.15 m
  (righthand) with foot lift up to .23 m: the prior contains high reaches.
- High-L/R kept .08/.04; ball crossing z median .99 m, highest hand median
  .94/.88 m, foot lift median 3-4 cm. Missed high shots: 52% hand never above
  .90 m, 45% hand within .15 m of the ball centre (closest hand ~8 cm below).
  Even z .85-.95 m, |y|<.3 m shots kept 0 (hand max .79 m).
- So the miss is behavioural (no raise, no jump), not a reach limit.

## Bug: foot contact force sign
mjlab `ContactSensor` (reduce="netforce") reports the force of the primary
(foot) on the ground: a standing foot reads z ~ -100 N. Upstream Isaac Gym
net contact force is the ground reaction, z > 0. All `force_z > 1` tests
(JumpState support, successland both/one-foot) were never true, so
`successland` was 0 for every run, not because the keeper never jumped (root
rise reached .1-.2 m in some high shots). Norm-based terms (feet_slippage,
penalize_sharpcontact, contact termination) were unaffected.

Fix: `SourceState.refresh` negates the foot force. After the fix, standing
reads +97..+100 N, and with model_12000 34.5% of high shots count as a jump
(0.75% elsewhere). Earlier notes that read "successland 0 = no jump" are
wrong on the cause.

## Reward parity with upstream (high regions)
All upstream terms with a non-zero scale exist with the same weights.
Not ported: postwaistdofpos, deviation_waist_pitch_joint (K1 has no waist);
airfeetorientation is defined upstream but has no scale. Differences: jump
detected at +.05 m root rise (upstream root z > 1.0 m, i.e. +.2 m on G1), and
shot ranges start at their bounds (upstream widens them with the curriculum).

## Resume control and checkpoint spread (truth eval, 1200 shots, +-.035)
Truth-only continuation `2026-11-17_21-48-09_source_k1_20k` (from model_12000,
old contact sign, no vision): 12400 .459, 13000 .407, 14000 .450, 16000 .440,
18000 .441, 20000 .407 (model_12000 .444). Resuming itself causes no drop; the
vision fine-tune dip (.28 at 12400) came from the input change. The total
plateaus at .41-.46 while regions swing .15-.3 between checkpoints (mid-L
.34-.63, low-L .73-.97); high stays <= .07 at every checkpoint.
jumpfix model_15000 (.412) is inside this spread.

Touch vs save (arm contact before the goal line), 12000 -> jumpfix 15000:
mid-L .85/.52 -> .73/.31, mid-R .96/.59 -> .78/.41, high-L .45/.05 -> .31/.05,
high-R .24/.02 -> .06/.00, low-L .88/.68 -> .98/.95, low-R .95/.79 -> 1.00/.75.
Without arm contact a save is ~0 except low-L.

Network (policy path ~560k params): history encoder 750-128-64-16 (105k),
ball and region estimators 750-128-32-6 (100k each), actor 98-512-256-256-22
(253k); critic 92-512-256-256-1 (245k). Same layout as upstream.

## highreach (K1 only, opt-in, 2026-11-22)
Per-region weighted step means at model_12000 (TensorBoard, last 50 logs):
eereach high-L/R .30/.31 vs mid/low .22-.25 while high kept is .02-.05; the
jump multiplier pays upward trunk velocity without a raised hand. stopball
credits a slowdown (vx change > max(.4|vx0|, .5)), so brushed high shots that
still score are paid like saves (high .0075/step vs mid .0071). AMP per region
.086-.099: not the blocker; amp_coef .4 fixed as upstream.

`K1Rewards._reward_highreach`, regions 2/3, ball x <= PHASE1_X and not behind
or deflected. Better hand vs end_target: q = (below/.10)^2 + (above-.10 over
/.15)^2 + (dy/.15)^2; score .5 exp(-q) + .5 exp(-q/9) (wide half gives a
gradient far from the ball); times uprightness. Weight 0 by default
(`--env.rewards.highreach.weight`). Parity test skips K1_ONLY_REWARDS.
model_12000: active ~14 steps per high shot; per-step mean .23/.11 (L/R),
best step median .38/.25. Suggested weight 20 (max ~280 per shot vs eereach
~1400 per episode). Compare against the jumpfix run (same contact sign).

### highreach weight 20 result (`2026-11-22_16-35-56_source_k1_highreach`, from model_12000)
1200 shots, touch/kept. Total: 13400 .449, 14000 .429, 14400 .418, 15000 .468
(model_12000 .444, jumpfix 15000 .412).
high-L: .16/.00, .33/.03, .42/.05, .38/.02 (12000 .45/.05).
high-R: .44/.01, .29/.04, .31/.04, .04/.00 (12000 .24/.02).
Hand max z high-L/R at 15000 .92/.85 (12000 .94/.88). No gain in high.
`Region/2,3/reward_highreach` flat at .010-.011 per step for all 3000 iters vs
eereach ~.30 in the same regions: weight 20 is ~4% of eereach, too small (the
~280/shot estimate assumed a full score; actual score ~.2). high-R jumps rose
to .75 at 15000 with touch .04: the eereach jump multiplier pays trunk vz
without a raised hand. 15000's total gain is mid-L (.62), inside the known
checkpoint spread.

### highreach weight 100 (`2026-11-22_21-04-22_source_k1_highreach100`, from model_12000)
Total 13400 .286 (low-L .31: early shock), 14000 .380, 14400 .441, 15000 .456.
high-L touch/kept .17/.01, .27/.00, .28/.03, .33/.04; high-R .11/.00, .10/.00,
.14/.01, .02/.00. Hand max z at 15000 .94/.84. No gain in high.
`reward_highreach` .055/step = exactly 5x the weight-20 run: the raw score
(~.00055 x weight) did not move in either run, the hands did not learn.
Likely cause: successland (upstream semantics, weight 4) pays 1 every step
after a jump and 6 every step with both feet down after it, independent of
the ball. In high regions it is .145/step (eereach .31, highreach100 .055,
stopball .008). It was 0 for model_12000 (contact sign bug); since the fix,
every run from 12000 (jumpfix, highreach 20, 100) ends with high-R jumped
.5-.75 and touch .02-.06 (12000: .24). K1's jump test (+.05 m root rise) is a
small hop; upstream needs root z > 1.0 m on G1 (~+.2 m).
