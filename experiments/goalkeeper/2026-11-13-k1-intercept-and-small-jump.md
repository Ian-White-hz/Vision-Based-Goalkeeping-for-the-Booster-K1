# K1 interception feasibility and small-jump feedback

## Changes
Production shots now jointly constrain goal-plane region, intercept-plane height,
and reaction time. Candidate origins x=1.5..2.5, y=-1..1, z=.15...8 m;
arrival at initial robot plane x=0 takes .7...9 s. Goal reference line moved
from -.8 to -.1 m. This geometric adaptation is necessary: original rear goal
plus long flight time raises the low shot above the permitted intercept band.
The initial proposal x=.1 intercept/goal=-.1 also failed the low-region constraint;
final intercept plane x=0 resolves it. Visual goal remains noncolliding, 1.5x1.1m.
Scene's otherwise unused guard-region bound is .05 to satisfy goal validation;
this does not constrain Source robot motion or change its termination/reward.

Intercept ball-centre heights: middle .4...75, high .8..1.05, low .15...35 m.
A margin absorbs .005s integration error. Goal bounds unchanged. Rejection sampler
has 512 attempts and explicit failure; no silent relaxation. Optional ball
perturbations and robot contact invalidate free-flight arrival guarantees.
The sampler derives ballistic intercept reward targets; upstream near-ball target
refresh and other reach formulas are retained. Candidates are conditioned by
acceptance, so final heights/times are not uniform across the rectangular ranges.

Added JumpState: prior bilateral support, >=2 consecutive control steps with both
feet off contact, >=.05m root rise relative to reset height. Detects small jumps
rather than G1 root z>1m. Partial reset clears history. Original repeated landing
reward semantics retained (not a one-shot landing bonus). Source vendor untouched.
Added missing postlinvel=1.0. Other task weights and AMP mixing unchanged.

Per-region TensorBoard step diagnostics: actual weighted task components,
weighted AMP, action-release time, first-visible time, first-near-phase time,
near-phase fraction, cumulative guide seconds. Unseen first-event times use
negative sentinel (-.02s); group means include sentinel and reset samples and
are diagnostic summaries, not completed-episode timing statistics.

## Validation
600 production shots (100/class), CPU MuJoCo with actual ball contacts:
all goal crossings in assigned region; all robot-plane crossings within specified
height and .7...9s timing; no ground contact before goal; max goal error .02328m.
Real six-env scene verifies distinct goal positions and region assignments.
Small-jump unit checks: unsupported initial air, debounce, insufficient rise,
one-foot support exclusion, valid takeoff, landing reward, partial reset.
Explicit exhausted sampler failure verified. Existing randomized environment
and real HIMPPO/AMP regression passed. Source-flow test passes original reward
parity except intentionally adapted successland, gradient updates, logging,
checkpoint save/load and continued learning. postlinvel participates in parity.

6000-shot statistics seed42: launch speed min/median/max 3.866/4.950/6.045 m/s;
intercept speed 3.265/4.628/5.679; time .700/.774/.900s.
These tests do not demonstrate a learned interception policy or maximum K1 reach.
Old checkpoints remain loadable; scene distribution and reward changed.
