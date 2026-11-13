# Split-coverage model_4000: guidance is active, lateral translation is small

Read-only diagnostic. No training or production reward changes.

Run: `2026-11-12_16-36-55_amp_test_side_movement_v2`, `model_4000.pt`.
Saved configs confirm split foot coverage, movement_weight=2, goal=-2, fall=-5,
seed123, resume=False, 4000 iterations. This is not the previous velocity-tracking
formula. Last100 iterations: body_intercept log=-0.000285068, equivalent to about
-0.00137 mixed reward per episode (log * 8s * .6), mean episode249.53 steps,
mean mixed return21.55. Signed averages can hide positive and negative terms.

## Independent saved evaluation

At iteration4000 validation/holdout each have120 cases:

| metric | validation | holdout |
| --- | ---: | ---: |
| safe_save | 36.67% | 36.67% |
| goal | 30.83% | 25.83% |
| fall | .83% | .83% |
| contact | 78.33% | 80.83% |

Ground classes4/5 safe_save: validation10%/15%, holdout20%/10%.
Holdout total safe_save at500/1000/2000/3000/4000 iterations:
23.33/27.5/32.5/40.83/36.67%. Overall task learning exists but is not monotonic,
and these metrics do not establish lateral walking. Do not attribute improvement
over older500-step runs solely to this reward change: budgets differ.

## Four targeted frozen-policy trials

Seeds5 and9, rolling classes4/5, strictly mirrored shots at |target_y|~.61m.
Three-second maximum, deterministic actor, no optimizer/expert updates, no
auto-reset; actor and AMP state unchanged assertions passed.

Numbers below cover only the active interception interval (~1.36–1.38s).
Reward sums include term weight, dt and task mixing. Positive y means left.

| seed/side | root net dy cm | foot-gap gate full-time fraction | gate ever off | final predicted surface gap cm | body reward | goal/contact |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| 5 left | +3.31 | 100% | no | 11.58 | +.3945 | yes/no |
| 5 right | -1.57 | 60.29% | no | 9.75 | +.1663 | yes/no |
| 9 left | -1.94 | 100% | no | 13.98 | -.2218 | yes/no |
| 9 right | -3.07 | 100% | no | 11.40 | +.3561 | yes/no |

The geometry is an approximate oriented mesh bound, not a contact measurement.
However, in these misses it does NOT erroneously report covered or suppress the
guidance. Adjusting the coverage-off threshold again does not address these cases.

Foot displacement shows mostly widening stance: seed5 left shot ends the active
phase with left foot y+18.56cm and right foot y-8.68cm. For the mirrored right
shot, left foot y+17.45cm and right foot y-20.26cm. Seed9 similarly moves left
foot +17.3–17.8cm and right foot -18.4–19.4cm regardless of side. Contact gaps and
small foot lifts exist; it is not literally zero leg motion, but not coordinated
translation toward the ball. Active yaw stays below11deg in these trials, so
large turning is not the main behavior here.

## Why the mean reward is near zero

Example seed5 right: positive body contributions+1.1720, negative-1.0057,
net+.1663. Both directional signs are used; the adapter is not disconnected.
With near-full gate and upright posture, the current formula integrates roughly
to `movement_weight * term_weight * task_factor * directed_displacement`, i.e.
`12 * displacement_in_meters`. Tiny net progress therefore earns tiny net reward;
no remaining-distance cost accumulates just because the foot is still far away.

In these four active intervals, AMP mixed sums are6.95–8.46, foot orientation plus
slippage3.48–4.08 (seed5 right3.29), and eereach.55–.61, despite no ball contact.
Those are benefits available to these trajectories, not proof that actual lateral
walking would score worse. Absolute reward sums do not establish causal inhibition
by AMP or stability rewards. AMP's joint-angle input alone also cannot certify
root displacement; a high score is not evidence of a side-step being achieved.

## Decision

Confirmed: new formula is loaded; foot guidance remains active on far misses;
directional feedback changes sign correctly; the policy mostly spreads the feet
with very little root translation. Not confirmed: AMP is the sole cause, a larger
coefficient solves the issue, or the policy lacks all latent sideways capability.

Do not keep shrinking coverage thresholds or simply extend training expecting
this gate fix to teach a new locomotion pattern. Preserve this stable model.
The next controlled experiment should distinguish locomotion acquisition from
task reward shaping (e.g. a bounded lateral command task using the same controller,
or a controlled achievable translation comparison scored by the frozen AMP).
If changing rewards, explicitly measure remaining interception error and sustained
root/foot progress rather than increasing rewards for leg spreading. Do not add
another large failure cost; no production changes were made during this audit.

Raw CSVs, mirror plots and checkpoint SHA256 are in local
`artifacts/goalkeeper_split_v2_audit/`; saved training summary is in that directory.
Four deterministic far trials are mechanism probes, not population success rates.
