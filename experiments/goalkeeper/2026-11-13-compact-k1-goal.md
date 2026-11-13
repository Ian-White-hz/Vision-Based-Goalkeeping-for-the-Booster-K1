# Compact K1 training goal

Replaced 3.6 x 1.8 m visual goal with 1.5 x 1.1 m, centralised as GOAL_WIDTH,
GOAL_HEIGHT and GOAL_X in source/settings.py. This is task difficulty scaling,
not a uniform morphological G1/K1 scale or a dynamic reach guarantee.
Real model default Trunk height .5125 m, Head_2 origin .7604 m (not head top),
left hand origin (.08155,.20498,.33573) m. These default-pose landmarks do not
measure maximum reach; learned stepping/jumping must be evaluated separately.

Region centres y ranges [.05,.55] and [-.55,-.05]; z ranges [.3,.65] middle,
[.65,.9] upper, [.12,.3] lower. Spawn height lowered .2..1 to .2...8 m.
Goal x=-.8 and ballistic launch convention unchanged. Regions retain clearance
from posts and bar for radius .1 ball. Training deliberately does not sample
every corner of the aperture. At robot plane the parabola may be higher than
at goal plane; small-jump full-goal coverage is not certified by these tests.

Validation: production sampler, 100 shots/region, 600/600 physical MuJoCo
crossings within correct regions, no early ground contact, max error .019615 m.
Full ball clearance bounds asserted against new goal dimensions. Six actual
mjlab environments retain distinct correctly aligned goal mocap positions.
Ruff and git diff --check passed. No reward or learning algorithm changes.
Old checkpoints are loadable but play now uses smaller goal/shot distribution.
