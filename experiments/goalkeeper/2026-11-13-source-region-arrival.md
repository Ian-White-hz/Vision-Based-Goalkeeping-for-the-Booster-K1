# Per-environment goals and verified region arrivals

Fixed Source reset to write each goal mocap pose at that environment's origin.
Previously goals retained the same default pose, overlapping while robots were
spread across the environment grid. Goal geometry remains non-colliding.

Regions now describe BALL CENTRE coordinates at the visible goal plane x=-0.8,
not an arbitrary endpoint behind the robot. Production sampler computes ballistic
velocity to an interior point with 4 cm margin. Reward target is independently
computed from that same parabola at x=0.1. Spawn x=3..5, y=-1..1, z=.2..1 m;
flight time .5.. .8 s. Ball perturbations remain off by default. Region guarantee
applies to undisturbed free flight, not after robot interception or opt-in noise.

K1 baseline regions (y,z metres):
0 [.1,.6], [.4,.75]; 1 [-.6,-.1], [.4,.75]
2 [.1,.6], [.75,1.05]; 3 [-.6,-.1], [.75,1.05]
4 [.1,.6], [.12,.3]; 5 [-.6,-.1], [.12,.3]
Train/eval/max bounds now identical; spatial curriculum expansion is capped.
This deliberately differs from upstream G1 metres. It does not certify K1 can
reach every shot, particularly because height at x=.1 exceeds goal-plane height.
Regions refer to centre, not requiring the entire sphere inside each rectangle.
Visual goal stays 3.6m by 1.8m; these are smaller training regions within it.

Validation: test_goalkeeper_source_region_arrival.py samples 100 shots per region
from production sampler and simulates all 600 in CPU MuJoCo with production ball
and terrain contact. Every interpolated goal-plane crossing is inside its assigned
rectangle, with no preceding terrain contact. Separately creates real six-env
mjlab scene, checks six distinct mocap origins equal env_origins, and region IDs
0 through 5. This is not a policy save-rate test or an exhaustive mathematical
proof for arbitrary numerical settings. Existing checkpoints load but were trained
on a different distribution; playback uses these new bounds.
