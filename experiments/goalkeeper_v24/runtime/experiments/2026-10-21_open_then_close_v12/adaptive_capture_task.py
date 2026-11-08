"""Selected receive-timing controller with the original v7 actor dimensions.

After a measured stop, use RGB TTC to choose a smooth 1.0--2.5 s handoff.
Below 2.8 s TTC use the short handoff; above 3.5 s retain the old slow handoff;
interpolate smoothly in between. Latch once per stop to avoid timing jitter.
Keep the tested 3 s descent, real post-pose dwell, and strict physical hold.
This module changes control timing, not trained weights or shot distribution.
"""
from timing_task import TimedCaptureEnv
from random_capture_task import make_cfg,raw_class,measurements,STAGE

class AdaptiveCaptureEnv(TimedCaptureEnv):
    impact_enabled=True
    deadline_enabled=False
    nominal_descent_duration=3.0
    handoff_duration=1.0

env_class=AdaptiveCaptureEnv
