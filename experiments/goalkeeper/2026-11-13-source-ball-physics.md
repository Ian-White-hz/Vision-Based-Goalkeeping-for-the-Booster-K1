# Source ball physics baseline

Restored the existing collision-free visual goal (3.6 m wide, 1.8 m high).
Preserved original six regions, spawn ranges, mass 0.1 kg and radius 0.1 m.
Replaced nonphysical source inertia 0.01 with thin-shell inertia 0.000667 kg m².
Ball contact now uses priority 1, solref (.04,.5), condim 6 and friction
(.5,.005,.0001). Priority makes ball contact parameters effective against terrain
and robot; consequently robot-ball contacts also change. This is a deliberate
MuJoCo adaptation, not exact upstream physics or real-ball calibration.

SourceSettings now independently exposes ball_drag, ball_force_noise and
ball_velocity_noise, all False by default. Robot randomization remains enabled.
CLI example: --env.source-settings.ball-drag True. Enable each independently
before combined trials. Original forces/noise can produce substantial trajectory
error because initial velocity assumes gravity alone. No reward changes.

Validation: tests/test_goalkeeper_source_ball_physics.py uses CPU MuJoCo and
production ball geometry. A 1 m centre-height drop rebounded to 0.1592 m; this
is deliberately modest bounce, not a measured football coefficient. Initial
horizontal speed 1 m/s travelled 0.6156 m in 1 s. Six target centres times three
launch heights (18 deterministic shots) all crossed x=.1. Maximum transverse
crossing error was .0409 m (discrete sampling/integration included).
Same shots with seeded ablations: drag 1.0179 m, force noise .4951 m,
velocity noise .1886 m, all 1.2504 m maximum error. These are no-robot tests,
not policy success rates; none missed the plane within 2 s.

Existing checkpoints remain loadable, but dynamics changed: compare playback
before deciding whether to resume training. Original high shot ranges were not
scaled down to K1. The goal is a visual reference, not a new scoring boundary.

Regression: test_goalkeeper_source_randomization.py passed, including a real
CPU HIMPPO/AMP update, randomized robot startup/reset, curriculum and PD delay.
