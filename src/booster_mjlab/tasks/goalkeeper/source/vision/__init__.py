"""Head-camera ball perception for the source goalkeeper (opt-in, off by default).

The only coupling to the task is the actor's ball input. `SourceSettings.ball_input`
selects it; everything else (rewards, critic, estimator labels, terminations,
curriculum) keeps using simulator truth, and the actor stays 75 x 10 = 750 dims.

    truth      sim position + uniform noise, upstream visibility rules (default)
    synthetic  geometric camera model -> frame clock -> tracker   (training)
    rendered   head RGB-D render -> detector -> localize -> clock -> tracker (evaluation)

Data flow (one policy step, batched over envs)::

    camera.py        intrinsics, Head_2 mount, OpenCV frame (x right, y down, z forward)
    measurement.py   GeometricMeasurement: FOV, range, detection prob., occlusion,
                     pixel/stereo/size noise, per-episode extrinsic and depth bias
                     localize(): box + depth image -> 3D centre and std (rendered, robot)
    occlusion.py     arm sphere chains vs. the line of sight
    frames.py        FrameClock: 30 Hz capture, random phase, latency queue
    tracker.py       BallTracker: ballistic Kalman filter with lagged updates, gating,
                     re-init after deflection, bounce, coasting limit
    pipeline.py      BallPerception = clock + front end + tracker (simulator-free)
    ball_input.py    VisionBallInput: the only module reading the env
    realtime.py      RealtimeBallInput: the same chain on the robot (no ROS)
    detectors.py     segmentation oracle, YOLO wrapper (lazy ultralytics import)
    augment.py       background / ball / photometric randomization of renders
    make_yolo_dataset.py, train_yolo.py, evaluate.py   tools

Task `Mjlab-Goalkeeper-Source-Vision-Flat-Booster-K1` is the source task with
ball_input="synthetic"; truth checkpoints load and fine-tune unchanged.
"""
