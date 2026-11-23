"""Parameters of the measurement model and the tracker.

Values are placeholders until calibrated: the stereo baseline, disparity noise,
frame rate and latency of the real camera are not known yet. `evaluate.py`
prints the rendered-pipeline error per distance bin next to this model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimingCfg:
    fps: float = 30.0
    latency: tuple[float, float] = (0.03, 0.09)  # per episode, s
    latency_jitter: float = 0.005  # per frame, s
    queue: int = 8  # pending frames per env


@dataclass
class MeasurementCfg:
    """Pinhole + stereo + size-ranging error model (OpenCV camera frame)."""

    ball_radius: float = 0.1
    min_range: float = 0.15
    max_range: float = 8.0
    # Detection probability: logistic in pixel radius.
    detect_r0: float = 2.0
    detect_slope: float = 3.0
    detect_max: float = 0.98
    # Ball counts as lost above this occluded fraction; below, p_det *= 1 - occ.
    max_occlusion: float = 0.5
    # Pixel noise of the detected centre and radius: a + b * r_px.
    center_px: tuple[float, float] = (0.4, 0.03)
    radius_px: tuple[float, float] = (0.3, 0.03)
    # Stereo depth: sigma_z = z^2 * disparity_px / (fx * baseline).
    baseline: float = 0.05
    disparity_px: float = 0.15
    stereo_max_range: float = 4.0
    outlier_prob: float = 0.01
    # Per-episode randomization.
    sigma_scale: tuple[float, float] = (0.7, 1.5)
    depth_scale_std: float = 0.02
    extrinsic_rot_std_deg: float = 1.0
    extrinsic_pos_std: float = 0.005


@dataclass
class TrackerCfg:
    accel_noise: float = 3.0  # m/s^2, white acceleration (drag, spin)
    init_vel_std: float = 8.0
    gate: float = 16.0  # Mahalanobis^2 on the 3D innovation
    reinit_after: int = 3  # consecutive rejected frames (deflection)
    confirm: int = 3  # accepted frames before the output is valid
    max_coast: float = 0.3  # s without an accepted frame before invalid
    max_sigma: float = 1.0  # m, position std above which the output is invalid
    ground_z: float | None = 0.0  # bounce plane in the tracking frame; None = off
    ball_radius: float = 0.1
    restitution: float = 0.6
    gravity: float = 9.81


@dataclass
class VisionCfg:
    timing: TimingCfg = field(default_factory=TimingCfg)
    measurement: MeasurementCfg = field(default_factory=MeasurementCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    # Random walk of the torso position estimate (leg odometry), m / sqrt(s).
    odom_drift: float = 0.02
    # Arm links checked for occlusion of the line of sight (sphere radius, m).
    occluder_radius: float = 0.045
    randomize: bool = True
