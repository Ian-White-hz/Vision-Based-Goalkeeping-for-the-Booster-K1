"""Vision package without the simulator: camera, clock, measurement model, tracker."""

import math

import torch

from booster_mjlab.tasks.goalkeeper.source.vision import camera
from booster_mjlab.tasks.goalkeeper.source.vision.config import (
    MeasurementCfg,
    TimingCfg,
    TrackerCfg,
    VisionCfg,
)
from booster_mjlab.tasks.goalkeeper.source.vision.frames import FrameClock
from booster_mjlab.tasks.goalkeeper.source.vision.measurement import (
    GeometricMeasurement,
    localize,
)
from booster_mjlab.tasks.goalkeeper.source.vision.occlusion import occluded_fraction
from booster_mjlab.tasks.goalkeeper.source.vision.pipeline import BallPerception
from booster_mjlab.tasks.goalkeeper.source.vision.tracker import BallTracker

torch.manual_seed(0)
G = torch.tensor([0.0, 0.0, -9.81])

# Camera: intrinsics and round trip; the mount looks along head +x.
assert abs(camera.FX - 208.7) < 0.1 and abs(camera.VFOV_DEG - 98.9) < 0.1
p = torch.tensor([[0.3, -0.2, 2.5]])
u, v = camera.project(p)
torch.testing.assert_close(camera.unproject(u, v, p[:, 2]), p)
pos, rot = camera.camera_pose(torch.zeros(1, 3), torch.eye(3)[None])
torch.testing.assert_close(rot[0, :, 2], torch.tensor([1.0, 0.0, 0.0]))  # forward
torch.testing.assert_close(rot[0, :, 1], torch.tensor([0.0, 0.0, -1.0]))  # down


def flight(t, p0=(5.0, 0.4, 0.6), v0=(-5.5, -0.4, 3.6)):
    p0, v0 = torch.tensor(p0), torch.tensor(v0)
    return p0 + v0 * t + 0.5 * G * t * t, v0 + G * t


# Tracker: exact lagged measurements converge to the true state.
tr = BallTracker(1, "cpu", TrackerCfg(ground_z=None))
dt, lag = 0.02, 0.06
R = (0.01**2 * torch.eye(3))[None]
for k in range(1, 40):
    tr.predict(dt)
    if k % 2 == 0:
        z, _ = flight(torch.tensor(k * dt - lag))
        tr.update(z[None], R, torch.tensor([lag]), torch.tensor([True]))
p_true, v_true = flight(torch.tensor(39 * dt))
assert (tr.position[0] - p_true).norm() < 0.02, tr.position
assert (tr.velocity[0] - v_true).norm() < 0.2, tr.velocity
assert tr.valid.all()
# Coasting past max_coast invalidates; a far outlier is rejected.
x_before = tr.x.clone()
tr.update(tr.position + 3.0, R, torch.tensor([0.0]), torch.tensor([True]))
torch.testing.assert_close(tr.x, x_before)
for _ in range(int(TrackerCfg().max_coast / dt) + 2):
    tr.predict(dt)
assert not tr.valid.any()

# Frame clock: ~30 Hz capture, each frame delivered once after its latency.
clock = FrameClock(2, 1, "cpu", TimingCfg(latency=(0.05, 0.05), latency_jitter=0.0))
now = torch.zeros(2)
clock.reset(torch.arange(2), now, randomize=False)
got, captured = [], 0
for k in range(150):
    now = torch.full((2,), k * 0.02)
    due = clock.due(now)
    captured += int(due[0])
    clock.push(due, now, now[:, None])
    payload, stamp, delivered = clock.pop(now)
    if delivered[0]:
        got.append((k * 0.02, float(stamp[0])))
assert abs(captured - 90) <= 2, captured
lags = torch.tensor([a - b for a, b in got])
assert (
    len(got) >= captured - 3
    and (lags >= 0.05 - 1e-6).all()
    and (lags < 0.05 + 0.02 + 1e-6).all()
)

# Measurement model: frustum, behind-camera, and error growth with range.
cfg = MeasurementCfg(outlier_prob=0.0)
gm = GeometricMeasurement(4000, "cpu", cfg)
gm.reset(torch.arange(4000), randomize=False)
for z_true, max_err in ((1.0, 0.03), (3.5, 0.25), (6.0, 0.8)):
    pts = torch.tensor([[0.2, 0.1, z_true]]).repeat(4000, 1)
    m, s, det = gm(pts)
    assert det.float().mean() > 0.9
    err = (m - pts)[det]
    assert err[:, 2].std() < max_err and err[:, 2].mean().abs() < 0.3 * max_err, (
        z_true,
        err.std(0),
    )
    # The reported std matches the actual error within 2x.
    ratio = err[:, 2].std() / s[det, 2].mean()
    assert 0.5 < ratio < 2.0, ratio
_, _, det = gm(torch.tensor([[0.0, 0.0, -2.0]]).repeat(4000, 1))
assert not det.any()
_, _, det = gm(torch.tensor([[5.0, 0.0, 1.0]]).repeat(4000, 1))  # 79 deg off axis
assert not det.any()

# Localize: exact box and planar depth of a sphere recover the centre.
H, W = camera.HEIGHT, camera.WIDTH
c = torch.tensor([[0.3, 0.1, 2.0]])
u, v = camera.project(c)
r = camera.FX * 0.1 / 2.0
box = torch.stack((u - r, v - r, u + r, v + r), -1)
depth = torch.full((1, H, W), 50.0)
ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
rr = ((xs - u) ** 2 + (ys - v) ** 2).sqrt() / r
inside = rr < 1
depth[0][inside] = 2.0 - 0.1 * torch.sqrt(1 - rr[inside] ** 2)
pos, std, ok = localize(box, torch.tensor([True]), depth)
assert ok.all() and (pos - c).norm() < 0.03, pos

# Occlusion: a hand on the line of sight hides the ball, one beside it does not.
cam, ball = torch.zeros(1, 3), torch.tensor([[3.0, 0.0, 0.0]])
on = occluded_fraction(cam, ball, torch.tensor([[[0.5, 0.0, 0.0]]]), 0.045, 0.1)
off = occluded_fraction(cam, ball, torch.tensor([[[0.5, 0.3, 0.0]]]), 0.045, 0.1)
assert on.item() > 0.9 and off.item() == 0.0

# Full pipeline on a ballistic shot: valid after a few frames, accurate near.
n = 256
cfg = VisionCfg()
cfg.measurement.outlier_prob = 0.0
bp = BallPerception(n, "cpu", cfg)
bp.reset(torch.arange(n), torch.zeros(n))
cam_pos, cam_rot = (
    torch.zeros(n, 3),
    camera.camera_pose(torch.zeros(n, 3), torch.eye(3).repeat(n, 1, 1))[1],
)
cam_pos[:, 2] = 0.9
first, errs = torch.full((n,), -1.0), []
for k in range(45):
    t = torch.tensor(k * 0.02)
    p_w, _ = flight(t)
    p_cam = (cam_rot.transpose(1, 2) @ (p_w - cam_pos)[:, :, None])[:, :, 0]

    def capture(mask):
        return gm_pipe(p_cam)

    if k == 0:
        gm_pipe = GeometricMeasurement(n, "cpu", cfg.measurement)
        gm_pipe.reset(torch.arange(n))
    bp.step(torch.full((n,), k * 0.02), 0.02, cam_pos, cam_rot, capture)
    first = torch.where((first < 0) & bp.valid, t, first)
    if p_w[0] < 1.5:
        errs.append((bp.position - p_w)[bp.valid].norm(dim=-1))
assert (first >= 0).float().mean() > 0.95 and first[first >= 0].median() < 0.2, (
    first.median()
)
errs = torch.cat(errs)
assert errs.median() < 0.1, errs.median()
print(
    f"vision units passed: first valid median {first.median():.2f}s, "
    f"near error median {errs.median():.3f} m p90 {errs.quantile(0.9):.3f} m"
)
assert math.isfinite(errs.mean().item())

# Robot-side input: frames with stamps and poses in, torso-frame ball out.
from booster_mjlab.tasks.goalkeeper.source.vision.realtime import RealtimeBallInput  # noqa: E402


class ProjectDetector:
    """Box of the true ball, set before each frame."""

    box = None

    def __call__(self, rgb, seg):
        return self.box[None], torch.tensor([True])


det = ProjectDetector()
rt = RealtimeBallInput(det)
cam_pos = torch.tensor([0.0, 0.0, 0.9])
cam_rot = camera.camera_pose(torch.zeros(1, 3), torch.eye(3)[None])[1][0]
torso_pos, torso_rot = torch.tensor([0.0, 0.0, 0.7]), torch.eye(3)
seen_at = None
for k in range(40):
    now = k * 0.02
    if k % 2 == 0 and k >= 2:
        stamp = now - 0.04
        p_w = flight(torch.tensor(stamp))[0]
        p_c = cam_rot.T @ (p_w - cam_pos)
        u, v = camera.project(p_c[None])
        r = camera.FX * 0.1 / p_c[2]
        det.box = torch.cat((u - r, v - r, u + r, v + r))
        rt.on_frame(
            torch.zeros(H, W, 3, dtype=torch.uint8), None, stamp, cam_pos, cam_rot
        )
    local, seen = rt.query(now, torso_pos, torso_rot)
    if seen and seen_at is None:
        seen_at = now
true_local = flight(torch.tensor(39 * 0.02))[0] - torso_pos
assert seen and seen_at < 0.2 and (local - true_local).norm() < 0.1, (local, true_local)
print(f"realtime input passed: error {(local - true_local).norm():.3f} m")
