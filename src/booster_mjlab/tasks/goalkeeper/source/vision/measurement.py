"""Ball measurements in the OpenCV camera frame: geometric model and image localization.

Both paths return the same thing per env: position [N,3], per-axis std [N,3]
and a detected flag [N]. The geometric model generates what a detector plus
depth/size ranging would report, from the true ball position, without images.
`localize` is the real computation from a detection box and a depth image and
is shared by rendered evaluation and the robot.
"""

from __future__ import annotations

import math

import torch

from . import camera
from .config import MeasurementCfg

PAYLOAD = 7  # pos 3, std 3, detected 1


def pack(pos, std, detected):
    return torch.cat((pos, std, detected[:, None].float()), -1)


def unpack(payload):
    return payload[:, :3], payload[:, 3:6], payload[:, 6] > 0.5


def _small_rotation(n, std_rad, device):
    """Random rotation matrices from axis-angle with per-axis std."""
    w = torch.randn(n, 3, device=device) * std_rad
    angle = w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / angle
    K = torch.zeros(n, 3, 3, device=device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K = K - K.transpose(1, 2)
    s, c = angle.sin()[:, :, None], angle.cos()[:, :, None]
    return torch.eye(3, device=device) + s * K + (1 - c) * K @ K


def fuse_range(z_size, s_size, z_stereo, s_stereo, stereo_ok):
    """Inverse-variance fusion of size ranging and stereo depth."""
    w_size = 1.0 / s_size.square()
    w_st = torch.where(stereo_ok, 1.0 / s_stereo.square(), torch.zeros_like(s_stereo))
    z = (w_size * z_size + w_st * torch.nan_to_num(z_stereo)) / (w_size + w_st)
    return z, (w_size + w_st).rsqrt()


class GeometricMeasurement:
    """What the camera pipeline would output, from the true ball position."""

    def __init__(self, num_envs: int, device, cfg: MeasurementCfg | None = None):
        self.cfg = cfg or MeasurementCfg()
        self.n, self.device = num_envs, device
        self.sigma_scale = torch.ones(num_envs, device=device)
        self.depth_scale = torch.ones(num_envs, device=device)
        self.rot_err = torch.eye(3, device=device).repeat(num_envs, 1, 1)
        self.pos_err = torch.zeros(num_envs, 3, device=device)

    def reset(self, ids, randomize: bool = True):
        c, m = self.cfg, len(torch.arange(self.n, device=self.device)[ids])
        if not randomize:
            self.sigma_scale[ids], self.depth_scale[ids] = 1.0, 1.0
            self.rot_err[ids] = torch.eye(3, device=self.device)
            self.pos_err[ids] = 0.0
            return
        lo, hi = c.sigma_scale
        self.sigma_scale[ids] = torch.empty(m, device=self.device).uniform_(lo, hi)
        self.depth_scale[ids] = (
            1.0 + torch.randn(m, device=self.device) * c.depth_scale_std
        )
        self.rot_err[ids] = _small_rotation(
            m, math.radians(c.extrinsic_rot_std_deg), self.device
        )
        self.pos_err[ids] = torch.randn(m, 3, device=self.device) * c.extrinsic_pos_std

    def __call__(self, p_cam: torch.Tensor, occlusion: torch.Tensor | None = None):
        """p_cam: true ball centre in the nominal camera frame [N,3]."""
        c, dev = self.cfg, self.device
        # The real camera sits slightly off its nominal pose.
        p = (self.rot_err @ p_cam[:, :, None])[:, :, 0] + self.pos_err
        z = p[:, 2]
        u, v = camera.project(p)
        r = camera.FX * c.ball_radius / z.clamp_min(1e-3)
        inside = (
            (z > c.min_range)
            & (z < c.max_range)
            & (u > -0.5 * r)
            & (u < camera.WIDTH + 0.5 * r)
            & (v > -0.5 * r)
            & (v < camera.HEIGHT + 0.5 * r)
        )
        p_det = c.detect_max * torch.sigmoid(c.detect_slope * (r - c.detect_r0))
        if occlusion is not None:
            p_det = p_det * (1 - occlusion).clamp(0, 1)
            inside &= occlusion <= c.max_occlusion
        detected = inside & (torch.rand(self.n, device=dev) < p_det)

        k = self.sigma_scale
        s_c = k * (c.center_px[0] + c.center_px[1] * r)
        s_r = k * (c.radius_px[0] + c.radius_px[1] * r)
        u_m = u + torch.randn_like(u) * s_c
        v_m = v + torch.randn_like(v) * s_c
        r_m = (r + torch.randn_like(r) * s_r).clamp_min(0.5)
        z_size = camera.FX * c.ball_radius / r_m
        s_size = z_size * s_r / r_m
        s_st = k * z.square() * c.disparity_px / (camera.FX * c.baseline)
        stereo_ok = z < c.stereo_max_range
        z_st = (z + torch.randn_like(z) * s_st) * self.depth_scale
        z_m, s_z = fuse_range(z_size, s_size, z_st, s_st, stereo_ok)
        pos = camera.unproject(u_m, v_m, z_m)
        s_xy = s_c * z_m / camera.FX + s_z * ((u_m - camera.CX).abs() / camera.FX)
        std = torch.stack(
            (
                s_xy,
                s_c * z_m / camera.FY + s_z * ((v_m - camera.CY).abs() / camera.FY),
                s_z,
            ),
            -1,
        )

        # Rare false detections anywhere in view.
        bad = detected & (torch.rand(self.n, device=dev) < c.outlier_prob)
        if bad.any():
            m = int(bad.sum())
            zb = torch.empty(m, device=dev).uniform_(0.5, 6.0)
            ub = torch.rand(m, device=dev) * camera.WIDTH
            vb = torch.rand(m, device=dev) * camera.HEIGHT
            pos[bad] = camera.unproject(ub, vb, zb)
        return pos, std, detected


def _ray(u, v):
    return camera.unproject(u, v, torch.ones_like(u))


def _angle(a, b):
    cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
    return torch.arccos(cos.clamp(-1.0, 1.0))


def localize(
    box: torch.Tensor,
    detected: torch.Tensor,
    depth: torch.Tensor | None,
    cfg: MeasurementCfg | None = None,
):
    """3D ball centre from a box [N,4] (x0, y0, x1, y1 px) and planar depth [N,H,W].

    Wide-angle safe: the sphere subtends a circular cone of half-angle
    asin(R / d) whatever its direction, so the range comes from the angle
    between the rays through opposite box edges (the larger of the two
    directions; occlusion only shrinks a box). Stereo uses the median ray
    distance over a disk of 0.6 r, moved from the front surface to the centre
    (median surface distance ~ d - 0.9 R, from rendered frames).
    """
    c = cfg or MeasurementCfg()
    u = 0.5 * (box[:, 0] + box[:, 2])
    v = 0.5 * (box[:, 1] + box[:, 3])
    r = (0.5 * torch.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1])).clamp_min(
        0.5
    )
    half = 0.5 * torch.maximum(
        _angle(_ray(box[:, 0], v), _ray(box[:, 2], v)),
        _angle(_ray(u, box[:, 1]), _ray(u, box[:, 3])),
    ).clamp(1e-4, 1.5)
    d_size = c.ball_radius / torch.sin(half)
    s_c = c.center_px[0] + c.center_px[1] * r
    s_r = c.radius_px[0] + c.radius_px[1] * r
    s_size = d_size * s_r / r
    d_st = torch.full_like(u, float("nan"))
    if depth is not None:
        h, w = depth.shape[1:3]
        ys = torch.arange(h, device=depth.device, dtype=depth.dtype)[None, :, None]
        xs = torch.arange(w, device=depth.device, dtype=depth.dtype)[None, None, :]
        disk = (xs + 0.5 - u[:, None, None]).square() + (
            ys + 0.5 - v[:, None, None]
        ).square() <= (0.6 * r)[:, None, None].square()
        good = (
            disk
            & (depth > c.min_range)
            & (depth < c.stereo_max_range)
            & detected[:, None, None]
        )
        # Planar depth to distance along each pixel ray.
        ray_len = torch.sqrt(
            1
            + ((xs + 0.5 - camera.CX) / camera.FX).square()
            + ((ys + 0.5 - camera.CY) / camera.FY).square()
        )
        vals = torch.where(good, depth * ray_len, torch.nan).flatten(1)
        d_st = vals.nanmedian(-1).values + 0.9 * c.ball_radius
    s_st = d_st.square() * c.disparity_px / (camera.FX * c.baseline)
    stereo_ok = ~torch.isnan(d_st)
    d, s_d = fuse_range(
        d_size, s_size, d_st, torch.nan_to_num(s_st, nan=1.0), stereo_ok
    )
    ray = _ray(u, v)
    ray = ray / ray.norm(dim=-1, keepdim=True)
    pos = ray * d[:, None]
    z = pos[:, 2]
    std = torch.stack(
        (
            s_c * z / camera.FX + s_d * ray[:, 0].abs(),
            s_c * z / camera.FY + s_d * ray[:, 1].abs(),
            s_d * ray[:, 2],
        ),
        -1,
    )
    return pos, std, detected & (z > c.min_range) & (d < c.max_range)
