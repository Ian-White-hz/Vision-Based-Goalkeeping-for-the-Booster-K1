"""Domain randomization of rendered head-camera frames, using the segmentation.

The simulator draws a white ball over a blue checker floor; a detector trained
on that alone learns the floor. Per frame, on GPU:
- background (everything but robot and ball): kept, replaced by a random photo
  from --backgrounds, or by a procedural scene (gradient, blobs, rectangles);
- ball: random tint and optional panel pattern, keeping the rendered shading;
- robot: brightness;
- whole image: brightness, contrast, gamma, blur, sensor noise.
Boxes stay valid since shapes are untouched (blur adds under a pixel).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class Backgrounds:
    """Random crops of user photos (any jpg/png under a directory)."""

    def __init__(self, root: str | None, height: int, width: int, device, limit=2000):
        self.images = []
        if root:
            files = sorted(
                p
                for p in Path(root).rglob("*")
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            )
            for p in files[:limit]:
                im = Image.open(p).convert("RGB")
                scale = max(height / im.height, width / im.width) * 1.2
                im = im.resize((int(im.width * scale) + 1, int(im.height * scale) + 1))
                self.images.append(torch.from_numpy(np.asarray(im)).to(device))
        self.h, self.w, self.device = height, width, device

    def __len__(self):
        return len(self.images)

    def sample(self) -> torch.Tensor:
        im = self.images[int(torch.randint(len(self.images), (1,), device="cpu"))]
        y = int(torch.randint(im.shape[0] - self.h + 1, (1,), device="cpu"))
        x = int(torch.randint(im.shape[1] - self.w + 1, (1,), device="cpu"))
        crop = im[y : y + self.h, x : x + self.w].float() / 255
        return crop.flip(1) if torch.rand(1, device="cpu") < 0.5 else crop


def _rand(n, lo, hi, device):
    return lo + (hi - lo) * torch.rand(n, device=device)


def procedural_background(n, h, w, device):
    """Gradient + low-frequency colour field + rectangles, [n,h,w,3] in [0,1]."""
    top, bottom = (
        torch.rand(n, 1, 1, 3, device=device),
        torch.rand(n, 1, 1, 3, device=device),
    )
    t = torch.linspace(0, 1, h, device=device)[None, :, None, None]
    img = (top * (1 - t) + bottom * t).expand(n, h, w, 3).clone()
    field = torch.rand(n, 3, 6, 8, device=device)
    field = F.interpolate(
        field, size=(h, w), mode="bicubic", align_corners=False
    ).permute(0, 2, 3, 1)
    img = 0.6 * img + 0.4 * field.clamp(0, 1)
    for _ in range(8):
        x0 = torch.randint(w, (n,), device=device)
        y0 = torch.randint(h, (n,), device=device)
        bw = torch.randint(10, w // 2, (n,), device=device)
        bh = torch.randint(10, h // 2, (n,), device=device)
        col = torch.rand(n, 1, 1, 3, device=device)
        xs, ys = (
            torch.arange(w, device=device)[None, None, :],
            torch.arange(h, device=device)[None, :, None],
        )
        inside = (
            (xs >= x0[:, None, None])
            & (xs < (x0 + bw)[:, None, None])
            & (ys >= y0[:, None, None])
            & (ys < (y0 + bh)[:, None, None])
        )
        keep = torch.rand(n, 1, 1, device=device) < 0.5
        img = torch.where((inside & keep)[..., None], col, img)
    return img


def augment(
    rgb,
    ball_mask,
    robot_mask,
    backgrounds: Backgrounds | None = None,
    p_keep_background=0.2,
):
    """rgb uint8 [n,h,w,3] -> randomized uint8 frames."""
    n, h, w, _ = rgb.shape
    dev = rgb.device
    img = rgb.float() / 255
    bg_mask = ~(ball_mask | robot_mask)

    # Background.
    bg = procedural_background(n, h, w, dev)
    if backgrounds is not None and len(backgrounds):
        use_photo = torch.rand(n, device=dev) < 0.7
        for i in torch.nonzero(use_photo).flatten().tolist():
            bg[i] = backgrounds.sample()
    keep = torch.rand(n, device=dev) < p_keep_background
    img = torch.where((bg_mask & ~keep[:, None, None])[..., None], bg, img)

    # Ball: tint the rendered shading, sometimes with dark panels.
    shade = img.mean(-1, keepdim=True)
    tint = torch.rand(n, 1, 1, 3, device=dev)
    white = (torch.rand(n, device=dev) < 0.4)[:, None, None, None]
    tint = torch.where(white, torch.ones_like(tint), 0.3 + 0.7 * tint)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij"
    )
    freq = _rand(n, 0.15, 0.6, dev)[:, None, None]
    phase = _rand(n, 0, 6.28, dev)[:, None, None]
    panels = (torch.sin(freq * xs + phase) * torch.sin(freq * ys + 2 * phase)) > 0.55
    panels &= (torch.rand(n, device=dev) < 0.5)[:, None, None]
    ball = shade * tint * torch.where(panels, 0.15, 1.0)[..., None]
    img = torch.where(ball_mask[..., None], ball, img)

    # Robot brightness.
    img = torch.where(
        robot_mask[..., None], img * _rand(n, 0.6, 1.3, dev)[:, None, None, None], img
    )

    # Photometric.
    contrast = _rand(n, 0.7, 1.3, dev)[:, None, None, None]
    bright = _rand(n, -0.15, 0.15, dev)[:, None, None, None]
    gamma = _rand(n, 0.7, 1.4, dev)[:, None, None, None]
    img = ((img - 0.5) * contrast + 0.5 + bright).clamp(0, 1) ** gamma
    blur = torch.rand(n, device=dev) < 0.3
    if blur.any():
        chw = img[blur].permute(0, 3, 1, 2)
        img[blur] = F.avg_pool2d(chw, 3, 1, 1, count_include_pad=False).permute(
            0, 2, 3, 1
        )
    noise = _rand(n, 0.0, 0.025, dev)[:, None, None, None]
    img = img + torch.randn(img.shape, device=dev) * noise
    return (img.clamp(0, 1) * 255).round().to(torch.uint8)
