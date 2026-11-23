"""Ball detectors: image batch -> box [N,4] (x0, y0, x1, y1 px), detected [N].

SegmentationDetector is the oracle from simulator labels (calibration and
upper-bound evaluation). YoloDetector wraps a trained Ultralytics model; the
package is imported lazily and is not a project dependency (AGPL-3.0).
"""

from __future__ import annotations

import torch

MIN_PIXELS = 4


def boxes_from_mask(mask: torch.Tensor, min_pixels: int = MIN_PIXELS):
    """Tight boxes of boolean masks [N,H,W]; detected if at least min_pixels."""
    n, h, w = mask.shape
    dev = mask.device
    rows, cols = mask.any(2), mask.any(1)
    big = 1 << 30
    ys = torch.arange(h, device=dev)
    xs = torch.arange(w, device=dev)
    y0 = torch.where(rows, ys, big).amin(1)
    y1 = torch.where(rows, ys, -1).amax(1) + 1
    x0 = torch.where(cols, xs, big).amin(1)
    x1 = torch.where(cols, xs, -1).amax(1) + 1
    detected = mask.sum((1, 2)) >= min_pixels
    box = torch.stack((x0, y0, x1, y1), -1).float()
    return torch.where(detected[:, None], box, 0.0), detected


class SegmentationDetector:
    def __init__(self, ball_geom_id: int, geom_type: int):
        self.ball_geom_id, self.geom_type = ball_geom_id, geom_type

    def __call__(self, rgb, segmentation):
        mask = (segmentation[..., 0] == self.ball_geom_id) & (
            segmentation[..., 1] == self.geom_type
        )
        return boxes_from_mask(mask)


class YoloDetector:
    def __init__(
        self, weights: str, conf: float = 0.25, imgsz: int = 544, device=None, chunk=64
    ):
        from ultralytics import YOLO  # optional, see module docstring

        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.chunk = chunk  # images per forward pass; bounds activation memory

    def __call__(self, rgb, segmentation=None):
        # Ultralytics treats numpy images as BGR.
        images = [im[..., ::-1].copy() for im in rgb.cpu().numpy()]
        results = []
        for k in range(0, len(images), self.chunk):
            results += self.model.predict(
                images[k : k + self.chunk],
                conf=self.conf,
                imgsz=self.imgsz,
                verbose=False,
                device=self.device,
            )
        box = torch.zeros(len(images), 4)
        detected = torch.zeros(len(images), dtype=torch.bool)
        for i, res in enumerate(results):
            if len(res.boxes):
                best = res.boxes.conf.argmax()
                box[i] = res.boxes.xyxy[best].cpu()
                detected[i] = True
        return box.to(rgb.device), detected.to(rgb.device)
