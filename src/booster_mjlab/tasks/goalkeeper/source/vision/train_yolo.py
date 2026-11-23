"""Fine-tune a COCO-pretrained YOLO on the rendered ball dataset.

Needs `ultralytics` (AGPL-3.0, not a project dependency; check the licence for
your use). Real-robot images labelled in the same YOLO format can be mixed in
by listing several image folders in a data.yaml.

    uv run --no-sync python -m booster_mjlab.tasks.goalkeeper.source.vision.train_yolo \
        --data artifacts/goalkeeper_k1/yolo_ball/data.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="artifacts/goalkeeper_k1/yolo_weights/yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--fraction", type=float, default=1.0, help="share of train images")
    ap.add_argument("--project", default="logs/yolo_ball")
    ap.add_argument("--name", default="k1_head")
    args = ap.parse_args()
    from ultralytics import YOLO

    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        # 544 = 17 * 32: full resolution for distant balls (letterboxed square;
        # rect batches would disable shuffling).
        imgsz=camera.WIDTH,
        # Absolute: Ultralytics nests relative projects under runs/detect/.
        project=str(Path(args.project).resolve()),
        name=args.name,
        fraction=args.fraction,
        # Colour and geometry are already randomized in the renderer; keep
        # flips (scene symmetric) and light mosaic, no rotation (horizon prior).
        degrees=0.0,
        fliplr=0.5,
        mosaic=0.5,
        scale=0.3,
        hsv_h=0.02,
        hsv_s=0.5,
        hsv_v=0.4,
    )


if __name__ == "__main__":
    main()
