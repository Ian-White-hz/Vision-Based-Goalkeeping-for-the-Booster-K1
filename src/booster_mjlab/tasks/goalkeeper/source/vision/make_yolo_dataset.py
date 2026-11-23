"""Render head-camera frames of real saves and write a YOLO ball dataset.

A trained policy plays the shots so arm occlusion, head motion and dives look
like deployment. Boxes come from the segmentation of the ball geom (visible
pixels). Frames are domain-randomized (augment.py) unless --no-augment; put
photos of the real environment under --backgrounds for best transfer. Each
split also gets meta_<split>.npz with the true ball centre in the OpenCV camera
frame, for calibrating the measurement model.

    uv run --no-sync python -m booster_mjlab.tasks.goalkeeper.source.vision.make_yolo_dataset \
        --checkpoint logs/rsl_rl/.../model_12000.pt --out artifacts/goalkeeper_k1/yolo_ball
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import mjlab  # noqa: F401
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from PIL import Image

from ..config import env_cfg, runner_cfg
from ..runner import SourceRunner
from . import camera
from .augment import Backgrounds, augment
from .detectors import boxes_from_mask

NEGATIVE_FRACTION = 0.1  # approximate share of ball-free frames kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="artifacts/goalkeeper_k1/yolo_ball")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=6, help="per env")
    ap.add_argument("--stride", type=int, default=3, help="policy steps per frame")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument(
        "--backgrounds", default=None, help="directory of background photos"
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev, n = "cuda:0", args.num_envs
    # The camera sensor is added by the rendered ball input; the actor still
    # sees the truth ball here (the policy was trained on it).
    cfg = env_cfg(num_envs=n, play=True, randomize=True)
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (camera.head_camera_cfg(),)
    env = ManagerBasedRlEnv(cfg, device=dev)
    wrapped = RslRlVecEnvWrapper(env)
    runner = SourceRunner(wrapped, asdict(runner_cfg()), device=dev)
    runner.load(args.checkpoint, load_cfg=True)
    policy = runner.alg.actor_critic
    sensor = env.scene.sensors[camera.CAMERA_NAME]
    geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    ball_geom = env.sim.mj_model.geom("ball/football_geom").id
    robot_geoms = env.scene["robot"].indexing.geom_ids.to(dev)
    ball = env.scene["ball"]
    backgrounds = Backgrounds(args.backgrounds, camera.HEIGHT, camera.WIDTH, dev)

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed)
    val_env = (torch.rand(n, generator=gen) < args.val_fraction).tolist()
    meta = {"train": [], "val": []}
    count = {"train": 0, "val": 0}
    pool = ThreadPoolExecutor(8)
    rng = np.random.default_rng(args.seed)
    steps = args.episodes * env.max_episode_length

    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    obs = wrapped.get_observations()
    with torch.no_grad():
        for t in range(steps):
            act = (
                policy.act(obs["actor"])[0]
                if args.stochastic
                else policy.act_inference(obs["actor"])
            )
            obs, _, _, _ = wrapped.step(act)
            if t % args.stride:
                continue
            data = sensor.data
            seg = data.segmentation
            is_geom = seg[..., 1] == geom_type
            ball_mask = is_geom & (seg[..., 0] == ball_geom)
            robot_mask = is_geom & torch.isin(seg[..., 0], robot_geoms)
            box, labelled = boxes_from_mask(ball_mask)
            pixels = ball_mask.sum((1, 2))
            rgb = (
                data.rgb
                if args.no_augment
                else augment(data.rgb, ball_mask, robot_mask, backgrounds)
            )
            # True ball centre in the OpenCV camera frame.
            cid = sensor.camera_idx
            rot = camera.mujoco_cam_to_cv(
                env.sim.data.cam_xmat[:, cid].reshape(n, 3, 3)
            )
            rel = ball.data.root_link_pos_w - env.sim.data.cam_xpos[:, cid]
            ball_cam = (rot.transpose(1, 2) @ rel[:, :, None])[:, :, 0]

            rgb, box, labelled = (
                rgb.cpu().numpy(),
                box.cpu().numpy(),
                labelled.cpu().numpy(),
            )
            pixels, ball_cam = pixels.cpu().numpy(), ball_cam.cpu().numpy()
            for i in range(n):
                # Partly visible balls under MIN_PIXELS are neither labels nor negatives.
                if not labelled[i] and (
                    pixels[i] > 0 or rng.random() > NEGATIVE_FRACTION / 3
                ):
                    continue
                split = "val" if val_env[i] else "train"
                name = f"{args.seed:02d}_{t:05d}_{i:04d}"
                label = ""
                if labelled[i]:
                    x0, y0, x1, y1 = box[i]
                    w, h = camera.WIDTH, camera.HEIGHT
                    label = f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}\n"
                (out / "labels" / split / f"{name}.txt").write_text(label)
                pool.submit(
                    Image.fromarray(rgb[i]).save,
                    out / "images" / split / f"{name}.jpg",
                    quality=int(rng.integers(70, 96)),
                )
                meta[split].append((t, i, pixels[i], *box[i], *ball_cam[i]))
                count[split] += 1
            if t % (args.stride * 50) == 0:
                print(f"step {t}/{steps} frames {count}", flush=True)
    pool.shutdown()
    cols = np.array("step env pixels x0 y0 x1 y1 ball_x ball_y ball_z".split())
    for split, rows in meta.items():
        np.savez(
            out / f"meta_{split}.npz",
            columns=cols,
            rows=np.array(rows, dtype=np.float64),
        )
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n"
    )
    print(f"done: {count}, written to {out}")
    env.close()


if __name__ == "__main__":
    main()
