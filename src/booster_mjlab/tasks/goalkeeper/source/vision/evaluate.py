"""Save rate and ball-input quality of a checkpoint under one actor ball input.

Run the same checkpoint with --ball-input truth / synthetic / rendered to see
what the camera costs. For synthetic and rendered it also prints the raw frame
error per range bin (detection rate, bias, spread, reported std): run both and
compare the tables to calibrate vision/config.py.

    uv run --no-sync python -m booster_mjlab.tasks.goalkeeper.source.vision.evaluate \
        --checkpoint logs/.../model_12000.pt --ball-input synthetic --num-envs 600
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import mjlab  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from ..config import env_cfg, runner_cfg
from ..runner import SourceRunner
from ..settings import GOAL_HEIGHT, GOAL_WIDTH, GOAL_X
from ..state import state

REGION_NAMES = ("mid-L", "mid-R", "high-L", "high-R", "low-L", "low-R")
RANGE_BINS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--ball-input", default="synthetic", choices=("truth", "synthetic", "rendered")
    )
    ap.add_argument("--num-envs", type=int, default=600)
    ap.add_argument(
        "--yolo",
        default=None,
        help="YOLO weights for rendered mode (default: segmentation oracle)",
    )
    ap.add_argument("--no-vision-randomize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev, n = "cuda:0", args.num_envs

    env = ManagerBasedRlEnv(
        env_cfg(num_envs=n, play=True, randomize=False, ball_input=args.ball_input),
        device=dev,
    )
    w = RslRlVecEnvWrapper(env)
    runner = SourceRunner(w, asdict(runner_cfg()), device=dev)
    runner.load(args.checkpoint, load_cfg=True)
    policy = runner.alg.actor_critic
    s = state(env)
    bi = s.ball_input
    if bi is not None:
        # Sensor randomization is part of the camera model, keep it in evaluation.
        bi.cfg.randomize = not args.no_vision_randomize
        if args.yolo:
            from .detectors import YoloDetector

            bi.detector = YoloDetector(args.yolo)
    torch.manual_seed(args.seed)
    env.reset(seed=args.seed)
    obs = w.get_observations()
    orig = env.scene.env_origins
    region = s.end_regions.clone()
    shape = (n,)
    done = torch.zeros(shape, dtype=torch.bool, device=dev)
    goal, crossed, fell = (
        torch.zeros_like(done),
        torch.zeros_like(done),
        torch.zeros_like(done),
    )
    first_x = torch.full(shape, float("nan"), device=dev)
    est_err, est_x, frames = [], [], []
    with torch.no_grad():
        for _ in range(env.max_episode_length - 1):
            obs, _, d, _ = w.step(policy.act_inference(obs["actor"]))
            alive = ~done
            b = s.ball_states
            bx, by, bz = b[:, 0] - orig[:, 0], b[:, 1] - orig[:, 1], b[:, 2]
            new = alive & ~crossed & (bx < GOAL_X)
            goal |= new & (by.abs() < GOAL_WIDTH / 2) & (bz < GOAL_HEIGHT)
            crossed |= new
            fell |= alive & d.bool() & ~crossed
            if bi is not None:
                seen = bi.seen & alive & ~crossed
                first_x = torch.where(seen & first_x.isnan(), bx, first_x)
                est_err.append((bi.ball_local - bi.true_local).norm(dim=-1)[seen])
                est_x.append(bx[seen])
                if bi.last_capture is not None:
                    mask, pos, std, det, true = bi.last_capture
                    m = mask & alive & ~crossed & (true[:, 2] > 0)
                    frames.append(
                        torch.cat((true[m], pos[m], std[m], det[m, None].float()), -1)
                    )
            done |= d.bool()
    valid = crossed | ~fell
    kept = ~goal
    print(
        f"ball input {args.ball_input}: kept out {kept[valid].float().mean():.3f} "
        f"(ended before crossing {(~valid).float().mean():.3f})"
    )
    print(
        "  per region:",
        " ".join(
            f"{r} {kept[valid & (region == i)].float().mean():.2f}"
            for i, r in enumerate(REGION_NAMES)
        ),
    )
    if bi is None:
        return
    fx = first_x[~first_x.isnan()]
    print(
        f"  tracker valid for {(~first_x.isnan()).float().mean():.2f} of shots; first valid at ball x "
        f"median {fx.median():.2f} m, p10 {fx.quantile(0.1):.2f} m"
    )
    e, x = torch.cat(est_err), torch.cat(est_x)
    print("  actor ball error |est - true| (m) by ball x:")
    for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        m = (x >= lo) & (x < hi)
        if m.any():
            print(
                f"    x {lo:.1f}-{hi:.1f}: median {e[m].median():.3f} p90 {e[m].quantile(0.9):.3f} (n={int(m.sum())})"
            )
    f = torch.cat(frames)
    true, pos, std, det = f[:, :3], f[:, 3:6], f[:, 6:9], f[:, 9] > 0.5
    z = true[:, 2]
    print("  raw frames, camera frame (x right, y down, z forward), by true range z:")
    print(
        "    z bin    det   outlier  median err x/y/z      robust std x/y/z      reported std x/y/z"
    )
    for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        m = (z >= lo) & (z < hi)
        if not m.any():
            continue
        md = m & det
        if md.sum() < 5:
            print(f"    {lo:.1f}-{hi:.1f}  {det[m].float().mean():.2f}")
            continue
        err = pos[md] - true[md]
        # Robust statistics: false detections would dominate a plain std.
        med = err.median(0).values
        rstd = 1.4826 * (err - med).abs().median(0).values
        outlier = ((err - med).abs() > 5 * rstd + 0.2).any(-1)
        fmt = lambda t: "/".join(f"{v:.3f}" for v in t.tolist())  # noqa: E731
        print(
            f"    {lo:.1f}-{hi:.1f}  {det[m].float().mean():.2f}  {outlier.float().mean():.3f}    "
            f"{fmt(med):20s}  {fmt(rstd):20s}  {fmt(std[md].median(0).values)}"
        )
    env.close()


if __name__ == "__main__":
    main()
