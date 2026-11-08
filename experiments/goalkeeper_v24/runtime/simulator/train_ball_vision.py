"""Collect simulated RGB labels and train a small football segmentation model.

MuJoCo segmentation is used for supervision and evaluation only. The neural
network receives RGB pixels, never object IDs or the football's simulator state.
"""

import argparse
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

import mujoco
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import CameraSensorCfg
from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from booster_mjlab.tasks.velocity.config.k1.rl_cfg import booster_k1_ppo_runner_cfg
from preview_incoming_camera import CAMERA_NAME, CHECKPOINT, camera_env_cfg


WIDTH, HEIGHT = 128, 96


def block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1), nn.ReLU(),
        nn.Conv2d(outputs, outputs, 3, padding=1), nn.ReLU(),
    )


class BallVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = block(3, 16)
        self.enc2 = block(16, 32)
        self.enc3 = block(32, 64)
        self.dec2 = block(96, 32)
        self.dec1 = block(48, 16)
        self.output = nn.Conv2d(16, 1, 1)

    def forward(self, rgb):
        e1 = self.enc1(rgb)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat((
            F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False),
            e2,
        ), dim=1))
        d1 = self.dec1(torch.cat((
            F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False),
            e1,
        ), dim=1))
        return self.output(d1)


def collect_split(env, wrapped, policy, samples, stride, seed, label):
    env.seed(seed)
    observations, _ = wrapped.reset()
    ball = env.scene["ball"]
    ball_id = mujoco.mj_name2id(
        env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball/football_geom"
    )
    assert ball_id >= 0
    images, masks = [], []
    count, batches = 0, 0
    started = time.monotonic()

    def capture():
        nonlocal count
        data = env.scene[CAMERA_NAME].data
        rgb, seg = data.rgb, data.segmentation
        assert rgb is not None and seg is not None
        mask = (seg[..., 0] == ball_id) & (
            seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        )
        take = min(samples - count, env.num_envs)
        images.append(rgb[:take].permute(0, 3, 1, 2).contiguous().cpu())
        masks.append(mask[:take, None].contiguous().cpu())
        count += take

    # The simulator mutates persistent delay buffers on resets; no_grad keeps
    # those tensors writable when switching to the held-out trajectories.
    with torch.no_grad():
        while count < samples:
            for _ in range(stride):
                observations, _, _, _ = wrapped.step(policy(observations))
            capture()
            batches += 1
            # Add real empty-camera examples: put the ball behind the robot,
            # render a labeled frame, then restore the rolling trajectory.
            if batches % 6 == 0 and count < samples:
                original_pose = ball.data.root_link_pose_w.clone()
                hidden_pose = original_pose.clone()
                robot = env.scene["robot"].data
                heading = robot.heading_w
                hidden_pose[:, :2] = robot.root_link_pos_w[:, :2] - torch.stack(
                    (torch.cos(heading), torch.sin(heading)), dim=1
                )
                hidden_pose[:, 2] = env.scene.env_origins[:, 2] + 0.112
                ball.write_root_link_pose_to_sim(hidden_pose)
                env.sim.forward()
                env.sim.sense()
                capture()
                ball.write_root_link_pose_to_sim(original_pose)
                env.sim.forward()
                env.sim.sense()
                observations = wrapped.get_observations()
            if batches % 16 == 0 or count >= samples:
                print(f"{label}: {count}/{samples} images, "
                      f"{time.monotonic() - started:.1f}s", flush=True)
    return torch.cat(images), torch.cat(masks)


def collect(args):
    checkpoint = Path(args.standing_checkpoint)
    if not checkpoint.is_file():
        raise SystemExit(f"Standing checkpoint missing: {checkpoint}")
    cfg = camera_env_cfg(play=True)
    cfg.scene.num_envs = args.num_envs
    cfg.scene.sensors = tuple(
        replace(s, width=WIDTH, height=HEIGHT) if isinstance(s, CameraSensorCfg) else s
        for s in cfg.scene.sensors
    )
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    try:
        agent = booster_k1_ppo_runner_cfg()
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = BoosterOnPolicyRunner(wrapped, asdict(agent), device="cuda:0")
        runner.load(str(checkpoint), load_cfg={"actor": True}, strict=True,
                    map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        train_rgb, train_mask = collect_split(
            env, wrapped, policy, args.samples, args.stride, args.seed, "Training data"
        )
        # A new seed and reset make the validation trajectories independent.
        val_rgb, val_mask = collect_split(
            env, wrapped, policy, args.val_samples, args.stride,
            args.seed + 10000, "Validation data"
        )
        output = Path(args.dataset)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "train_rgb": train_rgb, "train_mask": train_mask,
            "val_rgb": val_rgb, "val_mask": val_mask,
            "meta": {"width": WIDTH, "height": HEIGHT, "seed": args.seed,
                     "source": "head-camera RGB; simulator masks are labels only"},
        }, output)
        for name, mask in (("Training", train_mask), ("Validation", val_mask)):
            visible = mask.flatten(1).sum(1) >= 4
            print(f"{name}: visible {int(visible.sum())}/{len(visible)}", flush=True)
        print("Dataset saved:", output.resolve(), flush=True)
    finally:
        env.close()


def segmentation_loss(logits, targets):
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=logits.new_tensor(25.0)
    )
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).flatten(1).sum(1)
    denominator = probabilities.flatten(1).sum(1) + targets.flatten(1).sum(1)
    dice_loss = 1.0 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return bce + dice_loss


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    ious = []
    visible_total = visible_found = empty_total = empty_false = 0
    for rgb, target in loader:
        rgb = rgb.to(device).float() / 255.0
        target = target.to(device).bool()
        predicted = model(rgb).sigmoid() >= 0.5
        actual_visible = target.flatten(1).sum(1) >= 4
        found = predicted.flatten(1).sum(1) >= 4
        intersection = (predicted & target).flatten(1).sum(1)
        union = (predicted | target).flatten(1).sum(1)
        ious.extend((intersection[actual_visible] /
                     union[actual_visible].clamp_min(1)).cpu().tolist())
        visible_total += int(actual_visible.sum().item())
        # Count a detection only when it overlaps the actual football.
        # Unrelated pixels elsewhere in the image do not count as recall.
        matched = intersection / union.clamp_min(1) >= 0.25
        visible_found += int((actual_visible & found & matched).sum().item())
        empty_total += int((~actual_visible).sum().item())
        empty_false += int((~actual_visible & found).sum().item())
    return {
        "mean_iou": sum(ious) / max(1, len(ious)),
        "visible_recall": visible_found / max(1, visible_total),
        "empty_false_positive_rate": empty_false / max(1, empty_total),
        "visible_images": visible_total, "empty_images": empty_total,
    }


@torch.inference_mode()
def save_examples(model, rgb, masks, path, device):
    visible = masks.flatten(1).sum(1) >= 4
    indices = (visible.nonzero().flatten()[:6].tolist()
               + (~visible).nonzero().flatten()[:2].tolist())
    if not indices:
        return
    chosen = rgb[indices]
    predictions = model(chosen.to(device).float() / 255.0).sigmoid().cpu() >= 0.5
    tile_width, tile_height = WIDTH * 2, HEIGHT * 2 + 28
    canvas = Image.new("RGB", (tile_width * 4, tile_height * math.ceil(len(indices)/4)),
                       "white")
    draw = ImageDraw.Draw(canvas)
    for position, index in enumerate(indices):
        pixels = chosen[position].permute(1, 2, 0).numpy().copy()
        mask = predictions[position, 0].numpy()
        # Green marks the network prediction, not the simulator label.
        pixels[mask] = (pixels[mask].astype("float32") * 0.45
                        + [0, 140, 0]).clip(0, 255).astype("uint8")
        x, y = (position % 4) * tile_width, (position // 4) * tile_height
        canvas.paste(Image.fromarray(pixels).resize((WIDTH * 2, HEIGHT * 2)), (x, y+28))
        draw.text((x+4, y+5), f"RGB #{index}, predicted pixels={int(mask.sum())}",
                  fill="black")
    canvas.save(path)


def train(args):
    torch.manual_seed(args.seed)
    device = "cuda:0"
    data = torch.load(args.dataset, map_location="cpu", weights_only=True)
    train_data = TensorDataset(data["train_rgb"], data["train_mask"])
    val_data = TensorDataset(data["val_rgb"], data["val_mask"])
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            num_workers=0, pin_memory=True)
    model = BallVision().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for rgb, target in train_loader:
            rgb = rgb.to(device, non_blocking=True).float() / 255.0
            target = target.to(device, non_blocking=True)
            gain = torch.empty((rgb.shape[0], 3, 1, 1), device=device).uniform_(0.9, 1.1)
            rgb = (rgb * gain).clamp(0, 1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = segmentation_loss(model(rgb), target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item()) * rgb.shape[0]
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs}: loss={total_loss/len(train_data):.4f}, "
              f"val IoU={metrics['mean_iou']:.3f}, "
              f"recall={metrics['visible_recall']:.3f}, "
              f"empty false positives={metrics['empty_false_positive_rate']:.3f}, "
              f"elapsed={time.monotonic()-started:.1f}s", flush=True)
        if metrics["mean_iou"] > best_iou:
            best_iou = metrics["mean_iou"]
            torch.save({"model_state": model.state_dict(), "width": WIDTH,
                        "height": HEIGHT, "epoch": epoch+1, "metrics": metrics,
                        "input": "RGB only", "camera_downward_deg": 50.0},
                       output / "ball_vision.pt")
    best = torch.load(output / "ball_vision.pt", map_location=device, weights_only=True)
    model.load_state_dict(best["model_state"])
    model.eval()
    save_examples(model, data["val_rgb"], data["val_mask"],
                  output / "vision_validation.png", device)
    print("Best validation metrics:", best["metrics"], flush=True)
    print("Vision model:", (output / "ball_vision.pt").resolve(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("collect", "train"))
    parser.add_argument("--standing-checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--dataset", default="logs/vision/ball_dataset.pt")
    parser.add_argument("--output", default="logs/vision")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.samples, args.val_samples, args.num_envs, args.stride,
           args.epochs, args.batch_size) < 1:
        parser.error("Counts must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")
    if args.mode == "collect":
        collect(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
