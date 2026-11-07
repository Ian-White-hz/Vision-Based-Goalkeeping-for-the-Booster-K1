"""Validate goalkeeper AMP clips and render data/environment distribution plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from booster_mjlab.tasks.goalkeeper.amp_interfaces import (
    MOTION_NAMES,
    GoalkeeperExpertSampler,
)
from booster_mjlab.tasks.goalkeeper.scene.events import sample_six_region_shots
from booster_mjlab.tasks.goalkeeper.scene.settings import FullGoalSettings


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("artifacts/goalkeeper_k1/serial")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/goalkeeper_amp_diagnostics")
    )
    parser.add_argument("--num-shot-samples", type=int, default=10_000)
    parser.add_argument("--num-expert-samples", type=int, default=1_000)
    parser.add_argument("--frame-dt", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_shot_samples < 6 or args.num_expert_samples <= 0:
        parser.error("sample counts must be positive and include all six categories")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sampler = GoalkeeperExpertSampler(args.dataset_root, device=device)
    expert_ids = torch.arange(6, device=device).repeat_interleave(
        args.num_expert_samples
    )
    expert = sampler.sample(expert_ids, frame_dt=args.frame_dt)

    shot_ids = torch.arange(args.num_shot_samples, device=device) % 6
    shot_generator = torch.Generator(device=device).manual_seed(args.seed)
    settings = FullGoalSettings()
    shots = sample_six_region_shots(
        args.num_shot_samples,
        settings,
        device=device,
        generator=shot_generator,
        regions=shot_ids,
    )

    summary = {
        "dataset_root": str(args.dataset_root.resolve()),
        "joint_names": list(sampler.joint_names),
        "frame_dt": args.frame_dt,
        "categories": {},
    }
    expert_cpu = expert.cpu()
    shots_cpu = {name: value.cpu() for name, value in shots.items()}
    for motion_id, motion_name in enumerate(MOTION_NAMES):
        expert_mask = expert_ids.cpu() == motion_id
        shot_mask = shot_ids.cpu() == motion_id
        values = expert_cpu[expert_mask]
        summary["categories"][motion_name] = {
            "motion_id": motion_id,
            "expert_samples": int(expert_mask.sum()),
            "shot_samples": int(shot_mask.sum()),
            "rolling": bool(shots_cpu["is_rolling"][shot_mask].all()),
            "joint_min_rad": float(values.min()),
            "joint_max_rad": float(values.max()),
            "intercept_y_range_m": [
                float(shots_cpu["target_position_e"][shot_mask, 1].min()),
                float(shots_cpu["target_position_e"][shot_mask, 1].max()),
            ],
            "intercept_z_range_m": [
                float(shots_cpu["target_position_e"][shot_mask, 2].min()),
                float(shots_cpu["target_position_e"][shot_mask, 2].max()),
            ],
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    colors = plt.cm.tab10.colors
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for motion_id, motion_name in enumerate(MOTION_NAMES):
        mask = shot_ids.cpu() == motion_id
        axes[0].scatter(
            shots_cpu["target_position_e"][mask, 1],
            shots_cpu["target_position_e"][mask, 2],
            s=2,
            alpha=0.25,
            color=colors[motion_id],
            label=motion_name,
        )
        axes[1].scatter(
            shots_cpu["goal_target_position_e"][mask, 1],
            shots_cpu["goal_target_position_e"][mask, 2],
            s=2,
            alpha=0.25,
            color=colors[motion_id],
        )
    axes[0].set(title="Intercept plane", xlabel="y (m)", ylabel="z (m)")
    axes[1].set(title="Goal plane", xlabel="y (m)", ylabel="z (m)")
    axes[0].legend(markerscale=4, fontsize=8)
    figure.savefig(args.output_dir / "shot_intercepts.png", dpi=180)
    plt.close(figure)

    ranges = []
    for motion_id in range(6):
        values = expert_cpu[expert_ids.cpu() == motion_id]
        ranges.append((values.max(dim=0).values - values.min(dim=0).values).mean(dim=0))
    figure, axis = plt.subplots(figsize=(13, 4), constrained_layout=True)
    image = axis.imshow(torch.stack(ranges).numpy(), aspect="auto", cmap="viridis")
    axis.set_yticks(range(6), labels=MOTION_NAMES)
    axis.set_xticks(
        range(len(sampler.joint_names)), labels=sampler.joint_names, rotation=90
    )
    axis.set_title("Expert joint range by category (rad)")
    figure.colorbar(image, ax=axis)
    figure.savefig(args.output_dir / "expert_joint_ranges.png", dpi=180)
    plt.close(figure)

    print(f"Saved goalkeeper AMP diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
