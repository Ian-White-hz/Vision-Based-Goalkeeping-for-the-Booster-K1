"""Convert published Humanoid-Goalkeeper G1 clips into K1 motion assets."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np

from booster_mjlab.motion.goalkeeper_retarget import (
    CLIP_NAMES,
    RetargetOptions,
    retarget_clip,
)


def save_motion(path, motion):
    with path.open("wb") as file:
        pickle.dump(asdict(motion), file, protocol=pickle.HIGHEST_PROTOCOL)


def render_preview(path, motion, *, parallel):
    """Six evenly spaced FK snapshots; no torque tracking is implied."""
    import mujoco
    from PIL import Image, ImageDraw

    from booster_mjlab.motion.goalkeeper_retarget import _indices
    from booster_mjlab.robots.booster_k1.k1_constants import K1_JOINT_ORDER, K1_XML
    from booster_mjlab.robots.booster_k1.k1_parallel_constants import (
        K1_PARALLEL_JOINT_ORDER,
        K1_PARALLEL_XML,
    )

    spec = mujoco.MjSpec.from_file(str(K1_PARALLEL_XML if parallel else K1_XML))
    spec.worldbody.add_geom(
        name="preview_floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5, 5, 0.1],
        rgba=[0.2, 0.24, 0.28, 1],
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    qadr, _, _ = _indices(
        model, K1_PARALLEL_JOINT_ORDER if parallel else K1_JOINT_ORDER
    )
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -15, 2.5
    camera.lookat[:] = [*np.median(motion.root_pos[:, :2], axis=0), 0.65]
    canvas = Image.new("RGB", (3 * 480, 2 * 390), "white")
    draw = ImageDraw.Draw(canvas)
    with mujoco.Renderer(model, height=360, width=480) as renderer:
        for i, frame in enumerate(np.linspace(0, motion.num_frames - 1, 6, dtype=int)):
            data.qpos[:3] = motion.root_pos[frame]
            data.qpos[3:7] = np.asarray(motion.root_rot[frame])[[3, 0, 1, 2]]
            data.qpos[qadr] = motion.dof_pos[frame]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            x, y = (i % 3) * 480, (i // 3) * 390
            canvas.paste(Image.fromarray(renderer.render()), (x, y))
            draw.text(
                (x + 10, y + 365),
                f"frame {frame} / {frame / motion.fps:.2f} s",
                fill="black",
            )
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-repo", type=Path, default=Path("../Humanoid-Goalkeeper")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/goalkeeper_k1")
    )
    parser.add_argument(
        "--clips", nargs="+", choices=CLIP_NAMES, default=list(CLIP_NAMES)
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Source rate is supplied by upstream config, not embedded in .pt",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="<1 slows playback; changes fps without dropping frames",
    )
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--render-preview",
        action="store_true",
        help="Save a six-frame PNG for each variant; requires an offscreen GL backend",
    )
    args = parser.parse_args()
    repo = args.source_repo.resolve()
    dataset = repo / "legged_gym/resources/datasets/goalkeeper"
    mapping = dataset / "joint_id.txt"
    urdf = repo / "legged_gym/resources/robots/g1/urdf/g1_29.urdf"
    required = [mapping, urdf] + [dataset / f"{name}.pt" for name in args.clips]
    for path in required:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    output = args.output_dir.resolve()
    planned = [
        output / variant / f"{name}.pkl"
        for name in args.clips
        for variant in ("serial", "parallel")
    ]
    planned += [
        output / "reports" / f"{name}{suffix}"
        for name in args.clips
        for suffix in (".json", ".npz")
    ]
    planned += [output / "manifest.json"]
    if args.render_preview:
        planned += [
            output / "previews" / f"{name}_{variant}.png"
            for name in args.clips
            for variant in ("serial", "parallel")
        ]
    if not args.overwrite and any(path.exists() for path in planned):
        parser.error(
            "Output exists; choose another --output-dir or explicitly use --overwrite"
        )
    options = RetargetOptions(
        fps=args.fps, max_nfev=args.max_nfev, playback_speed=args.playback_speed
    )
    options.validate()
    for variant in ("serial", "parallel", "reports"):
        (output / variant).mkdir(parents=True, exist_ok=True)
    if args.render_preview:
        (output / "previews").mkdir(exist_ok=True)
    reports = []
    for name in args.clips:
        print(f"Retargeting {name}...", flush=True)
        serial, parallel, report, diagnostics = retarget_clip(
            dataset / f"{name}.pt", mapping, urdf, options
        )
        save_motion(output / "serial" / f"{name}.pkl", serial)
        save_motion(output / "parallel" / f"{name}.pkl", parallel)
        (output / "reports" / f"{name}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        np.savez_compressed(output / "reports" / f"{name}.npz", **diagnostics)
        if args.render_preview:
            render_preview(
                output / "previews" / f"{name}_serial.png", serial, parallel=False
            )
            render_preview(
                output / "previews" / f"{name}_parallel.png", parallel, parallel=True
            )
        reports.append(report)
        print(
            f"  {report['frames']} frames; mean point error {report['initial_mean_target_error_m']:.3f} -> {report['optimized_mean_target_error_m']:.3f} m; {report['acceptance']['status']}",
            flush=True,
        )
    manifest = {
        "source_repo": str(repo),
        "source_license": "Upstream README: CC BY-NC-SA 4.0. Preserve attribution and check authorization before redistribution/commercial use.",
        "source_project": "https://github.com/InternRobotics/Humanoid-Goalkeeper",
        "options": asdict(options),
        "root_rotation_order": "xyzw (MotionFile); wxyz only when writing MuJoCo qpos",
        "joint_layout": {"serial": 22, "parallel": 34},
        "reports": reports,
        "validation_scope": "Kinematic reference assets, not dynamically validated policies",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved assets and reports in {output}")


if __name__ == "__main__":
    main()
