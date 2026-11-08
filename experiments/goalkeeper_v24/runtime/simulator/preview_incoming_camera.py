"""Preview random incoming footballs with a head-mounted RGB camera.

The existing standing policy controls K1. Camera images are displayed, but are
not yet inputs to that policy. Segmentation is only a camera-debug label.
"""

import math
import sys
from pathlib import Path

import mujoco
import mjlab.tasks
from PIL import Image

from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab.viewer import ViewerConfig

from booster_mjlab.rl.runner import BoosterOnPolicyRunner
from booster_mjlab.tasks.velocity.config.k1.rl_cfg import (
    booster_k1_ppo_runner_cfg,
)
from play_random_ball import launch_ball
from train_stand import stand_env_cfg


TASK_NAME = "K1-Football-Incoming-Camera-Test"
CAMERA_NAME = "k1_head_camera"
CAMERA_DOWNWARD_DEG = 50.0
CHECKPOINT = Path(
    "logs/rsl_rl/k1_football_stand/"
    "2026-10-13_23-32-03_local_stand_2000/model_1999.pt"
)


def head_camera_quaternion():
    """Mount the optical axis below the head's forward axis (w, x, y, z)."""
    angle = math.radians(CAMERA_DOWNWARD_DEG) / 2.0
    c, s = math.cos(angle), math.sin(angle)
    return (0.5 * (c + s), 0.5 * (c - s),
            -0.5 * (c - s), -0.5 * (c + s))


def camera_env_cfg(play=False):
    cfg = stand_env_cfg(play=play)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 6.0
    cfg.events["launch_ball"] = EventTermCfg(func=launch_ball, mode="reset")

    # Keep rolling resistance small so slow incoming balls reach the robot.
    original_ball_spec = cfg.scene.entities["ball"].spec_fn

    def ball_spec():
        spec = original_ball_spec()
        spec.geom("football_geom").friction[2] = 0.0001
        return spec

    cfg.scene.entities["ball"].spec_fn = ball_spec

    # Tilt the camera itself down: the standing policy keeps the head close
    # to level. The forward offset reduces torso occlusion of the foot area.
    camera = CameraSensorCfg(
        name=CAMERA_NAME,
        parent_body="robot/Head_2",
        pos=(0.12, 0.0, 0.07),
        quat=head_camera_quaternion(),
        fovy=100.0,
        width=320,
        height=240,
        data_types=("rgb", "segmentation"),
        use_textures=True,
        use_shadows=False,
    )
    cfg.scene.sensors = (*cfg.scene.sensors, camera)
    cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    cfg.viewer.lookat = (0.7, 0.0, 0.5)
    cfg.viewer.distance = 3.2
    return cfg


def check_camera():
    env = ManagerBasedRlEnv(cfg=camera_env_cfg(play=True), device="cuda:0")
    try:
        observations, _ = env.reset(seed=7)
        data = env.scene[CAMERA_NAME].data
        rgb = data.rgb
        segmentation = data.segmentation
        assert rgb is not None and segmentation is not None
        ball_id = mujoco.mj_name2id(
            env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball/football_geom"
        )
        assert ball_id >= 0
        mask = (segmentation[0, :, :, 0] == ball_id) & (
            segmentation[0, :, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
        )
        image = rgb[0].cpu().numpy()
        output = Path("logs/camera_check.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(output)
        print("Camera RGB shape:", tuple(rgb.shape))
        print("RGB standard deviation:", float(rgb.float().std().item()))
        print("Ball pixels visible:", int(mask.sum().item()))
        print("Actor observation shape:", tuple(observations["actor"].shape))
        print("Camera image:", output.resolve())
        assert mask.any().item(), "Ball is outside the initial camera view"
        print("Camera check OK")
    finally:
        env.close()


def main():
    mode = sys.argv.pop(1) if len(sys.argv) > 1 else "play"
    if mode == "check":
        check_camera()
        return
    if mode != "play":
        raise SystemExit("Choose: check or play")

    agent = booster_k1_ppo_runner_cfg()
    agent.experiment_name = "k1_football_incoming_camera_test"
    agent.logger = "tensorboard"
    register_mjlab_task(
        task_id=TASK_NAME,
        env_cfg=camera_env_cfg(),
        play_env_cfg=camera_env_cfg(play=True),
        rl_cfg=agent,
        runner_cls=BoosterOnPolicyRunner,
    )
    if "--checkpoint-file" not in sys.argv:
        if not CHECKPOINT.is_file():
            raise SystemExit(f"Checkpoint missing: {CHECKPOINT.resolve()}")
        sys.argv.extend(["--checkpoint-file", str(CHECKPOINT)])
    sys.argv.insert(1, TASK_NAME)
    sys.argv.extend(["--agent", "trained", "--num-envs", "1",
                     "--device", "cuda:0", "--viewer", "viser"])
    from booster_mjlab.scripts.play import main as play_main
    play_main()


if __name__ == "__main__":
    main()
