"""Opt-in launcher for the reusable scene; the original run.py is unchanged."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("check", "list", "play", "train"):
        raise SystemExit("Usage: run_full_goal.py {check|list|play|train} [--camera] [task arguments]")
    command = sys.argv.pop(1)
    camera = "--camera" in sys.argv
    if camera:
        sys.argv.remove("--camera")
    import goalkeeper_sim.full_goal.task  # noqa: F401
    from goalkeeper_sim.full_goal import RGB_TASK_ID, TASK_ID
    task_id = RGB_TASK_ID if camera else TASK_ID
    if command == "list":
        print(TASK_ID, RGB_TASK_ID, sep="\n")
        return
    if command == "check":
        from mjlab.scene import Scene
        from mjlab.tasks.registry import load_env_cfg
        cfg = load_env_cfg(task_id)
        cfg.scene.num_envs = 1
        model = Scene(cfg.scene, device="cpu").compile()
        assert {"robot", "ball", "goal"} <= cfg.scene.entities.keys()
        assert model.nu == 22
        print("PASS: full-goal registry, K1/ball/goal scene, 22 actuators")
        return
    if len(sys.argv) < 2 or sys.argv[1] != task_id:
        sys.argv.insert(1, task_id)
    if command == "train":
        from booster_mjlab.scripts.train import main as entry
    else:
        from booster_mjlab.scripts.play import main as entry
    entry()


if __name__ == "__main__":
    main()
