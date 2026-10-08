"""Project launcher using the isolated vendored Booster environment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

TASK_ID = "Goalkeeper-K1-Sim"


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"train", "play", "list", "check"}:
        raise SystemExit("Usage: run.py {train|play|list|check} [task arguments]")

    command = sys.argv.pop(1)
    import goalkeeper_sim.task  # noqa: F401

    if command == "list":
        from mjlab.tasks.registry import list_tasks

        assert TASK_ID in list_tasks()
        print(TASK_ID)
        return

    if command == "check":
        from goalkeeper_sim.assets import get_ball_cfg
        from mjlab.entity import Entity
        from mjlab.tasks.registry import load_env_cfg

        cfg = load_env_cfg(TASK_ID)
        assert "robot" in cfg.scene.entities and "ball" in cfg.scene.entities
        assert "ball_state" in cfg.observations["actor"].terms
        ball_model = Entity(get_ball_cfg()).spec.compile()
        assert ball_model.nq == 7 and ball_model.nv == 6
        print("PASS: task registry, K1 scene, ball observation, free-body ball")
        return

    if len(sys.argv) < 2 or sys.argv[1] != TASK_ID:
        sys.argv.insert(1, TASK_ID)

    if command == "train":
        from booster_mjlab.scripts.train import main as entry
    else:
        from booster_mjlab.scripts.play import main as entry
    entry()


if __name__ == "__main__":
    main()
