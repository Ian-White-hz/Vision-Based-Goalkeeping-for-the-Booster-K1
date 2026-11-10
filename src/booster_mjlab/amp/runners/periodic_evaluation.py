"""Isolated evaluation and atomic, outcome-selected checkpoints at update boundaries."""

from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
import time

import numpy as np
import torch
from rsl_rl.utils import resolve_class

from booster_mjlab.amp.utils.interface_loading import resolve_amp_callable


@contextmanager
def preserve_rng():
    """mjlab seed_rng also changes PYTHONHASHSEED and all Torch generators."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    hash_seed = os.environ.get("PYTHONHASHSEED")
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if hash_seed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = hash_seed


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def frozen_policy(runner):
    # Construct from configuration: deepcopy() of the trained actor fails on
    # cached, non-leaf Gaussian distribution tensors after a PPO backward pass.
    actor_class, actor_cfg = resolve_class(runner.cfg["actor"])
    actor = actor_class(runner.env.get_observations().to(runner.device),
                        runner.cfg["obs_groups"], "actor", runner.env.num_actions,
                        **actor_cfg).to(runner.device)
    actor.load_state_dict(runner.alg.get_policy().state_dict())
    return actor.eval()


class PeriodicEvaluation:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.evaluator = None
        self.state: dict = {}
        interval = cfg.get("interval", 0)
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
            raise ValueError("evaluation.interval must be a nonnegative integer")

    def enabled(self, runner) -> bool:
        return bool(self.cfg.get("interval", 0) and runner.logger.log_dir is not None
                    and not runner.logger.disable_logs)

    def restore(self, state, checkpoint: str, runner) -> None:
        # New output directories establish a new baseline. A score without its
        # corresponding best artifact must never suppress saving the local best.
        if (isinstance(state, dict) and runner.logger.log_dir is not None
                and Path(checkpoint).resolve().parent == Path(runner.logger.log_dir).resolve()
                and (Path(runner.logger.log_dir) / "model_best.pt").is_file()):
            self.state = deepcopy(state)
            # A later evaluation may have improved best after this numbered
            # checkpoint was written. The actual best artifact is authoritative.
            artifact = torch.load(Path(runner.logger.log_dir) / "model_best.pt",
                                  map_location="cpu", weights_only=True)
            saved = (artifact.get("infos") or {}).get("periodic_evaluation", {})
            if saved.get("signature") == self.state.get("signature") and saved.get("best"):
                self.state["best"] = deepcopy(saved["best"])
        else:
            self.state = {}

    def run(self, runner, *, force=False) -> None:
        iteration = runner.current_learning_iteration
        if not self.enabled(runner):
            return
        if not force and iteration % self.cfg["interval"]:
            return
        if runner.alg.storage.step or runner.alg._amp_transition is not None:
            raise RuntimeError("Evaluate only at an empty rollout boundary")
        started = time.perf_counter()
        # Clone before eval(): neither normalizers nor module modes on the
        # training actor/critic/AMP are ever changed.
        with preserve_rng(), torch.inference_mode():
            if self.evaluator is None:
                factory = resolve_amp_callable(self.cfg.get("evaluator_factory"),
                                               field_name="evaluation.evaluator_factory")
                self.evaluator = factory(training_env=runner.env.unwrapped,
                                         config=self.cfg, device=runner.device)
            signature = self.evaluator.signature
            if self.state.get("signature") != signature:
                self.state = {}
            if self.state.get("last_iteration") == iteration:
                return
            actor = frozen_policy(runner)
            report = self.evaluator.evaluate(actor)
        score = report["selection_score"]
        if not isinstance(score, list) or not score or not all(
            isinstance(x, (int, float)) and math.isfinite(x) for x in score
        ):
            raise ValueError("Evaluation selection_score must be a finite numeric list")
        best = self.state.get("best")
        improved = best is None or tuple(score) > tuple(best["score"])
        report.update(iteration=iteration, signature=signature,
                      elapsed_seconds=time.perf_counter() - started,
                      selected_best=improved)
        directory = Path(runner.logger.log_dir)
        report_path = directory / "evaluation" / f"iteration_{iteration:08d}.json"
        atomic_json(report_path, report)
        old_state = deepcopy(self.state)
        self.state.update(signature=signature, last_iteration=iteration)
        if improved:
            self.state["best"] = {"iteration": iteration, "score": score,
                                  "report": str(report_path.relative_to(directory))}
            try:
                runner.save(str(directory / "model_best.pt"))
            except BaseException:
                self.state = old_state
                raise
        writer = runner.logger.writer
        if writer is not None:
            for suite in ("validation", "holdout"):
                for group, metrics in report[suite]["metrics"].items():
                    for name, value in metrics.items():
                        writer.add_scalar(f"Evaluation/{suite}/{group}/{name}", value, iteration)
            writer.add_scalar("Evaluation/elapsed_seconds", report["elapsed_seconds"], iteration)
            writer.add_scalar("Evaluation/best_iteration", self.state["best"]["iteration"], iteration)
        print(f"Evaluation at {iteration}: validation safe-save="
              f"{report['validation']['metrics']['macro']['safe_save_rate']:.3f}, "
              f"best={self.state['best']['iteration']}")
