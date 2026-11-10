"""Run a short, seeded learning-rate comparison from one Actor/Critic checkpoint."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import random

import numpy as np

import mjlab  # noqa: F401 -- initialize task entry points
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from booster_mjlab.amp.runners.categorical_on_policy_runner import CategoricalAmpOnPolicyRunner
from booster_mjlab.tasks.goalkeeper.config.k1.env_cfgs import booster_k1_goalkeeper_amp_env_cfg
from booster_mjlab.tasks.goalkeeper.config.k1.rl_cfg import booster_k1_goalkeeper_amp_runner_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--schedule", choices=("adaptive", "fixed"), required=True)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.checkpoint.is_file() or args.iterations <= 0:
        parser.error("A real checkpoint and positive iteration count are required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    cfg = booster_k1_goalkeeper_amp_env_cfg(num_envs=32)
    cfg.seed = 42
    rl = booster_k1_goalkeeper_amp_runner_cfg()
    rl.warm_start = True
    rl.amp.reward_mode = "direct"
    rl.algorithm.learning_rate = 1e-4 if args.schedule == "fixed" else 1e-3
    rl.algorithm.schedule = args.schedule
    rl.max_iterations = args.iterations
    rl.upload_model = False
    rl.logger = "tensorboard"
    config = {"env": asdict(cfg), "agent": asdict(rl),
              "checkpoint": str(args.checkpoint.resolve()), "device": args.device,
              "torch": torch.__version__, "seed": 42}
    (args.output_dir / "config.json").write_text(json.dumps(config, default=str, indent=2))
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=args.device))
    try:
        runner = CategoricalAmpOnPolicyRunner(
            env, asdict(rl), log_dir=str(args.output_dir), device=args.device)
        runner.load(str(args.checkpoint))
        started = time.monotonic()
        runner.learn(args.iterations, init_at_random_ep_len=True)
        summary = {"iterations": runner.current_learning_iteration,
                   "elapsed_seconds": time.monotonic() - started,
                   "final_iteration": runner._last_iteration_summary,
                   "ppo_diagnostics": runner.alg.last_ppo_diagnostics}
        (args.output_dir / "result.json").write_text(json.dumps(summary, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
