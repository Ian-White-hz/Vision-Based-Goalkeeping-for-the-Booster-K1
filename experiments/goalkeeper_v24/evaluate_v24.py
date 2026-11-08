"""Evaluate the frozen v24 policy on the original rolling-shot suites."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from bundle_support import BUNDLE_ROOT, bootstrap, model_path, verify_models

bootstrap()

import controller  # Registers the handoff modules before importing bridge.
import bridge
import evaluate as evaluation
from fluid_control import FluidCaptureEnv
from posture_acceptance import make_cfg
import random_capture_task
from review_posture_gate import review
import torch


SUITES = {'matched': 2026102723, 'fresh': 2026102824}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, default='fresh')
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--checkpoint', type=Path, default=model_path('goalkeeper_v24'))
    args = parser.parse_args()
    if args.episodes not in range(2, 33, 2):
        parser.error('--episodes must be an even number between 2 and 32')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')

    verify_models()
    torch.set_num_threads(4)
    bridge.GAIT_MODEL = model_path('receive_gait')
    random_capture_task.STAGE = 'goal_width_fast_edge_scoring_v2'
    random_capture_task.make_cfg = make_cfg
    random_capture_task.env_class = FluidCaptureEnv

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    run = BUNDLE_ROOT / 'runs' / f'eval-{args.suite}-{stamp}'
    run.mkdir(parents=True, exist_ok=False)

    def status(**fields):
        (run / 'status.json').write_text(json.dumps(fields, ensure_ascii=False, indent=2), encoding='utf-8')

    evaluation.status = status
    path = run / 'episodes.json'
    report = evaluation.evaluate(args.checkpoint.resolve(), args.episodes, path, SUITES[args.suite],
                                 task='random_anticipatory', audit=True, comparison=args.episodes < 32)
    gate = review(report, path)
    (run / 'posture_gate.json').write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Strict saves: {gate['qualified_successes']}/{report['episodes']}. Results: {run}")


if __name__ == '__main__':
    main()

