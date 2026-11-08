"""Re-run v24's high-level PPO fine-tuning from the packaged parent weights."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import random

from bundle_support import BUNDLE_ROOT, bootstrap, model_path, verify_models

bootstrap()

import controller  # Registers the handoff modules before importing bridge.
import bridge
from fluid_control import FluidCaptureEnv
from mjlab.rl import MjlabOnPolicyRunner
import numpy as np
from numerical_audit import audited_outcome
from posture_acceptance import RULE_ID, make_cfg
import random_capture_task
import torch
from train_bridge import config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--num-envs', type=int, default=16)
    parser.add_argument('--seed', type=int, default=2026102624)
    args = parser.parse_args()
    if args.iterations < 1 or args.num_envs < 4:
        parser.error('Use at least one iteration and four environments')

    verify_models()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a CUDA-capable GPU.')

    bridge.GAIT_MODEL = model_path('receive_gait')
    random_capture_task.STAGE = 'goal_width_fast_edge_mix_v2'
    cfg = make_cfg(args.num_envs)
    cfg.terminations['pose_success'].func = audited_outcome(
        cfg.terminations['pose_success'].func, random_capture_task.measurements)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    run = BUNDLE_ROOT / 'runs' / f'train-v24-{stamp}'
    run.mkdir(parents=True, exist_ok=False)
    raw = random_capture_task.raw_class(cfg=cfg, device='cuda:0')
    env = FluidCaptureEnv(raw)
    try:
        settings = asdict(config())
        settings['algorithm']['learning_rate'] = 2e-5
        runner = MjlabOnPolicyRunner(env, settings, str(run), 'cuda:0')
        runner.load(str(model_path('goalkeeper_v20')), load_cfg={'actor': True, 'critic': True},
                    map_location='cuda:0')
        raw.seed(args.seed)
        env.reset()
        setup = dict(iterations=args.iterations, num_envs=args.num_envs, seed=args.seed,
                     acceptance_rule=RULE_ID, curriculum=random_capture_task.STAGE,
                     parent='models/goalkeeper_v20/model.pt', gait='models/receive_gait/model.pt',
                     learning_rate=settings['algorithm']['learning_rate'])
        (run / 'setup.json').write_text(json.dumps(setup, indent=2), encoding='utf-8')
        runner.learn(args.iterations, init_at_random_ep_len=False)
        checkpoint = run / 'model_final.pt'
        runner.save(str(checkpoint))
        verify_models()
        print(f'Trained checkpoint: {checkpoint}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
