"""Resolve packaged project files and the separately installed Booster SDK."""

import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path
import sys

BUNDLE_ROOT = Path(__file__).resolve().parent
KEEPER = BUNDLE_ROOT / 'runtime/experiments/2026-10-21_open_then_close_v12'


def bootstrap():
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MUJOCO_GL', 'glfw')
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(BUNDLE_ROOT), str(KEEPER)]


def booster_asset(relative):
    spec = find_spec('booster_mjlab')
    if not spec or not spec.origin:
        raise RuntimeError('Install the course Booster package as described in README.md.')
    path = Path(spec.origin).resolve().parent / relative
    if not path.is_file():
        raise FileNotFoundError(f'Booster asset missing: {path}')
    return path


def model_path(name):
    return BUNDLE_ROOT / 'models' / name / 'model.pt'


def bundle_integrity():
    files = sorted((BUNDLE_ROOT / 'models').rglob('*')) + sorted((BUNDLE_ROOT / 'runtime').rglob('*'))
    return {p.relative_to(BUNDLE_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file() and p.suffix != '.pyc'}


def verify_models():
    manifest = json.loads((BUNDLE_ROOT / 'models/checksums.json').read_text(encoding='utf-8'))
    for relative, expected in manifest.items():
        path = BUNDLE_ROOT / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Missing or modified model: {relative}')

