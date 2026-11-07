"""Run script-style AMP tests with the same mjlab-first bootstrap as training."""

import contextlib
import io
import runpy
import traceback
from pathlib import Path

import mjlab  # noqa: F401 -- initialize task entry points before runner imports
import torch

torch.set_num_threads(1)
failures = []
scripts = sorted(Path(__file__).parent.glob("test_goalkeeper_amp_*.py"))
for script in scripts:
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            runpy.run_path(str(script), run_name="__main__")
    except Exception:
        failures.append(script.name)
        print(f"FAIL {script.name}\n{output.getvalue()}")
        traceback.print_exc()
    else:
        print(f"PASS {script.name}")
        if "SKIP:" in output.getvalue():
            print(output.getvalue())
print(f"AMP scripts: {len(scripts) - len(failures)}/{len(scripts)} passed")
if failures:
    raise SystemExit(1)
