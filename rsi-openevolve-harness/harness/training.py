"""Run the frozen runner in a subprocess with a timeout.

The subprocess keeps LLM-written policy code away from the evaluator process: a crash,
an endless loop or a memory blow-up ends one candidate, not the whole run.
"""
import os
import subprocess
import sys
import time

from . import PROJECT_ROOT


def run_training(policy_path, parent_checkpoint, train_path, validation_path, output_dir, stage_id,
                 max_epochs=50, timeout_s=600, cpu_threads=1, cache_dir=None):
    """Return {"returncode", "stdout", "stderr", "timed_out", "seconds"}."""
    command = [sys.executable, "-m", "harness.runner", "--policy", str(policy_path),
               "--parent", str(parent_checkpoint), "--train", str(train_path),
               "--validation", str(validation_path), "--output", str(output_dir),
               "--stage-id", str(stage_id), "--max-epochs", str(max_epochs)]
    if cache_dir:
        command += ["--cache-dir", str(cache_dir)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    environment["BASELINE_CPU_THREADS"] = str(cpu_threads)
    started = time.monotonic()
    try:
        done = subprocess.run(command, cwd=str(PROJECT_ROOT), env=environment, capture_output=True,
                              text=True, timeout=timeout_s)
        return {"returncode": done.returncode, "stdout": done.stdout, "stderr": done.stderr,
                "timed_out": False, "seconds": time.monotonic() - started}
    except subprocess.TimeoutExpired as expired:
        stderr = expired.stderr.decode("utf-8", "replace") if isinstance(expired.stderr, bytes) else (expired.stderr or "")
        return {"returncode": None, "stdout": "", "stderr": stderr, "timed_out": True,
                "seconds": time.monotonic() - started}
