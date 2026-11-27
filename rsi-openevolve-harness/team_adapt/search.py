"""Running one stage's search through the documented entry point, with stagnation monitoring.

EvolveSearch starts `scripts/run_evolve.py` as a subprocess and polls the stage's
records/iterations.jsonl. When the search stagnates, exceeds a budget or keeps failing,
it sends SIGTERM: OpenEvolve's own handler stops the loop, saves the best program and
run_evolve writes its summary, exactly as on a normal finish. Nothing inside the search
or the evaluator is changed.

Budgets are per stage and add up across interrupted attempts (records are append-only),
so resuming a stage never resets what it already spent.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from harness import PROJECT_ROOT
from harness.dataio import read_json
from harness.evaluator import code_digest

from . import stagnation


def read_records(work_dir):
    path = Path(work_dir) / "records" / "iterations.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def initial_score(work_dir):
    """Selection score of the stage's initial program, from the evaluator's report cache."""
    policy = Path(work_dir) / "initial_policy.py"
    if not policy.is_file():
        return None
    report = Path(work_dir) / "reports" / f"{code_digest(policy.read_text(encoding='utf-8'))}.json"
    if not report.is_file():
        return None
    data = read_json(report)
    return (data.get("result_metrics") or {}).get("combined_score") if data.get("status") == "ok" else None


def latest_openevolve_checkpoint(work_dir):
    root = Path(work_dir) / "openevolve_output" / "checkpoints"
    found = [p for p in root.glob("checkpoint_*") if p.is_dir() and p.name.split("_")[-1].isdigit()]
    return max(found, key=lambda p: int(p.name.split("_")[-1])) if found else None


class ContinuationSearch:
    """No model calls: the stage only trains the official continuation; the gate decides."""
    mode = "continuation"

    def run(self, stage, settings, previous=None, log=print):
        return {"mode": self.mode, "end_reason": "no_search", "attempts": [], "iterations": 0, "seconds": 0.0,
                "tokens": 0, "analysis": None}


class EvolveSearch:
    mode = "evolve"

    def __init__(self, evolve_script=None, python=None, analyzer=None):
        """analyzer(work_dir, records) -> state, optional; default: stagnation.analyze on combined_score.

        A custom analyzer returns the same keys as stagnation.analyze and may add "repaired": True
        (stop early because the stage's goal is met) and "description".
        """
        self.script = Path(evolve_script or PROJECT_ROOT / "scripts" / "run_evolve.py")
        self.python = python or sys.executable
        self.analyzer = analyzer

    def run(self, stage, settings, previous=None, log=print):
        """stage: {"run_config", "work_dir", "stage_id", "initial_policy", "log_dir"}; settings: merged config."""
        work = Path(stage["work_dir"])
        stag, budget, search = settings["stagnation"], settings["budget"], settings["search"]
        attempts = list((previous or {}).get("attempts") or [])
        spent_seconds = sum(a.get("seconds", 0.0) for a in attempts)

        def analysis():
            if self.analyzer is not None:
                return self.analyzer(work, read_records(work))
            return stagnation.analyze(read_records(work), stag["patience"], stag["min_delta"], initial_score(work),
                                      stag.get("max_consecutive_failures"))

        def stop_reason(state, elapsed):
            if state.get("repaired"):
                return "repaired"
            if state["stagnated"]:
                return "stagnated"
            if state["unhealthy"]:
                return "unhealthy"
            if state["records"] >= budget["stage_max_iterations"]:
                return "budget_iterations"
            if budget.get("stage_max_seconds") and elapsed >= budget["stage_max_seconds"]:
                return "budget_seconds"
            if budget.get("stage_max_tokens") and state["tokens"] >= budget["stage_max_tokens"]:
                return "budget_tokens"
            return None

        state = analysis()
        reason = stop_reason(state, spent_seconds)
        if reason:  # already finished in an earlier attempt
            return self._summary(reason, attempts, state)

        remaining = budget["stage_max_iterations"] - state["records"]
        command = [self.python, "-B", str(self.script), "--run-config", str(stage["run_config"]),
                   "--iterations", str(remaining), "--initial-policy", str(stage["initial_policy"])]
        for flag, key in (("--oe-config", "oe_config"), ("--api-base", "api_base"), ("--model", "model"),
                          ("--workers", "workers")):
            if search.get(key) is not None:
                command += [flag, str(search[key])]
        checkpoint = latest_openevolve_checkpoint(work)
        if checkpoint is not None:
            command += ["--resume", str(checkpoint)]
        env = dict(os.environ, HARNESS_RUN_ID=f"{stage['stage_id']}-a{len(attempts) + 1}")
        log_path = Path(stage["log_dir"]) / "evolve.log"
        started, sent, sent_at = time.time(), None, None
        with open(log_path, "a", encoding="utf-8") as stream:
            stream.write(f"\n$ {' '.join(command)}\n")
            stream.flush()
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, env=env)
            try:
                while True:
                    finished = process.poll() is not None
                    state = analysis()
                    if finished:
                        break
                    if sent is None:
                        sent = stop_reason(state, spent_seconds + time.time() - started)
                        if sent:
                            log(f"[team_adapt] 停止搜索：{sent}；{state.get('description') or stagnation.describe(state)}")
                            process.send_signal(signal.SIGTERM)
                            sent_at = time.time()
                    elif time.time() - sent_at > float(search.get("stop_grace_seconds") or 120):
                        process.kill()
                    time.sleep(float(search.get("poll_seconds") or 2.0))
            except BaseException:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    try:
                        process.wait(timeout=float(search.get("stop_grace_seconds") or 120))
                    except subprocess.TimeoutExpired:
                        process.kill()
                raise
        attempts.append({"run_id": env["HARNESS_RUN_ID"], "returncode": process.returncode,
                         "seconds": round(time.time() - started, 2), "stop_signal_reason": sent,
                         "resumed_from": str(checkpoint) if checkpoint else None})
        state = analysis()
        if sent:
            reason = sent
        elif state.get("repaired"):
            reason = "repaired"
        elif state["stagnated"]:
            reason = "stagnated"
        elif process.returncode == 0:
            reason = "budget_iterations"
        else:
            reason = "unhealthy" if state["unhealthy"] else "process_error"
        return self._summary(reason, attempts, state)

    def _summary(self, reason, attempts, state):
        return {"mode": self.mode, "end_reason": reason, "attempts": attempts, "iterations": state["records"],
                "seconds": round(sum(a.get("seconds", 0.0) for a in attempts), 2), "tokens": state["tokens"],
                "analysis": state, "description": state.get("description") or stagnation.describe(state)}
