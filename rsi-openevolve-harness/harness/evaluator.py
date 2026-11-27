"""Evaluator: the file OpenEvolve is pointed at (`evaluate(program_path)`).

For one policy file it
  1. statically checks the source,
  2. trains from the parent checkpoint with the frozen runner (subprocess, timeout),
  3. verifies the continuation rules on the produced checkpoint with the reference loader,
  4. recomputes all metrics itself on the selection split (self-reported numbers of the
     training process are never trusted),
  5. scores primary metric + guards, and
  6. writes the detailed feedback report and returns its text as an artifact.

It never raises: every failure becomes combined_score 0 plus an explanation.
All returned metrics are "larger is better", because OpenEvolve treats them that way.
"""
import contextlib
import hashlib
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.constraints import check_checkpoint, check_policy_source  # noqa: E402
from harness.dataio import read_json, write_json  # noqa: E402
from harness.feedback import build_report, render  # noqa: E402
from harness.runcfg import load_run_config  # noqa: E402
from harness.stage import get_context  # noqa: E402

FAILED_METRICS = {"combined_score": 0.0, "primary_ratio": 0.0}


def code_digest(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _tail(text, lines=12, chars=1200):
    return "\n".join(text.strip().splitlines()[-lines:])[-chars:]


def _finish(context, digest, report, metrics, cache=True):
    text = render(report, context.config["feedback"]["max_chars"])
    report.update({"code_sha256": digest, "result_metrics": metrics, "feedback_text": text})
    if cache:
        write_json(context.work / "reports" / f"{digest}.json", report)
    return {"metrics": dict(metrics), "artifacts": {"feedback": text}, "report": report}


@contextlib.contextmanager
def _code_lock(context, digest):
    """Serialise evaluations of identical code across worker processes (directory lock).

    Two workers given the same program would otherwise train into the same candidate
    directory. A lock older than the training timeout is treated as abandoned.
    """
    locks = context.work / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    lock = locks / f"candidate-{digest[:16]}.lock"
    stale_after = context.config["limits"]["train_timeout_s"] + 120
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > stale_after:
                    os.rmdir(lock)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.rmdir(lock)


def _cached_result(context, digest):
    cached = context.work / "reports" / f"{digest}.json"
    if not cached.is_file():
        return None
    report = read_json(cached)
    return {"metrics": dict(report["result_metrics"]), "artifacts": {"feedback": report["feedback_text"]},
            "report": report}


def evaluate_policy(program_path, context=None, use_cache=True):
    """Evaluate one policy file. Returns {"metrics", "artifacts", "report"}.

    Results are cached by the SHA-256 of the policy source: training is deterministic,
    so identical code always gives the identical result.
    """
    context = context or get_context(load_run_config())
    code = Path(program_path).read_text(encoding="utf-8")
    digest = code_digest(code)
    if use_cache:
        cached = _cached_result(context, digest)
        if cached:
            return cached
    with _code_lock(context, digest):
        if use_cache:
            cached = _cached_result(context, digest)  # another worker may just have finished it
            if cached:
                return cached
        return _evaluate_locked(program_path, code, digest, context)


def _evaluate_locked(program_path, code, digest, context):
    feedback_context = context.feedback_context()

    problems = check_policy_source(code)
    if problems:
        return _finish(context, digest, build_report(feedback_context, "policy_rejected", problems), FAILED_METRICS)

    directory = context.work / "candidates" / digest[:16]
    if directory.exists():
        shutil.rmtree(directory)
    outcome = context.train_candidate(program_path, directory)
    if outcome["timed_out"]:
        limit = context.config["limits"]["train_timeout_s"]
        report = build_report(feedback_context, "timeout", [f"训练超过 {limit} 秒被终止（检查是否有死循环或过大的轮数）"])
        return _finish(context, digest, report, FAILED_METRICS)
    if outcome["returncode"] != 0:
        policy_errors = [line[len("POLICY_ERROR: "):] for line in outcome["stderr"].splitlines()
                         if line.startswith("POLICY_ERROR: ")]
        report = build_report(feedback_context, "train_failed", policy_errors or ["训练进程异常退出"],
                              stderr_tail=None if policy_errors else _tail(outcome["stderr"]))
        return _finish(context, digest, report, FAILED_METRICS)

    model, problems = check_checkpoint(context.parent, directory / "checkpoint.npz")
    if problems:
        return _finish(context, digest, build_report(feedback_context, "constraint_violation", problems), FAILED_METRICS)

    bundle = context.analyze(model)
    candidate_metrics, scoring = context.score_bundle(bundle)
    runner_report = read_json(directory / "runner_report.json")
    report = build_report(feedback_context, "ok", bundle=bundle, scoring=scoring, runner_report=runner_report,
                          candidate_metrics=candidate_metrics)
    report["candidate_dir"] = str(directory)
    report["checkpoint_sha256"] = read_json(directory / "training.json")["checkpoint_sha256"]
    metrics = {"combined_score": scoring["combined_score"], "primary_ratio": scoring["primary_ratio"]}
    return _finish(context, digest, report, metrics)


def evaluate(program_path):
    """OpenEvolve entry point."""
    try:
        result = evaluate_policy(program_path)
    except Exception as error:  # never let an evaluator bug stop the evolution run
        text = f"[状态] 失败：评测器内部错误，combined_score=0\n{type(error).__name__}: {error}\n{_tail(traceback.format_exc())}"
        result = {"metrics": dict(FAILED_METRICS), "artifacts": {"feedback": text}}
    try:
        from openevolve.evaluation_result import EvaluationResult
    except ImportError:  # standalone use without OpenEvolve on the path
        return result
    return EvaluationResult(metrics=result["metrics"], artifacts=result["artifacts"])
