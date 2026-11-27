"""Per-iteration records: the data contract with the ACE module (docs/ACE_INTERFACE.md).

`on_iteration` is called by the patched OpenEvolve main loop once per finished
iteration (see docs/OPENEVOLVE_PATCH.md). It appends one JSON object per line to
<work_dir>/records/iterations.jsonl and, every `ace.batch_size` records, hands the
batch to the configured ACE module.

The file is append-only and written by a single process (the OpenEvolve main process).
"""
import importlib
import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from .dataio import read_json
from .evaluator import code_digest
from .feedback import _deltas
from .runcfg import load_run_config

SCHEMA = "harness.iteration.v1"
_USED = re.compile(r"USED_BULLETS:\s*\[(.*?)\]", re.S)
_HYPOTHESIS = re.compile(r"HYPOTHESIS:\s*(.+)")
_PLAYBOOK = re.compile(r"PLAYBOOK_VERSION:\s*(\S+)")
_state = {"count": 0, "batch": []}


def parse_used_bullets(response):
    match = _USED.search(response or "")
    if not match:
        return []
    return [item.strip().strip("'\"`") for item in match.group(1).split(",") if item.strip().strip("'\"`")]


def parse_hypothesis(response):
    match = _HYPOTHESIS.search(response or "")
    return match.group(1).strip()[:500] if match else None


def _report_for(work_dir, code):
    path = Path(work_dir) / "reports" / f"{code_digest(code)}.json"
    return read_json(path) if path.is_file() else None


def _pseudo_bundle(report):
    return {"overall": report["metrics"]["candidate_pooled"], **report["slices"]}


def build_record(event, config):
    """Turn an OpenEvolve iteration event into a record (see SCHEMA)."""
    child = event.get("child_program") or {}
    parent = event.get("parent_program") or {}
    response = event.get("llm_response") or ""
    prompt = event.get("prompt") or {}
    version = _PLAYBOOK.search(prompt.get("user", ""))
    record = {
        "schema": SCHEMA, "run_id": os.environ.get("HARNESS_RUN_ID", "run"),
        "stage_id": config["stage_id"], "task_name": config.get("task_name"),
        "iteration": event.get("iteration"), "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": event.get("kind"), "status": None, "outcome": "failed",
        "program_id": child.get("id"), "parent_id": event.get("parent_id"), "island": event.get("island"),
        "evolution_mode": event.get("evolution_mode"),
        "code_sha256": code_digest(child["code"]) if child.get("code") else None,
        "parent_code_sha256": code_digest(parent["code"]) if parent.get("code") else None,
        "hypothesis": parse_hypothesis(response), "used_bullets": parse_used_bullets(response),
        "playbook_version": version.group(1) if version else None,
        "changes": (child.get("metadata") or {}).get("changes"), "llm_response": response,
        "combined_score": None, "parent_combined_score": None, "delta": None, "metrics": child.get("metrics"),
        "primary": None, "flags": [], "guard_violations": [], "feedback_text": None, "report_path": None,
        "delta_vs_parent_program": None, "error": event.get("error"),
        "token_usage": event.get("token_usage"), "iteration_time": event.get("iteration_time"),
        "is_new_best": bool(event.get("is_new_best")),
        "admitted": event.get("kind") == "evaluated",
    }
    if config["records"].get("include_prompt"):
        record["prompt"] = prompt
    if event.get("kind") == "failed":
        record["status"] = "rejected_before_eval"
        return record

    report = _report_for(config["work_dir"], child.get("code", "")) if child.get("code") else None
    parent_metrics = (child.get("metadata") or {}).get("parent_metrics") or parent.get("metrics") or {}
    record["combined_score"] = (child.get("metrics") or {}).get("combined_score")
    record["parent_combined_score"] = parent_metrics.get("combined_score")
    record["status"] = report["status"] if report else "internal_error"
    if not report:
        record["feedback_text"] = (event.get("artifacts") or {}).get("feedback")
        record["error"] = (record["feedback_text"] or "evaluator produced no report")[:1000]
    if report:
        record["feedback_text"] = report.get("feedback_text")
        record["report_path"] = str(Path(config["work_dir"]) / "reports" / f"{record['code_sha256']}.json")
        record["flags"] = [flag["code"] for flag in report.get("flags", [])]
        if report["status"] != "ok":
            record["error"] = "; ".join(report.get("problems", []))[:1000]
    if report and report["status"] == "ok":
        name = report["task"]["primary"]
        record["guard_violations"] = report["score"]["violations"]
        record["primary"] = {"name": name, "candidate": report["metrics"]["candidate"][name],
                             "parent_checkpoint": report["metrics"]["parent_checkpoint"][name],
                             "reference_policy": (report["metrics"]["reference_policy"] or {}).get(name)}
        if record["combined_score"] is not None and record["parent_combined_score"] is not None:
            record["delta"] = record["combined_score"] - record["parent_combined_score"]
            record["outcome"] = ("improved" if record["delta"] > 1e-9 else
                                 "regressed" if record["delta"] < -1e-9 else "equal")
        parent_report = _report_for(config["work_dir"], parent["code"]) if parent.get("code") else None
        if parent_report and parent_report["status"] == "ok":
            record["primary"]["parent_program"] = parent_report["metrics"]["candidate"][name]
            record["delta_vs_parent_program"] = _deltas(
                _pseudo_bundle(report), _pseudo_bundle(parent_report),
                {"primary": name}, report["task"]["task_type"])
    return record


def append_record(record, config):
    path = Path(config["work_dir"]) / "records" / "iterations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_records(config, last=None):
    path = Path(config["work_dir"]) / "records" / "iterations.jsonl"
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in (lines[-last:] if last else lines) if line.strip()]


def on_iteration(event):
    """Iteration hook for OpenEvolve (OPENEVOLVE_ITERATION_HOOK=harness.records:on_iteration)."""
    config = load_run_config()
    record = build_record(event, config)
    append_record(record, config)
    _state["count"] += 1
    _state["batch"].append(record)
    batch_size = int(config["ace"].get("batch_size") or 0)
    if batch_size and len(_state["batch"]) >= batch_size:
        batch, _state["batch"] = _state["batch"], []
        module_name = config["ace"].get("module")
        if module_name:
            started = time.monotonic()
            try:
                importlib.import_module(module_name).on_batch(batch, config)
            except Exception:
                log = Path(config["work_dir"]) / "records" / "ace_errors.log"
                with open(log, "a", encoding="utf-8") as stream:
                    stream.write(f"--- batch ending at iteration {record['iteration']} "
                                 f"({time.monotonic() - started:.1f}s)\n{traceback.format_exc()}\n")
