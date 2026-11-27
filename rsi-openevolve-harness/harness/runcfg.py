"""Run configuration: one JSON file per stage tells the harness which data to use.

The evaluator, the prompt extension and the iteration hook all read the file named by
the environment variable HARNESS_RUN_CONFIG, so a new dataset or a new stage only needs
a new config file, never a code change.

Required keys: train_path, validation_path, parent_checkpoint, work_dir.
Relative paths are resolved against the directory of the config file.
"""
import copy
import json
import os
from pathlib import Path

ENV_VAR = "HARNESS_RUN_CONFIG"
PATH_KEYS = ("train_path", "validation_path", "parent_checkpoint", "work_dir", "spec_file",
             "split_manifest", "playbook_path")
DEFAULTS = {
    "task_name": None,            # "A" / "B" / "C" enables the known-task spec; None = defaults by task type
    "stage_id": "round-01",
    "seed": 42,
    "spec_file": None,            # optional JSON overriding the scoring spec (see harness/specs.py)
    "split_manifest": None,       # optional manifest written by the split script of a teammate
    "playbook_path": None,        # default: <work_dir>/playbook/playbook.md (owned by the ACE module)
    "limits": {"max_epochs": 50, "train_timeout_s": 600, "cpu_threads": 1},
    "split": {"min_groups": 4},
    "feedback": {"max_chars": 3500, "top_k": 5},
    "prompt": {"playbook_max_chars": 2500, "recent_failures": 3},
    "records": {"include_prompt": False},
    "ace": {"module": "harness.ace_stub", "batch_size": 10},
}


def _merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_run_config(path=None):
    """Load and normalise a run config (from `path` or from $HARNESS_RUN_CONFIG)."""
    path = path or os.environ.get(ENV_VAR)
    if not path:
        raise RuntimeError(f"no run config: pass a path or set {ENV_VAR}")
    path = Path(path).resolve()
    config = _merge(DEFAULTS, json.loads(path.read_text(encoding="utf-8")))
    for key in ("train_path", "validation_path", "parent_checkpoint", "work_dir"):
        if not config.get(key):
            raise ValueError(f"run config lacks required key '{key}'")
    for key in PATH_KEYS:
        if config.get(key):
            value = Path(config[key])
            config[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    if not config["playbook_path"]:
        config["playbook_path"] = str(Path(config["work_dir"]) / "playbook" / "playbook.md")
    config["config_path"] = str(path)
    return config


def write_resolved(config):
    """Write the normalised config into the work dir and return its path (absolute paths only)."""
    target = Path(config["work_dir"]) / "run_config.resolved.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: value for key, value in config.items() if key != "config_path"}
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(target)
