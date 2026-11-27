"""Loop state: which checkpoint is deployed now and how good it was when it was deployed.

<out>/state.json
  base_config      run config the loop takes limits, seed, split options and ACE from
  task_name, spec_file
  deployed         {"checkpoint", "policy", "since", "reference"}
                   reference = trigger.profile() of the deployed checkpoint, the level the
                   next batch is compared against
  history          one short entry per processed batch (the full record is in
                   batches/<id>/decision.json and decisions.jsonl)

The reference level changes only when a batch went through a retraining round (whatever
the gate decided). Skipped batches leave it alone, so slow drift adds up until it
triggers instead of being absorbed batch by batch.
"""
import json
import re
import shutil
import time
from pathlib import Path

from harness import PROJECT_ROOT
from harness.dataio import read_json, write_json
from harness.runcfg import PATH_KEYS

from . import trigger

BATCH_ID = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
INITIAL_POLICY = PROJECT_ROOT / "harness" / "policy_initial.py"


def state_path(out_dir):
    return Path(out_dir) / "state.json"


def read_base_config(path):
    """The raw base config with every path made absolute (it is copied into other directories)."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in PATH_KEYS:
        if raw.get(key):
            value = Path(raw[key])
            raw[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    ace = raw.get("ace")
    if isinstance(ace, dict) and ace.get("storage_root"):
        value = Path(ace["storage_root"])
        ace["storage_root"] = str(value if value.is_absolute() else (path.parent / value).resolve())
    return raw


def init(base_config, out_dir, reference_data=None):
    """Create the loop directory. The base config's parent checkpoint is the first deployed model.

    Its reference level is measured on `reference_data` (default: the base config's
    train_path, i.e. the data the model was built on).
    """
    out_dir = Path(out_dir)
    if state_path(out_dir).is_file():
        raise FileExistsError(f"{out_dir} 已经初始化过（state.json 存在）")
    raw = read_base_config(base_config)
    for key in ("train_path", "parent_checkpoint", "work_dir"):
        if not raw.get(key):
            raise ValueError(f"base config lacks {key}")
    deployed = out_dir / "deployed" / "initial"
    deployed.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw["parent_checkpoint"], deployed / "checkpoint.npz")
    shutil.copyfile(INITIAL_POLICY, deployed / "policy.py")
    from baselines.model import Model
    spec = trigger.spec_for(Model.load(deployed / "checkpoint.npz"), raw.get("task_name"), raw.get("spec_file"))
    data = reference_data or raw["train_path"]
    reference = trigger.profile(deployed / "checkpoint.npz", data, spec, cache_dir=out_dir / "cache")
    reference["measured_on"] = str(Path(data).resolve())
    state = {"version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "base_config": str(Path(base_config).resolve()), "task_name": raw.get("task_name"),
             "spec_file": raw.get("spec_file"), "stage_prefix": raw.get("stage_id", "loop"),
             "deployed": {"checkpoint": str(deployed / "checkpoint.npz"), "policy": str(deployed / "policy.py"),
                          "since": "initial", "reference": reference},
             "history": []}
    save(out_dir, state)
    write_current(out_dir, state)
    return state


def load(out_dir):
    path = state_path(out_dir)
    if not path.is_file():
        raise FileNotFoundError(f"{out_dir} 还没有初始化：先运行 python -m team_loop init")
    return read_json(path)


def save(out_dir, state):
    write_json(state_path(out_dir), state)


def write_current(out_dir, state):
    """<out>/current.json: the single place an outside system reads "which model is live"."""
    deployed = state["deployed"]
    write_json(Path(out_dir) / "current.json", {"checkpoint": deployed["checkpoint"], "policy": deployed["policy"],
                                                "since": deployed["since"]})


def done(state, batch_id):
    return any(entry["batch_id"] == batch_id for entry in state["history"])
