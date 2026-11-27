"""Preset experiences ("skills") for remaining-life regression, seeded into a stage's playbook.

The entries live in team_sim/seeds.json (same schema as team_gas/base_experiences.json). They are
written with team_gas.playbook.seed, which uses team_ace's own database functions when the stage
runs team_ace (so they survive re-publishing) and writes the playbook file otherwise. Every entry
shows up as 待验证; team_ace's counters, fed by gate outcomes, decide whether it earns trust.

Conditions understood here: task_type, task_names (optional), protocols (optional).
"""
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "seeds.json"


def load(path=None):
    return json.loads(Path(path or DEFAULT_PATH).read_text(encoding="utf-8"))["entries"]


def eligible(entry, context):
    """(ok, reasons); context: task_name, task_type, protocol."""
    cond = entry.get("conditions") or {}
    reasons = []
    if cond.get("task_type") and context.get("task_type") != cond["task_type"]:
        reasons.append(f"任务类型 {context.get('task_type')} 不是 {cond['task_type']}")
    if cond.get("task_names") and context.get("task_name") not in cond["task_names"]:
        reasons.append(f"任务 {context.get('task_name')} 不在 {cond['task_names']}")
    if cond.get("protocols") and context.get("protocol") not in cond["protocols"]:
        reasons.append(f"给法 {context.get('protocol')} 不在 {cond['protocols']}")
    return not reasons, reasons


def select(context, path=None):
    return [entry for entry in load(path) if eligible(entry, context)[0]]


def seed(run_config, context, path=None):
    """Write the eligible entries into the playbook this stage's prompts read."""
    from team_gas.playbook import seed as write_entries
    return write_entries(run_config, select(context, path))
