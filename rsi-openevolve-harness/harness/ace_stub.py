"""Placeholder for the ACE module (Reflector + Curator). Replace this file's behaviour by
pointing run config key ace.module at your own module; see docs/ACE_INTERFACE.md.

Contract:
    on_batch(records, config) -> None
        records: list of iteration records (schema "harness.iteration.v1"), oldest first
        config:  the resolved run config; config["playbook_path"] is the file you own

What the harness does for you:
  * reads config["playbook_path"] fresh for every prompt and injects it as {playbook};
  * asks the model to list the bullets it used ("USED_BULLETS: [...]") and stores them
    in each record together with the measured outcome.

What this stub does: it never writes the playbook. It only keeps evidence-based counters
per bullet and task in <work_dir>/playbook/bullet_stats.json, so that helpful/harmful
counts come from measured outcomes instead of an LLM's opinion. You may reuse
`update_bullet_stats` in your own module.
"""
import json
from pathlib import Path

OUTCOME_TO_COUNTER = {"improved": "helpful", "regressed": "harmful", "equal": "neutral", "failed": "failed"}


def update_bullet_stats(records, stats_path):
    """Add one count per (bullet, task, outcome) for every record that cites bullets."""
    stats_path = Path(stats_path)
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}
    for record in records:
        task = record.get("task_name") or "unknown"
        counter = OUTCOME_TO_COUNTER.get(record.get("outcome"), "failed")
        for bullet in record.get("used_bullets") or []:
            entry = stats.setdefault(bullet, {}).setdefault(
                task, {"helpful": 0, "harmful": 0, "neutral": 0, "failed": 0})
            entry[counter] += 1
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def on_batch(records, config):
    update_bullet_stats(records, Path(config["playbook_path"]).parent / "bullet_stats.json")
