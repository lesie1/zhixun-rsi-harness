"""Exposure ledger: every time a batch is read for training, development or final acceptance.

Entries are keyed by the file's SHA-256, so a batch keeps its history across experiments
and renamings. Together with the `exposure` list declared in the experiment config it
answers "has this batch already influenced a model or a decision?". A batch that has can
still be trained on, but it is never called an independent test set again.
"""
import json
import time
from pathlib import Path

from harness.dataio import write_json


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.entries = json.loads(self.path.read_text(encoding="utf-8"))["entries"] if self.path.is_file() else []

    def record(self, sha256, batch_id, use, experiment_id, stage_id=None, note=None):
        entry = {"sha256": sha256, "batch_id": batch_id, "use": use, "experiment_id": experiment_id,
                 "stage_id": stage_id, "note": note, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if not any(all(e.get(k) == entry[k] for k in ("sha256", "use", "experiment_id", "stage_id"))
                   for e in self.entries):
            self.entries.append(entry)
            write_json(self.path, {"schema": "team_adapt.exposure_ledger.v1", "entries": self.entries})
        return entry

    def uses(self, sha256, exclude_experiment=None):
        return [e for e in self.entries if e["sha256"] == sha256 and e["experiment_id"] != exclude_experiment]

    def final_used(self, sha256):
        return [e for e in self.entries if e["sha256"] == sha256 and e["use"] == "final"]


def exposure_status(batch, ledger, sha256, experiment_id):
    """Declared prior exposure plus ledger entries from other experiments."""
    declared = list(batch.get("exposure") or [])
    others = [f"{e['experiment_id']}:{e['use']}" + (f"@{e['stage_id']}" if e.get("stage_id") else "")
              for e in (ledger.uses(sha256, exclude_experiment=experiment_id) if sha256 else [])]
    return {"declared": declared, "ledger": others, "exposed": bool(declared or others)}
