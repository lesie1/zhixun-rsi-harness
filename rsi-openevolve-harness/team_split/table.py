"""Read the labelled data of one stage: the official validation file, nothing else.

Everything is read through the harness (harness.dataio, i.e. the reference loader of
baselines/data.py), so every schema and leakage check of the official code applies.
Task type, class names and feature paths come from the parent checkpoint, exactly as
harness.stage.StageContext does, so the split always matches what the harness evaluates.

Row order = file order, and file order is treated as time order.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import harness  # noqa: F401  (pins BLAS threads before numpy and puts the project on sys.path)
import numpy as np

from baselines.model import Model
from harness.dataio import file_sha256, load_dataset, load_meta


@dataclass(frozen=True)
class LabeledTable:
    ids: List[str]
    targets: np.ndarray            # class indices (classification) or values in original units
    meta_rows: List[dict]          # used for grouping only, never as model input
    task_type: str                 # "classification" | "regression"
    class_names: List[str] = field(default_factory=list)
    source_path: str = ""
    source_sha256: str = ""        # "sha256:" + hex digest of the raw file bytes

    def __len__(self):
        return len(self.ids)

    def labels(self):
        """Class names per row (classification) or None (regression)."""
        if self.task_type != "classification":
            return None
        return [self.class_names[int(t)] if self.class_names else str(int(t)) for t in self.targets]


def load_table(run_config):
    """Build the LabeledTable of a normalised run config (harness.runcfg.load_run_config)."""
    parent = Model.load(run_config["parent_checkpoint"])
    task, classes = parent.meta["task"], list(parent.meta["classes"])
    path = str(Path(run_config["validation_path"]).resolve())
    data = load_dataset(path, task, classes, parent.meta["paths"])
    ids, meta = load_meta(path)
    if ids != data["ids"]:
        raise ValueError("meta rows are not aligned with the validation data")
    sha = file_sha256(path)
    if sha != data["sha256"]:
        raise ValueError("validation file changed while it was being read")
    return LabeledTable(ids=list(data["ids"]), targets=data["y"], meta_rows=list(meta), task_type=task,
                        class_names=classes if task == "classification" else [], source_path=path,
                        source_sha256=sha)
