"""Everything that is fixed for one stage and shared by all candidates.

Built once (idempotently) per work dir:
  splits/manifest.json, select.jsonl, gate.jsonl   the frozen data split
  reference/                                       the unmodified policy = official continuation
and, in memory, the parent checkpoint and its evaluation, which every candidate is
scored against.
"""
import os
import shutil
import time
from pathlib import Path

import numpy as np

from baselines.model import Model, checkpoint_digest

from . import PROJECT_ROOT
from .analysis import analyze, group_regressions, scoring_metrics
from .dataio import file_sha256, load_dataset, load_meta, read_json, write_json, write_subset
from .groups import discover_groups
from .scoring import score
from .specs import resolve_spec
from .splits import make_split, validate_manifest
from .training import run_training

INITIAL_POLICY = PROJECT_ROOT / "harness" / "policy_initial.py"


class StageContext:
    def __init__(self, config):
        self.config = config
        self.work = Path(config["work_dir"])
        self.cache = self.work / "cache"
        self.parent_path = config["parent_checkpoint"]
        self.parent = Model.load(self.parent_path)
        self.parent_digest = checkpoint_digest(self.parent_path)
        self.task = self.parent.meta["task"]
        self.classes = self.parent.meta["classes"]
        self.spec = resolve_spec(self.task, config.get("task_name"), config.get("spec_file"))
        for name in ("cache", "splits", "candidates", "reports", "records", "playbook"):
            (self.work / name).mkdir(parents=True, exist_ok=True)
        self._prepare_split()
        self._prepare_parent()
        self._prepare_reference()
        self._write_summary()

    # ------------------------------------------------------------------ data
    def _load(self, path):
        return load_dataset(path, self.task, self.classes, self.parent.meta["paths"], cache_dir=self.cache)

    def _prepare_split(self):
        validation_path = self.config["validation_path"]
        full = self._load(validation_path)
        ids, meta = load_meta(validation_path, cache_dir=self.cache)
        if ids != full["ids"]:
            raise ValueError("meta rows are not aligned with the validation data")
        self._meta_of = dict(zip(ids, meta))
        manifest_path = Path(self.config.get("split_manifest") or self.work / "splits" / "manifest.json")
        if self.config.get("split_manifest") and not manifest_path.is_file():
            raise FileNotFoundError(f"split manifest not found: {manifest_path}")
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
        else:
            manifest = make_split(full["ids"], full["y"], meta, self.task, self.config["stage_id"],
                                  full["sha256"], seed=self.config["seed"],
                                  min_groups=self.config["split"]["min_groups"],
                                  class_names=self.classes if self.task == "classification" else None)
            self._locked_write(manifest_path, lambda: write_json(manifest_path, manifest))
            manifest = read_json(manifest_path)
        self.manifest = validate_manifest(manifest, full["ids"])
        self.select_path = self.work / "splits" / "select.jsonl"
        self.gate_path = self.work / "splits" / "gate.jsonl"
        marker = self.work / "splits" / "materialized.json"
        stamp = {"manifest_sha256": manifest["manifest_sha256"], "source_sha256": file_sha256(validation_path)}

        def materialize():
            write_subset(validation_path, manifest["select_ids"], self.select_path)
            write_subset(validation_path, manifest["gate_ids"], self.gate_path)
            write_json(marker, stamp)
        if not marker.is_file() or read_json(marker) != stamp:
            self._locked_write(marker, materialize, force=True)
        self.select = self._load(self.select_path)
        self.select_meta = [self._meta_of[i] for i in self.select["ids"]]
        self.group_info = discover_groups(self.select_meta)
        split_key = self.group_info["split_key"]
        groups = self.group_info["sizes"].get("+".join(split_key), 0) if split_key else 0
        mode = self.spec["aggregate"]
        if mode == "auto":
            mode = "group_mean" if manifest["method"] == "no_split" and groups >= 2 else "pooled"
        self.aggregate = mode if groups >= 2 or mode == "pooled" else "pooled"
        self.aggregate_key = split_key if self.aggregate == "group_mean" else None
        report_keys = self.group_info["report_keys"]
        self.guard_key = "+".join(report_keys[0]) if report_keys else None

    def load_gate(self):
        data = self._load(self.gate_path)
        return data, [self._meta_of[i] for i in data["ids"]]

    def _locked_write(self, target, action, force=False):
        """Run `action` unless another process already produced `target` (simple directory lock)."""
        locks = self.work / "locks"
        locks.mkdir(parents=True, exist_ok=True)
        lock = locks / (Path(target).parent.name + "-" + Path(target).name + ".lock")
        for _ in range(600):
            if target.is_file() and not force:
                return
            try:
                os.mkdir(lock)
            except FileExistsError:
                time.sleep(0.2)
                force = False  # whoever holds the lock is producing the file
                continue
            try:
                action()
            finally:
                os.rmdir(lock)
            return
        raise TimeoutError(f"gave up waiting for {lock}")

    # ------------------------------------------------------------ references
    def analyze(self, model, data=None, meta=None, group_info=None):
        return analyze(model, data or self.select, meta or self.select_meta, self.spec,
                       group_info or self.group_info, top_k=self.config["feedback"]["top_k"])

    def score_bundle(self, bundle, reference_bundle=None, reference_metrics=None):
        """Score an evaluation bundle against the parent checkpoint (or another reference)."""
        reference_bundle = reference_bundle or self.parent_bundle
        reference_metrics = reference_metrics or self.parent_metrics
        metrics = scoring_metrics(bundle, self.spec, self.aggregate_key)
        regressions = group_regressions(bundle, reference_bundle, self.spec, self.guard_key) if self.guard_key else None
        return metrics, score(self.spec, metrics, reference_metrics, regressions)

    def _prepare_parent(self):
        self.parent_bundle = self.analyze(self.parent)
        self.parent_metrics = scoring_metrics(self.parent_bundle, self.spec, self.aggregate_key)
        train = load_dataset(self.config["train_path"], self.task, self.classes, self.parent.meta["paths"],
                             cache_dir=self.cache)
        self.train_rows = len(train["ids"])
        if self.task == "classification":
            counts = np.bincount(train["y"], minlength=len(self.classes))
            self.train_stats = {"class_counts": {name: int(c) for name, c in zip(self.classes, counts)}}
        else:
            quartiles = np.quantile(train["y"], [0, .25, .5, .75, 1])
            self.train_stats = {"target_mean": float(train["y"].mean()), "target_std": float(train["y"].std()),
                                "target_quartiles": [float(q) for q in quartiles]}

    def train_candidate(self, policy_path, output_dir):
        limits = self.config["limits"]
        return run_training(policy_path, self.parent_path, self.config["train_path"], self.select_path,
                            output_dir, self.config["stage_id"], max_epochs=limits["max_epochs"],
                            timeout_s=limits["train_timeout_s"], cpu_threads=limits["cpu_threads"],
                            cache_dir=self.cache)

    def _prepare_reference(self):
        """The unmodified policy: what the official continuation command would produce."""
        directory = self.work / "reference"
        checkpoint = directory / "checkpoint.npz"

        def produce():
            if directory.exists():
                shutil.rmtree(directory)
            outcome = self.train_candidate(INITIAL_POLICY, directory)
            if outcome["returncode"] != 0:
                raise RuntimeError(f"reference training failed: {outcome['stderr'][-2000:]}")
        self._locked_write(checkpoint, produce)
        self.reference_dir = directory
        self.reference_model = Model.load(checkpoint)
        self.reference_bundle = self.analyze(self.reference_model)
        self.reference_metrics, self.reference_score = self.score_bundle(self.reference_bundle)

    # --------------------------------------------------------------- outputs
    def feedback_context(self):
        return {"task_name": self.config.get("task_name"), "task_type": self.task,
                "stage_id": self.config["stage_id"], "primary": self.spec["primary"], "unit": self.spec["unit"],
                "split_method": self.manifest["method"], "aggregate": self.aggregate, "spec": self.spec,
                "parent_metrics": self.parent_metrics, "parent_bundle": self.parent_bundle,
                "reference_metrics": self.reference_metrics, "reference_bundle": self.reference_bundle,
                "train_class_counts": self.train_stats.get("class_counts"),
                "split_warnings": self.manifest["warnings"]}

    def summary(self):
        manifest = self.manifest
        return {"task_name": self.config.get("task_name"), "task_type": self.task, "classes": self.classes,
                "stage_id": self.config["stage_id"], "spec": self.spec,
                "parent_checkpoint": self.parent_path, "parent_checkpoint_sha256": self.parent_digest,
                "parent_steps": self.parent.meta["steps"], "parent_epochs": self.parent.meta["epochs"],
                "train_rows": self.train_rows, "train_stats": self.train_stats,
                "split": {"method": manifest["method"], "group_key": manifest["group_key"],
                          "counts": manifest["counts"], "warnings": manifest["warnings"],
                          "manifest_sha256": manifest["manifest_sha256"],
                          "provisional": manifest.get("provisional", False)},
                "groups": {"report_keys": self.group_info["report_keys"], "split_key": self.group_info["split_key"],
                           "sizes": self.group_info["sizes"]},
                "aggregate": self.aggregate, "guard_group_key": self.guard_key,
                "parent_metrics": self.parent_metrics, "reference_metrics": self.reference_metrics,
                "reference_score": self.reference_score}

    def _write_summary(self):
        target = self.work / "stage_context.json"
        if not target.is_file():
            write_json(target, self.summary())


_CONTEXTS = {}


def get_context(config):
    """One StageContext per work dir and process."""
    key = config["work_dir"]
    if key not in _CONTEXTS:
        _CONTEXTS[key] = StageContext(config)
    return _CONTEXTS[key]
