"""Split the labelled validation data into a selection part and a gate part.

PROVISIONAL IMPLEMENTATION. A teammate owns the final split script; it only has to
produce the same manifest (see `MANIFEST_KEYS`) and can replace `make_split` or simply
drop a manifest file at the path named by run config key "split_manifest".

The split is part of the frozen zone: it is computed once per stage, never by the
evolving policy, and never from model predictions.

Methods:
  group_stratified  groups (e.g. engines) never straddle; balanced inside a coarser key
  group             same without a coarser key
  class_time        classification without groups: inside each class, earlier half ->
                    selection, later half -> gate (file order is treated as time order)
  time_blocks       regression without groups: contiguous blocks alternate
  no_split          fewer than `min_groups` groups: selection = gate = everything,
                    with a warning (no independent gate exists)
"""
import hashlib
import json
import random
from collections import Counter, defaultdict

from .groups import discover_groups, group_labels

MANIFEST_KEYS = ("stage_id", "source_sha256", "task_type", "method", "group_key", "seed",
                 "select_ids", "gate_ids", "counts", "warnings", "provisional", "manifest_sha256")


def _counts(ids, index_of, labels, groups):
    rows = [index_of[i] for i in ids]
    result = {"rows": len(rows)}
    if labels is not None:
        result["labels"] = dict(sorted(Counter(str(labels[r]) for r in rows).items()))
    if groups is not None:
        result["groups"] = len({groups[r] for r in rows})
    return result


def make_split(ids, targets, meta_rows, task_type, stage_id, source_sha256, seed=42, min_groups=4,
               class_names=None, time_blocks=10):
    """Return a manifest dict. `targets` are class indices or regression values."""
    n = len(ids)
    discovered = discover_groups(meta_rows)
    split_key = discovered["split_key"]
    groups = group_labels(meta_rows, split_key) if split_key else None
    group_order = list(dict.fromkeys(groups)) if groups else []
    warnings, select, gate = [], [], []
    rng = random.Random(seed)
    labels = None
    if task_type == "classification":
        labels = [class_names[int(t)] if class_names else str(int(t)) for t in targets]

    if groups and len(group_order) >= min_groups:
        coarse_key = next((key for key in discovered["report_keys"] if key != split_key), None)
        coarse = group_labels(meta_rows, coarse_key) if coarse_key else None
        buckets = defaultdict(list)
        first_row = {}
        for row, group in enumerate(groups):
            first_row.setdefault(group, row)
        for group in group_order:
            buckets[coarse[first_row[group]] if coarse else "all"].append(group)
        chosen = {}
        for bucket in sorted(buckets):
            members = sorted(buckets[bucket])
            rng.shuffle(members)
            for position, group in enumerate(members):
                chosen[group] = "select" if position % 2 == 0 else "gate"
        for row, group in enumerate(groups):
            (select if chosen[group] == "select" else gate).append(ids[row])
        method = "group_stratified" if coarse else "group"
    elif groups:
        method = "no_split"
        select, gate = list(ids), list(ids)
        warnings.append(f"only {len(group_order)} group(s) for key {'+'.join(split_key)}; "
                        "no independent gate split, selection and gate use the same data")
    elif task_type == "classification":
        method = "class_time"
        by_class = defaultdict(list)
        for row, label in enumerate(labels):
            by_class[label].append(row)
        part = {}
        for label, rows in by_class.items():
            half = (len(rows) + 1) // 2
            for position, row in enumerate(rows):
                part[row] = "select" if position < half else "gate"
            if len(rows) < 2:
                warnings.append(f"class {label} has a single sample; it is only in the selection split")
        for row in range(n):
            (select if part[row] == "select" else gate).append(ids[row])
    else:
        method = "time_blocks"
        block = max(1, -(-n // time_blocks))
        for row in range(n):
            (select if (row // block) % 2 == 0 else gate).append(ids[row])

    index_of = {identity: row for row, identity in enumerate(ids)}
    manifest = {
        "stage_id": stage_id, "source_sha256": source_sha256, "task_type": task_type, "method": method,
        "group_key": split_key, "seed": seed, "select_ids": select, "gate_ids": gate,
        "counts": {"select": _counts(select, index_of, labels, groups),
                   "gate": _counts(gate, index_of, labels, groups)},
        "warnings": warnings, "provisional": True,
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def manifest_digest(manifest):
    payload = {key: manifest[key] for key in MANIFEST_KEYS if key != "manifest_sha256" and key in manifest}
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_manifest(manifest, ids):
    """Raise ValueError when a manifest (ours or a teammate's) is unusable for `ids`."""
    missing = [key for key in MANIFEST_KEYS if key not in manifest and key != "provisional"]
    if missing:
        raise ValueError(f"split manifest lacks keys: {missing}")
    known = set(ids)
    select, gate = set(manifest["select_ids"]), set(manifest["gate_ids"])
    if not select or not gate:
        raise ValueError("split manifest has an empty side")
    if (select | gate) - known:
        raise ValueError("split manifest names ids that are not in the labelled data")
    if manifest["method"] != "no_split":
        if select & gate:
            raise ValueError("selection and gate ids overlap")
        if (select | gate) != known:
            raise ValueError("selection and gate ids do not cover the labelled data")
    return manifest
