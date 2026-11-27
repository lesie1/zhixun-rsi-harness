"""Assemble, digest and store a split manifest.

Keys are exactly harness.splits.MANIFEST_KEYS and the digest is harness.splits.manifest_digest,
so the harness (harness/stage.py) reads our manifests like its own.
"""
import json
from collections import Counter
from pathlib import Path

from harness.dataio import read_json, write_json
from harness.groups import group_labels
from harness.splits import MANIFEST_KEYS, manifest_digest


def compute_counts(table, rows, group_key):
    """Counts of one side, same structure as the provisional implementation."""
    result = {"rows": len(rows)}
    labels = table.labels()
    if labels is not None:
        result["labels"] = dict(sorted(Counter(str(labels[r]) for r in rows).items()))
    if group_key:
        groups = group_labels(table.meta_rows, group_key)
        result["groups"] = len({groups[r] for r in rows})
    return result


def build_manifest(table, assignment, stage_id, options):
    if not isinstance(stage_id, str) or not stage_id.strip():
        raise ValueError("stage_id must be a non-empty string")
    select_rows, gate_rows = sorted(assignment.select_rows), sorted(assignment.gate_rows)
    manifest = {
        "stage_id": stage_id,
        "source_sha256": table.source_sha256,
        "task_type": table.task_type,
        "method": assignment.method,
        "group_key": list(assignment.group_key) if assignment.group_key else None,
        "seed": options.seed,
        "select_ids": [table.ids[r] for r in select_rows],
        "gate_ids": [table.ids[r] for r in gate_rows],
        "counts": {"select": compute_counts(table, select_rows, assignment.group_key),
                   "gate": compute_counts(table, gate_rows, assignment.group_key)},
        "warnings": list(assignment.warnings),
        "provisional": False,
    }
    manifest["manifest_sha256"] = manifest_digest(manifest)
    assert tuple(manifest) == MANIFEST_KEYS
    return manifest


def read_manifest(path):
    return read_json(path)


def write_manifest(manifest, path, overwrite=False):
    """Atomic write. An existing file with different content is never replaced silently."""
    path = Path(path)
    if path.exists():
        try:
            same = read_json(path) == manifest
        except (OSError, json.JSONDecodeError):
            same = False
        if same:
            return path
        if not overwrite:
            raise FileExistsError(f"{path} 已存在且内容不同；清单被配置引用后不能修改，请换一个新文件名（例如 -v2）")
    write_json(path, manifest)
    return path
