"""Dataset loading with an on-disk cache, plus helpers for meta fields and subsets.

Parsing is always done by the reference loader (baselines.data.read_dataset), so every
schema and leakage check of the official code still applies. The cache only stores the
arrays that loader returned, keyed by the SHA-256 of the raw file.
"""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from baselines.data import read_dataset


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    os.close(handle)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, payload):
    def writer(target):
        with open(target, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    _atomic_write(path, writer)


def read_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def load_dataset(path, task, classes, expected_paths=None, cache_dir=None):
    """Return {"ids", "x", "y", "paths", "sha256"} exactly as the reference loader does."""
    if cache_dir is None:
        return read_dataset(path, task, classes, expected_paths)
    sha = file_sha256(path)
    key = hashlib.sha256(json.dumps([sha, task, list(classes or [])]).encode()).hexdigest()[:24]
    cache_file = Path(cache_dir) / f"dataset-{key}.npz"
    if cache_file.is_file():
        with np.load(cache_file, allow_pickle=False) as arrays:
            data = {"ids": [str(i) for i in arrays["ids"]], "x": arrays["x"].copy(), "y": arrays["y"].copy(),
                    "paths": json.loads(str(arrays["paths"])), "sha256": str(arrays["sha256"])}
        if data["sha256"] == sha:
            if expected_paths is not None and data["paths"] != expected_paths:
                raise ValueError("feature schema changed")
            return data
    data = read_dataset(path, task, classes, expected_paths)

    def writer(target):
        with open(target, "wb") as stream:
            np.savez(stream, ids=np.asarray(data["ids"]), x=data["x"], y=data["y"],
                     paths=np.array(json.dumps(data["paths"])), sha256=np.array(data["sha256"]))
    _atomic_write(cache_file, writer)
    return data


def read_rows(path):
    """Yield (raw_line, parsed_row) for every non-empty JSONL line, in file order."""
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield line.rstrip("\n"), json.loads(line)


def load_meta(path, cache_dir=None):
    """Return (ids, meta_rows) in file order. Meta is used for evaluation breakdowns only."""
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"meta-{file_sha256(path)[7:31]}.json"
        if cache_file.is_file():
            cached = read_json(cache_file)
            return cached["ids"], cached["meta"]
    ids, meta = [], []
    for _, row in read_rows(path):
        ids.append(row["id"])
        meta.append(row.get("meta") if isinstance(row.get("meta"), dict) else {})
    if cache_file is not None:
        write_json(cache_file, {"ids": ids, "meta": meta})
    return ids, meta


def write_subset(source_path, keep_ids, target_path):
    """Copy the lines of `source_path` whose id is in `keep_ids` (original order and bytes)."""
    keep = set(keep_ids)

    def writer(target):
        with open(target, "w", encoding="utf-8") as stream:
            for line, row in read_rows(source_path):
                if row["id"] in keep:
                    stream.write(line + "\n")
    _atomic_write(target_path, writer)
    return target_path
