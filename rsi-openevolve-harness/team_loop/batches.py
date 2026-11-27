"""Batch input: what one arriving batch looks like, and a simulator for local tests.

A batch is {"batch_id", "train_path", "validation_path"}:
  train_path       all labeled history the model may be trained on
  validation_path  the newest labeled batch: the trigger measures the deployed model on
                   it, and if a round runs, it is split into selection and gate data

A batch list file (batches.json) is {"batches": [batch, ...]} in arrival order; relative
paths are resolved against the file's directory.

`simulate` cuts the public validation file into contiguous chunks in file order (file
order is treated as time order, as everywhere in this project): batch k trains on the
official train file plus chunks 1..k-1 and is validated on chunk k.
"""
import json
from pathlib import Path

from harness.dataio import read_rows, write_json

from .state import BATCH_ID, read_base_config


def normalize(batch, base_dir=None):
    batch = dict(batch)
    for key in ("batch_id", "train_path", "validation_path"):
        if not batch.get(key):
            raise ValueError(f"batch lacks {key}")
    if not BATCH_ID.match(str(batch["batch_id"])):
        raise ValueError("batch_id 只能包含字母、数字、点、下划线和连字符（1–32 个字符）")
    for key in ("train_path", "validation_path"):
        path = Path(batch[key])
        if not path.is_absolute() and base_dir is not None:
            path = Path(base_dir) / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"batch {batch['batch_id']}: {key} {path} 不存在")
        batch[key] = str(path)
    return batch


def read_batches(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    batches = [normalize(b, path.parent) for b in payload["batches"]]
    ids = [b["batch_id"] for b in batches]
    if len(set(ids)) != len(ids):
        raise ValueError("batch_id 有重复")
    return batches


def simulate(base_config, chunks, out_dir):
    """Write chunk files, cumulative history files and batches.json; return the batch list."""
    if chunks < 2:
        raise ValueError("chunks must be >= 2")
    raw = read_base_config(base_config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [line for line, _ in read_rows(raw["validation_path"])]
    if len(lines) < chunks:
        raise ValueError("validation file has fewer rows than chunks")
    bounds = [round(i * len(lines) / chunks) for i in range(chunks + 1)]
    history = [line for line, _ in read_rows(raw["train_path"])]
    batches = []
    for k in range(chunks):
        batch_id = f"b{k + 1:02d}"
        chunk = lines[bounds[k]:bounds[k + 1]]
        train_file, chunk_file = out_dir / f"{batch_id}.train.jsonl", out_dir / f"{batch_id}.new.jsonl"
        train_file.write_text("".join(line + "\n" for line in history), encoding="utf-8")
        chunk_file.write_text("".join(line + "\n" for line in chunk), encoding="utf-8")
        batches.append({"batch_id": batch_id, "train_path": train_file.name, "validation_path": chunk_file.name,
                        "rows_train": len(history), "rows_new": len(chunk)})
        history = history + chunk
    write_json(out_dir / "batches.json", {"source": raw["validation_path"], "chunks": chunks, "batches": batches})
    return read_batches(out_dir / "batches.json")
