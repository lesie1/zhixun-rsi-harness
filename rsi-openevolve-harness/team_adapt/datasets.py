"""Batch files: identity, the training file of a stage, and overlap checks.

Rows are compared by sample ID and by content (a hash of the inputs alone), so the same
measurement appearing under two IDs, or in two batches, is caught as well.
"""
import hashlib
import json
from pathlib import Path

from harness.dataio import file_sha256, read_rows


def content_digest(row):
    return hashlib.sha256(json.dumps(row.get("inputs"), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def index(path):
    ids, contents, rows = set(), set(), 0
    for _, row in read_rows(path):
        ids.add(row["id"])
        contents.add(content_digest(row))
        rows += 1
    return {"path": str(path), "sha256": file_sha256(path), "rows": rows, "ids": ids, "contents": contents}


def overlap(a, b):
    return {"ids": len(a["ids"] & b["ids"]), "contents": len(a["contents"] & b["contents"])}


def materialize(batches, target):
    """Concatenate batch files (in the given time order) into `target`; refuse duplicate rows.

    `batches` is a list of (batch_id, path). Returns what was actually used.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    seen_ids, seen_contents, parts = {}, {}, []
    tmp = target.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as out:
        for batch_id, path in batches:
            rows = 0
            for line, row in read_rows(path):
                if row["id"] in seen_ids:
                    raise ValueError(f"样本 ID {row['id']} 同时出现在批次 {seen_ids[row['id']]} 和 {batch_id}")
                digest = content_digest(row)
                if digest in seen_contents:
                    raise ValueError(f"批次 {batch_id} 的样本 {row['id']} 与批次 {seen_contents[digest]} 的样本内容完全相同")
                seen_ids[row["id"]] = batch_id
                seen_contents[digest] = batch_id
                out.write(line + "\n")
                rows += 1
            parts.append({"batch_id": batch_id, "path": str(path), "sha256": file_sha256(path), "rows": rows})
    tmp.replace(target)
    return {"path": str(target), "sha256": file_sha256(target), "rows": sum(p["rows"] for p in parts), "parts": parts}
