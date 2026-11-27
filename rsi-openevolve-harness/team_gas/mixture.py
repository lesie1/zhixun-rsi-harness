"""Training-data composition for the data-mixing stage, with a hard 20 % cap on new data.

Counting rule: what matters is how often a sample is *presented* to the optimiser. The
runner presents every row of the training file exactly once per epoch (one permutation,
consecutive mini-batches, the last one partial, no padding) - `verify_presentations`
checks that from the runner's own step counts after training. A candidate file therefore
has exactly N rows (N = size of the history file): k of them are new-data presentations
(repeated new samples are separate rows and count each time), N - k are history rows.
Hence the new share is k / N in every epoch and over the whole run, and k is bounded by
floor(cap * N) with cap <= HARD_CAP = 0.20. Adding new rows *replaces* history rows, so
every candidate has the same N, the same number of update steps and the same shuffling
seed (the runner seeds the permutation from the checkpoint, not from the data).

Sample identity is fixed per stage: one seeded order of history rows (the first N - k are
kept), one seeded order per new-data class (the first k_c are drawn, cycling with repeats
when a class has fewer unique samples than requested).
"""
import json
import math
from pathlib import Path

import numpy as np

from harness.dataio import file_sha256, read_rows

HARD_CAP = 0.20


def floor_count(fraction, total):
    return int(math.floor(fraction * total + 1e-9))


def make_pools(history_path, new_path, degraded, confusion, seed):
    """Fixed identities for one stage: history order and the new samples of the two classes."""
    rng = np.random.default_rng(seed)
    history = [(line, row) for line, row in read_rows(history_path)]
    order = [history[i] for i in rng.permutation(len(history))]
    new = {"degraded": [], "confusion": []}
    for line, row in read_rows(new_path):
        if row["answer"] == degraded:
            new["degraded"].append((line, row))
        elif row["answer"] == confusion:
            new["confusion"].append((line, row))
    for key in new:
        new[key] = [new[key][i] for i in np.random.default_rng(seed + (1 if key == "degraded" else 2)).permutation(len(new[key]))]
    history_ids = {row["id"] for _, row in history}
    clash = history_ids & {row["id"] for part in new.values() for _, row in part}
    if clash:
        raise ValueError(f"新数据与历史数据的样本 ID 重叠：{sorted(clash)[:5]}")
    return {"history": order, "new": new, "degraded": degraded, "confusion": confusion,
            "history_path": str(history_path), "new_path": str(new_path),
            "history_sha256": file_sha256(history_path), "new_sha256": file_sha256(new_path), "seed": seed}


def plan(pools, fraction, degraded_share, cap):
    """Rows of one candidate (or None when the requested classes have no new samples)."""
    if cap > HARD_CAP + 1e-12:
        raise ValueError(f"新数据上限 {cap} 超过硬上限 {HARD_CAP}")
    if fraction > cap + 1e-12:
        raise ValueError(f"目标比例 {fraction} 超过上限 {cap}")
    total = len(pools["history"])
    limit = floor_count(cap, total)
    k = min(floor_count(fraction, total), limit)
    k_degraded = int(round(k * degraded_share))
    counts = {"degraded": k_degraded, "confusion": k - k_degraded}
    for key, count in counts.items():
        if count and not pools["new"][key]:
            return None
    rows, drawn = [], {"degraded": {}, "confusion": {}}
    for key, count in counts.items():
        pool = pools["new"][key]
        for i in range(count):
            line, row = pool[i % len(pool)]
            copy_index = i // len(pool)
            drawn[key][row["id"]] = drawn[key].get(row["id"], 0) + 1
            if copy_index:
                row = dict(row, id=f"{row['id']}~rep{copy_index}")
                line = json.dumps(row, ensure_ascii=False)
            rows.append(("new", key, line, row))
    for line, row in pools["history"][: total - k]:
        rows.append(("history", None, line, row))
    new_rows = sum(1 for r in rows if r[0] == "new")
    if new_rows > limit or new_rows / total > HARD_CAP + 1e-12 or len(rows) != total:
        raise AssertionError("新数据占比超过硬上限（不应发生）")
    by_class = {}
    for source, _, _, row in rows:
        entry = by_class.setdefault(row["answer"], {"history": 0, "new": 0})
        entry[source] += 1
    accounting = {
        "rows_per_epoch": total, "new_presentations_per_epoch": new_rows,
        "history_presentations_per_epoch": total - new_rows, "target_fraction": fraction,
        "actual_fraction": new_rows / total, "cap": cap, "cap_rows": limit,
        "degraded_share_target": degraded_share,
        "degraded_share_actual": (counts["degraded"] / new_rows) if new_rows else None,
        "new_unique": {key: len(drawn[key]) for key in drawn},
        "new_presentations": counts,
        "new_repeats": {key: counts[key] - len(drawn[key]) for key in drawn},
        "per_class_presentations_per_epoch": by_class,
        "per_batch_presentations_per_epoch": {"history": total - new_rows, "new": new_rows},
        "new_sample_ids": {key: sorted(drawn[key]) for key in drawn},
    }
    return {"rows": rows, "accounting": accounting}


def write(candidate, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for _, _, line, _ in candidate["rows"]), encoding="utf-8")
    return file_sha256(path)


def verify_presentations(runner_report, training, accounting, train_sha256=None):
    """Recompute the realised share from the runner's own counters; raise if it cannot be certified.

    runner_report = <model>/runner_report.json, training = <model>/training.json.
    """
    total = accounting["rows_per_epoch"]
    batch_size = runner_report["config"]["batch_size"]
    if training["train_rows"] != total:
        raise AssertionError("runner 读到的训练行数与计划不一致")
    if train_sha256 and training["train_sha256"].removeprefix("sha256:") != train_sha256.removeprefix("sha256:"):
        raise AssertionError("runner 训练用的文件不是计划写出的那份")
    per_batch_steps = math.ceil(total / batch_size)
    previous = runner_report["initial_steps"]
    for entry in runner_report["per_epoch"]:
        if entry["step"] - previous != per_batch_steps:
            raise AssertionError("某一轮的更新步数与“每行恰好一次”不符，无法核定实际占比")
        previous = entry["step"]
    epochs_run = len(runner_report["per_epoch"])
    epochs_used = runner_report["selected_epoch_index"] + 1
    new = accounting["new_presentations_per_epoch"]
    realised = {
        "epochs_run": epochs_run, "epochs_in_selected_checkpoint": epochs_used,
        "updates_per_epoch": per_batch_steps, "batch_size": batch_size,
        "new_presentations_total": new * epochs_run, "all_presentations_total": total * epochs_run,
        "new_presentations_in_selected_checkpoint": new * epochs_used,
        "all_presentations_in_selected_checkpoint": total * epochs_used,
        "per_epoch_fraction": new / total,
        "whole_run_fraction": (new * epochs_run) / (total * epochs_run),
    }
    cap_rows = floor_count(accounting["cap"], total)
    if new > cap_rows or realised["per_epoch_fraction"] > HARD_CAP + 1e-12 or realised["whole_run_fraction"] > HARD_CAP + 1e-12:
        raise AssertionError(f"实际新数据占比 {realised['whole_run_fraction']:.4f} 超过上限")
    return realised
