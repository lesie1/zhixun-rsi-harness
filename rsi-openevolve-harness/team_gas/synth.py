"""Synthetic batches of a six-gas classification stream, for offline end-to-end runs and tests.

Batch 1 is the history the start checkpoint is trained on. From batch 2 on, one gas
(Acetaldehyde) drifts next to another (Ethanol) - a model trained on batch 1 reads it as
Ethanol, and nothing in batch 1 can teach it otherwise; Ethanol itself moves slightly the
other way, so adding only Acetaldehyde pushes the boundary into Ethanol. From batch 5 on,
Toluene drifts next to Acetone in the same way. The class names only make the record
readable; the flow finds the classes and the pairs from the data.
"""
import json
from pathlib import Path

import numpy as np

from baselines.data import read_dataset
from baselines.model import Model

CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]
DISPLAY = {"Ethanol": "乙醇", "Ethylene": "乙烯", "Ammonia": "氨", "Acetaldehyde": "乙醛", "Acetone": "丙酮", "Toluene": "甲苯"}


def _unit(rng, dims, against=()):
    v = rng.normal(size=dims)
    for other in against:
        v -= (v @ other) * other
    return v / np.linalg.norm(v)


def make(out, seed=7, features=16, history_per_class=100, batch_per_class=40, batches=7, ace_storage=None):
    out = Path(out)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    means = {c: rng.normal(0, 2.5, features) for c in CLASSES}
    u1 = _unit(rng, features)
    u2 = _unit(rng, features, [u1])

    def centre(cls, k):
        if k >= 2 and cls == "Acetaldehyde":
            return means["Ethanol"] + 2.0 * u1
        if k >= 2 and cls == "Ethanol":
            return means["Ethanol"] - 0.6 * u1
        if k >= 5 and cls == "Toluene":
            return means["Acetone"] + 2.0 * u2
        if k >= 5 and cls == "Acetone":
            return means["Acetone"] - 0.6 * u2
        return means[cls]

    paths = {}
    for k in range(1, batches + 1):
        per_class = history_per_class if k == 1 else batch_per_class
        rows = []
        for cls in CLASSES:
            for vector in centre(cls, k) + rng.normal(0, 0.6, (per_class, features)):
                rows.append({"id": f"syn_b{k:02d}_{len(rows) + 1:05d}", "inputs": {"features": [round(float(v), 6) for v in vector]},
                             "answer": cls, "meta": {"source_batch": k, "source_index": len(rows) + 1}})
        order = rng.permutation(len(rows))
        rows = [rows[i] for i in order]
        path = data / f"b{k}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        paths[f"b{k}"] = path
    history = read_dataset(paths["b1"], "classification", CLASSES)
    model = Model("classification", history["paths"], CLASSES, history["x"], history["y"], hidden=32, seed=seed,
                  learning_rate=0.01, batch_size=32)
    model.fit(history["x"], history["y"], 20)
    checkpoint = out / "checkpoint.npz"
    if checkpoint.exists():
        checkpoint.unlink()
    model.save(checkpoint)
    base = {"task_name": "A", "stage_id": "synthetic-gas", "seed": 42, "train_path": str(paths["b1"]),
            "validation_path": str(paths["b2"]), "parent_checkpoint": str(checkpoint), "work_dir": str(out / "unused"),
            "limits": {"max_epochs": 5, "train_timeout_s": 120, "cpu_threads": 1}, "split": {"min_groups": 4},
            "records": {"include_prompt": True}}
    if ace_storage:
        base["ace"] = {"module": "team_ace", "batch_size": 1, "mode": "disabled", "storage_root": str(ace_storage),
                       "record_origin": "synthetic", "max_total_calls": 0, "live_calls_approved": False, "scope": {}}
    (out / "base.json").write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    uses = {1: ["train"], 2: ["alarm", "train"], 3: ["dev"], 4: ["final"], 5: ["alarm", "train"], 6: ["dev"], 7: ["final"]}
    flow = {"experiment_id": "synthetic-gas", "task_name": "A", "base_config": "base.json",
            "initial_train_batches": ["b1"],
            "batches": [{"id": f"b{k}", "order": k, "path": f"data/b{k}.jsonl", "labels_available": True,
                         "uses": uses.get(k, ["train"]), "exposure": []} for k in range(1, batches + 1)],
            "rounds": [{"alarm": "b2", "selection": "b3", "final": "b4"}, {"alarm": "b5", "selection": "b6", "final": "b7"}],
            "display_names": DISPLAY,
            "repair": {"search": {"mode": "continuation"}}}
    (out / "flow.json").write_text(json.dumps(flow, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"flow": out / "flow.json", "base": out / "base.json", "checkpoint": checkpoint, "batches": paths}
