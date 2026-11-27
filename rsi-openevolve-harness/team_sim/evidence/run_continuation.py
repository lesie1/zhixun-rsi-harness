"""Continue the OFFICIAL B checkpoint with the frozen harness runner, original vs capped target.

Policies are copies of harness/policy_initial.py with two text edits (epochs, and the regression
target inside output_grad). Training uses train.jsonl only; validation.jsonl is passed to the runner
as its selection split (never used for gradients) and is where we score, in TRUE minutes.
Usage: python run_continuation.py HARNESS_DIR PACKAGE_B_DIR OUT_DIR
"""
import json
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

HARNESS, PKG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(HARNESS))
from baselines.model import Model  # noqa: E402
from baselines.data import flatten  # noqa: E402

source = (HARNESS / "harness" / "policy_initial.py").read_text(encoding="utf-8")
assert '"epochs": 5,' in source and "    err = out[:, 0] - target\n" in source
parent = Model.load(str(PKG / "checkpoint.npz"))
mean, scale = parent.meta["target_mean"], parent.meta["target_scale"]


def policy(epochs, cap_minutes):
    text = source.replace('"epochs": 5,', f'"epochs": {epochs},')
    if cap_minutes is not None:
        cap = (cap_minutes - mean) / scale
        text = text.replace("    err = out[:, 0] - target\n",
                            f"    err = out[:, 0] - np.minimum(target, {cap!r})  # cap = {cap_minutes} min\n")
    return text


VARIANTS = OrderedDict([
    ("原目标 续训5轮", (5, None)), ("原目标 续训15轮", (15, None)),
    ("上限954 续训5轮", (5, mean)), ("上限954 续训15轮", (15, mean)), ("上限954 续训30轮", (30, mean)),
    ("上限1145 续训15轮", (15, 1145.0)),
])


def load(path):
    X, y, b = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                X.append(flatten(r["inputs"])[1]); y.append(float(r["answer"])); b.append(r["meta"]["bearing"])
    return np.asarray(X, float), np.asarray(y), np.asarray(b)


Xv, yv, bv = load(PKG / "public" / "validation.jsonl")
life = {n: int((bv == n).sum()) for n in OrderedDict.fromkeys(bv)}
late = np.array([1 - yv[i] / max(1, life[bv[i]] - 1) >= 0.8 for i in range(len(yv))])
rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))


def score(model):
    e = model.predict(Xv) - yv
    per = [rmse(e[bv == n]) for n in life]
    return rmse(e), float(np.mean(per)), rmse(e[~late]), rmse(e[late]), per


rows = [("官方检查点（不续训）", score(parent))]
OUT.mkdir(parents=True, exist_ok=True)
for i, (label, (epochs, cap)) in enumerate(VARIANTS.items()):
    pol = OUT / f"policy_{i}.py"; pol.write_text(policy(epochs, cap), encoding="utf-8")
    out = OUT / f"run_{i}"
    done = subprocess.run([sys.executable, "-m", "harness.runner", "--policy", str(pol),
                           "--parent", str(PKG / "checkpoint.npz"), "--train", str(PKG / "public" / "train.jsonl"),
                           "--validation", str(PKG / "public" / "validation.jsonl"), "--output", str(out),
                           "--stage-id", f"cap-test-{i}", "--max-epochs", "30"],
                          cwd=str(HARNESS), capture_output=True, text=True)
    if done.returncode != 0:
        print(label, "FAILED", done.stderr[-500:]); continue
    rows.append((label, score(Model.load(str(out / "checkpoint.npz")))))

print(f"{'方案':<16}{'按行RMSE':>9}{'等权RMSE':>9}{'健康段':>8}{'最后20%':>9}   " + " ".join(f"{n.replace('Bearing', 'B'):>6}" for n in life))
for label, (pooled, eq, h, l, per) in rows:
    print(f"{label:<16}{pooled:>9.0f}{eq:>9.0f}{h:>8.0f}{l:>9.0f}   " + " ".join(f"{p:>6.0f}" for p in per))
