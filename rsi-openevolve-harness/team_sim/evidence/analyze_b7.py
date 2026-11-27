"""Same 34-24-1 MLP and recipe as the official B baseline, trained from scratch with different targets.

  true      target = true RUL (the official recipe; sanity check against 681 / 448)
  cap_h     piecewise-linear target min(RUL, c_h), c_h = mean RUL of healthy training rows (first 80 % of life)
  cap_300   piecewise-linear with a small, literature-style cap (300 min)
  stage     healthy training rows -> c_h, last 20 % of life -> true RUL
Labels are always scored as the TRUE RUL in minutes.
Public package data only. Usage: python -I analyze_b7.py HARNESS_DIR PACKAGE_B_DIR
"""
import json
import sys
from collections import OrderedDict

import numpy as np

HARNESS, PKG = sys.argv[1], sys.argv[2]
sys.path.insert(0, HARNESS)
from baselines.model import Model  # noqa: E402
from baselines.data import flatten  # noqa: E402

PATHS = list(Model.load(f"{PKG}/checkpoint.npz").meta["paths"])


def load(path):
    X, y, b = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                X.append(flatten(r["inputs"])[1]); y.append(float(r["answer"])); b.append(r["meta"]["bearing"])
    return np.asarray(X, float), np.asarray(y), np.asarray(b)


Xa, ya, ba = load(f"{PKG}/public/train.jsonl")
Xv, yv, bv = load(f"{PKG}/public/validation.jsonl")
X = np.vstack([Xa, Xv]); y = np.concatenate([ya, yv]); b = np.concatenate([ba, bv])
bearings = list(OrderedDict.fromkeys(b))
life = {n: int((b == n).sum()) for n in bearings}
late = np.array([1 - y[i] / max(1, life[b[i]] - 1) >= 0.8 for i in range(len(y))])
rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
VARIANTS = ["true", "cap_h", "cap_300", "stage"]


def target(name, tr):
    c_h = y[tr & ~late].mean()
    if name == "true":
        return y[tr]
    if name == "cap_h":
        return np.minimum(y[tr], c_h)
    if name == "cap_300":
        return np.minimum(y[tr], 300.0)
    return np.where(late[tr], y[tr], c_h)


def run(tr, te, name, seed):
    t = target(name, tr)
    m = Model("regression", PATHS, [], X[tr], t, hidden=24, seed=seed, learning_rate=0.003, batch_size=128)
    m.fit(X[tr], t, 15)
    return m.predict(X[te])


def summarize(title, folds, seeds):
    print(f"\n== {title}（{len(seeds)} 个随机种子取平均）==")
    print(f"{'目标':<9}{'按行RMSE':>9}{'等权RMSE':>9}{'健康段':>8}{'最后20%':>9}")
    for name in VARIANTS:
        pooled, eq, hp, lp = [], [], [], []
        for s in seeds:
            errs, per, hs, ls = [], [], [], []
            for tr, te in folds:
                e = run(tr, te, name, s) - y[te]
                errs.append(e); hs.append(e[~late[te]]); ls.append(e[late[te]])
                per += [rmse(e[b[te] == n]) for n in OrderedDict.fromkeys(b[te])]
            pooled.append(rmse(np.concatenate(errs))); eq.append(np.mean(per))
            hp.append(rmse(np.concatenate(hs))); lp.append(rmse(np.concatenate(ls)))
        print(f"{name:<9}{np.mean(pooled):>9.0f}{np.mean(eq):>9.0f}{np.mean(hp):>8.0f}{np.mean(lp):>9.0f}")


train_mask = np.isin(b, list(OrderedDict.fromkeys(ba)))
summarize("官方划分：9 根训练 → 3 根验证（官方基线 checkpoint：按行 681，等权 448）",
          [(train_mask, ~train_mask)], seeds=[42, 1, 2])
summarize("留一根轴承，12 折", [(b != n, b == n) for n in bearings], seeds=[42, 1])
