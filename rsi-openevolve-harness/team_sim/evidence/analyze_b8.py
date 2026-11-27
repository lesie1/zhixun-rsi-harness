"""How sensitive is the capped-target MLP to the rule that sets the cap?

Cap rules, always computed from the TRAINING bearings of each fold:
  rows_f   mean RUL over all training snapshots in the first f of their bearing's life (row-weighted)
  bear_f   mean over bearings of the per-bearing mean RUL in the first f of life (bearing-equal)
Same 34-24-1 MLP / recipe as the official B baseline, trained from scratch; scored on TRUE minutes.
Public package data only. Usage: python -I analyze_b8.py HARNESS_DIR PACKAGE_B_DIR
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
frac = np.array([1 - y[i] / max(1, life[b[i]] - 1) for i in range(len(y))])
rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))


def cap_value(rule, tr):
    kind, f = rule.split("_")
    f = float(f)
    sel = tr & (frac < f + 1e-9)
    if kind == "rows":
        return y[sel].mean()
    return np.mean([y[sel & (b == n)].mean() for n in OrderedDict.fromkeys(b[tr])])


RULES = ["true", "rows_1.0", "rows_0.8", "rows_0.5", "bear_0.8"]


def run(tr, te, rule, seed):
    cap = None if rule == "true" else cap_value(rule, tr)
    t = y[tr] if cap is None else np.minimum(y[tr], cap)
    m = Model("regression", PATHS, [], X[tr], t, hidden=24, seed=seed, learning_rate=0.003, batch_size=128)
    m.fit(X[tr], t, 15)
    return m.predict(X[te]), cap


train_mask = np.isin(b, list(OrderedDict.fromkeys(ba)))
print("训练集 9 根轴承上各规则算出的上限（分钟）：" +
      "  ".join(f"{r}={cap_value(r, train_mask):.0f}" for r in RULES[1:]))
for title, folds, seeds in [("官方划分 9→3（3 个种子）", [(train_mask, ~train_mask)], [42, 1, 2]),
                            ("留一根轴承 12 折（2 个种子）", [(b != n, b == n) for n in bearings], [42, 1])]:
    print(f"\n== {title} ==")
    print(f"{'规则':<10}{'上限':>7}{'按行RMSE':>10}{'等权RMSE':>10}")
    for rule in RULES:
        pooled, eq, caps = [], [], []
        for s in seeds:
            errs, per = [], []
            for tr, te in folds:
                p, cap = run(tr, te, rule, s)
                e = p - y[te]; errs.append(e); caps.append(cap)
                per += [rmse(e[b[te] == n]) for n in OrderedDict.fromkeys(b[te])]
            pooled.append(rmse(np.concatenate(errs))); eq.append(np.mean(per))
        shown = "—" if rule == "true" else f"{np.mean(caps):.0f}"
        print(f"{rule:<10}{shown:>7}{np.mean(pooled):>10.0f}{np.mean(eq):>10.0f}")
