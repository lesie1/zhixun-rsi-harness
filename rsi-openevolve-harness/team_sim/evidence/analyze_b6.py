"""Can the healthy stage be improved in RMSE terms? Two-stage ideas vs plain regression.

Models (simple, for trends only):
  ridge      ridge regression on all rows (what a plain regressor does)
  const      one constant = row-weighted mean RUL of the training rows
  capped     ridge on the piecewise-linear target min(RUL, c_h), c_h = mean RUL of healthy training rows
  two_stage  p = P(last 20 % of life | x) from logistic regression;
             prediction = p * late_ridge(x) + (1 - p) * c_h
  oracle     same, but with the TRUE stage (upper bound if degradation detection were perfect)
Evaluated (i) on the official split: train on the 9 train bearings, test on the 3 validation bearings
(same data as the official baseline: pooled RMSE 681, bearing-equal 448), and (ii) leave one bearing out.
Public package data only. Usage: python -I analyze_b6.py HARNESS_DIR PACKAGE_B_DIR
"""
import json
import sys
from collections import OrderedDict

import numpy as np

HARNESS, PKG = sys.argv[1], sys.argv[2]
sys.path.insert(0, HARNESS)
from baselines.data import flatten  # noqa: E402


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
T = np.sign(X) * np.log1p(np.abs(X))
bearings = list(OrderedDict.fromkeys(b))
life = {n: int((b == n).sum()) for n in bearings}
frac = np.array([1 - y[i] / max(1, life[b[i]] - 1) for i in range(len(y))])
late = frac >= 0.8


def ridge(Xt, yt, lam=1.0):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    A = np.hstack([(Xt - mu) / sd, np.ones((len(Xt), 1))])
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yt)
    return lambda Xq: np.maximum(0, np.hstack([(Xq - mu) / sd, np.ones((len(Xq), 1))]) @ w)


def logistic(Xt, lt, steps=800, lr=0.5, lam=1e-2):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    A = np.hstack([(Xt - mu) / sd, np.ones((len(Xt), 1))]); w = np.zeros(A.shape[1]); t = lt.astype(float)
    for _ in range(steps):
        p = 1 / (1 + np.exp(-np.clip(A @ w, -30, 30)))
        w -= lr * (A.T @ (p - t) / len(t) + lam * w)
    return lambda Xq: 1 / (1 + np.exp(-np.clip(np.hstack([(Xq - mu) / sd, np.ones((len(Xq), 1))]) @ w, -30, 30)))


def models(tr):
    c_all = y[tr].mean()
    c_h = y[tr & ~late].mean()
    plain = ridge(T[tr], y[tr])
    capped = ridge(T[tr], np.minimum(y[tr], c_h))
    late_reg = ridge(T[tr & late], y[tr & late])
    p_late = logistic(T[tr], late[tr])
    return {
        "ridge": lambda q: plain(T[q]),
        "const": lambda q: np.full(q.sum(), c_all),
        "capped": lambda q: capped(T[q]),
        "two_stage": lambda q: (lambda p: p * late_reg(T[q]) + (1 - p) * c_h)(p_late(T[q])),
        "oracle": lambda q: np.where(late[q], late_reg(T[q]), c_h),
    }, c_h


rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
NAMES = ["ridge", "const", "capped", "two_stage", "oracle"]


def report(title, folds):
    """folds: list of (train_mask, test_mask)."""
    errs = {m: [] for m in NAMES}; per = {m: OrderedDict() for m in NAMES}; parts = {m: [[], []] for m in NAMES}
    chs = []
    for tr, te in folds:
        fitted, c_h = models(tr); chs.append(c_h)
        for m in NAMES:
            e = fitted[m](te) - y[te]
            errs[m].append(e)
            for n in OrderedDict.fromkeys(b[te]):
                sel = b[te] == n
                per[m][n] = rmse(e[sel])
            parts[m][0].append(e[~late[te]]); parts[m][1].append(e[late[te]])
    print(f"\n== {title}（健康期常数 c_h ≈ {np.mean(chs):.0f} 分钟）==")
    names = list(per["ridge"])
    print(f"{'模型':<10}{'按行RMSE':>9}{'等权RMSE':>9}{'健康段RMSE':>11}{'最后20%RMSE':>12}   " +
          " ".join(f"{n.replace('Bearing', 'B'):>6}" for n in names))
    for m in NAMES:
        allerr = np.concatenate(errs[m])
        print(f"{m:<10}{rmse(allerr):>9.0f}{np.mean(list(per[m].values())):>9.0f}"
              f"{rmse(np.concatenate(parts[m][0])):>11.0f}{rmse(np.concatenate(parts[m][1])):>12.0f}   " +
              " ".join(f"{per[m][n]:>6.0f}" for n in names))


train_mask = np.isin(b, list(OrderedDict.fromkeys(ba)))
report("(i) 官方划分：9 根训练 → 3 根验证（官方基线：按行 681，等权 448）", [(train_mask, ~train_mask)])
report("(ii) 留一根轴承，12 折", [(b != n, b == n) for n in bearings])
