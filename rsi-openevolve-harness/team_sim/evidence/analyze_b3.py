"""Do amplitude features carry the bearing's identity, and do shape features generalize better?

Public package data only. Usage: python -I analyze_b3.py HARNESS_DIR PACKAGE_B_DIR
"""
import json
import sys
from collections import OrderedDict

import numpy as np

HARNESS, PKG = sys.argv[1], sys.argv[2]
sys.path.insert(0, HARNESS)
from baselines.model import Model  # noqa: E402
from baselines.data import flatten  # noqa: E402

model = Model.load(f"{PKG}/checkpoint.npz")
PATHS = list(model.meta["paths"])


def load(path):
    X, y, b = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                paths, values = flatten(r["inputs"])
                X.append(values); y.append(float(r["answer"])); b.append(r["meta"]["bearing"])
    return np.asarray(X, float), np.asarray(y), np.asarray(b)


Xa, ya, ba = load(f"{PKG}/public/train.jsonl")
Xb, yb, bb = load(f"{PKG}/public/validation.jsonl")
X = np.vstack([Xa, Xb]); y = np.concatenate([ya, yb]); b = np.concatenate([ba, bb])
bearings = list(OrderedDict.fromkeys(b))
name = [json.loads(p)[-1][1] for p in PATHS]
AMPLITUDE = {"rms", "peak", "peak_to_peak", "mean_abs", "spectral_rms",
             "band1_energy_0_2k", "band2_energy_2k_6k", "band3_energy_6k_12k"}
groups = {
    "全部 34 个": np.arange(len(PATHS)),
    "形状+频率 18 个": np.array([i for i, n in enumerate(name) if n not in AMPLITUDE]),
    "幅值 16 个": np.array([i for i, n in enumerate(name) if n in AMPLITUDE]),
}
# signed log for heavy-tailed values, then z-score: a fair, simple preprocessing for every group
T = np.sign(X) * np.log1p(np.abs(X))
rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))


def ridge(Xt, yt, Xq, lam=1.0):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    A = np.hstack([(Xt - mu) / sd, np.ones((len(Xt), 1))])
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yt)
    return np.maximum(0, np.hstack([(Xq - mu) / sd, np.ones((len(Xq), 1))]) @ w)


def centroid_id(Xt, bt, Xq):
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
    C = np.stack([((Xt[bt == n] - mu) / sd).mean(0) for n in bearings])
    Q = (Xq - mu) / sd
    return np.array(bearings)[((Q[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)]


order_in_bearing = np.zeros(len(y), int)
for n in bearings:
    i = np.where(b == n)[0]; i = i[np.argsort(-y[i])]
    order_in_bearing[i] = np.arange(len(i)) // 10
even = order_in_bearing % 2 == 0

print(f"{'特征组':<16}{'认出见过的轴承(最近质心)':>20}{'留一根轴承 RUL 线性回归: 12根等权RMSE':>34}")
for label, cols in groups.items():
    F = T[:, cols]
    ident = (centroid_id(F[even], b[even], F[~even]) == b[~even]).mean()
    per = [rmse(ridge(F[b != n], y[b != n], F[b == n]) - y[b == n]) for n in bearings]
    print(f"{label:<16}{ident:>18.0%}{np.mean(per):>26.0f}   " + " ".join(f"{p:.0f}" for p in per))
print("（逐轴承顺序：" + " ".join(n.replace('Bearing', 'B') for n in bearings) + "）")
