"""Can healthy-stage features give a useful reference range for the remaining life?

(a) per bearing: lifetime vs median feature values over the first half of life
(b) leave one bearing out: for each healthy snapshot, the RUL range of its 50 most similar healthy
    snapshots from the other bearings, compared with the true RUL
Public package data only. Usage: python -I analyze_b5.py HARNESS_DIR PACKAGE_B_DIR
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
kind = [json.loads(p)[-1][1] for p in PATHS]
chan = [json.loads(p)[-2][1] for p in PATHS]


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
healthy = np.array([y[i] >= (life[b[i]] - 1) / 2 for i in range(len(y))])     # first half of life


def col(k, c="horizontal"):
    return next(i for i in range(len(PATHS)) if kind[i] == k and chan[i] == c)


def spearman(u, v):
    ru, rv = np.argsort(np.argsort(u)), np.argsort(np.argsort(v))
    return float(np.corrcoef(ru, rv)[0, 1])


print("== (a) 健康阶段（前半段）特征中位数 vs 寿命，按寿命排序（水平通道）==")
show = [("rms", "RMS"), ("kurtosis", "峭度"), ("crest_factor", "峰值因子"),
        ("band3_energy_6k_12k", "6-12.8kHz能量"), ("spectral_centroid", "频谱质心(Hz)")]
print(f"{'轴承':<8}{'寿命':>6}" + "".join(f"{lab:>14}" for _, lab in show))
for n in sorted(bearings, key=lambda n: life[n]):
    m = (b == n) & healthy
    print(f"{n.replace('Bearing', 'B'):<8}{life[n]:>6}" + "".join(f"{np.median(X[m, col(k)]):>14.3g}" for k, _ in show))
corr = []
for i in range(len(PATHS)):
    med = [np.median(X[(b == n) & healthy, i]) for n in bearings]
    corr.append((abs(spearman(med, [life[n] for n in bearings])), spearman(med, [life[n] for n in bearings]), kind[i], chan[i]))
corr.sort(reverse=True)
print("与寿命的秩相关（12 根轴承，|ρ| 最大的 5 个；|ρ|≥0.58 才算 5% 水平显著）：")
print("  " + "  ".join(f"{k}/{c[0]}:{r:+.2f}" for _, r, k, c in corr[:5]))

print("\n== (b) 留一根轴承：找其他轴承里最像的 50 个健康快照，看它们的剩余寿命范围 ==")
T = np.sign(X) * np.log1p(np.abs(X))
print(f"{'轴承':<8}{'真值(健康期均值)':>14}{'相似快照RUL 10%~90%':>24}{'真值落在范围内':>14}{'中位数-真值':>12}")
truth_mean, guess_mean = [], []
for n in bearings:
    q = (b == n) & healthy; r = (b != n) & healthy
    mu, sd = T[r].mean(0), T[r].std(0) + 1e-9
    Q, R = (T[q] - mu) / sd, (T[r] - mu) / sd
    d = (Q ** 2).sum(1)[:, None] - 2 * Q @ R.T + (R ** 2).sum(1)[None, :]
    nn = np.argpartition(d, 50, axis=1)[:, :50]
    ry = y[r][nn]
    lo, mid, hi = np.percentile(ry, 10, axis=1), np.median(ry, axis=1), np.percentile(ry, 90, axis=1)
    inside = ((y[q] >= lo) & (y[q] <= hi)).mean()
    truth_mean.append(y[q].mean()); guess_mean.append(mid.mean())
    print(f"{n.replace('Bearing', 'B'):<8}{y[q].mean():>14.0f}{np.median(lo):>14.0f} ~ {np.median(hi):<8.0f}{inside:>12.0%}{mid.mean() - y[q].mean():>+12.0f}")
print(f"相似快照给出的估计 vs 真实寿命尺度（12 根轴承的秩相关）：{spearman(guess_mean, truth_mean):+.2f}")
