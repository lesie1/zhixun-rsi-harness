"""Which features carry RUL information that is shared across bearings?

1. per feature type: does it separate the last 20 % of life from the rest, bearing by bearing (AUC)?
2. which feature types the official baseline MLP actually relies on (permutation importance on the
   3 validation bearings it never saw).
Public package data only. Usage: python -I analyze_b4.py HARNESS_DIR PACKAGE_B_DIR
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
kind = [json.loads(p)[-1][1] for p in PATHS]          # feature type, e.g. "rms"
chan = [json.loads(p)[-2][1] for p in PATHS]          # "horizontal" / "vertical"
TYPES = list(OrderedDict.fromkeys(kind))
CN = {"rms": "RMS", "peak": "绝对峰值", "peak_to_peak": "峰峰值", "mean_abs": "绝对均值", "kurtosis": "峭度",
      "skewness": "偏度", "crest_factor": "峰值因子", "shape_factor": "波形因子", "impulse_factor": "脉冲因子",
      "margin_factor": "裕度因子", "spectral_centroid": "频谱质心", "spectral_rms": "频谱RMS",
      "spectral_kurtosis": "功率谱峭度", "dominant_freq": "主频", "band1_energy_0_2k": "0-2kHz能量",
      "band2_energy_2k_6k": "2-6kHz能量", "band3_energy_6k_12k": "6-12.8kHz能量"}


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
Xv, yv, bv = load(f"{PKG}/public/validation.jsonl")
X = np.vstack([Xa, Xv]); y = np.concatenate([ya, yv]); b = np.concatenate([ba, bv])
bearings = list(OrderedDict.fromkeys(b))
life = {n: int((b == n).sum()) for n in bearings}
frac = np.array([1 - y[i] / max(1, life[b[i]] - 1) for i in range(len(y))])
late = frac >= 0.8


def auc(score, positive):
    order = np.argsort(score, kind="mergesort"); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    p, q = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - p * (p + 1) / 2) / (p * q)


print("== 1. 各类特征能否区分「寿命最后 20%」（每根轴承单独算 AUC，两个通道取较好的那个）==")
rows = []
for t in TYPES:
    cols = [i for i, k in enumerate(kind) if k == t]
    per = []
    for n in bearings:
        m = b == n
        per.append(max(auc(X[m, c], late[m]) for c in cols))
    per = np.array(per)
    rows.append((np.mean(per), t, int((per >= 0.8).sum()), per))
rows.sort(reverse=True)
print(f"{'特征':<14}{'12根平均AUC':>10}{'AUC≥0.8的轴承数':>16}   最差的三根")
for mean, t, good, per in rows:
    worst = ", ".join(f"{bearings[i].replace('Bearing', 'B')}:{per[i]:.2f}" for i in np.argsort(per)[:3])
    print(f"{CN[t]:<14}{mean:>10.2f}{good:>12}/12      {worst}")

print("\n== 2. 官方基线在 3 根验证轴承上依赖哪些特征（打乱该类特征后 RMSE 增加多少，分钟）==")
rng = np.random.default_rng(0)
rmse = lambda p, t: float(np.sqrt(np.mean((p - t) ** 2)))
base_pred = model.predict(Xv)
base_pooled = rmse(base_pred, yv)
base_eq = np.mean([rmse(base_pred[bv == n], yv[bv == n]) for n in OrderedDict.fromkeys(bv)])
imp = []
for t in TYPES:
    cols = [i for i, k in enumerate(kind) if k == t]
    pooled, eq = [], []
    for _ in range(5):
        Xp = Xv.copy(); perm = rng.permutation(len(Xv))
        Xp[:, cols] = Xv[perm][:, cols]
        p = model.predict(Xp)
        pooled.append(rmse(p, yv) - base_pooled)
        eq.append(np.mean([rmse(p[bv == n], yv[bv == n]) for n in OrderedDict.fromkeys(bv)]) - base_eq)
    imp.append((np.mean(pooled), np.mean(eq), t))
imp.sort(reverse=True)
print(f"基线：按行 RMSE {base_pooled:.0f}，三根等权 {base_eq:.0f}")
for pooled, eq, t in imp:
    print(f"  {CN[t]:<14} 按行 {pooled:+7.0f}   等权 {eq:+7.0f}")
