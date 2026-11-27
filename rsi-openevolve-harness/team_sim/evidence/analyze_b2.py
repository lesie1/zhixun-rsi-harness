"""What can the 34 B features tell apart: bearings, lifetime scale, degradation stage?

Public package data only. Usage: python -I analyze_b2.py HARNESS_DIR PACKAGE_B_DIR
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
                assert list(paths) == PATHS
                X.append(values); y.append(float(r["answer"])); b.append(r["meta"]["bearing"])
    return np.asarray(X, float), np.asarray(y), np.asarray(b)


Xa, ya, ba = load(f"{PKG}/public/train.jsonl")
Xb, yb, bb = load(f"{PKG}/public/validation.jsonl")
X = np.vstack([Xa, Xb]); y = np.concatenate([ya, yb]); b = np.concatenate([ba, bb])
Z = model.normalized(X)
bearings = list(OrderedDict.fromkeys(b))
life = {n: int((b == n).sum()) for n in bearings}
frac = np.array([1 - y[i] / max(1, life[b[i]] - 1) for i in range(len(y))])   # 0 = first snapshot, 1 = last
short = lambda n: n.replace("Bearing", "B")


def softmax_fit(Zt, labels, classes, steps=500, lr=0.5, lam=1e-3):
    Y = np.stack([(labels == c).astype(float) for c in classes], axis=1)
    A = np.hstack([Zt, np.ones((len(Zt), 1))]); W = np.zeros((A.shape[1], len(classes)))
    weight = 1.0 / np.maximum(Y.sum(axis=0), 1)
    for _ in range(steps):
        S = A @ W; P = np.exp(S - S.max(axis=1, keepdims=True)); P /= P.sum(axis=1, keepdims=True)
        W -= lr * (A.T @ ((P - Y) * weight) + lam * W)
    def predict(Zq):
        S = np.hstack([Zq, np.ones((len(Zq), 1))]) @ W
        return np.array(classes)[S.argmax(axis=1)], S
    return predict


# ------------------------------------------------------------- 1. seen bearings: is there a per-bearing signature?
print("== 1. 认得出见过的轴承吗（12 类）==")
block = np.zeros(len(y), int)
for n in bearings:
    i = np.where(b == n)[0]; i = i[np.argsort(frac[i])]
    block[i] = np.arange(len(i)) // 10
for label, train_mask in [("交错切分（每 10 个快照一块，奇偶块分开）", block % 2 == 0),
                          ("时间外推（前半段训练，后半段测试）", frac < 0.5)]:
    pred, _ = softmax_fit(Z[train_mask], b[train_mask], bearings)(Z[~train_mask])
    ok = pred == b[~train_mask]
    per = [f"{short(n)}:{ok[b[~train_mask] == n].mean():.0%}" for n in bearings]
    print(f"  {label}: 按行 {ok.mean():.0%}，按轴承平均 {np.mean([ok[b[~train_mask] == n].mean() for n in bearings]):.0%}")
    print("    " + " ".join(per))

# ------------------------------------------------------------- 2. unseen bearing: lifetime scale from healthy snapshots?
print("\n== 2. 没见过的轴承，从健康阶段（前半段）能看出寿命长短吗（留一根轴承）==")
cls = {n: ("短" if life[n] < 200 else "中" if life[n] < 1000 else "长") for n in bearings}
print("  分组：" + "  ".join(f"{k}={[short(n) for n in bearings if cls[n] == k]}" for k in "短中长"))
healthy = frac < 0.5
lab = np.array([cls[n] for n in b])
right, votes = 0, []
for n in bearings:
    hold = (b == n) & healthy; tr = (b != n) & healthy
    pred, _ = softmax_fit(Z[tr], lab[tr], list("短中长"))(Z[hold])
    right += int((pred == cls[n]).sum())
    vote = max("短中长", key=list(pred).count)
    votes.append(f"{short(n)}({cls[n]})→{vote}{'✓' if vote == cls[n] else '✗'}")
print(f"  按行准确率 {right / healthy.sum():.0%}（瞎猜约 33%）")
print("  按轴承投票：" + " ".join(votes))

# ------------------------------------------------------------- 3. unseen bearing: degradation stage?
print("\n== 3. 没见过的轴承，能看出是否进入寿命最后 20% 吗（留一根轴承，AUC）==")


def auc(score, positive):
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    p, q = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - p * (p + 1) / 2) / (p * q) if p and q else float("nan")


late = frac >= 0.8
aucs = []
for n in bearings:
    hold = b == n
    _, S = softmax_fit(Z[~hold], np.where(late[~hold], "late", "early"), ["early", "late"])(Z[hold])
    aucs.append(auc(S[:, 1] - S[:, 0], late[hold]))
print("  " + " ".join(f"{short(n)}:{a:.2f}" for n, a in zip(bearings, aucs)) + f"  | 平均 {np.nanmean(aucs):.2f}（0.5=瞎猜，1=完全分开）")

# ------------------------------------------------------------- 4. how flat is the healthy stage?
k = PATHS.index(next(p for p in PATHS if '"rms"' in p and '"horizontal"' in p))
print("\n== 4. 水平通道 RMS：前半段中位数 → 最后 10% 中位数 ==")
print("  " + "  ".join(f"{short(n)}:{np.median(X[(b == n) & (frac < 0.5), k]):.2f}→{np.median(X[(b == n) & (frac >= 0.9), k]):.2f}"
                       for n in bearings))
