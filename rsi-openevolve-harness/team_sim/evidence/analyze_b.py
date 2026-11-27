"""Does mixing the three XJTU-SY operating conditions hurt the B baseline?

Public package data only (train = _1.._3, validation = _4 of each condition).
Usage: python -I analyze_b.py HARNESS_DIR PACKAGE_B_DIR
"""
import json
import sys
from collections import OrderedDict

import numpy as np

HARNESS, PKG = sys.argv[1], sys.argv[2]
sys.path.insert(0, HARNESS)
from baselines.model import Model  # noqa: E402  (the team's copy of the official code)
from baselines.data import flatten  # noqa: E402

model = Model.load(f"{PKG}/checkpoint.npz")
PATHS = list(model.meta["paths"])


def load(path):
    X, y, b = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            paths, values = flatten(r["inputs"])
            assert list(paths) == PATHS
            X.append(values)
            y.append(float(r["answer"]))
            b.append(r["meta"]["bearing"])
    return np.asarray(X, float), np.asarray(y), np.asarray(b)


Xt, yt, bt = load(f"{PKG}/public/train.jsonl")
Xv, yv, bv = load(f"{PKG}/public/validation.jsonl")
X = np.vstack([Xt, Xv]); y = np.concatenate([yt, yv]); b = np.concatenate([bt, bv])
split = np.array(["train"] * len(yt) + ["val"] * len(yv))
cond = np.array([name[7] for name in b])          # "Bearing2_3" -> "2"
bearings = list(OrderedDict.fromkeys(b))
rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))

# ---------------------------------------------------------------- 1. baseline errors per bearing
pred = model.predict(X)
print("== 1. 官方基线逐轴承误差（分钟；train 轴承是训练内，val 是没见过的）==")
print(f"{'轴承':<11}{'集合':<6}{'寿命':>6}{'RMSE':>8}{'偏差':>8}  前半段:真值均值/预测均值/预测标准差   最后10%:真值/预测")
for name in bearings:
    i = np.where(b == name)[0]
    i = i[np.argsort(-y[i])]                       # time order: RUL counts down
    n = len(i); first = i[: n // 2]; last = i[int(n * 0.9):]
    e = pred[i] - y[i]
    print(f"{name:<11}{split[i[0]]:<6}{n:>6}{rmse(e):>8.0f}{e.mean():>+8.0f}  "
          f"{y[first].mean():>8.0f} / {pred[first].mean():>6.0f} / {pred[first].std():>5.0f}"
          f"        {y[last].mean():>5.0f} / {pred[last].mean():>5.0f}")
v = split == "val"
print(f"验证集整体 RMSE {rmse(pred[v] - y[v]):.0f}；三根等权平均 RMSE "
      f"{np.mean([rmse(pred[(b == n)] - y[(b == n)]) for n in bearings if split[b == n][0] == 'val']):.0f}")

# ---------------------------------------------------------------- 2. can the condition be read from features?
Z = model.normalized(X)
print("\n== 2. 只看振动特征能不能认出工况（最近质心，留一根轴承）==")
correct_rows = 0
line = []
for name in bearings:
    hold = b == name
    cents = {c: Z[(~hold) & (cond == c)].mean(axis=0) for c in "123"}
    keys = sorted(cents)
    d = np.stack([((Z[hold] - cents[c]) ** 2).sum(axis=1) for c in keys], axis=1)
    guess = np.array(keys)[d.argmin(axis=1)]
    acc = float((guess == cond[hold]).mean()); correct_rows += int((guess == cond[hold]).sum())
    vote = max(set(guess), key=list(guess).count)
    line.append(f"{name}:{acc:.0%}(投票→工况{vote})")
print("  " + "  ".join(line))
print(f"  按行总体准确率 {correct_rows / len(y):.0%}")
k = PATHS.index(next(p for p in PATHS if '"dominant_freq"' in p and '"horizontal"' in p))
for c in "123":
    print(f"  工况{c} 水平通道主频中位数 {np.median(X[cond == c, k]):.0f} Hz，"
          f"寿命 {[int((b == n).sum()) for n in bearings if n[7] == c]}")

# stronger classifiers, still leave-one-bearing-out
def softmax_fit(Za, ca, steps=400, lr=0.5, lam=1e-3):
    Y = np.stack([(ca == c).astype(float) for c in "123"], axis=1)
    A = np.hstack([Za, np.ones((len(Za), 1))]); W = np.zeros((A.shape[1], 3))
    weight = 1.0 / Y.sum(axis=0)                     # balance the classes
    for _ in range(steps):
        P = np.exp(A @ W - (A @ W).max(axis=1, keepdims=True)); P /= P.sum(axis=1, keepdims=True)
        W -= lr * (A.T @ ((P - Y) * weight) + lam * W)
    return lambda Zb: np.array(list("123"))[(np.hstack([Zb, np.ones((len(Zb), 1))]) @ W).argmax(axis=1)]


for label, make in [("15-近邻", None), ("逻辑回归", softmax_fit)]:
    right, line = 0, []
    for name in bearings:
        hold = b == name
        if make is None:
            Zt, ct = Z[~hold], cond[~hold]
            guess = []
            for s in range(0, hold.sum(), 512):
                q = Z[hold][s:s + 512]
                d = (q ** 2).sum(1)[:, None] - 2 * q @ Zt.T + (Zt ** 2).sum(1)[None, :]
                nn = np.argpartition(d, 15, axis=1)[:, :15]
                guess += [max("123", key=list(row).count) for row in ct[nn]]
            guess = np.array(guess)
        else:
            guess = make(Z[~hold], cond[~hold])(Z[hold])
        ok = guess == cond[hold]; right += int(ok.sum())
        line.append(f"{name.replace('Bearing', 'B')}:{ok.mean():.0%}")
    print(f"  {label}: " + " ".join(line) + f"  | 按行总体 {right / len(y):.0%}")

# ---------------------------------------------------------------- 3. leave-one-bearing-out, simple models
print("\n== 3. 留一根轴承交叉验证（12 折，简单模型，只看趋势）==")


def ridge_fit(Xa, ya, lam=1.0):
    mu, sd = Xa.mean(axis=0), Xa.std(axis=0) + 1e-9
    A = np.hstack([(Xa - mu) / sd, np.ones((len(Xa), 1))])
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ya)
    return lambda Xb: np.maximum(0, np.hstack([(Xb - mu) / sd, np.ones((len(Xb), 1))]) @ w)


onehot = np.stack([(cond == c).astype(float) for c in "123"], axis=1)
names = ["常数(不分工况)", "常数(按工况)", "线性(不分工况)", "线性+工况编码", "线性(每工况单独)"]
table = {m: {} for m in names}
for name in bearings:
    hold = b == name; tr = ~hold; c = cond[hold][0]; same = tr & (cond == c)
    preds = {
        names[0]: np.full(hold.sum(), y[tr].mean()),
        names[1]: np.full(hold.sum(), y[same].mean()),
        names[2]: ridge_fit(X[tr], y[tr])(X[hold]),
        names[3]: ridge_fit(np.hstack([X[tr], onehot[tr]]), y[tr])(np.hstack([X[hold], onehot[hold]])),
        names[4]: ridge_fit(X[same], y[same])(X[hold]),
    }
    for m in names:
        table[m][name] = rmse(preds[m] - y[hold])
print(f"{'模型':<16}" + "".join(f"{n.replace('Bearing', 'B'):>7}" for n in bearings) + f"{'等权均值':>9}")
for m in names:
    print(f"{m:<16}" + "".join(f"{table[m][n]:>7.0f}" for n in bearings) + f"{np.mean(list(table[m].values())):>9.0f}")
