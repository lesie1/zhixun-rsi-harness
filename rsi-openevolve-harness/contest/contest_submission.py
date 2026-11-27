"""智训引擎 · 工业模型自进化 Harness（比赛镜像 v1）。

平台每轮调用 train(rnd)：读本轮公开数据和父模型 → 从父模型便宜续训出几个候选 → 在公开验证集上用
配对 bootstrap 验收（候选必须以 ≥90% 的概率优于父模型且护栏不破）→ 调一次 Qwen 审阅这个决定
（Qwen 只能否决上线，不能强行上线）→ 提交候选或保留父模型 / 退回静态基线 → 把本轮摘要存进 /state。
predict(sample, model_dir) 用选中的检查点逐行预测。

v1 的取舍：OpenEvolve 搜索和 ACE 经验库不在镜像里运行（单次 Qwen 调用 1–10 分钟、平台每轮时限未知），
候选策略是团队实验里定下的固定几种：官方续训、目标截断续训（回归，B 任务的 v3 规则）、
学习率减半衰减 + 按验证集选 epoch（C 任务搜索得到的种子策略）。

v1.1（2026-11-27）把经验卡 A-00-BASE-CLASSMIX-01（原 GASMIX 卡的通用版）编译成候选：
- 类别加权（分类）：每类在损失里的总份量相同，样本少的类不被多数类淹没；权重只用训练集统计。
- 新数据加权：/state 里记下以前各轮训练集出现过的样本（id 的哈希），本轮没见过的就是新样本；
  另写一份训练文件，新样本各多复制 NEW_DATA_COPIES 份（按 1 + NEW_DATA_COPIES 条计）。第 1 轮、
  或本轮全是新样本 / 没有新样本时不训练这类候选。
新候选和原有候选一样都要过验收门，不通过就不上线。每轮的行数和新样本数写进记录（configuration.data）。
"""
import gzip
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parent
RSI = APP / "rsi"
for path in (RSI, APP):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TASKS = ("classification", "regression")
PREDICT_ARGV = ["python3", "/app/harness.py", "predict"]
ADOPT_PROBABILITY = 0.90          # P(candidate better than parent) needed to deploy it
RESAMPLES = 1000                  # paired bootstrap resamples
GUARD = {"classification": ("accuracy", 0.02), "regression": ("mae", 0.05)}   # tolerated relative worsening
PRIMARY = {"classification": "macro_f1", "regression": "rmse"}
EPOCHS = 5
MAX_EPOCHS = 30
SEED = 42
NEW_DATA_COPIES = 2               # a training row unseen in earlier rounds counts 1 + 2 = 3 times in "recent" candidates
MAX_TRAIN_BYTES = 120 * 1024 * 1024   # the reference loader refuses files over 128 MiB
CARD = "A-00-BASE-CLASSMIX-01"  # the experience card the class / new-data weighting candidates come from
SEEN_FILE = "seen_train_ids.txt.gz"


def log(message):
    print(f"[rsi] {message}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------- candidates
UNIFORM_WEIGHTS = "def sample_weights(y, task, num_classes):\n    return np.ones(len(y))\n"
CLASS_WEIGHTS = """def sample_weights(y, task, num_classes):
    if task != "classification":
        return np.ones(len(y))
    counts = np.bincount(y, minlength=num_classes).astype(float)
    w = (1.0 / np.maximum(counts, 1.0))[y]
    return w / w.mean()
"""


def class_weighted(source):
    """The same policy with class-balanced sample weights: every class gets the same total weight in the loss."""
    if source.count(UNIFORM_WEIGHTS) != 1:
        raise ValueError("policy has no uniform sample_weights to replace")
    return source.replace(UNIFORM_WEIGHTS, CLASS_WEIGHTS)


def candidate_policies(task, label_mean, target_mean, target_scale, recent_path=None):
    """(name, policy source, one-line description, training file or None). Order = training order.

    recent_path: training file in which rows unseen in earlier rounds appear 1 + NEW_DATA_COPIES times.
    """
    from team_sim.policy import render
    initial = (RSI / "harness" / "policy_initial.py").read_text(encoding="utf-8")
    pick = (RSI / "policies" / "policy_pick.py").read_text(encoding="utf-8")
    items = [("continue", initial, f"官方续训 {EPOCHS} 轮（参考做法）", None),
             ("pick", pick, f"续训 {EPOCHS} 轮，学习率线性降到一半，按公开验证集选最好的一轮", None)]
    if task == "classification":
        items += [("balanced", class_weighted(initial),
                   f"官方续训 {EPOCHS} 轮，按类别加权：每类在损失里的总份量相同（经验卡 {CARD}）", None),
                  ("pick+balanced", class_weighted(pick), f"选轮续训 + 按类别加权（经验卡 {CARD}）", None)]
    if task == "regression" and target_scale:
        items.append(("cap1.0", render(1.0, label_mean, target_mean, target_scale, epochs=EPOCHS),
                      f"续训 {EPOCHS} 轮，训练目标截断在本轮训练标签均值（{label_mean:.1f}）以内，评分仍用真实标签", None))
    if recent_path is not None:
        weight = 1 + NEW_DATA_COPIES
        if task == "classification":
            items.append(("recent+balanced", class_weighted(pick),
                          f"选轮续训 + 按类别加权 + 新样本每条按 {weight} 条计（经验卡 {CARD}）", recent_path))
        else:
            items.append(("recent+pick", pick, f"选轮续训，新样本每条按 {weight} 条计（经验卡 {CARD}）", recent_path))
    return items


# ---------------------------------------------------------------- new training rows across rounds
def row_key(identity):
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]


def load_seen(state_dir):
    path = Path(state_dir) / SEEN_FILE
    try:
        with gzip.open(path, "rt", encoding="ascii") as stream:
            return {line.strip() for line in stream if line.strip()}
    except (OSError, EOFError, UnicodeDecodeError):
        return set()


def save_seen(state_dir, keys):
    with gzip.open(Path(state_dir) / SEEN_FILE, "wt", encoding="ascii") as stream:
        stream.write("\n".join(sorted(keys)) + "\n")


def write_recent(train_path, new_keys, out_path, copies=NEW_DATA_COPIES):
    """Copy of the training file in which every new row appears 1 + copies times (copies get an id suffix,
    since sample IDs must be unique). Returns the path, or None when the file would be too large."""
    out_path = Path(out_path)
    total = Path(train_path).stat().st_size
    with open(train_path, "r", encoding="utf-8") as source, open(out_path, "w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            target.write(line if line.endswith("\n") else line + "\n")
            row = json.loads(line)
            if row_key(row["id"]) in new_keys:
                for n in range(1, copies + 1):
                    extra = json.dumps(dict(row, id=f"{row['id']}#recent{n}"), ensure_ascii=False) + "\n"
                    total += len(extra.encode("utf-8"))
                    if total > MAX_TRAIN_BYTES:
                        target.close()
                        out_path.unlink()
                        return None
                    target.write(extra)
    return str(out_path)


# ---------------------------------------------------------------- evaluation
def evaluate(task, num_classes, y, predictions):
    from harness.metrics import classification_metrics, regression_metrics
    if task == "classification":
        return classification_metrics(y, predictions, num_classes)
    return regression_metrics(y, predictions, with_nasa=False)


def better(task, a, b):
    return a > b if task == "classification" else a < b


def predict_array(checkpoint, x):
    from baselines.model import Model
    model = Model.load(str(checkpoint))
    out = model.predict(x)
    return out.argmax(axis=1) if model.meta["task"] == "classification" else out


def decide(task, num_classes, y, pool, round_index):
    """pool: {name: predictions}; "parent" must be in it. Returns (choice, table)."""
    from harness.gate import probability_better
    primary = PRIMARY[task]
    guard_name, guard_tol = GUARD[task]
    metrics = {name: evaluate(task, num_classes, y, pred) for name, pred in pool.items()}
    base = metrics["parent"]
    table = {"parent": {"metrics": metrics["parent"]}}
    eligible = []
    for name, pred in pool.items():
        if name == "parent":
            continue
        p = probability_better(task, primary, y, pred, pool["parent"], num_classes, mode="sample",
                               resamples=RESAMPLES, seed=SEED + round_index)
        g_new, g_old = metrics[name][guard_name], base[guard_name]
        if task == "classification":
            guard_ok = g_new >= g_old * (1 - guard_tol)
        else:
            guard_ok = g_new <= g_old * (1 + guard_tol)
        passes = p >= ADOPT_PROBABILITY and guard_ok and better(task, metrics[name][primary], base[primary])
        table[name] = {"metrics": metrics[name], "p_better_than_parent": round(p, 4), "guard_ok": guard_ok,
                       "passes": passes}
        if passes:
            eligible.append(name)
    if not eligible:
        return "parent", table
    best = eligible[0]
    for name in eligible[1:]:
        if better(task, metrics[name][primary], metrics[best][primary]):
            best = name
    return best, table


# ---------------------------------------------------------------- the one model call per round
def ask_model(rnd, summary):
    system = ("你是工业模型自进化系统的上线审阅者。系统已用固定规则（配对 bootstrap，候选以至少 90% 的概率"
              "优于父模型且护栏不破才上线）给出本轮决定。请审阅数据，只回复 JSON："
              '{"agree": true 或 false, "reason": "一句中文理由"}。你只能否决上线，不能要求上线规则以外的模型。')
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(summary, ensure_ascii=False, default=str)}]
    limit = max(1, min(400, int(rnd.max_output_tokens)))
    try:
        receipt = rnd.model(messages, max_tokens=limit, temperature=0, enable_thinking=False)
    except Exception as error:      # option rejected or provider error: retry once with plain options
        if "OUTCOME_UNKNOWN" in str(getattr(error, "code", "")) or "OUTCOME_UNKNOWN" in str(error):
            raise
        log(f"model call with enable_thinking=False failed ({type(error).__name__}: {error}); retrying plain")
        receipt = rnd.model(messages, max_tokens=limit, temperature=0)
    text = receipt.text or ""
    verdict = {"agree": True, "reason": "", "raw": text[:500], "call_id": receipt.call_id}
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        parsed = json.loads(text[start:end])
        verdict["agree"] = bool(parsed.get("agree", True))
        verdict["reason"] = str(parsed.get("reason", ""))[:300]
    except (ValueError, json.JSONDecodeError):
        verdict["reason"] = text.strip()[:300]
    return verdict


# ---------------------------------------------------------------- one round
def train(rnd):
    import numpy as np
    from baselines.model import Model
    from harness import runner
    from harness.dataio import load_dataset

    started = time.time()
    wall = float(rnd.wall_seconds) if rnd.wall_seconds else 1800.0
    train_budget = started + 0.5 * wall
    work = Path(os.environ.get("CONTEST_WORK", "/workspace")) / f"rsi-round-{rnd.round_index:02d}"
    work.mkdir(parents=True, exist_ok=True)
    log(f"dataset {rnd.dataset_key} round {rnd.round_index} task {rnd.task.get('task')} wall {wall:.0f}s")

    try:
        history = rnd.history()
    except Exception as error:
        log(f"history unavailable: {error}")
        history = []
    state_dir = rnd.state_workspace(work / "state")
    memory_path = state_dir / "memory.json"
    try:
        memory = json.loads(memory_path.read_text(encoding="utf-8")) if memory_path.is_file() else {}
    except (OSError, json.JSONDecodeError):
        memory = {}
    static_id = memory.get("static_artifact_id") or rnd.parent.artifact_id

    previous_root = os.environ.get("CONTEST_PREVIOUS", "/previous")
    parent_rel = rnd.parent.descriptor["checkpoint_paths"][0]
    parent_ckpt = rnd.parent.path(parent_rel, root=previous_root)
    parent = Model.load(str(parent_ckpt))
    task, classes = parent.meta["task"], parent.meta["classes"]
    num_classes = len(classes)
    train_path, validation_path = rnd.data.train_path, rnd.data.validation_path
    cache = work / "cache"
    tr = load_dataset(train_path, task, classes, parent.meta["paths"], cache_dir=cache)
    va = load_dataset(validation_path, task, classes, parent.meta["paths"], cache_dir=cache)
    label_mean = float(np.mean(tr["y"])) if task == "regression" else 0.0

    # New training rows = rows whose id did not appear in the training data of earlier rounds (/state).
    keys = [row_key(identity) for identity in tr["ids"]]
    seen = load_seen(state_dir)
    new_mask = np.array([key not in seen for key in keys]) if seen else np.zeros(len(keys), dtype=bool)
    new_keys = {key for key, new in zip(keys, new_mask) if new}
    recent_path = None
    if seen and 0 < len(new_keys) < len(keys):
        try:
            recent_path = write_recent(train_path, new_keys, work / "train_recent.jsonl")
        except Exception as error:
            log(f"new-data training file failed: {type(error).__name__}: {error}")
    data = {"train_rows": len(keys), "train_new_rows": int(new_mask.sum()) if seen else None,
            "validation_rows": int(len(va["y"])), "recent_candidates": recent_path is not None}
    if task == "regression":
        data["train_label_mean"] = round(label_mean, 3)
        if new_mask.any():
            data["new_label_mean"] = round(float(np.mean(tr["y"][new_mask])), 3)
    save_seen(state_dir, seen | set(keys))
    log(f"rows train {data['train_rows']} (new {data['train_new_rows']}) validation {data['validation_rows']}")

    pool = {"parent": predict_array(parent_ckpt, va["x"])}
    checkpoints = {"parent": parent_ckpt}
    notes = {"parent": "父模型（上一轮选定的模型）"}
    for name, source, note, train_file in candidate_policies(task, label_mean, parent.meta["target_mean"],
                                                             parent.meta["target_scale"], recent_path=recent_path):
        if time.time() > train_budget:
            log(f"time budget reached; skip candidate {name}")
            break
        policy_path = work / "policies" / f"{name}.py"
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(source, encoding="utf-8")
        out = work / "candidates" / name
        try:
            runner.train(str(policy_path), str(parent_ckpt), train_file or train_path, validation_path, str(out),
                         stage_id=f"r{rnd.round_index:02d}-{name}", max_epochs=MAX_EPOCHS, cache_dir=str(cache))
            checkpoints[name] = out / "checkpoint.npz"
            pool[name] = predict_array(checkpoints[name], va["x"])
            notes[name] = note
        except Exception as error:
            log(f"candidate {name} failed: {type(error).__name__}: {error}")

    if static_id != rnd.parent.artifact_id and any(a.get("artifact_id") == static_id for a in rnd.artifacts):
        try:
            static_root = rnd.restore_artifact(static_id, work / "static")
            static_desc = next(a for a in rnd.artifacts if a.get("artifact_id") == static_id)["artifact"]
            checkpoints["static"] = static_root / static_desc["checkpoint_paths"][0]
            pool["static"] = predict_array(checkpoints["static"], va["x"])
            notes["static"] = "静态基线（第 1 轮的官方模型），用于回滚"
        except Exception as error:
            log(f"static baseline unavailable: {type(error).__name__}: {error}")

    choice, table = decide(task, num_classes, va["y"], pool, rnd.round_index)
    primary = PRIMARY[task]
    summary = {"dataset": rnd.dataset_key, "round": rnd.round_index, "task": task, "primary_metric": primary,
               "validation_rows": int(len(va["y"])), "train_rows": int(len(tr["y"])),
               "train_new_rows": data["train_new_rows"],
               "rule": "候选以 ≥90% 概率优于父模型、护栏不破且点估计更好才上线；否则保留父模型",
               "candidates": {name: {"what": notes[name], **{k: (round(v, 6) if isinstance(v, float) else v)
                                                            for k, v in table[name].items() if k != "metrics"},
                                     primary: round(float(table[name]["metrics"][primary]), 6)}
                              for name in table},
               "rule_decision": choice,
               "recent_rounds": memory.get("rounds", [])[-3:]}
    verdict = ask_model(rnd, summary)
    final = choice
    if choice != "parent" and not verdict["agree"]:
        final = "parent"
        log(f"model vetoed deploying {choice}: {verdict['reason']}")

    if final == "parent":
        selected = rnd.parent.artifact_id
        change = "保留父模型，本轮不上线新模型。"
    elif final == "static":
        selected = static_id
        change = "回滚到静态基线（第 1 轮的官方模型）。"
    else:
        submitted = rnd.submit_candidate(
            checkpoints[final].parent,
            hypothesis=f"候选 {final} 在本轮公开验证集上以 {table[final]['p_better_than_parent']:.0%} 的概率优于父模型，预计在下一阶段也更好。",
            change_summary=notes[final],
            configuration={"candidate": final, "epochs": EPOCHS, "adopt_probability": ADOPT_PROBABILITY},
            seed=SEED,
            checkpoint_paths=["checkpoint.npz"],
            code_paths=[],
            predict_argv=PREDICT_ARGV,
        )
        selected = submitted["artifact_id"]
        change = notes[final]
        log(f"platform measurement for {final}: {json.dumps(submitted.get('summary'), ensure_ascii=False)[:500]}")

    record = {"round": rnd.round_index, "choice": final, "rule_decision": choice, "selected": selected,
              "primary": primary, "metrics": {n: round(float(t["metrics"][primary]), 6) for n, t in table.items()},
              "model_agree": verdict["agree"], "seconds": round(time.time() - started, 1), "data": data}
    memory.setdefault("rounds", []).append(record)
    memory["rounds"] = memory["rounds"][-20:]
    memory["static_artifact_id"] = static_id
    memory_path.write_text(json.dumps(memory, ensure_ascii=False, indent=1), encoding="utf-8")
    rnd.save_state(state_dir)

    reason = (f"规则决定：{choice}；最终选择：{final}。{primary}：" +
              "，".join(f"{n} {float(t['metrics'][primary]):.4f}" for n, t in table.items()) +
              f"。Qwen 审阅：{'同意' if verdict['agree'] else '否决'}，{verdict['reason'] or '无补充'}")
    rnd.finish(
        selected_artifact_id=selected,
        hypothesis="只上线在公开验证集上有把握更好的模型；没把握就保留父模型，父模型不如静态基线时回滚。",
        change_summary=change,
        reason=reason[:3900],
        configuration={"candidates": sorted(pool), "choice": final, "rule_decision": choice,
                       "adopt_probability": ADOPT_PROBABILITY, "resamples": RESAMPLES, "epochs": EPOCHS,
                       "new_data_weight": 1 + NEW_DATA_COPIES, "data": data},
        seed=SEED,
        history_ids=[item["id"] for item in history],
        model_call_ids=list(rnd.call_ids),
    )
    log(f"round done in {time.time() - started:.1f}s: {final}")


# ---------------------------------------------------------------- prediction
_MODEL = {}


def _load(model_dir):
    if "model" not in _MODEL:
        from baselines.model import Model
        path = os.environ.get("CONTEST_ARTIFACT_PATH") or str(Path(model_dir) / "checkpoint.npz")
        model = Model.load(path)
        _MODEL.update(model=model, paths=list(model.meta["paths"]))
    return _MODEL["model"], _MODEL["paths"]


def predict(sample, model_dir):
    import numpy as np
    from baselines.data import flatten
    model, paths = _load(model_dir)
    names, values = flatten(sample.inputs)
    lookup = dict(zip(names, values))
    x = np.array([[lookup[p] for p in paths]], dtype=np.float64)
    out = model.predict(x)
    if model.meta["task"] == "classification":
        probs = [float(v) for v in out[0]]
        top = max(range(len(probs)), key=lambda i: probs[i])
        if sum(1 for p in probs if p == probs[top]) > 1:            # the SDK needs one unique highest score
            probs = [p * 0.999999 if i != top and p == probs[top] else p for i, p in enumerate(probs)]
        return [{"label": label, "score": min(1.0, max(0.0, p))} for label, p in zip(model.meta["classes"], probs)]
    value = float(out[0])
    return {"value": value if math.isfinite(value) and value > 0 else 0.0}
