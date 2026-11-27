"""Which class degraded on the newly arrived batch, and what is it mistaken for?

The same deployed checkpoint is scored on reference data (what it was known to handle) and
on the new labelled batch. Per class we compare recall and precision; classes come from the
checkpoint, nothing is hard-coded. A class is

    degraded               recall (or precision) fell by at least the configured amount, both
                           sides have at least `min_support` samples, and a bootstrap over the
                           new samples says the fall is real (P >= confidence): "性能退化，疑似漂移"
    insufficient_evidence  the fall is there but one side has too few samples: "样本太少，证据不足"
    stable                 otherwise

For a recall-degraded class D the confusion direction is the class C that receives most of
D's new errors ("真实 D 经常被判为 C", with counts and shares). The overall check reuses
team_loop.trigger (Macro-F1 / guard drop with bootstrap), so a single-class failure is
reported even when the overall score looks stable.
"""
import numpy as np

from baselines.model import Model
from harness.dataio import load_dataset
from harness.metrics import classification_metrics
from team_loop import trigger as loop_trigger


def profile(checkpoint, path, cache_dir=None):
    model = Model.load(checkpoint)
    classes = model.meta["classes"]
    data = load_dataset(path, model.meta["task"], classes, model.meta["paths"], cache_dir=cache_dir)
    predicted = model.predict(data["x"]).argmax(axis=1)
    return _from_predictions(classes, data["y"], predicted)


def _from_predictions(classes, expected, predicted):
    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    np.add.at(matrix, (expected, predicted), 1)
    per = {}
    for i, name in enumerate(classes):
        support, guessed, hits = int(matrix[i].sum()), int(matrix[:, i].sum()), int(matrix[i, i])
        recall = hits / support if support else None
        precision = hits / guessed if guessed else None
        f1 = 2 * hits / (support + guessed) if support + guessed else None
        per[name] = {"support": support, "predicted": guessed, "correct": hits,
                     "recall": recall, "precision": precision, "f1": f1}
    overall = classification_metrics(expected, predicted, n)
    return {"rows": int(len(expected)), "classes": per, "confusion": matrix.tolist(), "class_order": list(classes),
            "macro_f1": overall["macro_f1"], "accuracy": overall["accuracy"],
            "_expected": expected, "_predicted": predicted}


def _p_worse(hits, total, reference_rate, resamples, rng):
    """P(the new rate is below the reference rate) by bootstrap over the new samples."""
    if total == 0 or reference_rate is None:
        return None
    sample = np.zeros(total)
    sample[:hits] = 1
    draws = rng.integers(0, total, size=(resamples, total))
    return float(np.mean(sample[draws].mean(axis=1) < reference_rate))


def compare(reference, new, params, display=lambda c: c):
    """Pure comparison of two profiles (both from the same checkpoint)."""
    rng = np.random.default_rng(params.get("seed", 42))
    order = new["class_order"]
    matrix = np.asarray(new["confusion"])
    classes, alarms, insufficient = {}, [], []
    for i, cls in enumerate(order):
        ref, cur = reference["classes"][cls], new["classes"][cls]
        entry = {"reference": {k: ref[k] for k in ("support", "recall", "precision")},
                 "new": {k: cur[k] for k in ("support", "predicted", "correct", "recall", "precision", "f1")},
                 "recall_drop": None, "precision_drop": None, "status": "stable"}
        enough = ref["support"] >= params["min_support"] and cur["support"] >= params["min_support"]
        if ref["recall"] is not None and cur["recall"] is not None:
            entry["recall_drop"] = ref["recall"] - cur["recall"]
        if ref["precision"] is not None and cur["precision"] is not None:
            entry["precision_drop"] = ref["precision"] - cur["precision"]
        kinds = []
        if entry["recall_drop"] is not None and entry["recall_drop"] >= params["recall_drop"]:
            kinds.append("recall")
        if entry["precision_drop"] is not None and entry["precision_drop"] >= params["precision_drop"]:
            kinds.append("precision")
        if kinds and not enough:
            entry["status"] = "insufficient_evidence"
            insufficient.append({"class": cls, "kinds": kinds, "reference_support": ref["support"],
                                 "new_support": cur["support"],
                                 "text": f"{display(cls)}：指标下降，但样本太少（参照 {ref['support']}，新批次 {cur['support']}，"
                                         f"至少需要 {params['min_support']}），证据不足"})
        elif kinds:
            p_recall = _p_worse(cur["correct"], cur["support"], ref["recall"], params["resamples"], rng) if "recall" in kinds else None
            p_precision = (_p_worse(cur["correct"], cur["predicted"], ref["precision"], params["resamples"], rng)
                           if "precision" in kinds else None)
            entry["p_worse"] = {"recall": p_recall, "precision": p_precision}
            confirmed = [k for k, p in (("recall", p_recall), ("precision", p_precision)) if p is not None and p >= params["confidence"]]
            entry["status"] = "degraded" if confirmed else "insufficient_evidence"
            if not confirmed:
                insufficient.append({"class": cls, "kinds": kinds, "text": f"{display(cls)}：指标下降但统计上不显著，证据不足"})
            for kind in confirmed:
                alarm = {"class": cls, "kind": kind, "status": "degraded"}
                if kind == "recall":
                    errors = matrix[i].copy()
                    errors[i] = 0
                    j = int(errors.argmax())
                    wrong = int(errors.sum())
                    alarm.update({"recall_reference": ref["recall"], "recall_new": cur["recall"],
                                  "support_new": cur["support"], "errors_new": wrong})
                    if wrong and errors[j] / wrong >= params["min_confusion_share"]:
                        alarm.update({"confused_as": order[j], "confused_count": int(errors[j]),
                                      "share_of_class": errors[j] / cur["support"], "share_of_errors": errors[j] / wrong})
                        alarm["text"] = (f"真实 {display(cls)} 在新批次中 {int(errors[j])}/{cur['support']}"
                                         f"（{errors[j] / cur['support']:.1%}）被判为 {display(order[j])}；"
                                         f"召回 {ref['recall']:.2f} → {cur['recall']:.2f}")
                    else:
                        alarm["text"] = (f"真实 {display(cls)} 召回 {ref['recall']:.2f} → {cur['recall']:.2f}，"
                                         f"错误分散到多个类别（{wrong} 个错误）")
                else:
                    column = matrix[:, i].copy()
                    column[i] = 0
                    j = int(column.argmax())
                    alarm.update({"precision_reference": ref["precision"], "precision_new": cur["precision"],
                                  "false_positives_new": int(column.sum()), "mostly_from": order[j] if column.sum() else None})
                    alarm["text"] = (f"被判为 {display(cls)} 的样本里有 {int(column.sum())}/{cur['predicted']} 是错的"
                                     + (f"，主要来自真实 {display(order[j])}（{int(column[j])} 个）" if column.sum() else "")
                                     + f"；精确率 {ref['precision']:.2f} → {cur['precision']:.2f}")
                alarms.append(alarm)
        classes[cls] = entry
    recall_alarms = [a for a in alarms if a["kind"] == "recall" and a.get("confused_as")]
    pair = None
    if recall_alarms:
        top = max(recall_alarms, key=lambda a: (a["recall_reference"] - a["recall_new"], a["support_new"]))
        pair = {"degraded": top["class"], "confusion": top["confused_as"], "count": top["confused_count"],
                "support": top["support_new"], "share_of_class": top["share_of_class"]}
    status = "degraded" if alarms else ("insufficient_evidence" if insufficient else "stable")
    return {"status": status, "status_text": {"degraded": "性能退化，疑似漂移", "insufficient_evidence": "样本太少或不显著，证据不足",
                                              "stable": "未发现单类退化"}[status],
            "macro_f1": {"reference": reference["macro_f1"], "new": new["macro_f1"]},
            "classes": classes, "alarms": alarms, "insufficient": insufficient, "pair": pair,
            "confusion_new": new["confusion"], "class_order": order}


def diagnose(checkpoint, reference_path, new_path, params, spec, display=lambda c: c, cache_dir=None):
    reference = profile(checkpoint, reference_path, cache_dir)
    new = profile(checkpoint, new_path, cache_dir)
    result = compare(reference, new, params, display)
    reference_level = loop_trigger.profile(checkpoint, reference_path, spec, cache_dir=cache_dir)
    overall = loop_trigger.check(checkpoint, new_path, reference_level, spec, {"seed": params.get("seed", 42)},
                                 cache_dir=cache_dir)
    result["overall_check"] = {"retrain": overall["retrain"], "reasons": overall["reasons"],
                               "metrics_new": overall["metrics"]}
    if result["status"] == "stable" and overall["retrain"]:
        result["status"], result["status_text"] = "overall_degraded", "整体指标下降，但没有定位到单个类别"
    return result
