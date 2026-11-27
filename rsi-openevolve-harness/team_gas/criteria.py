"""One rule for "the targeted class recovered without breaking the others", used everywhere.

Inputs are per-class dicts {class: {"recall", "precision", "support"}} measured on the same
data for a baseline and a candidate. The candidate passes when
    recall(D) rose by at least `gain`,
    the confusion class C lost at most `confusion_guard` of recall AND of precision
        (catches "D recovered but C is now mislabelled as D"),
    every other class with samples lost at most `other_guard` of recall.
"""


def _value(stats, key):
    return None if stats is None else stats.get(key)


def targeted(base, cand, degraded, confusion, gain, confusion_guard, other_guard):
    checks, notes = {}, []
    d_base, d_cand = _value(base.get(degraded), "recall") or 0.0, _value(cand.get(degraded), "recall") or 0.0
    checks["degraded_gain"] = d_cand - d_base
    checks["degraded_ok"] = checks["degraded_gain"] >= gain
    c_recall_drop = (_value(base.get(confusion), "recall") or 0.0) - (_value(cand.get(confusion), "recall") or 0.0)
    base_p, cand_p = _value(base.get(confusion), "precision"), _value(cand.get(confusion), "precision")
    c_precision_drop = (base_p - (cand_p or 0.0)) if base_p is not None else 0.0
    checks.update({"confusion_recall_drop": c_recall_drop, "confusion_precision_drop": c_precision_drop})
    checks["confusion_ok"] = c_recall_drop <= confusion_guard and c_precision_drop <= confusion_guard
    others = {}
    for cls, stats in base.items():
        if cls in (degraded, confusion) or not stats.get("support"):
            continue
        drop = (stats.get("recall") or 0.0) - (_value(cand.get(cls), "recall") or 0.0)
        others[cls] = drop
    checks["other_recall_drops"] = others
    checks["others_ok"] = all(drop <= other_guard for drop in others.values())
    if checks["degraded_ok"] and not checks["confusion_ok"]:
        notes.append(f"退化类 {degraded} 恢复了，但混淆类 {confusion} 被大量误判"
                     f"（召回 {-c_recall_drop:+.2f}，精确率 {-c_precision_drop:+.2f}）")
    bad_others = [c for c, d in others.items() if d > other_guard]
    if bad_others:
        notes.append(f"其他类别召回下降超过护栏：{bad_others}")
    checks["ok"] = checks["degraded_ok"] and checks["confusion_ok"] and checks["others_ok"]
    checks["notes"] = notes
    return checks


def per_class_from_report(report):
    """Per-class dict from a harness evaluator report (selection split)."""
    return {row["class"]: {"recall": row.get("recall"), "precision": row.get("precision"),
                           "support": row.get("support"), "f1": row.get("f1")}
            for row in (report.get("slices") or {}).get("per_class", [])}


def per_class_from_profile(profile):
    return {cls: {"recall": v["recall"], "precision": v["precision"], "support": v["support"], "f1": v["f1"]}
            for cls, v in profile["classes"].items()}
