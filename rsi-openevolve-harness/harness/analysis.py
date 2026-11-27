"""Detailed evaluation of one checkpoint on one data split.

Produces the "evaluation bundle" used by scoring (headline numbers), by the feedback
report (slices, internals, worst samples) and by the gate. Predictions always come from
the reference `Model.predict`; meta fields are used here for breakdowns only and never
reach the training code.
"""
import numpy as np

from .groups import group_labels
from .metrics import (classification_metrics, confusion_matrix, nasa_score, per_class_stats,
                      regression_extras, regression_metrics)


def _ranking_metrics(task, expected, predicted, num_classes, with_nasa):
    if task == "classification":
        return classification_metrics(expected, predicted, num_classes)
    return regression_metrics(expected, predicted, with_nasa=with_nasa)


def analyze(model, data, meta_rows, spec, group_info, top_k=5):
    """Return the evaluation bundle of `model` on `data` ({"ids", "x", "y"})."""
    task, classes = model.meta["task"], model.meta["classes"]
    expected = data["y"]
    output = model.predict(data["x"])
    hidden, _ = model.forward(model.normalized(data["x"]))
    bundle = {"n": int(len(expected)), "task": task,
              "saturation": float(np.mean(np.abs(hidden) > 0.99))}
    if task == "classification":
        predicted = output.argmax(axis=1)
        confidence = output.max(axis=1)
        wrong = predicted != expected
        stats = per_class_stats(expected, predicted, len(classes))
        matrix = confusion_matrix(expected, predicted, len(classes))
        pairs = [(int(matrix[i, j]), classes[i], classes[j])
                 for i in range(len(classes)) for j in range(len(classes)) if i != j and matrix[i, j]]
        bundle.update({
            "per_class": [{"class": classes[i], **stats[i]} for i in range(len(classes))],
            "confusion": matrix.tolist(),
            "top_confusions": [{"true": t, "pred": p, "count": c} for c, t, p in sorted(pairs, reverse=True)[:top_k]],
            "true_share": {classes[i]: stats[i]["support"] / len(expected) for i in range(len(classes))},
            "pred_share": {classes[i]: stats[i]["predicted"] / len(expected) for i in range(len(classes))},
            "extras": {"mean_confidence": float(confidence.mean()),
                       "mean_confidence_when_wrong": float(confidence[wrong].mean()) if wrong.any() else None},
        })
        order = np.argsort(-np.where(wrong, confidence, -1.0))[:top_k]
        bundle["worst"] = [{"id": data["ids"][i], "true": classes[int(expected[i])],
                            "pred": classes[int(predicted[i])], "confidence": float(confidence[i])}
                           for i in order if wrong[i]]
    else:
        predicted = output
        error = predicted - expected
        bundle["extras"] = regression_extras(expected, predicted)
        edges = np.quantile(expected, [0, .25, .5, .75, 1])
        quartiles = []
        for position, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (expected >= low) & ((expected <= high) if position == 3 else (expected < high))
            if mask.any():
                quartiles.append({"low": float(low), "high": float(high), "n": int(mask.sum()),
                                  "rmse": float(np.sqrt(np.mean(error[mask] ** 2))),
                                  "bias": float(np.mean(error[mask]))})
        bundle["by_quantile"] = quartiles
        order = np.argsort(-np.abs(error))[:top_k]
        bundle["worst"] = [{"id": data["ids"][i], "true": float(expected[i]), "pred": float(predicted[i]),
                            "error": float(error[i])} for i in order]
        if spec["with_nasa"]:
            exponent = np.where(error < 0, -error / 13.0, error / 10.0)
            contributions = np.exp(np.minimum(exponent, 700.0)) - 1.0
            total = float(contributions.sum())
            top = np.sort(contributions)[-max(1, len(contributions) // 100):]
            bundle["nasa"] = {"sum": total, "top1pct_share": float(top.sum() / total) if total > 0 else 0.0,
                              "over_estimate_share": float(contributions[error > 0].sum() / total) if total > 0 else 0.0}
    bundle["overall"] = _ranking_metrics(task, expected, predicted, len(classes), spec["with_nasa"])

    bundle["by_group"] = {}
    keys = list(group_info["report_keys"])
    if group_info["split_key"] and group_info["split_key"] not in keys:
        keys.append(group_info["split_key"])
    for key in keys:
        labels = np.asarray(group_labels(meta_rows, key))
        rows = []
        for group in dict.fromkeys(labels.tolist()):
            mask = labels == group
            entry = {"group": group, "n": int(mask.sum()), "share": float(mask.mean()),
                     **_ranking_metrics(task, expected[mask], predicted[mask], len(classes), spec["with_nasa"])}
            if task == "regression":
                entry["bias"] = float(np.mean(predicted[mask] - expected[mask]))
                entry["target_max"] = float(expected[mask].max())
            rows.append(entry)
        bundle["by_group"]["+".join(key)] = rows
    return bundle


def scoring_metrics(bundle, spec, group_key=None):
    """Metrics that enter the score: pooled, or the mean over groups of `group_key`."""
    if group_key is None:
        return dict(bundle["overall"])
    rows = bundle["by_group"].get("+".join(group_key))
    if not rows or len(rows) < 2:
        return dict(bundle["overall"])
    return {name: float(np.mean([row[name] for row in rows])) for name in bundle["overall"]}


def group_regressions(candidate, reference, spec, key_name):
    """[(group, relative worsening of the primary metric)] for the groups of `key_name`."""
    from .scoring import relative_worsening
    reference_rows = {row["group"]: row for row in reference["by_group"].get(key_name, [])}
    result = []
    for row in candidate["by_group"].get(key_name, []):
        if row["group"] in reference_rows:
            result.append((row["group"], relative_worsening(
                spec["primary"], row[spec["primary"]], reference_rows[row["group"]][spec["primary"]])))
    return result
