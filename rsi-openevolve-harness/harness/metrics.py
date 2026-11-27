"""Metric library. `macro_f1`, `accuracy`, `rmse` and `mae` reproduce the formulas of the
reference trainer (baselines/train.py::metrics) operation for operation.

Every metric is registered with its direction so that scoring never has to guess
whether larger or smaller is better.
"""
import numpy as np

# name -> True when larger is better
HIGHER_IS_BETTER = {
    "macro_f1": True,
    "macro_f1_all": True,
    "weighted_f1": True,
    "accuracy": True,
    "balanced_accuracy": True,
    "rmse": False,
    "mae": False,
    "nasa_score": False,
}

CLASSIFICATION_METRICS = ("macro_f1", "macro_f1_all", "weighted_f1", "accuracy", "balanced_accuracy")
REGRESSION_METRICS = ("rmse", "mae", "nasa_score")


def confusion_matrix(expected, labels, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (expected, labels), 1)
    return matrix


def per_class_stats(expected, labels, num_classes):
    """Per class: true positives, predicted count, support, precision, recall, F1 (None when undefined)."""
    stats = []
    for label in range(num_classes):
        tp = int(np.sum((labels == label) & (expected == label)))
        predicted = int(np.sum(labels == label))
        support = int(np.sum(expected == label))
        stats.append({
            "tp": tp, "predicted": predicted, "support": support,
            "precision": tp / predicted if predicted else None,
            "recall": tp / support if support else None,
            "f1": 2 * tp / (predicted + support) if predicted + support else None,
        })
    return stats


def classification_metrics(expected, labels, num_classes):
    """expected and labels are integer class indices."""
    expected = np.asarray(expected)
    labels = np.asarray(labels)
    # Reference definition: average F1 over the classes that occur in truth or prediction.
    observed = sorted(set(expected.tolist()) | set(labels.tolist()))
    f1 = [2 * np.sum((labels == label) & (expected == label)) /
          (np.sum(labels == label) + np.sum(expected == label)) for label in observed]
    result = {"accuracy": float(np.mean(labels == expected)), "macro_f1": float(np.mean(f1))}
    stats = per_class_stats(expected, labels, num_classes)
    # Alternative definition: average over every configured class, undefined F1 counted as 0.
    result["macro_f1_all"] = float(np.mean([s["f1"] or 0.0 for s in stats]))
    total = sum(s["support"] for s in stats)
    result["weighted_f1"] = float(sum((s["f1"] or 0.0) * s["support"] for s in stats) / total) if total else 0.0
    recalls = [s["recall"] for s in stats if s["recall"] is not None]
    result["balanced_accuracy"] = float(np.mean(recalls)) if recalls else 0.0
    return result


def nasa_score(expected, predicted):
    """NASA C-MAPSS asymmetric score, summed over samples; late predictions cost more.

    d = predicted - expected; score = exp(-d/13) - 1 if d < 0 else exp(d/10) - 1.
    The exponent is capped at 700 so that the sum stays finite.
    """
    d = np.asarray(predicted, dtype=np.float64) - np.asarray(expected, dtype=np.float64)
    exponent = np.where(d < 0, -d / 13.0, d / 10.0)
    return float(np.sum(np.exp(np.minimum(exponent, 700.0)) - 1.0))


def regression_metrics(expected, predicted, with_nasa=False):
    expected = np.asarray(expected, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - expected
    result = {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2)))}
    if with_nasa:
        result["nasa_score"] = nasa_score(expected, predicted)
    return result


def regression_extras(expected, predicted):
    """Diagnostics that are not ranking metrics."""
    expected = np.asarray(expected, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - expected
    return {"bias": float(np.mean(error)), "over_rate": float(np.mean(error > 0)),
            "clipped_zero_rate": float(np.mean(predicted <= 0))}


def predict_labels(model, x):
    """Class indices for classification, values in original units for regression."""
    output = model.predict(x)
    if model.meta["task"] == "classification":
        return output.argmax(axis=1), output
    return output, None


def overall_metrics(model, data, with_nasa=False):
    """Headline metrics of `model` on `data` ({"x", "y"}), same numbers as the reference trainer."""
    predicted, _ = predict_labels(model, data["x"])
    if model.meta["task"] == "classification":
        return classification_metrics(data["y"], predicted, len(model.meta["classes"]))
    return regression_metrics(data["y"], predicted, with_nasa=with_nasa)
