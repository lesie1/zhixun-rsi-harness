"""Should a new batch trigger retraining? A simple rule on metrics the platform also uses.

Measure the deployed model on the newest labeled batch and compare with its "reference
level" (what the same model achieved when it was deployed, see state.py). Retrain when

    the primary metric got worse by more than `primary_drop` (relative, e.g. 0.05 = 5 %)
    or an auxiliary (guard) metric got worse by more than its tolerance in the task spec,

AND a row bootstrap says the metric really got worse (P(worse) >= `confidence`), so a
small, noisy batch does not start an expensive search by chance. A batch with fewer than
`min_rows` rows never triggers; a model without a reference level always does.

Input drift and label shift are measured and reported, but they do not trigger on their
own: labels are always available, so the metrics say directly whether the drift hurts.

All metrics are the harness formulas, except that nasa_score is divided by the number of
rows here (it is a sum, and batches differ in size).
"""
import numpy as np

from baselines.model import Model
from harness.dataio import load_dataset
from harness.metrics import classification_metrics, regression_metrics
from harness.scoring import relative_worsening
from harness.specs import resolve_spec

DEFAULTS = {"primary_drop": 0.05, "use_guards": True, "min_rows": 50, "confidence": 0.90,
            "resamples": 500, "seed": 42}


def options(overrides=None):
    result = dict(DEFAULTS)
    for key, value in (overrides or {}).items():
        if key not in DEFAULTS:
            raise ValueError(f"unknown trigger option {key!r}")
        if value is not None:
            result[key] = value
    if not 0 <= result["primary_drop"] < 1 or not 0 < result["confidence"] <= 1:
        raise ValueError("primary_drop must be in [0, 1) and confidence in (0, 1]")
    if int(result["resamples"]) < 50 or int(result["min_rows"]) < 1:
        raise ValueError("resamples must be >= 50 and min_rows >= 1")
    return result


def spec_for(model, task_name=None, spec_file=None):
    return resolve_spec(model.meta["task"], task_name, spec_file)


def _metrics(model, expected, predicted, spec):
    if model.meta["task"] == "classification":
        return classification_metrics(expected, predicted, len(model.meta["classes"]))
    result = regression_metrics(expected, predicted, with_nasa=spec["with_nasa"])
    if "nasa_score" in result:
        # The NASA score is a sum over rows; batches of different sizes are compared per row.
        result["nasa_score"] = result["nasa_score"] / max(1, len(expected))
    return result


def _predict(model, x):
    output = model.predict(x)
    return output.argmax(axis=1) if model.meta["task"] == "classification" else output


def profile(model_path, data_path, spec, cache_dir=None):
    """Metrics and data profile of a checkpoint on a labeled file (used as reference level)."""
    model = Model.load(model_path)
    data = load_dataset(data_path, model.meta["task"], model.meta["classes"], model.meta["paths"], cache_dir=cache_dir)
    predicted = _predict(model, data["x"])
    result = {"rows": len(data["ids"]), "metrics": _metrics(model, data["y"], predicted, spec),
              "data_sha256": data["sha256"]}
    result.update(_data_profile(model, data))
    return result


def _data_profile(model, data):
    z = model.normalized(data["x"])
    info = {"feature_shift_sd": float(np.mean(np.abs(z.mean(axis=0))))}
    if model.meta["task"] == "classification":
        counts = np.bincount(data["y"], minlength=len(model.meta["classes"]))
        info["class_share"] = {name: float(c / max(1, len(data["y"]))) for name, c in zip(model.meta["classes"], counts)}
    else:
        info["target_mean"] = float(np.mean(data["y"]))
        info["target_std"] = float(np.std(data["y"]))
    return info


def _probability_worse(model, expected, predicted, spec, name, reference_value, resamples, seed):
    rng = np.random.default_rng(seed)
    n = len(expected)
    worse = 0
    for _ in range(int(resamples)):
        rows = rng.integers(0, n, n)
        value = _metrics(model, expected[rows], predicted[rows], spec)[name]
        if relative_worsening(name, value, reference_value) > 0:
            worse += 1
    return worse / int(resamples)


def check(model_path, data_path, reference, spec, opts=None, cache_dir=None):
    """Decide whether `data_path` (the newest labeled batch) should trigger retraining.

    `reference` is the stored reference level ({"metrics": {...}, ...}) or None.
    Returns a JSON-serialisable report whose key "retrain" is the decision.
    """
    opts = options(opts)
    model = Model.load(model_path)
    data = load_dataset(data_path, model.meta["task"], model.meta["classes"], model.meta["paths"], cache_dir=cache_dir)
    expected = data["y"]
    predicted = _predict(model, data["x"])
    current = _metrics(model, expected, predicted, spec)
    report = {"rows": len(expected), "metrics": current, "options": opts, "primary": spec["primary"],
              "reference": reference, "checks": [], "retrain": False, "reasons": []}
    report.update(_data_profile(model, data))
    if reference and model.meta["task"] == "classification" and reference.get("class_share"):
        names = model.meta["classes"]
        old = np.array([reference["class_share"].get(n, 0.0) for n in names])
        new = np.array([report["class_share"][n] for n in names])
        report["label_shift_tv"] = float(0.5 * np.abs(old - new).sum())
    elif reference and reference.get("target_std"):
        report["target_shift_sd"] = abs(report["target_mean"] - reference["target_mean"]) / reference["target_std"]

    if reference is None:
        report["retrain"], report["reasons"] = True, ["没有参照水平（第一次检查），直接重训"]
        return report
    if len(expected) < opts["min_rows"]:
        report["reasons"] = [f"新批次只有 {len(expected)} 行（少于 {opts['min_rows']}），证据不足，不重训"]
        return report

    watched = [(spec["primary"], opts["primary_drop"], "主指标")]
    if opts["use_guards"]:
        watched += [(name, tolerance, "辅助指标") for name, tolerance in spec["guards"].items()]
    for name, tolerance, kind in watched:
        if name not in current or name not in reference["metrics"]:
            continue
        old = reference["metrics"][name]
        drop = relative_worsening(name, current[name], old)
        entry = {"metric": name, "kind": kind, "reference": old, "current": current[name],
                 "relative_worsening": drop, "tolerance": tolerance, "probability_worse": None, "fires": False}
        if drop > tolerance:
            entry["probability_worse"] = _probability_worse(model, expected, predicted, spec, name, old,
                                                            opts["resamples"], opts["seed"])
            entry["fires"] = entry["probability_worse"] >= opts["confidence"]
        report["checks"].append(entry)
        if entry["fires"]:
            report["reasons"].append(f"{kind} {name} 从 {old:.4g} 变为 {current[name]:.4g}，相对变差 {drop:.1%}"
                                     f"（阈值 {tolerance:.0%}，变差概率 {entry['probability_worse']:.2f}）")
    report["retrain"] = bool(report["reasons"])
    if not report["retrain"]:
        report["reasons"] = ["各项指标的变差都没有超过阈值，或统计上不显著，不重训"]
    return report


def describe(report):
    lines = [f"[重训判断] {'重训' if report['retrain'] else '不重训'}：" + "；".join(report["reasons"]),
             f"  新批次 {report['rows']} 行；{report['primary']}={report['metrics'][report['primary']]:.4g}"
             f"；输入平均偏移 {report['feature_shift_sd']:.2f} 个训练标准差"]
    if "label_shift_tv" in report:
        lines.append(f"  类别占比变化（总变差距离）{report['label_shift_tv']:.3f}")
    if "target_shift_sd" in report:
        lines.append(f"  目标均值变化 {report['target_shift_sd']:.2f} 个标准差")
    for entry in report["checks"]:
        probability = "" if entry["probability_worse"] is None else f"，变差概率 {entry['probability_worse']:.2f}"
        lines.append(f"  {entry['metric']}: 参照 {entry['reference']:.4g} → 现在 {entry['current']:.4g}，"
                     f"相对变差 {entry['relative_worsening']:+.1%}（阈值 {entry['tolerance']:.0%}{probability}）"
                     f"{' → 触发' if entry['fires'] else ''}")
    return "\n".join(lines)
