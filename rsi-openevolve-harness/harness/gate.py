"""Acceptance gate: decide which checkpoint becomes the next selected model.

The search (OpenEvolve) only ever sees the selection split and picks the highest of many
noisy scores, so its winner is optimistically biased. The gate re-examines three
checkpoints on the gate split, which never influenced the search:

    parent     the checkpoint this stage started from ("do nothing")
    reference  the unmodified policy = official continuation
    candidate  the evolved policy's checkpoint

A challenger replaces the parent only if a paired bootstrap says it is better on the
primary metric with probability >= threshold AND no guard is violated on the gate split.
The candidate must also be at least as good as the reference; otherwise the reference
(if it passes) is adopted, else the parent stays. "Keep the parent" is the rollback.

When no independent gate split exists (too few groups, method "no_split") the gate runs
on the same data as the search, uses a stricter threshold and says so in its output.
"""
import numpy as np

from .analysis import group_regressions, scoring_metrics
from .groups import discover_groups, group_labels
from .metrics import HIGHER_IS_BETTER, classification_metrics, regression_metrics
from .scoring import score

THRESHOLD = 0.90
THRESHOLD_NO_INDEPENDENT_GATE = 0.95
MIN_CLUSTERS = 8


def _metric(task, name, expected, predicted, num_classes, with_nasa):
    if task == "classification":
        return classification_metrics(expected, predicted, num_classes)[name]
    return regression_metrics(expected, predicted, with_nasa=with_nasa)[name]


def _predict(model, x):
    output = model.predict(x)
    return output.argmax(axis=1) if model.meta["task"] == "classification" else output


def probability_better(task, name, expected, challenger, incumbent, num_classes, groups=None,
                       mode="sample", resamples=2000, seed=42, with_nasa=False):
    """P(challenger beats incumbent on metric `name`) under a paired bootstrap.

    mode "sample":  resample rows.
    mode "cluster": resample whole groups (rows of one engine stay together).
    mode "group_mean": resample rows inside each group, compare the mean of per-group metrics.
    """
    rng = np.random.default_rng(seed)
    n = len(expected)
    higher = HIGHER_IS_BETTER[name]
    members = None
    if mode in ("cluster", "group_mean"):
        labels = np.asarray(groups)
        members = [np.flatnonzero(labels == label) for label in dict.fromkeys(labels.tolist())]

    def value(predicted, rows):
        return _metric(task, name, expected[rows], predicted[rows], num_classes, with_nasa)

    wins = ties = 0
    for _ in range(resamples):
        if mode == "cluster":
            rows = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
            a, b = value(challenger, rows), value(incumbent, rows)
        elif mode == "group_mean":
            parts = [member[rng.integers(0, len(member), len(member))] for member in members]
            a = float(np.mean([value(challenger, rows) for rows in parts]))
            b = float(np.mean([value(incumbent, rows) for rows in parts]))
        else:
            rows = rng.integers(0, n, n)
            a, b = value(challenger, rows), value(incumbent, rows)
        if a == b:
            ties += 1
        elif (a > b) == higher:
            wins += 1
    return (wins + 0.5 * ties) / resamples


def decide(context, candidate_model=None, resamples=2000, seed=42):
    """Run the gate. `candidate_model` may be None (then only reference vs parent is decided)."""
    data, meta = context.load_gate()
    group_info = discover_groups(meta)
    spec, task, primary = context.spec, context.task, context.spec["primary"]
    independent = context.manifest["method"] != "no_split"
    threshold = THRESHOLD if independent else THRESHOLD_NO_INDEPENDENT_GATE
    split_key = group_info["split_key"]
    groups = group_labels(meta, split_key) if split_key else None
    clusters = len(set(groups)) if groups else 0
    if context.aggregate == "group_mean" and clusters >= 2:
        mode = "group_mean"
    elif clusters >= MIN_CLUSTERS:
        mode = "cluster"
    else:
        mode = "sample"
    aggregate_key = split_key if mode == "group_mean" else None
    report_keys = group_info["report_keys"]
    guard_key = "+".join(report_keys[0]) if report_keys else None

    models = {"parent": context.parent, "reference": context.reference_model}
    if candidate_model is not None:
        models["candidate"] = candidate_model
    bundles = {name: context.analyze(model, data, meta, group_info) for name, model in models.items()}
    metrics = {name: scoring_metrics(bundle, spec, aggregate_key) for name, bundle in bundles.items()}
    predictions = {name: _predict(model, data["x"]) for name, model in models.items()}

    def compare(challenger, incumbent):
        regressions = group_regressions(bundles[challenger], bundles[incumbent], spec, guard_key) if guard_key else None
        scoring = score(spec, metrics[challenger], metrics[incumbent], regressions)
        probability = probability_better(task, primary, data["y"], predictions[challenger], predictions[incumbent],
                                         len(context.classes), groups=groups, mode=mode, resamples=resamples,
                                         seed=seed, with_nasa=spec["with_nasa"])
        return {"probability_better": probability, "primary_ratio": scoring["primary_ratio"],
                "guard_ok": scoring["guard_ok"], "violations": scoring["violations"],
                "passes": probability >= threshold and scoring["guard_ok"]}

    comparisons = {"reference_vs_parent": compare("reference", "parent")}
    decision, reason = "keep_parent", "没有候选通过验收门，保留起点检查点（回滚）"
    reference_ok = comparisons["reference_vs_parent"]["passes"]
    if candidate_model is not None:
        comparisons["candidate_vs_parent"] = compare("candidate", "parent")
        comparisons["candidate_vs_reference"] = compare("candidate", "reference")
        candidate_ok = comparisons["candidate_vs_parent"]["passes"]
        not_worse = comparisons["candidate_vs_reference"]["primary_ratio"] >= 1.0
        if candidate_ok and (not_worse or not reference_ok):
            decision, reason = "adopt_candidate", "进化出的策略通过验收门，且不差于官方续训"
        elif reference_ok:
            decision = "adopt_reference"
            reason = ("进化出的策略没有通过验收门" if not candidate_ok else "进化出的策略不如官方续训") + "；官方续训通过，采用官方续训"
        elif candidate_ok:
            decision, reason = "adopt_candidate", "进化出的策略通过验收门"
    elif reference_ok:
        decision, reason = "adopt_reference", "官方续训通过验收门"

    warnings = list(context.manifest["warnings"])
    if not independent:
        warnings.append("验收用的数据与搜索用的数据相同，结论带有选择偏差；阈值已提高到 "
                        f"{THRESHOLD_NO_INDEPENDENT_GATE}")
    return {"decision": decision, "reason": reason, "primary": primary, "threshold": threshold,
            "independent_gate": independent, "bootstrap": {"mode": mode, "resamples": resamples, "seed": seed,
                                                           "clusters": clusters},
            "gate_rows": len(data["ids"]), "aggregate": "group_mean" if aggregate_key else "pooled",
            "metrics": metrics, "comparisons": comparisons, "warnings": warnings}


def describe(result):
    """Short Chinese summary of a gate result."""
    primary = result["primary"]
    lines = [f"[验收门] 决定: {result['decision']} —— {result['reason']}",
             f"  数据: 验收集 {result['gate_rows']} 行，{'独立于搜索' if result['independent_gate'] else '与搜索用的数据相同'}；"
             f"自助法方式 {result['bootstrap']['mode']}，重采样 {result['bootstrap']['resamples']} 次，阈值 {result['threshold']}"]
    for name, values in result["metrics"].items():
        lines.append(f"  {name}: {primary}={values[primary]:.6g}")
    for name, comparison in result["comparisons"].items():
        lines.append(f"  {name}: P(更好)={comparison['probability_better']:.3f}，主指标比值={comparison['primary_ratio']:.4f}，"
                     f"护栏{'通过' if comparison['guard_ok'] else '违规'} → {'通过' if comparison['passes'] else '不通过'}")
    lines += [f"  注意: {warning}" for warning in result["warnings"]]
    return "\n".join(lines)
