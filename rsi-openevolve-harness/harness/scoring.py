"""combined_score = primary-metric ratio - guard penalties + a tiny tie-break.

1.0 means "same as the reference" (the parent checkpoint), above 1.0 is better.
Auxiliary metrics never buy back a loss on the primary metric: they only subtract
when they get worse than their tolerance, and add at most ~0.001 as a tie-break.
"""
from .metrics import HIGHER_IS_BETTER

EPS = 1e-12
MAX_RATIO = 10.0
TIE_WEIGHT = 0.001


def improvement_ratio(name, candidate, reference):
    """>1 when the candidate is better than the reference on metric `name`."""
    if HIGHER_IS_BETTER[name]:
        ratio = candidate / reference if reference > EPS else 1.0 + (candidate - reference)
    else:
        ratio = reference / candidate if candidate > EPS else MAX_RATIO
    return float(min(max(ratio, 0.0), MAX_RATIO))


def relative_worsening(name, candidate, reference):
    """Positive when the candidate is worse than the reference (0.10 = 10% worse)."""
    if HIGHER_IS_BETTER[name]:
        return float(1.0 - candidate / reference) if reference > EPS else 0.0
    return float(candidate / reference - 1.0) if reference > EPS else float(candidate - reference)


def score(spec, candidate, reference, group_regressions=None):
    """Score candidate metrics against reference metrics.

    group_regressions: optional list of (group_name, relative_worsening) for the primary
    metric per group; worsening beyond spec["group_guard"] is penalised like a guard.
    """
    primary = spec["primary"]
    ratio = improvement_ratio(primary, candidate[primary], reference[primary])
    penalty, violations = 0.0, []
    for name, tolerance in spec["guards"].items():
        if name not in candidate or name not in reference:
            continue
        worse = relative_worsening(name, candidate[name], reference[name])
        if worse > tolerance:
            penalty += worse - tolerance
            violations.append({"guard": name, "worsening": worse, "tolerance": tolerance,
                               "candidate": candidate[name], "reference": reference[name]})
    for group, worse in group_regressions or []:
        if worse > spec["group_guard"]:
            penalty += (worse - spec["group_guard"]) / max(1, len(group_regressions))
            violations.append({"guard": f"group:{group}", "worsening": worse, "tolerance": spec["group_guard"]})
    tie = TIE_WEIGHT * sum(improvement_ratio(name, candidate[name], reference[name]) - 1.0
                           for name in spec["tiebreak"] if name in candidate and name in reference)
    combined = max(0.0, ratio - spec["penalty_weight"] * penalty + tie)
    return {"combined_score": float(combined), "primary_ratio": ratio, "penalty": float(penalty),
            "tie": float(tie), "violations": violations, "guard_ok": not violations}
