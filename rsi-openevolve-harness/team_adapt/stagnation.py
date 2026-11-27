"""Search stagnation on the selection split, computed from the harness iteration records.

Only *valid* candidates (evaluator status "ok", i.e. trained and scored on the selection
split) can show stagnation. Everything else is counted separately and never moves the
stagnation counter:

    invalid_code    the policy was rejected (static check, edit outside the block, bad diff,
                    checkpoint rule violated)
    api_error       the model call failed or timed out; no candidate was produced
    train_failure   training crashed or exceeded its time limit
    internal_error  the evaluator produced no report

Rule (like early stopping): `best` starts at the score of the stage's initial program and
moves only when a valid candidate beats it by at least `min_delta`. The search is
"stagnated" once `patience` valid candidates in a row have not done so. A long streak of
failures is reported as "unhealthy" instead; it says nothing about the data.

Stagnation means "this search on this data stopped finding improvements". It is not
evidence of drift and is never reported as such.
"""
from collections import Counter

API_MARKERS = ("llm generation failed", "llm returned none", "timed out", "timeout", "connection", "rate limit",
               "api error", "apierror", "service unavailable")
INVALID_STATUSES = {"policy_rejected", "constraint_violation"}
TRAIN_FAILURES = {"train_failed", "timeout"}
CATEGORIES = ("valid", "invalid_code", "api_error", "train_failure", "internal_error")


def classify(record):
    status = record.get("status")
    if status == "ok":
        return "valid"
    if status in INVALID_STATUSES:
        return "invalid_code"
    if status in TRAIN_FAILURES:
        return "train_failure"
    if status == "rejected_before_eval":
        error = (record.get("error") or "").lower()
        return "api_error" if any(marker in error for marker in API_MARKERS) else "invalid_code"
    return "internal_error"


def tokens(record):
    usage = record.get("token_usage") or {}
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, (int, float)):
        return int(total)
    return int((usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0))


def analyze(records, patience, min_delta, initial_score=None, max_consecutive_failures=None):
    """Return the stagnation state after `records` (in the order they were written)."""
    best = initial_score
    best_source = "initial_program" if initial_score is not None else None
    best_seen = initial_score
    since = 0
    streak = longest_streak = 0
    counts = Counter()
    first_stagnation = None
    for position, record in enumerate(records):
        category = classify(record)
        counts[category] += 1
        if category != "valid":
            streak += 1
            longest_streak = max(longest_streak, streak)
            continue
        streak = 0
        score = float(record.get("combined_score") or 0.0)
        best_seen = score if best_seen is None else max(best_seen, score)
        if best is None:  # no initial score known: the first valid candidate sets the baseline
            best, best_source, since = score, f"iteration {record.get('iteration')}", 0
            continue
        if score >= best + min_delta:
            best, best_source, since = score, f"iteration {record.get('iteration')}", 0
        else:
            since += 1
            if since >= patience and first_stagnation is None:
                first_stagnation = {"position": position, "iteration": record.get("iteration"),
                                    "run_id": record.get("run_id")}
    stagnated = since >= patience
    unhealthy = bool(max_consecutive_failures) and streak >= max_consecutive_failures
    return {
        "stagnated": stagnated, "valid_since_last_improvement": since, "patience": patience,
        "min_delta": min_delta, "best_score": best, "best_from": best_source, "best_seen": best_seen,
        "initial_score": initial_score, "records": len(records),
        "counts": {name: counts.get(name, 0) for name in CATEGORIES},
        "current_failure_streak": streak, "longest_failure_streak": longest_streak,
        "unhealthy": unhealthy, "first_stagnation": first_stagnation if stagnated else None,
        "tokens": sum(tokens(r) for r in records),
        "note": "停滞只说明在这份挑选集上搜索不再找到足够的改进，不代表已确认数据漂移",
    }


def describe(state):
    c = state["counts"]
    text = (f"有效候选 {c['valid']}，连续 {state['valid_since_last_improvement']}/{state['patience']} 个未超过当前最好 "
            f"{state['best_score'] if state['best_score'] is None else round(state['best_score'], 4)} + {state['min_delta']}；"
            f"无效代码 {c['invalid_code']}，API 失败 {c['api_error']}，训练失败 {c['train_failure']}，"
            f"内部错误 {c['internal_error']}")
    if state["stagnated"]:
        text = "搜索停滞（考虑新数据适应；不等于已确认漂移）：" + text
    elif state["unhealthy"]:
        text = f"搜索不健康（连续 {state['current_failure_streak']} 次失败，与数据无关）：" + text
    return text
