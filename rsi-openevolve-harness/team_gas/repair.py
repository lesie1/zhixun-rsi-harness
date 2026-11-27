"""Repair on the current training data: alarm-aware stagnation and the repair gate.

AlarmAnalyzer is plugged into team_adapt's EvolveSearch monitor. A valid candidate counts
as progress only if
    the alarm class's recall (selection split) rose by >= alarm_min_delta over the best so far, or
    the combined score rose by >= min_delta AND the alarm recall did not fall below its best,
so a small overall gain cannot hide a target class that keeps getting worse. `patience`
valid candidates without progress = stagnated. Invalid code, API failures/timeouts and
training failures are counted separately (team_adapt.stagnation.classify) and never count.
As soon as one candidate meets the targeted-fix rule on the selection split the search is
stopped early ("repaired").
"""
import json
from pathlib import Path

from baselines.model import Model
from harness.dataio import read_json, write_json
from harness.evaluator import code_digest
from team_adapt import stagnation
from team_adapt.search import initial_score

from . import criteria, diagnose


class AlarmAnalyzer:
    def __init__(self, degraded, confusion, parent_checkpoint, stagnation_cfg, success_cfg, cache_dir=None):
        self.degraded, self.confusion = degraded, confusion
        self.parent = parent_checkpoint
        self.stag, self.success = stagnation_cfg, success_cfg
        self.cache_dir = cache_dir
        self._baseline = None
        self._reports = {}

    def baseline(self, work):
        if self._baseline is None and (Path(work) / "splits" / "select.jsonl").is_file():
            self._baseline = criteria.per_class_from_profile(
                diagnose.profile(self.parent, Path(work) / "splits" / "select.jsonl", self.cache_dir))
        return self._baseline

    def _report(self, work, digest):
        if digest not in self._reports:
            path = Path(work) / "reports" / f"{digest}.json"
            if not path.is_file():
                return None
            self._reports[digest] = read_json(path)
        return self._reports[digest]

    def __call__(self, work, records):
        work = Path(work)
        base = self.baseline(work)
        init_score = initial_score(work)
        init_alarm = None
        policy = work / "initial_policy.py"
        if policy.is_file():
            report = self._report(work, code_digest(policy.read_text(encoding="utf-8")))
            if report and report.get("status") == "ok":
                init_alarm = criteria.per_class_from_report(report).get(self.degraded, {}).get("recall")
        if init_alarm is None and base:
            init_alarm = base.get(self.degraded, {}).get("recall")
        best_score, best_alarm = init_score, init_alarm
        since, streak, longest, trace, repaired = 0, 0, 0, [], None
        counts = {c: 0 for c in stagnation.CATEGORIES}
        for record in records:
            category = stagnation.classify(record)
            counts[category] += 1
            if category != "valid":
                streak += 1
                longest = max(longest, streak)
                continue
            streak = 0
            report = self._report(work, record.get("code_sha256")) if record.get("code_sha256") else None
            per_class = criteria.per_class_from_report(report) if report else {}
            score = float(record.get("combined_score") or 0.0)
            alarm = (per_class.get(self.degraded) or {}).get("recall") or 0.0
            if best_score is None:
                best_score, best_alarm, progress = score, alarm, True
            else:
                best_alarm = alarm if best_alarm is None else best_alarm
                progress = (alarm >= best_alarm + self.stag["alarm_min_delta"]) or \
                           (score >= best_score + self.stag["min_delta"] and alarm >= best_alarm - 1e-12)
            if progress:
                best_score, best_alarm, since = max(best_score, score), max(best_alarm, alarm), 0
            else:
                since += 1
            fix = criteria.targeted(base, per_class, self.degraded, self.confusion, self.success["target_recall_gain"],
                                    self.success["confusion_guard"], self.success["other_guard"]) if base and per_class else None
            trace.append({"iteration": record.get("iteration"), "score": score, "alarm_recall": alarm,
                          "confusion_recall": (per_class.get(self.confusion) or {}).get("recall"),
                          "confusion_precision": (per_class.get(self.confusion) or {}).get("precision"),
                          "progress": progress, "targeted_fix": bool(fix and fix["ok"])})
            if fix and fix["ok"] and repaired is None:
                repaired = {"iteration": record.get("iteration"), "code_sha256": record.get("code_sha256"),
                            "candidate_dir": (report or {}).get("candidate_dir"), "checks": fix}
        stagnated = since >= self.stag["patience"]
        unhealthy = bool(self.stag.get("max_consecutive_failures")) and streak >= self.stag["max_consecutive_failures"]
        state = {"stagnated": stagnated and not repaired, "repaired": bool(repaired), "repaired_candidate": repaired,
                 "unhealthy": unhealthy, "records": len(records), "tokens": sum(stagnation.tokens(r) for r in records),
                 "counts": counts, "valid_since_last_improvement": since, "patience": self.stag["patience"],
                 "min_delta": self.stag["min_delta"], "alarm_min_delta": self.stag["alarm_min_delta"],
                 "best_score": best_score, "best_alarm_recall": best_alarm, "initial_score": init_score,
                 "initial_alarm_recall": init_alarm, "current_failure_streak": streak, "longest_failure_streak": longest,
                 "alarm_class": self.degraded, "trace": trace,
                 "note": "停滞只说明在这份挑选集上找不到足够的改进，不代表已确认数据漂移"}
        state["description"] = (
            ("已修复：" if repaired else "搜索停滞（考虑新数据适应；不等于已确认漂移）：" if state["stagnated"] else "")
            + f"有效候选 {counts['valid']}，连续 {since}/{self.stag['patience']} 个既没让报警类 {self.degraded} 的召回提高 "
            f"{self.stag['alarm_min_delta']}，也没在不降低它的前提下让总分提高 {self.stag['min_delta']}；"
            f"报警类召回 起点 {init_alarm if init_alarm is None else round(init_alarm, 3)} / 最好 "
            f"{best_alarm if best_alarm is None else round(best_alarm, 3)}；无效代码 {counts['invalid_code']}，"
            f"API 失败 {counts['api_error']}，训练失败 {counts['train_failure']}，内部错误 {counts['internal_error']}")
        return state


def find_policy(work, digest):
    """Source of a candidate program (best program or an OpenEvolve checkpoint), if still on disk."""
    work = Path(work)
    best = work / "openevolve_output" / "best" / "best_program.py"
    if best.is_file() and code_digest(best.read_text(encoding="utf-8")) == digest:
        return best
    for program in sorted((work / "openevolve_output" / "checkpoints").glob("checkpoint_*/programs/*.json")):
        try:
            code = json.loads(program.read_text(encoding="utf-8")).get("code")
        except (OSError, json.JSONDecodeError):
            continue
        if code and code_digest(code) == digest:
            target = work / "repaired_policy.py"
            target.write_text(code, encoding="utf-8")
            return target
    return None


def gate(run_config_path, preferred=None, cache_dir=None):
    """Apply the harness acceptance gate (same rule as scripts/run_gate.py) in-process.

    `preferred` = {"code_sha256", "candidate_dir"} of a candidate that met the targeted-fix
    rule; otherwise the search's best program is the candidate (if there is one).
    """
    from harness.evaluator import evaluate_policy
    from harness.gate import decide, describe
    from harness.runcfg import load_run_config
    from harness.stage import get_context

    config = load_run_config(run_config_path)
    context = get_context(config)
    work = Path(config["work_dir"])
    candidate_model, candidate_dir, policy, source = None, None, None, None
    if preferred and preferred.get("candidate_dir") and (Path(preferred["candidate_dir"]) / "checkpoint.npz").is_file():
        candidate_dir = Path(preferred["candidate_dir"])
        candidate_model = Model.load(candidate_dir / "checkpoint.npz")
        policy = find_policy(work, preferred["code_sha256"])
        source = "targeted_fix_candidate"
    else:
        best = work / "openevolve_output" / "best" / "best_program.py"
        if best.is_file():
            evaluation = evaluate_policy(best, context)
            if evaluation["report"]["status"] == "ok":
                candidate_dir = Path(evaluation["report"]["candidate_dir"])
                candidate_model = Model.load(candidate_dir / "checkpoint.npz")
                policy, source = best, "best_program"
    result = decide(context, candidate_model, resamples=2000, seed=config["seed"])
    chosen = {"adopt_candidate": candidate_dir, "adopt_reference": context.reference_dir}.get(result["decision"])
    result.update({"candidate_source": source, "candidate_dir": str(candidate_dir) if candidate_dir else None,
                   "candidate_policy": str(policy) if policy else None,
                   "selected_dir": str(chosen) if chosen else None,
                   "parent_checkpoint": config["parent_checkpoint"]})
    (work / "gate").mkdir(exist_ok=True)
    write_json(work / "gate" / "decision.json", result)
    result["description"] = describe(result)
    return result
