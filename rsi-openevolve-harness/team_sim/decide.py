"""Update policies compared by the simulator, and the statistics they use.

  never     the static baseline forever (what the "static baseline" of the rules does)
  always    continue every round that brought new training labels, deploy without a gate
  v1        the existing team_loop rule: compare with the model's own "reference level"
            (team_loop/trigger.py, unchanged), retrain on a >5 % drop with P >= 0.9, then a gate
  v3        the agreed rule (third version + two refinements):
            - every round with new training labels: cheap continuations with target caps
              {1.0, 0.8, 1.2, none} x mean training label (harness/policy_rul_cap.py)
            - the static baseline is always in the pool
            - relative advantage over the static baseline on the SAME newest labelled data,
              (static RMSE - model RMSE) / static RMSE, is the yardstick across rounds
            - red: the deployed model is worse than the static baseline, P >= 0.8
                   -> roll back at once to the best of {deployed, candidates, static};
                      ask for a search only if no candidate beats the static baseline
            - yellow: the advantage shrank below half of the mean of the two previous rounds
                   and by >= 5 points, P >= 0.8 -> ask for a search and meanwhile deploy the
                   best of {deployed, candidates, static} on this round's validation data
            - green: a candidate replaces the deployed model only with P >= 0.9; a cap other
                   than 1.0 must beat the 1.0 candidate by >= 2 % with P >= 0.9
  v3_cap1 / v3_nocap   ablations of v3 with a single candidate (cap 1.0 / no cap)

All probabilities are paired row bootstraps on the round's data (the same resamples for every
model of a round, so comparisons are paired). Thresholds are starting points to calibrate.
"""
from dataclasses import dataclass, field

import numpy as np

from baselines.model import Model
from harness import PROJECT_ROOT
from team_loop import trigger

from .policy import render

INITIAL_POLICY = (PROJECT_ROOT / "harness" / "policy_initial.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- statistics
def rmse(pred, y):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y)) ** 2)))


def unit_rmse(pred, y, units):
    """Mean over units of each unit's RMSE (every bearing counts once)."""
    return float(np.mean([rmse(pred[units == u], y[units == u]) for u in dict.fromkeys(units.tolist())]))


def resample_index(n, resamples, seed):
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n), dtype=np.int32)


def p_better(pred_a, pred_b, y, index):
    """P(RMSE of a < RMSE of b) over paired row resamples."""
    ea, eb = (pred_a - y) ** 2, (pred_b - y) ** 2
    return float(np.mean(ea[index].mean(axis=1) < eb[index].mean(axis=1)))


def advantage(pred_model, pred_static, y):
    base = rmse(pred_static, y)
    return (base - rmse(pred_model, y)) / base if base > 0 else 0.0


def advantage_samples(pred_model, pred_static, y, index):
    base = np.sqrt(((pred_static - y) ** 2)[index].mean(axis=1))
    model = np.sqrt(((pred_model - y) ** 2)[index].mean(axis=1))
    return np.where(base > 0, (base - model) / np.where(base > 0, base, 1.0), 0.0)


def cap_label(multiplier):
    return "cap_none" if multiplier is None else f"cap_{multiplier:g}"


# ---------------------------------------------------------------- policies
class Never:
    name = "never"

    def start(self, sim, state):
        pass

    def round(self, sim, k, ctx, state):
        return {"action": "keep_static", "reason": "从不更新：一直使用静态基线"}


class Always:
    name = "always"

    def start(self, sim, state):
        pass

    def round(self, sim, k, ctx, state):
        if not ctx["new_train"]:
            return {"action": "keep", "reason": "本轮训练集没有新标签"}
        source = render(None, epochs=sim.epochs)          # original target, exactly policy_initial
        state["deployed"] = sim.train(state["deployed"], source, k, "always")
        state["deployed_label"] = "always"
        return {"action": "deploy_unconditionally", "reason": "每轮都续训并直接上线（无验收）"}


class TriggerV1:
    """team_loop's rule, reusing team_loop/trigger.py as is."""
    name = "v1"

    def __init__(self, adopt_prob=0.9):
        self.adopt_prob = adopt_prob

    def start(self, sim, state):
        state["reference"] = None

    def round(self, sim, k, ctx, state):
        spec = trigger.spec_for(Model.load(sim.static), sim.task)
        if state["reference"] is None:   # team_loop/state.py: reference level measured on the training data
            state["reference"] = trigger.profile(sim.static, ctx["files"]["train"], spec, cache_dir=sim.cache_dir)
        newest = ctx["files"].get("newest")
        if not newest:
            return {"action": "keep", "reason": "没有新到的带标签批次，不做重训判断"}
        report = trigger.check(state["deployed"], newest, state["reference"], spec, cache_dir=sim.cache_dir)
        decision = {"trigger_retrain": report["retrain"], "trigger_reasons": report["reasons"]}
        if not report["retrain"]:
            return dict(decision, action="keep", reason="重训判断：不重训")
        candidate = sim.train(state["deployed"], render(None, epochs=sim.epochs), k, "v1")
        p = p_better(sim.predict(candidate, "validation"), sim.predict(state["deployed"], "validation"),
                     sim.y("validation"), sim.index("validation"))
        decision["p_candidate_better"] = p
        if p >= self.adopt_prob:
            state["deployed"], state["deployed_label"] = candidate, "v1"
            decision.update(action="adopt", reason=f"重训判断触发，候选在验证数据上更好的概率 {p:.2f}")
        else:
            decision.update(action="keep", reason=f"重训判断触发，但候选更好的概率只有 {p:.2f}，保留")
        # team_loop/step.py: after a retraining round the reference level is re-measured on this batch
        state["reference"] = trigger.profile(state["deployed"], newest, spec, cache_dir=sim.cache_dir)
        return decision


@dataclass
class V3Config:
    caps: tuple = (1.0, 0.8, 1.2, None)
    default_cap: object = 1.0
    cap_margin: float = 0.02
    cap_prob: float = 0.9
    adopt_prob: float = 0.9
    red_prob: float = 0.8
    yellow_ratio: float = 0.5
    yellow_points: float = 0.05
    yellow_prob: float = 0.8
    extra: dict = field(default_factory=dict)


class V3:
    def __init__(self, name="v3", **overrides):
        self.name = name
        self.cfg = V3Config(**overrides)

    def start(self, sim, state):
        state["adv_history"] = []

    def round(self, sim, k, ctx, state):
        cfg = self.cfg
        incumbent, static = state["deployed"], sim.static
        watch = "newest" if sim.has("newest") else "validation"
        y_w, idx_w = sim.y(watch), sim.index(watch)
        y_g, idx_g = sim.y("validation"), sim.index("validation")

        # 1. how is the deployed model doing against the static baseline on the newest data?
        p_inc, p_sta = sim.predict(incumbent, watch), sim.predict(static, watch)
        adv_now = advantage(p_inc, p_sta, y_w)
        samples = advantage_samples(p_inc, p_sta, y_w, idx_w)
        p_worse = float(np.mean(samples < 0)) if incumbent != static else 0.0
        red = incumbent != static and p_worse >= cfg.red_prob
        history = state["adv_history"]
        yellow, threshold = False, None
        if not red and incumbent != static and len(history) >= 2:
            base = float(np.mean(history[-2:]))
            if base > 0:
                threshold = min(cfg.yellow_ratio * base, base - cfg.yellow_points)
                yellow = float(np.mean(samples < threshold)) >= cfg.yellow_prob
        history.append(adv_now)
        decision = {"watch_data": watch, "incumbent": state["deployed_label"], "advantage_newest": adv_now,
                    "p_worse_than_static": p_worse, "red": red, "yellow": yellow, "yellow_threshold": threshold}

        # 2. cheap continuations, one per cap, from the deployed model
        pool = {"incumbent": incumbent, "static": static}
        trained = {}
        if ctx["new_train"]:
            for multiplier in cfg.caps:
                source = render(multiplier, ctx["label_mean"], sim.target_mean, sim.target_scale, epochs=sim.epochs)
                trained[cap_label(multiplier)] = sim.train(incumbent, source, k, f"v3-{cap_label(multiplier)}")
            pool.update(trained)
        scores = {label: rmse(sim.predict(path, "validation"), y_g) for label, path in pool.items()}
        decision["gate_rmse"] = scores

        challenger = None
        if trained:
            default = cap_label(cfg.default_cap) if cap_label(cfg.default_cap) in trained else next(iter(trained))
            challenger = default
            for label in trained:
                if label == default:
                    continue
                gain = 1 - scores[label] / scores[default] if scores[default] > 0 else 0.0
                p = p_better(sim.predict(trained[label], "validation"), sim.predict(trained[default], "validation"),
                             y_g, idx_g)
                if p >= cfg.cap_prob and gain >= cfg.cap_margin and scores[label] < scores[challenger]:
                    challenger = label
        decision["challenger"] = challenger

        # 3. choose what to deploy
        chosen, rollback = "incumbent", False
        if red:          # roll back at once: best of {deployed, candidates, static} on this round's data
            chosen, rollback = min(scores, key=scores.get), True
        elif yellow:     # ask for a search; meanwhile deploy the best of this round's pool
            chosen = min(scores, key=scores.get)
        elif challenger:
            p = p_better(sim.predict(pool[challenger], "validation"), sim.predict(incumbent, "validation"), y_g, idx_g)
            decision["p_challenger_better"] = p
            if p >= cfg.adopt_prob:
                chosen = challenger
        chosen = self._safe(sim, pool, chosen, watch, y_w, idx_w, decision)
        search_requested = (red and chosen == "static") or yellow or bool(getattr(sim, "force_search", False))
        decision["search_requested"] = search_requested

        # 4. optional search (OpenEvolve + ACE) when asked for and configured
        if search_requested and sim.search is not None:
            policy_source = sim.search.run(sim, k, incumbent, render(1.0, ctx["label_mean"], sim.target_mean,
                                                                      sim.target_scale, epochs=sim.epochs))
            decision["search_ran"] = policy_source is not None      # a new candidate came out and is trained
            decision["search"] = {key: sim.search.last.get(key) for key in
                                  ("returncode", "stopped", "seconds", "best_program", "best_is_initial")}
            if policy_source:
                pool["search"] = sim.train(incumbent, policy_source, k, "v3-search")
                scores["search"] = rmse(sim.predict(pool["search"], "validation"), y_g)
                p = p_better(sim.predict(pool["search"], "validation"), sim.predict(pool[chosen], "validation"), y_g, idx_g)
                decision["p_search_better"] = p
                if p >= cfg.adopt_prob:
                    chosen = self._safe(sim, pool, "search", watch, y_w, idx_w, decision)

        state["deployed"] = pool[chosen]
        if chosen not in ("incumbent",):
            state["deployed_label"] = chosen
        decision.update(action=chosen, rollback=rollback, trained=sorted(trained))
        decision["reason"] = self._reason(decision)
        return decision

    def _safe(self, sim, pool, chosen, watch, y_w, idx_w, decision):
        """Never deploy a model that loses to the static baseline on the newest data (P >= red_prob)."""
        if chosen == "static":
            return chosen
        samples = advantage_samples(sim.predict(pool[chosen], watch), sim.predict(sim.static, watch), y_w, idx_w)
        if float(np.mean(samples < 0)) >= self.cfg.red_prob:
            decision["safety_rollback"] = chosen
            return "static"
        return chosen

    @staticmethod
    def _reason(d):
        parts = [f"当前模型相对静态基线 {d['advantage_newest']:+.1%}"]
        if d["red"]:
            parts.append(f"红灯（更差的概率 {d['p_worse_than_static']:.2f}），立即回退")
        elif d["yellow"]:
            parts.append("黄灯（优势明显缩水）")
        else:
            parts.append("绿灯")
        parts.append(f"上线 {d['action']}")
        if d.get("safety_rollback"):
            parts.append(f"{d['safety_rollback']} 打不过静态基线，改用静态基线")
        if d["search_requested"]:
            parts.append("请求大模型搜索")
        return "；".join(parts)


POLICIES = {
    "never": Never,
    "always": Always,
    "v1": TriggerV1,
    "v3": lambda: V3("v3"),
    "v3_cap1": lambda: V3("v3_cap1", caps=(1.0,), default_cap=1.0),
    "v3_nocap": lambda: V3("v3_nocap", caps=(None,), default_cap=None),
}


def make(name):
    if name not in POLICIES:
        raise ValueError(f"unknown policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name]()
