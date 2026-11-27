"""Configuration of the A-type (gas classification) adaptation flow.

Batches are declared once, in time order, with the roles they may play:
    train   may be used for gradient training (history, or new samples in a data-mixing stage)
    alarm   may be used to diagnose the deployed model and to score/gate the repair search
    dev     may be used to choose among data-mixing candidates
    final   reserved; used once, after a mixing candidate has been frozen (exclusive role)

A round is {"alarm", "selection", "final"} (batch ids, strictly increasing in time). Rounds are
processed in order; round r+1 starts from whatever round r deployed.
"""
import copy
import json
import os
from pathlib import Path

from harness import PROJECT_ROOT
from harness.dataio import file_sha256

from . import mixture

ROLES = ("train", "alarm", "dev", "final")
DEFAULTS = {
    "diagnosis": {"min_support": 15, "recall_drop": 0.3, "precision_drop": 0.3, "confidence": 0.9,
                  "resamples": 1000, "min_confusion_share": 0.3, "seed": 42},
    "repair": {
        "search": {"mode": "continuation", "oe_config": None, "api_base": None, "model": None, "workers": 1,
                   "poll_seconds": 2.0, "stop_grace_seconds": 120},
        "stagnation": {"patience": 5, "min_delta": 0.002, "alarm_min_delta": 0.02, "max_consecutive_failures": 8},
        "budget": {"stage_max_iterations": 30, "stage_max_seconds": 3600, "stage_max_tokens": None},
        "success": {"target_recall_gain": 0.25, "confusion_guard": 0.1, "other_guard": 0.1},
        "advance_on": None,
    },
    "mixture": {"fractions": [0.0, 0.02, 0.05, 0.10, 0.15, 0.20], "degraded_shares": [1.0, 0.75, 0.5, 0.25],
                "cap": 0.20, "tie_tolerance": 0.005, "patience_levels": 2, "seed": 42, "policy": None,
                "min_gain": 0.1, "confusion_guard": 0.1, "other_guard": 0.1,
                "min_new_samples": {"degraded": 3, "confusion": 1}},
    "acceptance": {"target_recall_gain": 0.25, "confusion_guard": 0.1, "other_guard": 0.1, "threshold": 0.9,
                   "resamples": 2000, "seed": 42},
    "playbook": {"base_experiences": str(PROJECT_ROOT / "team_gas" / "base_experiences.json")},
    "display_names": {},
    "exposure_ledger": None,
    "initial_checkpoint": None,
    "initial_policy": None,
}


class ConfigError(ValueError):
    pass


def _merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = _merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def _resolve(value, base_dir):
    path = Path(os.path.expandvars(str(value)))
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def load(path):
    path = Path(path).resolve()
    cfg = _merge(DEFAULTS, json.loads(path.read_text(encoding="utf-8")))
    cfg["config_path"] = str(path)
    base_dir = path.parent
    for key in ("base_config", "initial_checkpoint", "initial_policy", "exposure_ledger"):
        if cfg.get(key):
            cfg[key] = _resolve(cfg[key], base_dir)
    if cfg["repair"]["search"].get("oe_config"):
        cfg["repair"]["search"]["oe_config"] = _resolve(cfg["repair"]["search"]["oe_config"], base_dir)
    if cfg["mixture"].get("policy"):
        cfg["mixture"]["policy"] = _resolve(cfg["mixture"]["policy"], base_dir)
    cfg["playbook"]["base_experiences"] = _resolve(cfg["playbook"]["base_experiences"], base_dir)
    for batch in cfg.get("batches") or []:
        if isinstance(batch, dict) and batch.get("path"):
            batch["path"] = _resolve(batch["path"], base_dir)
    if cfg["repair"]["advance_on"] is None:
        cfg["repair"]["advance_on"] = ["stagnated"] if cfg["repair"]["search"]["mode"] == "evolve" else ["no_search"]
    problems = validate(cfg)
    if problems:
        raise ConfigError("A 流程配置有问题：\n  - " + "\n  - ".join(problems))
    cfg["batches"] = sorted(cfg["batches"], key=lambda b: b["order"])
    base = json.loads(Path(cfg["base_config"]).read_text(encoding="utf-8"))
    if not cfg.get("initial_checkpoint"):
        parent = Path(base["parent_checkpoint"])
        cfg["initial_checkpoint"] = str(parent if parent.is_absolute() else (Path(cfg["base_config"]).parent / parent).resolve())
    if not cfg.get("initial_policy"):
        cfg["initial_policy"] = str(PROJECT_ROOT / "harness" / "policy_initial.py")
    if not cfg["mixture"].get("policy"):
        cfg["mixture"]["policy"] = str(PROJECT_ROOT / "harness" / "policy_initial.py")
    return cfg


def validate(cfg):
    problems = []
    for key in ("experiment_id", "task_name", "base_config"):
        if not cfg.get(key):
            problems.append(f"{key} 必填")
    if cfg.get("base_config") and not Path(cfg["base_config"]).is_file():
        problems.append(f"base_config 不存在：{cfg['base_config']}")
    batches = cfg.get("batches")
    if not isinstance(batches, list) or not batches:
        return problems + ["batches 必须是非空列表"]
    by_id, orders = {}, set()
    for i, b in enumerate(batches):
        where = f"batches[{i}]"
        if not isinstance(b, dict) or not b.get("id") or not b.get("path"):
            problems.append(f"{where} 需要 id 和 path")
            continue
        if b["id"] in by_id:
            problems.append(f"批次 id 重复：{b['id']}")
        by_id[b["id"]] = b
        if not isinstance(b.get("order"), (int, float)) or isinstance(b.get("order"), bool) or b["order"] in orders:
            problems.append(f"{where}.order 必须是不重复的数字（时间顺序）")
        orders.add(b.get("order"))
        if not isinstance(b.get("labels_available"), bool):
            problems.append(f"{where}.labels_available 必须是 true/false")
        uses = b.get("uses")
        if not isinstance(uses, list) or any(u not in ROLES for u in uses):
            problems.append(f"{where}.uses 必须是 {list(ROLES)} 的子集（可以为空：只登记、不使用）")
        elif "final" in uses and len(uses) > 1:
            problems.append(f"{where}：final 批次不能兼任其他用途")
        b.setdefault("exposure", [])
    if problems:
        return problems
    initial = cfg.get("initial_train_batches") or []
    if not initial or any(x not in by_id or "train" not in by_id[x]["uses"] for x in initial):
        problems.append("initial_train_batches 必须是允许 train 的已声明批次")
    rounds = cfg.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return problems + ["rounds 必须是非空列表，每轮写 alarm / selection / final"]
    last_alarm = max((by_id[x]["order"] for x in initial if x in by_id), default=None)
    finals = set()
    for i, r in enumerate(rounds, 1):
        ids = {k: r.get(k) for k in ("alarm", "selection", "final")}
        if any(v not in by_id for v in ids.values()):
            problems.append(f"第 {i} 轮的 alarm/selection/final 必须是已声明批次")
            continue
        need = {"alarm": "alarm", "selection": "dev", "final": "final"}
        for key, role in need.items():
            if role not in by_id[ids[key]]["uses"]:
                problems.append(f"第 {i} 轮的 {key}={ids[key]} 没有声明 {role} 用途")
        a, s, f = (by_id[ids[k]]["order"] for k in ("alarm", "selection", "final"))
        if not (a < s < f):
            problems.append(f"第 {i} 轮必须满足时间顺序 alarm < selection < final")
        if last_alarm is not None and a <= last_alarm:
            problems.append(f"第 {i} 轮的报警批次必须晚于此前的训练/报警批次")
        last_alarm = a
        if ids["final"] in finals:
            problems.append(f"最终批次 {ids['final']} 被多轮使用")
        finals.add(ids["final"])
        for other in rounds:
            if ids["final"] in (other.get("alarm"), other.get("selection")):
                problems.append(f"最终批次 {ids['final']} 不能在任何一轮里兼作报警或开发批次")
    cap = cfg["mixture"].get("cap")
    if not isinstance(cap, (int, float)) or not 0 <= cap <= mixture.HARD_CAP:
        problems.append(f"mixture.cap 必须在 0 到 {mixture.HARD_CAP} 之间（20% 是硬上限，只能调低）")
    fractions = cfg["mixture"].get("fractions") or []
    if not fractions or sorted(fractions) != list(fractions) or fractions[0] != 0 or any(f < 0 for f in fractions):
        problems.append("mixture.fractions 必须从 0 开始递增（0 是同预算旧数据对照）")
    elif max(fractions) > cap + 1e-12:
        problems.append(f"mixture.fractions 里有超过上限 {cap} 的值")
    shares = cfg["mixture"].get("degraded_shares") or []
    if not shares or any(not 0 <= x <= 1 for x in shares):
        problems.append("mixture.degraded_shares 必须是 0..1 之间的值（新数据中退化类所占份额）")
    if cfg["repair"]["search"].get("mode") not in ("evolve", "continuation"):
        problems.append("repair.search.mode 必须是 evolve 或 continuation")
    return problems


def batch(cfg, batch_id):
    return next(b for b in cfg["batches"] if b["id"] == batch_id)


def ready(cfg, *batch_ids):
    reasons = []
    for bid in batch_ids:
        b = batch(cfg, bid)
        if not b["labels_available"]:
            reasons.append(f"批次 {bid} 的标签还不可用")
        elif not Path(b["path"]).is_file():
            reasons.append(f"批次 {bid} 的文件不存在：{b['path']}")
    return not reasons, reasons


def check_sha256(b):
    actual = file_sha256(b["path"])
    expected = b.get("sha256")
    if expected and expected.removeprefix("sha256:") != actual.removeprefix("sha256:"):
        raise ValueError(f"批次 {b['id']} 的文件摘要与声明不一致：声明 {expected}，实际 {actual}")
    return actual


def name(cfg, cls):
    shown = cfg.get("display_names", {}).get(cls)
    return f"{shown}({cls})" if shown else cls
