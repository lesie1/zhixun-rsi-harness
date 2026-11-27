"""Experiment configuration: batches in time order, what each batch may be used for, and the stage plan.

A batch is declared once with
  id, order        unique; `order` is the time order and the only processing order
  path             JSONL in the official format; ${VAR} and paths relative to the config file allowed
  labels_available whether its labels may be read now (false = the controller waits, never reads it)
  uses             subset of {"train", "dev", "final"}; "final" excludes the other two
  exposure         prior uses outside this experiment (honest record; non-empty = not independent)

A stage trains on `train` batches and is scored/gated on one later `dev` batch. Stages come
either from an explicit list ("stages": [{"new_train": [...], "dev": "..."}]) or are derived:
stage 1 trains on `initial_train_batches`; each next stage takes the next train-capable batch
in time order, and the first dev-capable batch after it. In "replay" mode a stage trains on
all earlier training batches plus its new ones; in "new_only" mode on its new ones only.
"""
import copy
import hashlib
import json
import os
import re
from pathlib import Path

from harness import PROJECT_ROOT

USES = ("train", "dev", "final")
BATCH_ID = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
TRAINING_MODES = ("new_only", "replay")
SEARCH_MODES = ("evolve", "continuation")
END_REASONS = ("stagnated", "budget_iterations", "budget_seconds", "budget_tokens", "unhealthy",
               "process_error", "no_search")

DEFAULTS = {
    "training": {"mode": "replay"},
    "stagnation": {"patience": 6, "min_delta": 0.002, "max_consecutive_failures": 8},
    "budget": {"stage_max_iterations": 30, "stage_max_seconds": 3600, "stage_max_tokens": None,
               "max_stages": 6, "max_total_iterations": None},
    "search": {"mode": "continuation", "oe_config": None, "api_base": None, "model": None, "workers": 1,
               "poll_seconds": 2.0, "stop_grace_seconds": 120},
    "advance_on": None,
    "final": None,
    "focus_classes": [],
    "forgetting_tolerance": 0.02,
    "exposure_ledger": None,
    "stages": None,
    "initial_checkpoint": None,
    "initial_policy": None,
}


class ConfigError(ValueError):
    pass


def _merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve(value, base_dir):
    path = Path(os.path.expandvars(str(value)))
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def load_config(path):
    """Read, normalise and validate an experiment config. Raises ConfigError with all problems."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    cfg = _merge(DEFAULTS, raw)
    cfg["config_path"] = str(path)
    base_dir = path.parent
    for key in ("base_config", "initial_checkpoint", "initial_policy", "exposure_ledger"):
        if cfg.get(key):
            cfg[key] = _resolve(cfg[key], base_dir)
    if cfg.get("search", {}).get("oe_config"):
        cfg["search"]["oe_config"] = _resolve(cfg["search"]["oe_config"], base_dir)
    for batch in cfg.get("batches") or []:
        if isinstance(batch, dict) and batch.get("path"):
            batch["path"] = _resolve(batch["path"], base_dir)
    if cfg["advance_on"] is None:
        cfg["advance_on"] = ["stagnated"] if cfg["search"]["mode"] == "evolve" else ["no_search"]
    problems = validate(cfg)
    if problems:
        raise ConfigError("实验配置有问题：\n  - " + "\n  - ".join(problems))
    cfg["batches"] = sorted(cfg["batches"], key=lambda b: b["order"])
    if not cfg.get("initial_policy"):
        cfg["initial_policy"] = str(PROJECT_ROOT / "harness" / "policy_initial.py")
    base = json.loads(Path(cfg["base_config"]).read_text(encoding="utf-8"))
    if not cfg.get("initial_checkpoint"):
        parent = Path(base["parent_checkpoint"])
        cfg["initial_checkpoint"] = str(parent if parent.is_absolute() else (Path(cfg["base_config"]).parent / parent).resolve())
    if base.get("task_name") not in (None, cfg["task_name"]):
        raise ConfigError(f"base_config 的 task_name 是 {base.get('task_name')!r}，实验配置是 {cfg['task_name']!r}")
    cfg["plan"] = make_plan(cfg)
    return cfg


def validate(cfg):
    problems = []
    if not EXPERIMENT_ID.match(str(cfg.get("experiment_id") or "")):
        problems.append("experiment_id 必填，只能包含字母、数字、点、下划线和连字符")
    if not cfg.get("task_name"):
        problems.append("task_name 必填")
    if not cfg.get("base_config") or not Path(cfg["base_config"]).is_file():
        problems.append(f"base_config 不存在：{cfg.get('base_config')}")
    batches = cfg.get("batches")
    if not isinstance(batches, list) or not batches:
        return problems + ["batches 必须是非空列表"]
    ids, orders = set(), set()
    for i, b in enumerate(batches):
        where = f"batches[{i}]"
        if not isinstance(b, dict):
            problems.append(f"{where} 必须是对象")
            continue
        if not BATCH_ID.match(str(b.get("id") or "")):
            problems.append(f"{where}.id 无效")
        elif b["id"] in ids:
            problems.append(f"批次 id 重复：{b['id']}")
        ids.add(b.get("id"))
        if not isinstance(b.get("order"), (int, float)) or isinstance(b.get("order"), bool):
            problems.append(f"{where}.order 必须是数字（时间顺序）")
        elif b["order"] in orders:
            problems.append(f"{where}.order 与其他批次重复：{b['order']}")
        else:
            orders.add(b["order"])
        if not b.get("path"):
            problems.append(f"{where}.path 必填")
        if not isinstance(b.get("labels_available"), bool):
            problems.append(f"{where}.labels_available 必须是 true/false")
        uses = b.get("uses")
        if not isinstance(uses, list) or not uses or any(u not in USES for u in uses):
            problems.append(f"{where}.uses 必须是 {list(USES)} 的非空子集")
        elif "final" in uses and len(uses) > 1:
            problems.append(f"{where}：final 批次不能同时用于 train/dev")
        if not isinstance(b.get("exposure", []), list):
            problems.append(f"{where}.exposure 必须是列表（记录此前的使用情况，没有就写 []）")
        b.setdefault("exposure", [])
    if problems:
        return problems
    by_id = {b["id"]: b for b in batches}
    initial = cfg.get("initial_train_batches")
    if not isinstance(initial, list) or not initial:
        problems.append("initial_train_batches 必填：起点检查点是用哪些批次训练的")
    else:
        for bid in initial:
            if bid not in by_id or "train" not in by_id[bid]["uses"]:
                problems.append(f"initial_train_batches 里的 {bid} 不存在或不允许用于训练")
    mode = cfg["training"].get("mode")
    if mode not in TRAINING_MODES:
        problems.append(f"training.mode 必须是 {TRAINING_MODES} 之一")
    if cfg["search"].get("mode") not in SEARCH_MODES:
        problems.append(f"search.mode 必须是 {SEARCH_MODES} 之一")
    if any(r not in END_REASONS for r in cfg["advance_on"]):
        problems.append(f"advance_on 只能取 {END_REASONS}")
    stag = cfg["stagnation"]
    if not isinstance(stag.get("patience"), int) or stag["patience"] < 1:
        problems.append("stagnation.patience 必须是 >= 1 的整数")
    if not isinstance(stag.get("min_delta"), (int, float)) or stag["min_delta"] < 0:
        problems.append("stagnation.min_delta 必须 >= 0")
    budget = cfg["budget"]
    for key in ("stage_max_iterations", "max_stages"):
        if not isinstance(budget.get(key), int) or budget[key] < 1:
            problems.append(f"budget.{key} 必须是 >= 1 的整数")
    final = cfg.get("final")
    if final is not None:
        if not isinstance(final, dict) or final.get("batch") not in by_id:
            problems.append("final.batch 必须指向一个已声明的批次")
        elif by_id[final["batch"]]["uses"] != ["final"]:
            problems.append("final.batch 的 uses 必须只有 final")
    if not problems:
        try:
            plan = make_plan(cfg)
        except ConfigError as error:
            problems.append(str(error))
        else:
            used = {bid for stage in plan for bid in stage["train"] + [stage["dev"]]}
            if final and final.get("batch") in by_id:
                f = by_id[final["batch"]]
                if f["id"] in used:
                    problems.append("final 批次不能出现在任何阶段里")
                later = [bid for bid in used if by_id[bid]["order"] >= f["order"]]
                if later:
                    problems.append(f"final 批次必须晚于所有训练/开发批次，以下批次不早于它：{sorted(later)}")
    return problems


def make_plan(cfg):
    """List of stages: {"index", "new_train", "train", "dev"} in time order."""
    by_id = {b["id"]: b for b in cfg["batches"]}
    order = lambda bid: by_id[bid]["order"]
    ordered = sorted(cfg["batches"], key=lambda b: b["order"])
    explicit = cfg.get("stages")
    stages = []
    if explicit:
        for i, spec in enumerate(explicit, 1):
            new, dev = list(spec.get("new_train") or []), spec.get("dev")
            if not new or any(b not in by_id or "train" not in by_id[b]["uses"] for b in new):
                raise ConfigError(f"stages[{i - 1}].new_train 必须是允许训练的已声明批次")
            if dev not in by_id or "dev" not in by_id[dev]["uses"]:
                raise ConfigError(f"stages[{i - 1}].dev 必须是允许开发验证的已声明批次")
            stages.append((sorted(new, key=order), dev))
    else:
        new = sorted(cfg["initial_train_batches"], key=order)
        while True:
            dev = next((b["id"] for b in ordered if "dev" in b["uses"] and b["order"] > max(map(order, new))), None)
            if dev is None:
                break
            stages.append((new, dev))
            nxt = next((b["id"] for b in ordered if "train" in b["uses"] and b["order"] > max(map(order, new))), None)
            if nxt is None:
                break
            new = [nxt]
    plan, pool = [], list(cfg["initial_train_batches"])
    previous_max = None
    for index, (new, dev) in enumerate(stages, 1):
        if order(dev) <= max(map(order, new)):
            raise ConfigError(f"第 {index} 阶段的开发批次 {dev} 必须晚于它的训练批次 {new}")
        if previous_max is not None and min(map(order, new)) <= previous_max:
            raise ConfigError(f"第 {index} 阶段的新训练批次 {new} 必须晚于上一阶段的训练批次（严格按时间顺序）")
        previous_max = max(map(order, new))
        pool = sorted(set(pool) | set(new), key=order)
        train = list(pool) if cfg["training"]["mode"] == "replay" else list(new)
        plan.append({"index": index, "new_train": list(new), "train": train, "dev": dev})
    if not plan:
        raise ConfigError("按声明的批次排不出任何阶段（需要至少一个晚于初始训练批次的 dev 批次）")
    return plan


def readiness(cfg, stage):
    """(ready, reasons): every batch the stage needs has readable labels now."""
    by_id = {b["id"]: b for b in cfg["batches"]}
    reasons = []
    for bid in stage["train"] + [stage["dev"]]:
        b = by_id[bid]
        if not b["labels_available"]:
            reasons.append(f"批次 {bid} 的标签还不可用（labels_available=false）")
        elif not Path(b["path"]).is_file():
            reasons.append(f"批次 {bid} 的文件不存在：{b['path']}")
    return not reasons, reasons


def fingerprint(cfg):
    """What must not change once a run has started (appending batches or flipping labels is fine)."""
    keep = {k: cfg[k] for k in ("experiment_id", "task_name", "base_config", "initial_train_batches", "training",
                                "stagnation", "initial_checkpoint", "initial_policy")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def batch_identity(batch):
    return {k: batch[k] for k in ("id", "order", "path", "uses")}


def check_sha256(batch, actual):
    """Raise if the batch declares a SHA-256 and the file on disk differs (wrong or altered file)."""
    expected = batch.get("sha256")
    if expected and expected.removeprefix("sha256:") != actual.removeprefix("sha256:"):
        raise ValueError(f"批次 {batch['id']} 的文件摘要与声明不一致：声明 {expected}，实际 {actual}")
