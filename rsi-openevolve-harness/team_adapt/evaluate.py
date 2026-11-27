"""Comparing models on explicitly labelled evaluation sets.

Arms of a stage (all evaluated on the same sets, never across sets):
    pre_update             the stage's starting checkpoint (no training on the new data)
    official_continuation  unmodified policy trained on the same stage data from the same start
    evolved                the search's best program trained the same way (if one was valid)
    selected               what the stage's acceptance gate chose (equals one of the above)

Every set carries its role, so a report can never present a set the model was trained or
selected on as independent evidence:
    train      stage training data (in-sample; never evidence of adaptation)
    dev_select selection split of the stage's dev batch (search scores and stagnation)
    dev_gate   gate split of the stage's dev batch (used once by the stage's gate)
    old:sNN    gate split of an earlier stage (old-scenario / forgetting check; already used)
    final      the reserved final batch (used once, after all stages)
"""
from baselines.model import Model
from harness.analysis import analyze
from harness.dataio import load_dataset, load_meta
from harness.groups import discover_groups
from harness.metrics import HIGHER_IS_BETTER
from harness.scoring import relative_worsening

ROLE_TEXT = {
    "train": "本阶段训练数据（样本内，不能作为适应有效的证据）",
    "dev_select": "本阶段挑选集（搜索打分和停滞判断用过）",
    "dev_gate": "本阶段验收集（本阶段验收决定用过一次）",
    "old": "早先阶段的验收集（旧场景退化检查；已被当时的验收用过）",
    "final": "预留的最终验收批次（全流程只用一次）",
}


def evaluate_model(checkpoint, path, spec, cache_dir=None):
    model = Model.load(checkpoint)
    data = load_dataset(path, model.meta["task"], model.meta["classes"], model.meta["paths"], cache_dir=cache_dir)
    _, meta = load_meta(path, cache_dir=cache_dir)
    groups = discover_groups(meta)
    bundle = analyze(model, data, meta, spec, groups, top_k=3)
    result = {"rows": bundle["n"], "overall": bundle["overall"]}
    if model.meta["task"] == "classification":
        result["per_class"] = {row["class"]: {k: row.get(k) for k in ("support", "predicted", "recall", "precision", "f1")}
                               for row in bundle["per_class"]}
        result["top_confusions"] = bundle.get("top_confusions", [])
    else:
        result["by_group"] = {key: [{k: row.get(k) for k in ("group", "n", spec["primary"], "bias")} for row in rows]
                              for key, rows in bundle["by_group"].items()}
    return result


def compare(arms, sets, spec, cache_dir=None, forgetting_tolerance=0.02, focus=()):
    """arms: {name: checkpoint path}; sets: {name: {"path", "role", "batch"}} -> comparison table."""
    primary = spec["primary"]
    table = {"primary": primary, "higher_is_better": HIGHER_IS_BETTER[primary], "focus": list(focus), "sets": {}}
    for set_name, info in sets.items():
        entry = {"role": info["role"], "role_text": ROLE_TEXT[info["role"].split(":")[0]], "batch": info.get("batch"),
                 "path": info["path"], "arms": {}}
        for arm, checkpoint in arms.items():
            if checkpoint:
                entry["arms"][arm] = evaluate_model(checkpoint, info["path"], spec, cache_dir=cache_dir)
        base = entry["arms"].get("pre_update")
        if base:
            for arm, result in entry["arms"].items():
                if arm == "pre_update":
                    continue
                delta = {"primary": result["overall"][primary] - base["overall"][primary],
                         "primary_relative_worsening": relative_worsening(primary, result["overall"][primary],
                                                                         base["overall"][primary])}
                for cls in focus:
                    if cls in result.get("per_class", {}) and cls in base.get("per_class", {}):
                        delta[f"recall:{cls}"] = (result["per_class"][cls]["recall"] or 0.0) - (base["per_class"][cls]["recall"] or 0.0)
                result["vs_pre_update"] = delta
                if info["role"].startswith("old"):
                    result["regressed"] = delta["primary_relative_worsening"] > forgetting_tolerance
        table["sets"][set_name] = entry
    return table


def verdict(table):
    """Evidence level for "the selected model adapted": only gate/final sets can support it."""
    primary = table["primary"]
    out = {}
    for set_name, entry in table["sets"].items():
        selected, base = entry["arms"].get("selected"), entry["arms"].get("pre_update")
        if not selected or not base or entry["role"] in ("train", "dev_select"):
            continue
        better = (selected["overall"][primary] > base["overall"][primary]) == table["higher_is_better"] and \
            selected["overall"][primary] != base["overall"][primary]
        out[set_name] = {"role": entry["role"], "selected_better_than_pre_update": bool(better),
                         "selected": selected["overall"][primary], "pre_update": base["overall"][primary]}
    return out
