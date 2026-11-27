"""Detailed feedback report for one evaluated policy.

Six levels of evidence, each computed by code (never by an LLM):
  1 outcome and hard constraints      4 model internals (saturation, clipping, drift of weights)
  2 training dynamics per epoch       5 worst samples
  3 data slices (class/group/range)   6 change against fixed references
plus symptom flags that map evidence to the hook most likely responsible.

`build_report` returns the full JSON-serialisable report (stored on disk, part of the ACE
record). `render` turns it into the compact Chinese text placed in the LLM prompt.
Only selection-split results are reported; gate results never reach the prompt.
"""
from .metrics import HIGHER_IS_BETTER
from .scoring import relative_worsening

SCHEMA = "harness.feedback.v1"
HEADLINE = {"classification": ("macro_f1", "accuracy"), "regression": ("rmse", "mae")}


def _better(name, a, b):
    """True when value a is strictly better than b for metric `name`."""
    return a > b if HIGHER_IS_BETTER[name] else a < b


def _curve(runner_report, spec, task):
    epochs = runner_report["per_epoch"]
    metric = spec["primary"] if spec["primary"] in epochs[0]["val"] else HEADLINE[task][0]
    values = [entry["val"][metric] for entry in epochs]
    best = max(range(len(values)), key=lambda i: values[i] if HIGHER_IS_BETTER[metric] else -values[i])
    return {"metric": metric, "epochs": [entry["epoch"] for entry in epochs],
            "train_loss": [entry["train_loss"] for entry in epochs], "val": values,
            "before": runner_report["validation_before"].get(metric),
            "best_index": best, "selected_index": runner_report["selected_epoch_index"],
            "learning_rate": [entry["learning_rate"] for entry in epochs]}


def _deltas(candidate, other, spec, task):
    """Per-slice change of `candidate` against `other` (both evaluation bundles)."""
    primary = spec["primary"]
    result = {"overall": {name: candidate["overall"][name] - other["overall"][name] for name in candidate["overall"]}}
    if task == "classification":
        reference = {row["class"]: row for row in other["per_class"]}
        result["per_class"] = [
            {"class": row["class"],
             "recall": None if row["recall"] is None or reference[row["class"]]["recall"] is None
             else row["recall"] - reference[row["class"]]["recall"],
             "f1": (row["f1"] or 0.0) - (reference[row["class"]]["f1"] or 0.0)}
            for row in candidate["per_class"]]
    else:
        result["by_quantile"] = [
            {"low": row["low"], "high": row["high"], "rmse": row["rmse"] - ref["rmse"], "bias": row["bias"] - ref["bias"]}
            for row, ref in zip(candidate["by_quantile"], other["by_quantile"])]
    result["by_group"] = {}
    for key, rows in candidate["by_group"].items():
        reference = {row["group"]: row for row in other["by_group"].get(key, [])}
        result["by_group"][key] = [{"group": row["group"], primary: row[primary] - reference[row["group"]][primary]}
                                   for row in rows if row["group"] in reference]
    return result


def _flags(report, bundle, runner_report, spec, context):
    task, primary, flags = context["task_type"], spec["primary"], []

    def add(code, detail, hint):
        flags.append({"code": code, "detail": detail, "hint": hint})

    score, curve = report["score"], report["curve"]
    for violation in score["violations"]:
        add("GUARD_VIOLATION",
            f"{violation['guard']} 变差 {violation['worsening']:.1%}，超过容忍度 {violation['tolerance']:.0%}",
            "主指标的提升不能靠牺牲这一项换来；缩小改动幅度，或针对变差的部分单独处理")
    if score["primary_ratio"] < 1.0:
        add("WORSE_THAN_START", f"{primary} 比起点检查点更差（比值 {score['primary_ratio']:.4f}）",
            "这一轮训练让模型变差了：降低 learning_rate、减少 CONFIG['epochs']，或用 pick_epoch 选更早的一轮")
    reference = context.get("reference_metrics")
    if reference and _better(primary, reference[primary], report["metrics"]["candidate"][primary]):
        add("WORSE_THAN_REFERENCE", f"{primary} 不如官方续训（初始策略）的 {reference[primary]:.4g}",
            "这次改动不如什么都不改；回到初始写法再换一个方向")

    values, best, selected = curve["val"], curve["best_index"], curve["selected_index"]
    metric = curve["metric"]
    if best < len(values) - 1 and relative_worsening(metric, values[-1], values[best]) > 0.01:
        chosen_note = "而且实际选用的不是最好的一轮" if selected != best else "但已通过 pick_epoch 选到最好的一轮"
        add("LATE_EPOCH_DEGRADATION",
            f"{metric} 在第 {best + 1}/{len(values)} 轮最好（{values[best]:.4g}），最后一轮退到 {values[-1]:.4g}；{chosen_note}",
            "用 pick_epoch 选挑选集上最好的一轮，或减少轮数、让 learning_rate 逐轮衰减")
    elif best == len(values) - 1 and len(values) >= 2 and relative_worsening(metric, values[-2], values[-1]) > 0.005:
        add("STILL_IMPROVING", f"{metric} 到最后一轮仍在变好（{values[-2]:.4g} → {values[-1]:.4g}）",
            "可以增加 CONFIG['epochs']")
    losses = curve["train_loss"]
    if len(losses) >= 2 and losses[-1] >= losses[0]:
        add("NO_TRAIN_PROGRESS", f"训练损失没有下降（{losses[0]:.4g} → {losses[-1]:.4g}）",
            "学习率可能过大或过小，或损失/权重的改动让优化变难")
    elif len(losses) >= 2 and losses[-1] < 0.95 * losses[0] and relative_worsening(metric, values[-1], values[0]) > 0.01:
        add("TRAIN_VAL_DIVERGE", "训练损失在下降，但挑选集指标在变差（过拟合或分布漂移）",
            "提前停止（pick_epoch）、用 extra_grad 加 L2 正则，或调低学习率")

    last = runner_report["per_epoch"][runner_report["selected_epoch_index"]]
    if bundle["saturation"] > 0.30:
        add("HIGH_SATURATION", f"{bundle['saturation']:.0%} 的隐藏单元输出接近 ±1（tanh 饱和）",
            "饱和单元几乎没有梯度：调低学习率，或用 extra_grad 对 w1 加 L2 抑制权重增大")
    clip_rate = sum(e["clip_rate"] for e in runner_report["per_epoch"]) / len(runner_report["per_epoch"])
    if clip_rate > 0.5:
        add("GRAD_CLIP_FREQUENT", f"{clip_rate:.0%} 的更新步触发了梯度裁剪",
            "梯度经常超过 CONFIG['grad_clip']：检查样本权重或损失的量级是否过大")
    if last["weight_change"] > 0.5:
        add("LARGE_WEIGHT_CHANGE", f"权重相对起点变化了 {last['weight_change']:.0%}",
            "离起点检查点很远，容易遗忘已学到的内容；减小学习率或轮数")

    if task == "classification":
        train_counts = context.get("train_class_counts") or {}
        for row in bundle["per_class"]:
            if row["support"] and not row["predicted"]:
                add("CLASS_NEVER_PREDICTED",
                    f"{row['class']} 一次都没被预测出来（挑选集 {row['support']} 个，训练集 {train_counts.get(row['class'], '?')} 个）",
                    "在 sample_weights 里提高该类样本的权重（例如按类别频率的倒数加权）")
            elif row["support"] >= 5 and row["recall"] is not None and row["recall"] < 0.5:
                add("CLASS_LOW_RECALL", f"{row['class']} 召回率只有 {row['recall']:.2f}（挑选集 {row['support']} 个）",
                    "提高该类权重，或检查它主要被错分成哪一类（见混淆）")
        train_total = sum(train_counts.values())
        if train_total:
            distance = 0.5 * sum(abs(bundle["true_share"][name] - train_counts.get(name, 0) / train_total)
                                 for name in bundle["true_share"])
            if distance > 0.2:
                add("LABEL_SHIFT", f"挑选集与训练集的类别占比差异很大（总变差距离 {distance:.2f}）",
                    "训练集的类别比例不代表新数据：不要让模型过度偏向训练集里的大类")
    else:
        quartiles = bundle["by_quantile"]
        if quartiles:
            low, high = quartiles[0], quartiles[-1]
            if low["bias"] > 0.5 * low["rmse"]:
                add("OVERESTIMATE_LOW_TARGET",
                    f"目标最小的四分之一（{low['low']:.0f}–{low['high']:.0f}）平均高估 {low['bias']:+.1f}",
                    "临近失效时高估寿命：在 output_grad 里加大高估（预测>真值）的惩罚，或在 sample_weights 里提高小目标样本的权重")
            if high["bias"] < -0.5 * high["rmse"]:
                add("UNDERESTIMATE_HIGH_TARGET",
                    f"目标最大的四分之一（{high['low']:.0f}–{high['high']:.0f}）平均低估 {high['bias']:+.1f}",
                    "长寿命样本被系统性低估：提高大目标样本的权重，或检查是否训练轮数不足")
        nasa = bundle.get("nasa")
        if nasa and nasa["top1pct_share"] > 0.9:
            add("NASA_DOMINATED_BY_FEW",
                f"NASA 分数的 {nasa['top1pct_share']:.0%} 来自误差最大的 1% 样本（其中高估占 {nasa['over_estimate_share']:.0%}）",
                "少数大幅高估决定了 NASA 分数：重点压低大的正误差")
    for key, rows in bundle["by_group"].items():
        for row in rows:
            if row["share"] > 0.7 and len(rows) > 1:
                add("GROUP_DOMINATES", f"分组 {key}={row['group']} 占挑选集 {row['share']:.0%}，总指标基本由它决定",
                    "不要只为这一组调参；注意其他组是否变差")
    for warning in context.get("split_warnings") or []:
        add("SPLIT_WARNING", warning, "挑选集与验收集相同，分数存在选择偏差；改动要对每个分组都成立")
    return flags


def build_report(context, status, problems=None, bundle=None, scoring=None, runner_report=None,
                 candidate_metrics=None, stderr_tail=None):
    """Assemble the full report. `context` is StageContext.feedback_context()."""
    report = {"schema": SCHEMA, "status": status, "problems": list(problems or []),
              "task": {**{key: context[key] for key in ("task_name", "task_type", "stage_id", "primary", "unit",
                                                        "split_method", "aggregate")},
                       "ranking_metrics": [context["spec"]["primary"], *context["spec"]["guards"]]},
              "flags": []}
    if stderr_tail:
        report["stderr_tail"] = stderr_tail
    if status != "ok":
        return report
    spec, task = context["spec"], context["task_type"]
    report["score"] = scoring
    report["metrics"] = {"candidate": candidate_metrics, "parent_checkpoint": context["parent_metrics"],
                         "reference_policy": context.get("reference_metrics"),
                         "candidate_pooled": bundle["overall"]}
    report["constraints"] = {"passed": True,
                             "steps": [runner_report["initial_steps"], runner_report["final_steps"]],
                             "epochs": [runner_report["initial_epochs"], runner_report["final_epochs"]]}
    report["curve"] = _curve(runner_report, spec, task)
    report["slices"] = {key: bundle[key] for key in ("per_class", "top_confusions", "true_share", "pred_share",
                                                     "by_quantile", "by_group", "nasa", "extras") if key in bundle}
    report["deltas"] = {"vs_parent_checkpoint": _deltas(bundle, context["parent_bundle"], spec, task)}
    if context.get("reference_bundle"):
        report["deltas"]["vs_reference_policy"] = _deltas(bundle, context["reference_bundle"], spec, task)
    selected = runner_report["per_epoch"][runner_report["selected_epoch_index"]]
    report["internals"] = {
        "saturation": bundle["saturation"], "saturation_before": runner_report["saturation_before"],
        "clip_rate": [entry["clip_rate"] for entry in runner_report["per_epoch"]],
        "grad_norm_mean": [entry["grad_norm_mean"] for entry in runner_report["per_epoch"]],
        "weight_change": selected["weight_change"], "config": runner_report["config"],
        "sample_weights": runner_report["sample_weights"], "training_seconds": runner_report["training_seconds"]}
    report["worst"] = bundle["worst"]
    report["flags"] = _flags(report, bundle, runner_report, spec, context)
    return report


# ---------------------------------------------------------------------------------- rendering

def _num(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _series(values, digits=4):
    return " ".join(_num(float(v), digits) for v in values)


def _signed(value, digits=3):
    return "-" if value is None else f"{value:+.{digits}g}"


def render(report, max_chars=3500):
    """Compact Chinese text for the prompt. Sections are dropped from the end to fit `max_chars`."""
    task = report["task"]
    if report["status"] != "ok":
        names = {"policy_rejected": "策略代码未通过静态检查，没有运行", "train_failed": "训练过程出错",
                 "timeout": "训练超时被终止", "constraint_violation": "产出的检查点违反续训规则",
                 "internal_error": "评测器内部错误"}
        lines = [f"[状态] 失败：{names.get(report['status'], report['status'])}，combined_score=0"]
        lines += [f"  - {problem}" for problem in report["problems"][:8]]
        if report.get("stderr_tail"):
            lines.append("[错误输出]\n" + report["stderr_tail"])
        lines.append("[提示] 只修改 EVOLVE-BLOCK 内的代码，保持钩子的名字、参数和返回值形状不变；只能 import numpy 和 math。")
        return "\n".join(lines)[:max_chars]

    score, metrics, curve = report["score"], report["metrics"], report["curve"]
    primary, unit = task["primary"], f" {task['unit']}" if task["unit"] else ""
    candidate, parent, reference = metrics["candidate"], metrics["parent_checkpoint"], metrics["reference_policy"]
    sections = []
    steps, epochs = report["constraints"]["steps"], report["constraints"]["epochs"]
    sections.append(f"[状态] 评估成功；续训规则全部通过（步数 {steps[0]}→{steps[1]}，轮次 {epochs[0]}→{epochs[1]}，"
                    "标准化和网络结构未改）")
    line = (f"[得分] combined_score={score['combined_score']:.4f}（1.0=与起点检查点持平）  "
            f"{primary}={_num(candidate[primary])}{unit}，起点 {_num(parent[primary])}")
    if reference:
        line += f"，官方续训 {_num(reference[primary])}"
    if task["aggregate"] == "group_mean":
        line += f"\n  （无独立验收集：以上为各分组等权平均；按样本合并的 {primary}={_num(metrics['candidate_pooled'][primary])}）"
    others = [name for name in candidate if name != primary and name in parent
              and name in ("accuracy", "mae", "nasa_score")]
    if others:
        line += "\n  辅助指标: " + "；".join(f"{name}={_num(candidate[name])}（起点 {_num(parent[name])}）" for name in others)
    line += "\n  护栏: " + ("全部通过" if score["guard_ok"] else f"有 {len(score['violations'])} 项违规，已扣分 {score['penalty']:.3f}")
    sections.append(line)

    if report["flags"]:
        sections.append("[症状与建议]\n" + "\n".join(
            f"  - {flag['code']}: {flag['detail']}。建议：{flag['hint']}" for flag in report["flags"][:8]))

    selected = curve["selected_index"]
    sections.append(
        f"[训练曲线] 本轮训练 {len(curve['val'])} 轮，选用第 {selected + 1} 轮\n"
        f"  训练损失: {_series(curve['train_loss'])}\n"
        f"  挑选集 {curve['metric']}: 起点 {_num(curve['before'])} → {_series(curve['val'])}\n"
        f"  学习率: {_series(curve['learning_rate'], 3)}")

    slices = report["slices"]
    if "per_class" in slices:
        rows = [f"  {row['class']}: P={_num(row['precision'], 3)} R={_num(row['recall'], 3)} F1={_num(row['f1'], 3)} "
                f"挑选集{row['support']}个 预测{row['predicted']}个" for row in slices["per_class"]]
        text = "[按类别]\n" + "\n".join(rows)
        if slices.get("top_confusions"):
            text += "\n  主要混淆: " + "；".join(f"{c['true']}→{c['pred']} {c['count']}个" for c in slices["top_confusions"][:4])
        sections.append(text)
    if "by_quantile" in slices:
        rows = [f"  真值 {row['low']:.0f}–{row['high']:.0f}（{row['n']}个）: RMSE={_num(row['rmse'])} 平均偏差={row['bias']:+.4g}"
                for row in slices["by_quantile"]]
        extras = slices.get("extras", {})
        text = "[按目标区间]（偏差>0 表示高估）\n" + "\n".join(rows)
        text += (f"\n  整体: 平均偏差={extras.get('bias', 0):+.4g}，高估比例={extras.get('over_rate', 0):.0%}，"
                 f"预测被截到0的比例={extras.get('clipped_zero_rate', 0):.1%}")
        if slices.get("nasa"):
            text += f"；NASA分数的 {slices['nasa']['over_estimate_share']:.0%} 来自高估"
        sections.append(text)
    for key, rows in slices.get("by_group", {}).items():
        if len(rows) > 12:
            continue
        body = "\n".join(f"  {row['group']}（{row['n']}个，占{row['share']:.0%}）: {primary}={_num(row[primary])}"
                         + (f" 偏差={row['bias']:+.4g}" if "bias" in row else "") for row in rows)
        sections.append(f"[按分组 {key}]\n{body}")

    deltas = report["deltas"].get("vs_reference_policy") or report["deltas"]["vs_parent_checkpoint"]
    label = "官方续训" if "vs_reference_policy" in report["deltas"] else "起点检查点"
    parts = [f"{name} {_signed(value)}" for name, value in deltas["overall"].items()
             if name in task["ranking_metrics"]]
    text = f"[相对{label}的变化] " + "，".join(parts)
    if "per_class" in deltas:
        text += "\n  各类召回率变化: " + "，".join(f"{row['class']} {_signed(row['recall'])}" for row in deltas["per_class"])
    if "by_quantile" in deltas:
        text += "\n  各目标区间 RMSE 变化: " + "，".join(
            f"{row['low']:.0f}–{row['high']:.0f} {_signed(row['rmse'])}" for row in deltas["by_quantile"])
    for key, rows in deltas["by_group"].items():
        if 1 < len(rows) <= 12:
            text += f"\n  各分组({key}) {primary} 变化: " + "，".join(f"{row['group']} {_signed(row[primary])}" for row in rows)
    sections.append(text)

    internals = report["internals"]
    weights = internals["sample_weights"]
    weight_note = "全部相同" if weights["cv"] < 1e-9 else f"最小 {_num(weights['min'], 3)} 最大 {_num(weights['max'], 3)}"
    sections.append(
        f"[模型内部] tanh 饱和 {internals['saturation']:.1%}（起点 {internals['saturation_before']:.1%}）；"
        f"梯度裁剪触发率 {_series([100 * r for r in internals['clip_rate']], 3)} %；"
        f"权重相对起点变化 {internals['weight_change']:.1%}；样本权重 {weight_note}；"
        f"batch={internals['config']['batch_size']}，训练耗时 {internals['training_seconds']:.2f}s")

    if report["worst"]:
        if "confidence" in report["worst"][0]:
            pairs = {}
            for w in report["worst"]:
                pairs.setdefault((w["true"], w["pred"]), []).append(w["confidence"])
            rows = [f"{true}→{pred} ×{len(c)}（置信度最高 {max(c):.2f}）" for (true, pred), c in pairs.items()]
            sections.append("[最自信的错误] " + "；".join(rows))
        else:
            rows = [f"真值 {w['true']:.0f} 预测 {w['pred']:.0f}" for w in report["worst"][:5]]
            sections.append("[误差最大的样本] " + "；".join(rows))

    text = "\n".join(sections)
    while len(text) > max_chars and len(sections) > 3:
        sections.pop()
        text = "\n".join(sections)
    return text[:max_chars]
