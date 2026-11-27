"""Build the system message: fixed rules (prompts/system_message.md) + a task brief computed
from the stage context, so the same code serves any dataset in the official format."""
from . import PROJECT_ROOT
from .metrics import HIGHER_IS_BETTER

TEMPLATE = PROJECT_ROOT / "prompts" / "system_message.md"


def _fmt(value):
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def task_brief(context):
    spec, meta, manifest = context.spec, context.parent.meta, context.manifest
    primary = spec["primary"]
    direction = "越大越好" if HIGHER_IS_BETTER[primary] else "越小越好"
    name = context.config.get("task_name") or "（未命名的新任务）"
    lines = ["## 当前任务", f"- 任务: {name}" + (f"（{spec['description']}）" if spec["description"] else "")]
    if context.task == "classification":
        lines.append(f"- 类型: 分类，{len(context.classes)} 类: {', '.join(context.classes)}")
    else:
        lines.append("- 类型: 回归（非负的剩余寿命）" + (f"，单位: {spec['unit']}" if spec["unit"] else "")
                     + "；runner 里的 target 是按训练集均值和标准差标准化后的值")
    lines.append(f"- 主指标: {primary}（{direction}）")
    guards = [f"{metric} 最多变差 {tolerance:.0%}" for metric, tolerance in spec["guards"].items()]
    if context.guard_key:
        guards.append(f"按 {context.guard_key} 分组，任何一组的 {primary} 最多变差 {spec['group_guard']:.0%}")
    lines.append("- 护栏: " + ("；".join(guards) if guards else "无"))
    counts = manifest["counts"]["select"]
    lines.append(f"- 数据: 训练集 {context.train_rows} 行；挑选集 {counts['rows']} 行（评测只在挑选集上进行，"
                 f"划分方式 {manifest['method']}）；网络: {len(meta['paths'])} 个输入 → {meta['hidden']} 个 tanh 隐藏单元 → "
                 f"{len(context.classes) if context.task == 'classification' else 1} 个输出")
    lines.append(f"- 起点检查点: 已训练 {meta['epochs']} 轮 / {meta['steps']} 步，学习率 {meta['learning_rate']}，"
                 f"batch {meta['batch_size']}；挑选集上 {primary}={_fmt(context.parent_metrics[primary])}")
    lines.append(f"- 官方续训（未修改的初始策略，再训 5 轮）: {primary}={_fmt(context.reference_metrics[primary])}，"
                 f"combined_score={context.reference_score['combined_score']:.4f}。你的改动应当超过它。")
    if context.task == "classification":
        counts = context.train_stats["class_counts"]
        lines.append("- 训练集各类样本数: " + "，".join(f"{label} {count}" for label, count in counts.items()))
    else:
        stats = context.train_stats
        lines.append(f"- 训练集目标: 均值 {stats['target_mean']:.4g}，标准差 {stats['target_std']:.4g}，"
                     f"四分位 {' / '.join(f'{q:.4g}' for q in stats['target_quartiles'])}")
    if context.aggregate == "group_mean":
        lines.append("- 注意: 有标签的验证数据分组太少，无法留出独立的验收集；得分按各分组等权平均，"
                     "改动必须对每个分组都成立，不能只照顾样本最多的那一组。")
    return "\n".join(lines)


def system_message(context):
    return TEMPLATE.read_text(encoding="utf-8").replace("{task_brief}", task_brief(context))
