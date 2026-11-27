"""Markdown summary of an adaptation experiment (generated from state, comparisons and final report)."""
import json
from pathlib import Path

from harness.dataio import read_json

ARM_TEXT = {"pre_update": "更新前模型", "official_continuation": "官方续训（同数据同预算）", "evolved": "进化策略续训",
            "selected": "验收门选中", "initial": "实验起点", "last_pre_update": "最后阶段更新前",
            "last_official_continuation": "最后阶段官方续训", "last_evolved": "最后阶段进化策略"}


def _fmt(value):
    return "-" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))


def _set_table(entry, primary, focus):
    head = ["模型", primary] + [f"{c} 召回" for c in focus] + ["相对更新前"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for arm, result in entry["arms"].items():
        row = [ARM_TEXT.get(arm, arm), _fmt(result["overall"].get(primary))]
        for cls in focus:
            row.append(_fmt((result.get("per_class", {}).get(cls) or {}).get("recall")))
        delta = result.get("vs_pre_update")
        row.append("-" if not delta else f"{delta['primary']:+.4f}" + (" 退化" if result.get("regressed") else ""))
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build(out_dir):
    out = Path(out_dir)
    state = read_json(out / "state.json")
    events = [json.loads(l) for l in (out / "events.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lines = [f"# 持续适应实验 {state['experiment_id']}（任务 {state['task_name']}）", "",
             f"状态：{state['status']}" + (f"（{state['message']}）" if state.get("message") else ""), "",
             "说明：所有比较都在同一个评估集内进行；不同数据集上的分数不直接相比。搜索停滞只表示在该挑选集上找不到足够改进，"
             "不等于已确认数据漂移。训练集上的分数不能作为适应有效的证据。", ""]
    for stage in state["stages"]:
        plan = stage["plan"]
        lines += [f"## 第 {stage['index']} 阶段 {stage['stage_id']}", "",
                  f"- 训练数据：{' + '.join(plan['train'])}（新数据 {' + '.join(plan['new_train'])}，方式 {stage['training_mode']}，"
                  f"{stage['data']['train']['rows']} 行）；开发批次 {plan['dev']}（{stage['data']['dev']['rows']} 行）",
                  f"- 起点：{stage['parent']['from']}（sha256 {stage['parent']['sha256'][7:19]}…）"]
        for warning in stage.get("warnings") or []:
            lines.append(f"- 注意：{warning}")
        search = stage.get("search") or {}
        if search:
            lines.append(f"- 搜索结束原因：{search.get('end_reason')}；迭代 {search.get('iterations')}，"
                         f"{search.get('seconds')} 秒，token {search.get('tokens')}")
            if search.get("description"):
                lines.append(f"- {search['description']}")
        if stage.get("gate"):
            lines.append(f"- 验收门：{stage['gate']['decision']}（{stage['gate']['reason']}）→ 下一阶段起点 {stage['selected']['from']}")
        if stage.get("comparison"):
            table = read_json(stage["comparison"])
            focus = table.get("focus", [])
            for name, entry in table["sets"].items():
                lines += ["", f"**{name}**：{entry['role_text']}", ""] + _set_table(entry, table["primary"], focus)
            for name, why in (table.get("missing_arms") or {}).items():
                lines.append(f"\n（{ARM_TEXT.get(name, name)}：{why}）")
        lines.append("")
    final = state.get("final")
    if final:
        report = read_json(final["path"])
        lines += ["## 最终验收", "",
                  f"批次 {report['batch']}（{report['rows']} 行）；{'独立' if report['independent'] else '非独立（已暴露或重复使用）'}", ""]
        entry = report["table"]["sets"]["final"]
        lines += _set_table(entry, report["table"]["primary"], report["table"].get("focus", []))
    switches = [e for e in events if e["event"] == "phase_switch"]
    if switches:
        lines += ["", "## 阶段切换", ""] + [f"- {e['at']}：{e['from_stage']} → 第 {e['to_index']} 阶段，触发原因 {e['trigger']}，"
                                         f"新训练批次 {e['new_train']}，开发批次 {e['dev']}" for e in switches]
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text
