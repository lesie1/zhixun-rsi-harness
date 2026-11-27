"""Readable record of one round: what degraded, into what, why data was switched, what was tried,
what was adopted or rejected, and which experience the model's prompt contained."""
from pathlib import Path

from harness.dataio import read_json


def _f(value, digits=3):
    return "-" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def _class_table(flow, profiles, focus):
    classes = flow.classes
    lines = ["| 模型 | macro_f1 | " + " | ".join(f"{flow.display(c)} 召回/精确/F1" for c in classes) + " |",
             "|---|---|" + "---|" * len(classes)]
    names = {"pre_update": "更新前模型", "control_0pct": "0% 新数据对照（同预算）", "selected": "选出的针对性混合"}
    for arm, prof in profiles.items():
        cells = []
        for c in classes:
            s = prof["classes"][c]
            cell = f"{_f(s['recall'], 2)}/{_f(s['precision'], 2)}/{_f(s['f1'], 2)}"
            cells.append(f"**{cell}**" if c in focus else cell)
        lines.append(f"| {names.get(arm, arm)} | {_f(prof['macro_f1'], 4)} | " + " | ".join(cells) + " |")
    return lines


def write_round(flow, rnd):
    folder = Path(rnd["folder"])
    lines = [f"# 第 {rnd['index']} 轮记录（报警 {rnd['alarm']}，开发 {rnd['selection']}，最终验收 {rnd['final']}）", ""]
    diag = rnd.get("diagnosis") or {}
    lines += ["## 1. 诊断", "", f"- 结论：{diag.get('status_text', '-')}（同一个部署检查点，参照数据 vs 新批次）",
              f"- 整体 macro_f1：参照 {_f((diag.get('macro_f1') or {}).get('reference'), 4)} → 新批次 "
              f"{_f((diag.get('macro_f1') or {}).get('new'), 4)}；team_loop 整体检查："
              f"{'；'.join((diag.get('overall_check') or {}).get('reasons', []))}"]
    lines += [f"- {a['text']}" for a in diag.get("alarms", [])]
    lines += [f"- 证据不足：{i['text']}" for i in diag.get("insufficient", [])]
    pair = diag.get("pair")
    if pair:
        lines.append(f"- 处理对象：退化类 **{flow.display(pair['degraded'])}**，主要混淆类 **{flow.display(pair['confusion'])}**"
                     f"（{pair['count']}/{pair['support']}，{pair['share_of_class']:.1%}）")
    stage = rnd.get("repair")
    if stage:
        search = stage.get("search") or {}
        lines += ["", "## 2. 先在现有训练集上修复", "",
                  f"- 阶段 {stage['stage_id']}，起点 {stage['parent']['from']}，训练数据 = 当前部署模型的训练集",
                  f"- 搜索结束原因：{search.get('end_reason', '-')}；迭代 {search.get('iterations', '-')}"]
        if search.get("description"):
            lines.append(f"- {search['description']}")
        trace = (search.get("analysis") or {}).get("trace") or []
        if trace:
            lines += ["", "| 迭代 | 总分 | 报警类召回 | 混淆类召回 | 混淆类精确率 | 算进展 | 达到修复标准 |", "|---|---|---|---|---|---|---|"]
            lines += [f"| {t['iteration']} | {_f(t['score'], 4)} | {_f(t['alarm_recall'], 2)} | {_f(t['confusion_recall'], 2)} | "
                      f"{_f(t['confusion_precision'], 2)} | {'是' if t['progress'] else '否'} | {'是' if t['targeted_fix'] else '否'} |"
                      for t in trace]
        gate = stage.get("gate") or {}
        lines += ["", f"- 修复阶段验收门：{gate.get('decision', '-')}（{gate.get('reason', '')}）；"
                      f"报警类是否修复：{'是' if stage.get('repaired') else '否'}"]
        pb = stage.get("playbook") or {}
        lines += ["", "## 3. 模型提示里读到的经验", "",
                  f"- playbook 文件：{stage.get('playbook_path')}（{pb.get('mode')}，版本 {pb.get('version')}）",
                  f"- 写入的基础经验：{pb.get('seeded')}；真实提示（OpenEvolve PromptSampler + harness.prompt_extras）中出现次数："
                  f"{pb.get('occurrences_in_prompt')}；{pb.get('prompt_version_line')}",
                  f"- 提示预览：{Path(stage['folder']) / 'prompt_preview.txt'}"]
        seen = stage.get("prompts_seen") or {}
        if seen.get("prompts"):
            for key, value in seen.items():
                if key != "prompts":
                    lines.append(f"- 搜索中实际发给模型的 {seen['prompts']} 条提示里，{value['prompts_containing']} 条含有 {key}，"
                                 f"单条提示中最多出现 {value['max_per_prompt']} 次")
        for skipped in pb.get("skipped") or []:
            lines.append(f"- 未写入 {skipped['id']}：{'；'.join(skipped['reasons'])}")
    if rnd.get("switch"):
        lines += ["", "## 4. 为什么换数据", "", f"- {rnd['switch']['why']}", "- 停滞只说明旧数据上找不到足够改进，不等于已确认漂移"]
    mix = rnd.get("mixture")
    if mix:
        D, C = mix["pair"]["degraded"], mix["pair"]["confusion"]
        lines += ["", "## 5. 尝试过的实际配比（开发批次 " + mix["selection_batch"] + "）", "",
                  f"更新前模型在开发批次上：macro_f1 {_f(mix['pre_update_dev']['macro_f1'], 4)}，{flow.display(D)} 召回 "
                  f"{_f(mix['pre_update_dev']['per_class'][D]['recall'], 2)}，{flow.display(C)} 召回 "
                  f"{_f(mix['pre_update_dev']['per_class'][C]['recall'], 2)}", "",
                  "| 方案 | 目标占比 | 实际新数据次数/每轮总次数 | 实际占比 | 新数据中退化类份额 | 独立样本（退化/混淆） | 重复次数 | "
                  f"{flow.display(D)} 召回 | {flow.display(C)} 召回/精确 | macro_f1 | 结论 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in mix["candidates"]:
            if c.get("status") != "ok":
                lines.append(f"| {c['id']} | {c['fraction']:.0%} | - | - | - | - | - | - | - | - | {c.get('status')} |")
                continue
            acc, dev = c["accounting"], c["dev"]
            verdict = c.get("role") or ("通过护栏" if c.get("eligible") else "未通过：" + "；".join((c.get("check_vs_control") or {}).get("notes") or ["退化类改善不足"]))
            if c["id"] == mix.get("chosen"):
                verdict = "**选中并冻结**"
            lines.append(f"| {c['id']} | {c['fraction']:.0%} | {acc['new_presentations_per_epoch']}/{acc['rows_per_epoch']} | "
                         f"{acc['actual_fraction']:.2%} | {_f(acc['degraded_share_actual'], 2)} | "
                         f"{acc['new_unique']['degraded']}/{acc['new_unique']['confusion']} | "
                         f"{acc['new_repeats']['degraded'] + acc['new_repeats']['confusion']} | "
                         f"{_f(dev['per_class'][D]['recall'], 2)} | {_f(dev['per_class'][C]['recall'], 2)}/"
                         f"{_f(dev['per_class'][C]['precision'], 2)} | {_f(dev['macro_f1'], 4)} | {verdict} |")
        if mix.get("stopped_early"):
            lines.append(f"\n提前停止：比例 {mix['stopped_early']['after_fraction']:.0%} 之后连续没有更好的方案，跳过 "
                         f"{[f'{x:.0%}' for x in mix['stopped_early']['skipped']]}")
        lines.append("\n每个方案都从同一个检查点出发，训练文件行数相同（新数据替换历史行），更新步数和打乱顺序相同；"
                     "实际占比按 runner 记录的每轮步数核对（每行每轮恰好一次）")
    final = rnd.get("final_acceptance")
    if final:
        data = read_json(folder / "final_acceptance.json")
        focus = [mix["pair"]["degraded"], mix["pair"]["confusion"]]
        lines += ["", f"## 6. 最终验收（批次 {final['batch']}，{final['rows']} 行，只用一次）", ""]
        lines += _class_table(flow, data["profiles"], focus)
        lines += ["", f"- 相对更新前：宏平均 F1 更好的概率 {final['p_macro_f1_better_than_pre_update']:.3f}（门槛 {final['threshold']}）；"
                      f"退化类召回 {final['check_vs_pre_update']['degraded_gain']:+.2f}；混淆类召回 "
                      f"{-final['check_vs_pre_update']['confusion_recall_drop']:+.2f}、精确率 "
                      f"{-final['check_vs_pre_update']['confusion_precision_drop']:+.2f}"]
        lines += [f"- 注意：{n}" for n in final["check_vs_pre_update"]["notes"]]
        lines.append(f"- 结果：{'**采用**' if final['accepted'] else '**拒绝，保留更新前模型**'}")
    lines += ["", f"## 本轮结论：{rnd.get('outcome', '-')}", "", rnd.get("outcome_text", "")]
    text = "\n".join(lines) + "\n"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "record.md").write_text(text, encoding="utf-8")
    return text
