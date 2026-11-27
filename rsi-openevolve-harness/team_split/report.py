"""Quality report of a split: is the gate side representative of the selection side?

Warnings here are advice only; they never block writing a manifest.
"""
import numpy as np

from harness.groups import discover_groups, group_labels

GATE_SHARE_RANGE = (0.30, 0.70)
MIN_CLASS_ON_GATE = 5
MAX_CLASS_TV = 0.15
MAX_MEAN_SHIFT_SD = 0.25


def _rows(manifest, table):
    index_of = {identity: row for row, identity in enumerate(table.ids)}
    return {side: sorted(index_of[i] for i in manifest[f"{side}_ids"] if i in index_of) for side in ("select", "gate")}


def split_report(manifest, table):
    rows = _rows(manifest, table)
    total = len(table)
    method = manifest["method"]
    report = {"method": method, "group_key": manifest.get("group_key"), "total_rows": total,
              "sides": {}, "warnings": []}
    for side in ("select", "gate"):
        r = rows[side]
        report["sides"][side] = {"rows": len(r), "share_of_all": len(r) / total if total else 0.0,
                                 "first_line": r[0] + 1 if r else None, "last_line": r[-1] + 1 if r else None}
    labels = table.labels()
    if labels is not None:
        names = sorted(set(labels))
        per_side = {}
        for side in ("select", "gate"):
            counts = {name: 0 for name in names}
            for r in rows[side]:
                counts[labels[r]] += 1
            per_side[side] = counts
        report["classes"] = per_side
        shares = {side: np.array([per_side[side][n] for n in names], float) / max(1, len(rows[side]))
                  for side in per_side}
        report["class_tv_distance"] = float(0.5 * np.abs(shares["select"] - shares["gate"]).sum())
    else:
        targets = np.asarray(table.targets, float)
        report["targets"] = {}
        for side in ("select", "gate"):
            values = targets[rows[side]]
            report["targets"][side] = {"mean": float(values.mean()), "max": float(values.max()),
                                       "quartiles": [float(q) for q in np.quantile(values, [0, .25, .5, .75, 1])]}
        sd = float(targets.std())
        report["targets"]["all_std"] = sd
        report["targets"]["mean_shift_sd"] = (abs(report["targets"]["select"]["mean"] - report["targets"]["gate"]["mean"])
                                              / sd if sd > 0 else 0.0)
    key = manifest.get("group_key")
    if key and all(all(name in row for name in key) for row in table.meta_rows):
        groups = group_labels(table.meta_rows, key)
        report["groups"] = {side: len({groups[r] for r in rows[side]}) for side in rows}
        coarse_key = next((k for k in discover_groups(table.meta_rows)["report_keys"] if k != key), None)
        if coarse_key:
            coarse = group_labels(table.meta_rows, coarse_key)
            breakdown = {}
            for value in dict.fromkeys(coarse):
                breakdown[value] = {side: {"groups": len({groups[r] for r in rows[side] if coarse[r] == value}),
                                           "rows": sum(1 for r in rows[side] if coarse[r] == value)}
                                    for side in rows}
            report["coarse_key"] = coarse_key
            report["coarse_breakdown"] = breakdown

    warn = report["warnings"]
    if method == "no_split":
        warn.append("没有独立验收集（no_split）：验收结论带有选择偏差")
    else:
        share = len(rows["gate"]) / total if total else 0.0
        if not GATE_SHARE_RANGE[0] <= share <= GATE_SHARE_RANGE[1]:
            warn.append(f"验收集占全部有标签数据的 {share:.0%}，不在 {GATE_SHARE_RANGE[0]:.0%}–{GATE_SHARE_RANGE[1]:.0%} 之间")
        if labels is not None:
            for name, count in report["classes"]["gate"].items():
                if count < MIN_CLASS_ON_GATE:
                    warn.append(f"类别 {name} 在验收集只有 {count} 个（少于 {MIN_CLASS_ON_GATE}），这一类的验收结论不可靠")
            if report["class_tv_distance"] > MAX_CLASS_TV:
                warn.append(f"两边的类别占比差异较大（总变差距离 {report['class_tv_distance']:.2f} > {MAX_CLASS_TV}）")
        elif report["targets"]["mean_shift_sd"] > MAX_MEAN_SHIFT_SD:
            warn.append(f"两边目标均值相差 {report['targets']['mean_shift_sd']:.2f} 个标准差（> {MAX_MEAN_SHIFT_SD}）")
    report["warnings"] = warn
    return report


def _fmt(value):
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def render_report(report):
    names = {"select": "挑选集", "gate": "验收集"}
    lines = [f"划分方式: {report['method']}" + (f"，分组键: {'+'.join(report['group_key'])}" if report.get("group_key") else ""),
             f"有标签数据共 {report['total_rows']} 行"]
    for side, info in report["sides"].items():
        line = f"  {names[side]}: {info['rows']} 行（{info['share_of_all']:.1%}），文件第 {info['first_line']}–{info['last_line']} 行之间"
        if "groups" in report:
            line += f"，{report['groups'][side]} 组"
        lines.append(line)
    if "classes" in report:
        lines.append("  各类别数量（挑选 / 验收）: " + "，".join(
            f"{name} {report['classes']['select'][name]}/{report['classes']['gate'][name]}" for name in report["classes"]["select"]))
        lines.append(f"  两边类别占比的总变差距离: {report['class_tv_distance']:.3f}")
    if "targets" in report:
        for side in ("select", "gate"):
            t = report["targets"][side]
            lines.append(f"  {names[side]}目标: 均值 {_fmt(t['mean'])}，四分位 {' / '.join(_fmt(q) for q in t['quartiles'])}")
        lines.append(f"  两边目标均值相差 {report['targets']['mean_shift_sd']:.3f} 个标准差")
    if "coarse_breakdown" in report:
        lines.append(f"  按 {'+'.join(report['coarse_key'])} 细分（组数/行数，挑选 | 验收）:")
        for value, info in report["coarse_breakdown"].items():
            lines.append(f"    {value}: {info['select']['groups']}/{info['select']['rows']} | {info['gate']['groups']}/{info['gate']['rows']}")
    lines.append("质量提示: 无" if not report["warnings"] else "质量提示:")
    lines += [f"  - {w}" for w in report["warnings"]]
    return "\n".join(lines)
