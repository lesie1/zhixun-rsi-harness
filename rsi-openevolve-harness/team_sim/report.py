"""Collect every run's summary.json into <out>/summary.json and a Chinese report.md."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import PROTOCOL_NAMES

COLUMNS = [
    ("mean_advantage", "平均相对优势", "{:+.1%}"),
    ("overall_advantage", "全程按行优势", "{:+.1%}"),
    ("mean_unit_advantage", "按轴承等权优势", "{:+.1%}"),
    ("worst_advantage", "最差一轮", "{:+.1%}"),
    ("rounds_below_static", "低于基线轮数", "{:.1f}"),
    ("mean_rounds_to_recover", "平均恢复轮数", "{:.1f}"),
    ("advantage_std", "各轮波动", "{:.3f}"),
    ("trainings", "训练次数", "{:.1f}"),
    ("rollbacks", "回退次数", "{:.1f}"),
    ("search_requests", "搜索请求", "{:.1f}"),
    ("searches_run", "搜出新候选", "{:.1f}"),
    ("search_adopted", "搜索候选上线", "{:.1f}"),
    ("search_time_limited", "搜索到时限", "{:.1f}"),
]


def _cell(row, key, fmt):
    return "—" if row.get(key) is None else fmt.format(row[key])


def build(out_dir):
    out = Path(out_dir)
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("f*/p*/*/summary.json"))]
    groups = defaultdict(list)
    for run in runs:
        groups[(run["protocol"], run["policy"])].append(run)
    table = []
    for (protocol, policy), members in sorted(groups.items()):
        row = {"protocol": protocol, "policy": policy, "folds": sorted(m["fold"] for m in members)}
        for key, _, _ in COLUMNS:   # summaries written before a column existed show "—"
            values = [m.get(key) for m in members]
            row[key] = None if None in values else float(np.mean(values))
        table.append(row)
    (out / "summary.json").write_text(json.dumps({"runs": runs, "table": table}, ensure_ascii=False, indent=1) + "\n",
                                      encoding="utf-8")
    lines = ["# 数据流模拟结果（公开数据，第一层）", "",
             "相对优势 = (静态基线 RMSE − 模型 RMSE) / 静态基线 RMSE，在同一轮的同一批测试数据上计算；正数表示比从不更新的伪基线好。",
             "每格是各折的平均值。数字只来自公开数据上的模拟，不代表平台成绩。", ""]
    for protocol in sorted({row["protocol"] for row in table}):
        lines += [f"## 给法 {PROTOCOL_NAMES.get(protocol, protocol)}", "",
                  "| 策略 | 折数 | " + " | ".join(title for _, title, _ in COLUMNS) + " |",
                  "|---|---|" + "---|" * len(COLUMNS)]
        for row in (r for r in table if r["protocol"] == protocol):
            lines.append(f"| {row['policy']} | {len(row['folds'])} | " +
                         " | ".join(_cell(row, key, fmt) for key, _, fmt in COLUMNS) + " |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"runs": len(runs), "table": table, "report": str(out / "report.md")}
