"""Strict verification of a split manifest against the labelled data of a stage.

harness.splits.validate_manifest only checks keys, empty sides, unknown ids and (outside
no_split) overlap/coverage. This module checks everything docs/SPLIT_INTERFACE.md asks for
and never raises: it returns a list of problems.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass

from harness.groups import group_labels
from harness.splits import MANIFEST_KEYS, manifest_digest

from .manifest import build_manifest, compute_counts
from .strategies import GROUP_METHODS, KNOWN_METHODS, SplitOptions, split


@dataclass
class Problem:
    code: str
    severity: str   # "error" | "warning"
    message: str

    def to_dict(self):
        return {"code": self.code, "severity": self.severity, "message": self.message}


def has_errors(problems):
    return any(p.severity == "error" for p in problems)


def _is_str_list(value):
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def verify_manifest(manifest, table, stage_id=None, options=None):
    problems = []

    def add(code, message, severity="error"):
        problems.append(Problem(code, severity, message))

    try:
        _verify(manifest, table, stage_id, options, add)
    except Exception as error:  # a malformed manifest must still produce a report, not a crash
        add("TYPE_INVALID", f"校验中途出错（清单结构异常）：{type(error).__name__}: {error}")
    return problems


def _verify(manifest, table, stage_id, options, add):
    if not isinstance(manifest, dict):
        add("TYPE_INVALID", "清单必须是 JSON 对象")
        return
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    if [key for key in missing if key != "provisional"]:
        add("KEYS_MISSING", f"缺少字段: {[key for key in missing if key != 'provisional']}")
    if "provisional" in missing:
        add("KEYS_MISSING", "缺少 provisional 字段；harness 允许缺失，但交接时应明确写出 true 或 false", "warning")

    # ---- types
    types_ok = True
    checks = {
        "stage_id": lambda v: isinstance(v, str) and bool(v.strip()),
        "source_sha256": lambda v: isinstance(v, str) and v.startswith("sha256:") and len(v) == 71,
        "task_type": lambda v: v in ("classification", "regression"),
        "method": lambda v: v in KNOWN_METHODS,
        "group_key": lambda v: v is None or (_is_str_list(v) and len(v) > 0),
        "seed": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "select_ids": _is_str_list,
        "gate_ids": _is_str_list,
        "counts": lambda v: isinstance(v, dict) and isinstance(v.get("select"), dict) and isinstance(v.get("gate"), dict),
        "warnings": _is_str_list,
        "provisional": lambda v: isinstance(v, bool),
        "manifest_sha256": lambda v: isinstance(v, str) and v.startswith("sha256:"),
    }
    for key, check in checks.items():
        if key in manifest and not check(manifest[key]):
            add("TYPE_INVALID", f"字段 {key} 的类型或取值不合法: {str(manifest[key])[:80]!r}"
                + (f"（已知方法: {', '.join(KNOWN_METHODS)}）" if key == "method" else ""))
            if key in ("select_ids", "gate_ids", "method", "group_key", "counts"):
                types_ok = False
    if any(key in missing for key in ("select_ids", "gate_ids", "method")):
        types_ok = False

    # ---- digest and provenance
    if isinstance(manifest.get("manifest_sha256"), str) and manifest_digest(manifest) != manifest["manifest_sha256"]:
        add("DIGEST_MISMATCH", "manifest_sha256 与按 manifest_digest 重算的结果不一致（清单被改过，或摘要算法不对）")
    if "source_sha256" in manifest and manifest["source_sha256"] != table.source_sha256:
        add("SOURCE_MISMATCH", f"source_sha256 与本阶段验证文件不一致：清单 {str(manifest['source_sha256'])[:23]}…，"
                               f"实际 {table.source_sha256[:23]}…")
    if "task_type" in manifest and manifest["task_type"] != table.task_type:
        add("TASK_MISMATCH", f"task_type 是 {manifest['task_type']}，本阶段是 {table.task_type}")
    if stage_id is not None and "stage_id" in manifest and manifest["stage_id"] != stage_id:
        add("STAGE_MISMATCH", f"stage_id 是 {manifest['stage_id']!r}，本阶段是 {stage_id!r}")
    if not types_ok:
        return

    # ---- ids
    method = manifest["method"]
    known = set(table.ids)
    index_of = {identity: row for row, identity in enumerate(table.ids)}
    sides = {"select": manifest["select_ids"], "gate": manifest["gate_ids"]}
    names = {"select": "挑选集", "gate": "验收集"}
    for side, ids in sides.items():
        duplicated = [identity for identity, count in Counter(ids).items() if count > 1]
        if duplicated:
            add("DUPLICATE_ID", f"{names[side]}有 {len(duplicated)} 个重复 ID，例如 {duplicated[:3]}")
        unknown = [identity for identity in ids if identity not in known]
        if unknown:
            add("UNKNOWN_ID", f"{names[side]}有 {len(unknown)} 个 ID 不在本阶段的验证文件里，例如 {unknown[:3]}")
        if not ids:
            add("EMPTY_SIDE", f"{names[side]}是空的")
    select, gate = set(sides["select"]), set(sides["gate"])
    if method == "no_split":
        for side, ids in sides.items():
            if set(ids) != known or len(ids) != len(known):
                add("NO_SPLIT_INCOMPLETE", f"no_split 要求{names[side]}恰好是全部 {len(known)} 个 ID，实际 {len(ids)} 个")
        if not manifest.get("warnings"):
            add("NO_SPLIT_INCOMPLETE", "no_split 必须在 warnings 里说明没有独立验收集")
    else:
        overlap = select & gate
        if overlap:
            add("OVERLAP", f"两边有 {len(overlap)} 个相同的 ID，例如 {sorted(overlap)[:3]}")
        uncovered = known - select - gate
        if uncovered:
            add("NOT_COVERING", f"有 {len(uncovered)} 个验证样本两边都没有，例如 {sorted(uncovered)[:3]}")

    rows = {side: sorted(index_of[i] for i in set(ids) if i in index_of) for side, ids in sides.items()}

    # ---- counts
    group_key = manifest["group_key"]
    fields_ok = True
    if group_key:
        missing_fields = [name for name in group_key if any(name not in row for row in table.meta_rows)]
        if missing_fields:
            fields_ok = False
            add("GROUP_KEY_INVALID", f"group_key 里的字段 {missing_fields} 不是每一行的 meta 都有")
    elif method in GROUP_METHODS:
        fields_ok = False
        add("GROUP_KEY_INVALID", f"{method} 是分组方法，但 group_key 为空")
    if fields_ok:
        for side in ("select", "gate"):
            expected = compute_counts(table, rows[side], group_key)
            if manifest["counts"].get(side) != expected:
                add("COUNTS_MISMATCH", f"{names[side]}的 counts 与实际不符：清单 {manifest['counts'].get(side)}，实际 {expected}")

    # ---- groups never straddle
    if method in GROUP_METHODS and group_key and fields_ok:
        labels = group_labels(table.meta_rows, group_key)
        straddling = sorted({labels[r] for r in rows["select"]} & {labels[r] for r in rows["gate"]})
        if straddling:
            add("GROUP_STRADDLES", f"{len(straddling)} 个组同时出现在两边（按 {'+'.join(group_key)}），例如 {straddling[:3]}")

    # ---- classes
    labels = table.labels()
    if labels is not None:
        if method == "class_time":
            positions = defaultdict(lambda: {"select": [], "gate": []})
            for side in ("select", "gate"):
                for r in rows[side]:
                    positions[labels[r]][side].append(r)
            for label, where in sorted(positions.items()):
                if where["select"] and where["gate"] and max(where["select"]) > min(where["gate"]):
                    add("CLASS_ORDER_VIOLATED", f"类别 {label}：挑选集里有样本排在验收集样本之后"
                                                f"（挑选集最后一行 {max(where['select']) + 1}，验收集第一行 {min(where['gate']) + 1}）")
        if method != "no_split":
            present = {side: {labels[r] for r in rows[side]} for side in rows}
            for label in sorted(set(labels)):
                for side, other in (("select", "gate"), ("gate", "select")):
                    if label not in present[side] and label in present[other]:
                        add("CLASS_MISSING_ON_SIDE", f"类别 {label} 只出现在{names[other]}，{names[side]}里没有", "warning")

    # ---- reproducibility
    if options is not None:
        try:
            rebuilt = build_manifest(table, split(table, options), manifest["stage_id"], options)
            if rebuilt["manifest_sha256"] != manifest.get("manifest_sha256"):
                differ = [key for key in MANIFEST_KEYS if rebuilt.get(key) != manifest.get(key)]
                add("NOT_REPRODUCIBLE", f"用给定参数 {options.to_dict()} 重建的清单不同，差异字段: {differ}")
        except Exception as error:
            add("NOT_REPRODUCIBLE", f"用给定参数无法重建清单：{error}")
