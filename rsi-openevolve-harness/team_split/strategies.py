"""Split strategies: which validation rows go to selection, which to the gate.

Each strategy is a pure function (table, groups_info, options) -> Assignment registered in
STRATEGIES. Group discovery always comes from harness.groups (one implementation for the
whole project). With default options the result is identical, id for id, to the
provisional harness.splits.make_split, so every number validated so far stays valid.

Methods (manifest "method" values):
  group_stratified       whole groups alternate after a seeded shuffle, inside each coarser group
  group                  same without a coarser key
  group_stratified_rows  like group_stratified, but groups are placed greedily to balance row counts
  group_rows             like group, balanced by rows
  class_time             classification without groups: inside each class earlier half -> selection
  time_blocks            regression without groups: contiguous blocks alternate
  no_split               too few groups: selection = gate = everything, with a warning

The "_rows" variants exist only when SplitOptions.balance == "rows". The balancing is
recorded in the method name rather than in `warnings`, because the harness turns every
manifest warning into a "selection equals gate" symptom in the LLM feedback
(harness/feedback.py, SPLIT_WARNING), which would be wrong for a balanced split.
"""
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from harness.groups import discover_groups, group_labels

BASE_METHODS = ("group_stratified", "group", "class_time", "time_blocks", "no_split")
ROW_BALANCED = ("group_stratified_rows", "group_rows")
KNOWN_METHODS = BASE_METHODS + ROW_BALANCED
GROUP_METHODS = ("group_stratified", "group") + ROW_BALANCED


@dataclass
class SplitOptions:
    seed: int = 42
    min_groups: int = 4
    time_blocks: int = 10
    method: str = "auto"          # "auto" or one of BASE_METHODS
    balance: str = "groups"       # "groups" (default, alternate) or "rows" (greedy by row count)

    def validate(self):
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(self.min_groups, bool) or not isinstance(self.min_groups, int) or self.min_groups < 2:
            raise ValueError("min_groups must be an integer >= 2")
        if isinstance(self.time_blocks, bool) or not isinstance(self.time_blocks, int) or self.time_blocks < 2:
            raise ValueError("time_blocks must be an integer >= 2")
        if self.method != "auto" and self.method not in BASE_METHODS:
            raise ValueError(f"method must be auto or one of {BASE_METHODS}")
        if self.balance not in ("groups", "rows"):
            raise ValueError("balance must be groups or rows")
        return self

    def to_dict(self):
        return asdict(self)


@dataclass
class Assignment:
    method: str
    group_key: Optional[List[str]]
    select_rows: List[int]
    gate_rows: List[int]
    warnings: List[str] = field(default_factory=list)


def base_method(method):
    """Strip the row-balancing suffix: group_stratified_rows -> group_stratified."""
    return method[:-len("_rows")] if method in ROW_BALANCED else method


def options_for_manifest(manifest, defaults=None):
    """SplitOptions that should reproduce `manifest` (seed and method/balance from the manifest)."""
    base = SplitOptions(**(defaults.to_dict() if defaults else {}))
    base.seed = manifest.get("seed", base.seed)
    method = manifest.get("method", "auto")
    base.balance = "rows" if method in ROW_BALANCED else "groups"
    base.method = base_method(method) if method in KNOWN_METHODS else "auto"
    return base


# ------------------------------------------------------------------------- helpers

def _coarse_key(groups_info, split_key):
    return next((key for key in groups_info["report_keys"] if key != split_key), None)


def _group_assign(table, groups_info, options, use_coarse):
    """Shared implementation of the four group methods."""
    split_key = groups_info["split_key"]
    groups = group_labels(table.meta_rows, split_key)
    group_order = list(dict.fromkeys(groups))
    coarse_key = _coarse_key(groups_info, split_key) if use_coarse else None
    coarse = group_labels(table.meta_rows, coarse_key) if coarse_key else None
    rng = random.Random(options.seed)
    first_row, sizes = {}, defaultdict(int)
    for row, group in enumerate(groups):
        first_row.setdefault(group, row)
        sizes[group] += 1
    buckets = defaultdict(list)
    for group in group_order:
        buckets[coarse[first_row[group]] if coarse else "all"].append(group)
    chosen = {}
    for bucket in sorted(buckets):
        members = sorted(buckets[bucket])
        rng.shuffle(members)
        if options.balance == "rows":
            # Greedy longest-first: biggest group goes to the side with fewer rows so far;
            # the seeded shuffle above only decides between groups of equal size.
            order = sorted(range(len(members)), key=lambda i: (-sizes[members[i]], i))
            totals = {"select": 0, "gate": 0}
            for i in order:
                side = "select" if totals["select"] <= totals["gate"] else "gate"
                chosen[members[i]] = side
                totals[side] += sizes[members[i]]
        else:
            for position, group in enumerate(members):
                chosen[group] = "select" if position % 2 == 0 else "gate"
    select = [row for row, group in enumerate(groups) if chosen[group] == "select"]
    gate = [row for row, group in enumerate(groups) if chosen[group] == "gate"]
    method = "group_stratified" if coarse_key else "group"
    if options.balance == "rows":
        method += "_rows"
    return Assignment(method, list(split_key), select, gate, [])


# ------------------------------------------------------------------------ strategies

def group_stratified(table, groups_info, options):
    return _group_assign(table, groups_info, options, use_coarse=True)


def group(table, groups_info, options):
    return _group_assign(table, groups_info, options, use_coarse=False)


def class_time(table, groups_info, options):
    labels = table.labels()
    by_class = defaultdict(list)
    for row, label in enumerate(labels):
        by_class[label].append(row)
    part, warnings = {}, []
    for label, rows in by_class.items():
        half = (len(rows) + 1) // 2
        for position, row in enumerate(rows):
            part[row] = "select" if position < half else "gate"
        if len(rows) < 2:
            warnings.append(f"类别 {label} 只有 1 个样本，只能放进挑选集，验收集里没有这一类")
    select = [row for row in range(len(table)) if part[row] == "select"]
    gate = [row for row in range(len(table)) if part[row] == "gate"]
    return Assignment("class_time", None, select, gate, warnings)


def time_blocks(table, groups_info, options):
    n = len(table)
    block = max(1, -(-n // options.time_blocks))
    select = [row for row in range(n) if (row // block) % 2 == 0]
    gate = [row for row in range(n) if (row // block) % 2 == 1]
    return Assignment("time_blocks", None, select, gate, [])


def no_split(table, groups_info, options):
    split_key = groups_info["split_key"]
    rows = list(range(len(table)))
    if split_key:
        count = groups_info["sizes"].get("+".join(split_key), 0)
        warning = (f"按 {'+'.join(split_key)} 只有 {count} 组（少于 {options.min_groups} 组），无法留出独立验收集；"
                   "挑选集和验收集是同一份数据，验收结论带有选择偏差")
    else:
        warning = "指定了不划分（no_split）：挑选集和验收集是同一份数据，验收结论带有选择偏差"
    return Assignment("no_split", list(split_key) if split_key else None, rows, list(rows), [warning])


STRATEGIES = {"group_stratified": group_stratified, "group": group, "class_time": class_time,
              "time_blocks": time_blocks, "no_split": no_split}


# ------------------------------------------------------------------------- choosing

def _group_count(groups_info):
    key = groups_info["split_key"]
    return groups_info["sizes"].get("+".join(key), 0) if key else 0


def choose_method(table, groups_info, options):
    """The rule of docs/SPLIT_INTERFACE.md ("当前临时算法")."""
    if groups_info["split_key"]:
        if _group_count(groups_info) >= options.min_groups:
            return "group_stratified" if _coarse_key(groups_info, groups_info["split_key"]) else "group"
        return "no_split"
    return "class_time" if table.task_type == "classification" else "time_blocks"


def check_applicable(method, table, groups_info, options):
    """Raise ValueError when a forced method cannot be used on this data."""
    if method in ("group_stratified", "group"):
        if not groups_info["split_key"]:
            raise ValueError(f"{method} 需要分组字段，但这份数据的 meta 里没有发现可用的分组")
        if _group_count(groups_info) < options.min_groups:
            raise ValueError(f"{method} 至少需要 {options.min_groups} 组，实际只有 {_group_count(groups_info)} 组")
        if method == "group_stratified" and not _coarse_key(groups_info, groups_info["split_key"]):
            raise ValueError("group_stratified 需要一个比分组更粗的报告键（例如子集），这份数据没有；请用 group")
    if method == "class_time" and table.task_type != "classification":
        raise ValueError("class_time 只适用于分类任务")


def split(table, options=None):
    """Discover groups, choose (or check) the method, run it."""
    options = (options or SplitOptions()).validate()
    groups_info = discover_groups(table.meta_rows)
    method = choose_method(table, groups_info, options) if options.method == "auto" else options.method
    check_applicable(method, table, groups_info, options)
    assignment = STRATEGIES[method](table, groups_info, options)
    if not assignment.select_rows or not assignment.gate_rows:
        raise ValueError(f"{assignment.method} 产生了空的一边（数据只有 {len(table)} 行）")
    return assignment
