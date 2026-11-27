"""Discover grouping fields from the `meta` objects of a dataset, without hard-coding names.

Shared by the split module (a group never straddles two splits) and by the feedback
report (per-group breakdowns), so both always mean the same thing by "one bearing" or
"one engine".

Rules:
  1. ignore fields with a single value (constants) and fields that differ on every row;
  2. a field whose equal values sit next to each other in file order is a group key;
  3. a non-contiguous field combined with a contiguous one may form a composite key
     (C-MAPSS: unit 71 exists in FD001 and FD003, the engine is (source_subset, unit));
  4. keys with 2..max_report_groups groups are used for report breakdowns, the finest key
     is used for splitting.
"""


def _column(meta_rows, key):
    return [str(row.get(key)) for row in meta_rows]


def _runs(values):
    return 1 + sum(1 for a, b in zip(values, values[1:]) if a != b) if values else 0


def discover_groups(meta_rows, max_report_groups=50):
    """Return {"keys": [...], "report_keys": [...], "split_key": [...] or None, "fields": {...}}.

    Each key is a list of meta field names (length 1, or 2 for a composite key).
    """
    n = len(meta_rows)
    names = sorted({key for row in meta_rows for key in row})
    fields, contiguous, scattered = {}, [], []
    for name in names:
        values = _column(meta_rows, name)
        distinct, runs = len(set(values)), _runs(values)
        fields[name] = {"distinct": distinct, "runs": runs}
        if distinct <= 1 or distinct >= n:
            continue
        (contiguous if runs == distinct else scattered).append(name)
    keys = [[name] for name in contiguous]
    for name in scattered:
        for outer in contiguous:
            pairs = list(zip(_column(meta_rows, outer), _column(meta_rows, name)))
            distinct = len(set(pairs))
            if _runs(pairs) == distinct and distinct * 2 <= n:
                keys.append([outer, name])
                break
    sizes = {tuple(key): len(set(group_labels(meta_rows, key))) for key in keys}
    report_keys = [key for key in keys if 2 <= sizes[tuple(key)] <= max_report_groups]
    split_key = max(keys, key=lambda key: sizes[tuple(key)]) if keys else None
    return {"keys": keys, "report_keys": report_keys, "split_key": split_key,
            "sizes": {"+".join(key): size for key, size in sizes.items()}, "fields": fields}


def group_labels(meta_rows, key):
    """One label per row for group key `key` (a list of field names)."""
    if not key:
        return ["all"] * len(meta_rows)
    columns = [_column(meta_rows, name) for name in key]
    return ["/".join(parts) for parts in zip(*columns)]
