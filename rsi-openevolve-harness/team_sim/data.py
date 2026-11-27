"""Data side of the simulator: units, folds, life segments and the files of one round.

Only the public package is read. Unit names from `meta` are bookkeeping for the simulator
(fold roles, per-unit scores). The files handed to training keep only id / inputs / answer,
like the platform's files, so nothing downstream can use meta as a model input.

Roles inside one condition (B has 4 public bearings per condition), rotated over 4 folds:
  train       2 units, full life: the fold's pseudo-baseline is trained on them from scratch
  stream      1 unit: delivers new labelled data period by period (protocol ②), or serves as a
              fixed validation unit (protocol ①)
  hidden      1 unit: tested period by period, like the platform's hidden test
  For protocol 2c one of the two train units becomes stream_val (validation only, never trained),
  so the pseudo-baseline of 2c is trained on the remaining unit (train_2c).

"Period" k = the k-th of 10 contiguous segments of each unit's own life (relative life stage).

Protocols (round k = 1..10):
  "1"   ① test-then-train: the hidden segment tested in round k-1 comes back with labels as
        validation in round k and joins the training data in round k+1 (validation never feeds
        gradients, as in the SDK). Test = hidden segment k (the next period).
  "2a"  ② new data + same-period test, discarded after scoring; the new period's data goes to
        train.jsonl: first 70 % of stream segment k trains this round, the last 30 % validates.
  "2b"  ② as above, but the new period's data is only validation this round and trains from
        the next round on.
  "2c"  ② with unit-disjoint validation: the new period of the stream unit trains this round,
        the new period of a separate stream_val unit validates and never trains. In 2a / 2b the
        validation units also feed training, so a gate there rewards memorising those units,
        which does not carry over to unseen hidden units; 2c measures what the hidden test needs.
"""
import json
from collections import OrderedDict
from pathlib import Path

SEGMENTS = 10
PROTOCOLS = ("1", "2a", "2b", "2c")
PROTOCOL_NAMES = {"1": "① 先测后训", "2a": "②a 当期新数据进训练，测完作废", "2b": "②b 当期新数据只作验证，测完作废",
                  "2c": "②c 当期新数据进训练，验证用从不参与训练的设备，测完作废"}


def _bearing(meta):
    return meta["bearing"]


def _condition(unit):
    return unit.split("_")[0][-1]          # "Bearing2_3" -> "2"


TASKS = {"B": {"unit": _bearing, "group": _condition, "profile": "B"}}


def load_rows(package_dir, task="B"):
    if task not in TASKS:
        raise ValueError(f"the simulator currently supports {sorted(TASKS)} only")
    spec = TASKS[task]
    rows = []
    for source in ("train", "validation"):
        with (Path(package_dir) / "public" / f"{source}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    unit = spec["unit"](record["meta"])
                    rows.append({"id": record["id"], "inputs": record["inputs"], "answer": float(record["answer"]),
                                 "unit": unit, "group": spec["group"](unit), "source": source})
    return rows


def by_unit(rows):
    """unit -> its rows in time order (the remaining life counts down)."""
    units = OrderedDict()
    for row in rows:
        units.setdefault(row["unit"], []).append(row)
    for members in units.values():
        members.sort(key=lambda row: -row["answer"])
    return units


def make_folds(units, n_folds=None):
    groups = OrderedDict()
    for unit, members in units.items():
        groups.setdefault(members[0]["group"], []).append(unit)
    for members in groups.values():
        members.sort()
    size = min(len(members) for members in groups.values())
    if size < 4:
        raise ValueError("every group needs at least 4 units (train, stream, stream_val/train, hidden)")
    count = size if n_folds is None else max(1, min(int(n_folds), size))
    folds = []
    for fold in range(count):
        hidden, stream, train, stream_val, train_2c = [], [], [], [], []
        for members in groups.values():
            n = len(members)
            h, s, v = members[fold % n], members[(fold + 1) % n], members[(fold + 2) % n]
            hidden.append(h)
            stream.append(s)
            train += [m for m in members if m not in (h, s)]
            stream_val.append(v)
            train_2c += [m for m in members if m not in (h, s, v)]
        folds.append({"fold": fold, "hidden": hidden, "stream": stream, "train": train,
                      "stream_val": stream_val, "train_2c": train_2c})
    return folds


def segment(rows, k, n=SEGMENTS):
    """k-th (1-based) of n contiguous segments of a time-ordered row list."""
    if not 1 <= k <= n:
        return []
    m = len(rows)
    return rows[(k - 1) * m // n: k * m // n]


def round_rows(units, fold, protocol, k, n=SEGMENTS, train_share=0.7):
    """Rows of round k: train / validation / newest (newest labelled block) / test."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    init = [row for unit in init_units(fold, protocol) for row in units[unit]]

    def part(role, j):
        return [row for unit in fold[role] for row in segment(units[unit], j, n)]

    test = part("hidden", k)
    if protocol == "1":
        newest = part("hidden", k - 1) if k >= 2 else []
        train = init + [row for j in range(1, k - 1) for row in part("hidden", j)]
        validation = [row for unit in fold["stream"] for row in units[unit]] + newest
    elif protocol == "2b":
        newest = part("stream", k)
        train = init + [row for j in range(1, k) for row in part("stream", j)]
        validation = newest
    elif protocol == "2c":
        train = init + [row for j in range(1, k + 1) for row in part("stream", j)]
        newest = validation = part("stream_val", k)
    else:  # "2a"
        head, tail = [], []
        for unit in fold["stream"]:
            current = segment(units[unit], k, n)
            cut = int(len(current) * train_share)
            head += current[:cut]
            tail += current[cut:]
        train = init + [row for j in range(1, k) for row in part("stream", j)] + head
        newest = validation = tail
    return {"train": train, "validation": validation, "newest": newest, "test": test}


def init_units(fold, protocol):
    """Units the pseudo-baseline of this protocol is trained on (full life)."""
    return fold["train_2c"] if protocol == "2c" else fold["train"]


def write_jsonl(path, rows):
    """Platform format: id, inputs, answer only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps({"id": row["id"], "inputs": row["inputs"], "answer": row["answer"]},
                                    ensure_ascii=False, allow_nan=False) + "\n")
    return str(path)
