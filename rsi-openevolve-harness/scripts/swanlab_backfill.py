"""Backfill existing experiment artifacts into SwanLab (one SwanLab experiment per record).

What it uploads (nothing is retrained, no number is changed):
  train   every directory holding training.json (+ history.jsonl): one industrial-model training run.
          Per-epoch train loss and public-validation metrics as curves, the run's settings as config,
          the policy source as text when it sits next to the run.
  gate    gate/decision.json (harness acceptance gate) and final_acceptance.json (team_gas): the decision,
          its reason, the compared metrics and bootstrap probabilities.
  sim     team_sim rounds.jsonl (+ summary.json): per round data sizes, decision, test scores.
  search  records/iterations.jsonl (OpenEvolve search): per iteration score, status, hypothesis.
  record  any other result file named with --extra (results.json, comparison.json, …): its numbers as
          scalars and the whole file as text.

Every experiment is marked as backfilled: its description names the source file and the original time,
its config carries source_path / original_time / backfilled=true.

Never uploads checkpoints, data files or keys. Login once with `swanlab login` (or SWANLAB_API_KEY in
the environment); the key is never read or written by this script.

Usage:
  python scripts/swanlab_backfill.py --root runs --dry-run
  python scripts/swanlab_backfill.py --root runs --root ../other/runs --project zhixun-rsi --public
A ledger (default state/swanlab_ledger.json) remembers what was uploaded; a rerun skips unchanged files.
"""
import argparse
import fnmatch
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SKIP_DIRS = {"cache", "locks", "__pycache__", "data", ".git", "openevolve_output", "swanlog"}
MAX_SCALARS = 300
MAX_TEXT = 20000


def log(message):
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- discovery
def discover(roots, include_tests=False, include=(), exclude=(), extra_names=()):
    extra_names = set(extra_names)
    items = []
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            log(f"[backfill] 跳过不存在的目录: {root}")
            continue
        for current, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and (include_tests or d != "tests"))
            here = Path(current)
            names = set(files)
            if "training.json" in names:
                items.append(("train", root, here / "training.json"))
            if "decision.json" in names and here.name == "gate":
                items.append(("gate", root, here / "decision.json"))
            if "final_acceptance.json" in names:
                items.append(("gate", root, here / "final_acceptance.json"))
            if "rounds.jsonl" in names:
                items.append(("sim", root, here / "rounds.jsonl"))
            if "iterations.jsonl" in names:
                items.append(("search", root, here / "iterations.jsonl"))
            for name in sorted(names & extra_names):
                if name == "summary.json" and "rounds.jsonl" in names:
                    continue        # already part of the sim record of this directory
                items.append(("record", root, here / name))
    selected = []
    for kind, root, path in items:
        text = str(path)
        if include and not any(fnmatch.fnmatch(text, pattern) for pattern in include):
            continue
        if any(fnmatch.fnmatch(text, pattern) for pattern in exclude):
            continue
        selected.append((kind, root, path))
    return selected


# ---------------------------------------------------------------- helpers
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def number(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    return None


def flatten_numbers(obj, prefix="", out=None, limit=MAX_SCALARS):
    """Numeric leaves of a JSON object as {"a/b/c": value}; bools become 0/1; lists of numbers are skipped."""
    out = {} if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(obj, dict):
        for key, value in obj.items():
            flatten_numbers(value, f"{prefix}/{key}" if prefix else str(key), out, limit)
    else:
        value = number(obj)
        if value is not None and prefix:
            out[prefix] = value
    return out


def scalars_config(obj):
    """Top-level scalar fields of a dict (for SwanLab config)."""
    return {key: value for key, value in obj.items()
            if isinstance(value, (str, int, float, bool)) or value is None}


def text(value):
    body = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=1, default=str)
    return body if len(body) <= MAX_TEXT else body[:MAX_TEXT] + "\n…（截断）"


def mtime_iso(path):
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


TASK_TOKENS = {"A": "A", "B": "B", "C": "C", "gas": "A", "team_gas": "A"}


def guess_task(rel, data=None):
    profile = (data or {}).get("profile") or (data or {}).get("task_name")
    if profile in ("A", "B", "C"):
        return profile
    for part in rel.parts:
        for token in (part, part.split("-")[0], part.split("_")[0]):
            if token in TASK_TOKENS:
                return TASK_TOKENS[token]
    return "unknown"


def short_name(kind, task, rel, limit=90):
    body = f"{kind}-{task}-{rel.as_posix()}"
    if len(body) <= limit:
        return body
    tag = hashlib.sha1(body.encode()).hexdigest()[:8]
    return f"{kind}-{task}-…{rel.as_posix()[-(limit - len(kind) - len(task) - 14):]}-{tag}"


# ---------------------------------------------------------------- SwanLab
class Uploader:
    def __init__(self, args):
        self.args = args
        if args.dry_run:
            self.sw = None
            return
        try:
            import swanlab
        except ImportError:
            sys.exit("未安装 swanlab：先运行 python -m pip install swanlab")
        self.sw = swanlab

    def open(self, name, group, tags, description, config):
        a = self.args
        kwargs = dict(project=a.project, name=name, group=group, tags=tags, description=description,
                      config=config, mode=a.mode, reinit=True)
        if a.workspace:
            kwargs["workspace"] = a.workspace
        if a.log_dir:
            kwargs["log_dir"] = a.log_dir
        if a.public:
            kwargs["public"] = True
        try:
            return self.sw.init(**kwargs)
        except TypeError:          # older SwanLab: experiment_name / logdir / mode "cloud"
            old = dict(project=a.project, experiment_name=name, description=description, config=config,
                       mode="cloud" if a.mode == "online" else a.mode)
            if a.workspace:
                old["workspace"] = a.workspace
            if a.log_dir:
                old["logdir"] = a.log_dir
            return self.sw.init(**old)

    def log(self, data, step=None):
        clean = {}
        for key, value in data.items():
            if isinstance(value, str):
                clean[key] = self.sw.Text(value)
            else:
                v = number(value)
                if v is not None:
                    clean[key] = v
        if clean:
            if step is None:
                self.sw.log(clean)
            else:
                self.sw.log(clean, step=int(step))

    def close(self):
        self.sw.finish()


# ---------------------------------------------------------------- one experiment per record
def describe(kind, rel, original):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return (f"补录（backfill）：原始记录 {rel.as_posix()}，原始时间 {original}。"
            f"由 scripts/swanlab_backfill.py 于 {now} 从本地产物导入，数值未改动。类型：{kind}")


def plan_train(path, root):
    data = read_json(path)
    rel = path.parent.relative_to(root)
    task = guess_task(rel, data)
    original = data.get("created_at") or mtime_iso(path)
    config = scalars_config(data)
    config.update({"configuration": data.get("configuration"), "learning_rate_schedule": data.get("learning_rate_schedule"),
                   "source_path": str(path.parent), "original_time": original, "backfilled": True, "record_kind": "train"})
    report = path.parent / "runner_report.json"
    if report.is_file():
        try:
            runner = read_json(report)
            config["runner_status"] = runner.get("status") if isinstance(runner, dict) else None
        except (OSError, json.JSONDecodeError):
            pass
    history_path = path.parent / "history.jsonl"
    history = read_jsonl(history_path) if history_path.is_file() else []
    steps = []
    for i, row in enumerate(history):
        values = {"train/loss": row.get("loss", row.get("train_loss"))}
        for key, value in (row.get("validation") or row.get("val") or {}).items():
            values[f"val/{key}"] = value
        if "selected" in row:
            values["train/selected_epoch"] = row["selected"]
        steps.append((row.get("epoch", i), values))
    last = steps[-1][0] if steps else 0
    summary = {f"summary/val_before_{k}": v for k, v in (data.get("validation_before") or {}).items()}
    summary.update({f"summary/val_after_{k}": v for k, v in (data.get("validation_after") or {}).items()})
    summary.update({"summary/training_seconds": data.get("training_seconds"), "summary/train_rows": data.get("train_rows"),
                    "summary/validation_rows": data.get("validation_rows"), "summary/trained_epochs": data.get("trained_epochs")})
    steps.append((last, summary))
    policy = None
    for candidate in (path.parent / "policy.py", path.parent.parent / f"{path.parent.name}.policy.py"):
        if candidate.is_file():
            policy = candidate.read_text(encoding="utf-8", errors="replace")
            break
    if policy:
        steps.append((last, {"policy/source": text(policy)}))
    return task, rel, original, config, steps


def plan_gate(path, root):
    data = read_json(path)
    rel = path.relative_to(root)
    task = guess_task(rel, data if isinstance(data, dict) else None)
    original = mtime_iso(path)
    config = scalars_config(data) if isinstance(data, dict) else {}
    if isinstance(data, dict) and isinstance(data.get("bootstrap"), dict):
        config.update({f"bootstrap_{k}": v for k, v in data["bootstrap"].items()})
    config.update({"source_path": str(path), "original_time": original, "backfilled": True, "record_kind": "gate"})
    values = flatten_numbers(data)
    values["decision/record"] = text(data)
    return task, rel, original, config, [(0, values)]


def plan_sim(path, root):
    rows = read_jsonl(path)
    rel = path.parent.relative_to(root)
    task = guess_task(rel, rows[0] if rows else None)
    original = mtime_iso(path)
    config = {"source_path": str(path.parent), "original_time": original, "backfilled": True, "record_kind": "sim"}
    summary_path = path.parent / "summary.json"
    steps = []
    if summary_path.is_file():
        summary = read_json(summary_path)
        config.update({f"summary_{k}": v for k, v in scalars_config(summary).items()})
    for i, row in enumerate(rows):
        step = row.get("round", i + 1)
        values = flatten_numbers({key: value for key, value in row.items() if key != "decision"})
        values.update(flatten_numbers(row.get("decision") or {}, "decision"))
        values["decision/record"] = text(row.get("decision") or {})
        steps.append((step, values))
    if summary_path.is_file():
        steps.append((steps[-1][0] if steps else 0, flatten_numbers(read_json(summary_path), "summary")))
    return task, rel, original, config, steps


def plan_search(path, root):
    rows = read_jsonl(path)
    rel = path.parent.relative_to(root)
    first = rows[0] if rows else {}
    task = guess_task(rel, first)
    original = mtime_iso(path)
    config = {key: first.get(key) for key in ("run_id", "stage_id", "task_name", "schema") if key in first}
    config.update({"source_path": str(path.parent), "original_time": original, "backfilled": True,
                   "record_kind": "search", "iterations": len(rows)})
    steps = []
    for i, row in enumerate(rows):
        step = row.get("iteration") if isinstance(row.get("iteration"), int) else i
        values = flatten_numbers({key: value for key, value in row.items() if key not in ("metrics",)})
        values["iteration/status"] = f"{row.get('status')} / {row.get('outcome')}"
        if row.get("hypothesis"):
            values["iteration/hypothesis"] = text(row["hypothesis"])
        steps.append((step, values))
    return task, rel, original, config, steps


def plan_record(path, root):
    """Any other JSON / JSONL result file named with --extra: numbers as scalars, the whole file as text."""
    rel = path.relative_to(root)
    original = mtime_iso(path)
    config = {"source_path": str(path), "original_time": original, "backfilled": True, "record_kind": "record"}
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
        task = guess_task(rel, rows[0] if rows and isinstance(rows[0], dict) else None)
        steps = []
        for i, row in enumerate(rows):
            values = flatten_numbers(row) if isinstance(row, dict) else {}
            values["record/line"] = text(row)
            steps.append((i, values))
        return task, rel, original, config, steps
    data = read_json(path)
    task = guess_task(rel, data if isinstance(data, dict) else None)
    if isinstance(data, dict):
        config.update(scalars_config(data))
    values = flatten_numbers(data) if isinstance(data, dict) else {}
    values["record/content"] = text(data)
    return task, rel, original, config, [(0, values)]


PLANNERS = {"train": plan_train, "gate": plan_gate, "sim": plan_sim, "search": plan_search, "record": plan_record}


# ---------------------------------------------------------------- main
def load_ledger(path):
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def save_ledger(path, ledger):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", action="append", required=True, help="实验产物根目录，可重复")
    parser.add_argument("--project", default="zhixun-industrial-rsi", help="SwanLab 项目名")
    parser.add_argument("--workspace", help="SwanLab 组织/个人空间，默认个人")
    parser.add_argument("--mode", default="online", choices=["online", "offline", "local", "disabled"])
    parser.add_argument("--log-dir", help="SwanLab 本地日志目录（offline/local 时用）")
    parser.add_argument("--public", action="store_true", help="新建项目时设为公开（组委会要求记录公开）")
    parser.add_argument("--kinds", default="train,gate,sim,search,record", help="要上传的类型，逗号分隔")
    parser.add_argument("--extra", action="append", default=[],
                        help="另外按文件名收录的结果文件（如 results.json、comparison.json），作为 record 类型；可重复")
    parser.add_argument("--include", action="append", default=[], help="只要路径匹配这些通配符的记录")
    parser.add_argument("--exclude", action="append", default=[], help="排除路径匹配这些通配符的记录")
    parser.add_argument("--include-tests", action="store_true", help="也上传 tests/ 下的测试产物（默认跳过）")
    parser.add_argument("--max", type=int, default=0, help="最多上传多少条（0 = 不限）")
    parser.add_argument("--ledger", default="state/swanlab_ledger.json", help="已上传记录的台账")
    parser.add_argument("--dry-run", action="store_true", help="只列出会上传什么，不连 SwanLab")
    args = parser.parse_args(argv)

    kinds = {kind.strip() for kind in args.kinds.split(",") if kind.strip()}
    items = [item for item in discover(args.root, args.include_tests, args.include, args.exclude, args.extra)
             if item[0] in kinds]
    ledger_path = Path(args.ledger)
    ledger = load_ledger(ledger_path)
    todo = []
    for kind, root, path in items:
        digest = sha256_file(path)
        if ledger.get(str(path), {}).get("sha256") == digest:
            continue
        todo.append((kind, root, path, digest))
    if args.max:
        todo = todo[:args.max]

    counts = Counter(kind for kind, *_ in todo)
    groups = Counter((kind, "/".join(path.relative_to(root).parts[:2])) for kind, root, path, _ in todo)
    log(f"[backfill] 找到 {len(items)} 条记录，待上传 {len(todo)} 条：{dict(counts)}（台账里已上传且未变的跳过）")
    for (kind, group), n in sorted(groups.items()):
        log(f"  {kind:6s} {n:5d}  {group}")
    if args.dry_run:
        for kind, root, path, _ in todo[:20]:
            log(f"  例：{kind} {path}")
        return 0

    uploader = Uploader(args)
    done = failed = 0
    started = time.time()
    for kind, root, path, digest in todo:
        try:
            task, rel, original, config, steps = PLANNERS[kind](path, root)
            name = short_name(kind, task, rel)
            group = "/".join(rel.parts[:3]) or kind
            uploader.open(name, group, [kind, f"task-{task}", "backfill"], describe(kind, rel, original), config)
            try:
                for step, values in steps:
                    uploader.log(values, step)
            finally:
                uploader.close()
            ledger[str(path)] = {"sha256": digest, "kind": kind, "name": name,
                                 "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                 "project": args.project, "mode": args.mode}
            save_ledger(ledger_path, ledger)
            done += 1
            if done % 20 == 0:
                log(f"[backfill] 已上传 {done}/{len(todo)}，用时 {time.time() - started:.0f} 秒")
        except Exception as error:      # one bad record must not stop the rest
            failed += 1
            log(f"[backfill] 失败 {kind} {path}: {type(error).__name__}: {error}")
    log(f"[backfill] 完成：成功 {done}，失败 {failed}，用时 {time.time() - started:.0f} 秒。项目：{args.project}")
    return 1 if failed and not done else 0


if __name__ == "__main__":
    sys.exit(main())
