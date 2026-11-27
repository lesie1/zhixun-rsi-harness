"""Process the queued ACE jobs of one finished search round again, on a copy of its ACE state.

  python -m team_sim ace-retry --search-dir runs/B-live-01/f0/p2c/v3/round-10/search \
      --ace-approve-live --ace-no-thinking --ace-max-jobs 2

For trying new ACE call settings on real search evidence without rerunning the searches. The
round's ACE folder (located through its search/run.json) is copied to state/sim/retry-<time>-<hash>/
and only the copy is processed, so the finished run stays exactly as it was. Model, endpoint, key
variable and call budget come from run.json (--ace-max-calls changes the copy's budget), and the copy
keeps the calls that round already spent.
Jobs that failed once are left alone unless include_failed (team_ace tries a job at most twice).
A run without --keep-data has deleted the round's data files; with --package they are rebuilt into
the retry's directory and checked against the hashes recorded in the ACE scope.
"""
import copy
import hashlib
import itertools
import json
import os
import shutil
import sqlite3
import time
from pathlib import Path

from harness import PROJECT_ROOT


def _snapshot(folder):
    db = sqlite3.connect(f"file:{Path(folder) / 'memory.sqlite3'}?mode=ro", uri=True)
    try:
        return {"jobs": dict(db.execute("SELECT id, status FROM jobs").fetchall()),
                "last_call": db.execute("SELECT coalesce(max(id), 0) FROM calls").fetchone()[0],
                "entries": [row[0] for row in db.execute("SELECT id FROM entries")]}
    finally:
        db.close()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rebuild_round_data(config, search_dir, package, folder):
    """Write the round's train/validation files again (a run without --keep-data deletes them).

    team_ace checks a scope by the content hashes of these files, not by their paths, so they are
    written into `folder` (the retry's own directory) and must hash exactly as recorded in the scope.
    Fold, protocol and round come from the path <out>/f<fold>/p<protocol>/<policy>/round-<k>/search.
    """
    from . import data as simdata
    parts = Path(search_dir).parts
    if len(parts) < 5 or parts[-1] != "search" or not parts[-2].startswith("round-") \
            or not parts[-4].startswith("p") or not parts[-5].startswith("f"):
        raise ValueError(f"cannot tell fold/protocol/round from {search_dir}")
    fold_id, protocol, k = int(parts[-5][1:]), parts[-4][1:], int(parts[-2].split("-", 1)[1])
    units = simdata.by_unit(simdata.load_rows(package, config["task_name"]))
    fold = next((f for f in simdata.make_folds(units) if f["fold"] == fold_id), None)
    if fold is None:
        raise ValueError(f"fold {fold_id} does not exist in {package}")
    rows = simdata.round_rows(units, fold, protocol, k)
    paths = {"train_path": simdata.write_jsonl(Path(folder) / "train.jsonl", rows["train"]),
             "validation_path": simdata.write_jsonl(Path(folder) / "validation.jsonl", rows["validation"])}
    recorded = config["ace"]["scope"]["source_hashes"]
    for key, path in paths.items():
        if _sha256(path) != recorded[key]:
            raise ValueError(f"rebuilt {key} differs from the data of that run (another --package?)")
    return paths


def retry(search_dir, approve_live=False, thinking=None, request_timeout=300, max_tokens=4096,
          worker_seconds=630, max_jobs=1, include_failed=False, ask_key=None, package=None, max_calls=None):
    """Copy the round's ACE state, then let the bounded team_ace worker process up to max_jobs jobs there.

    thinking: None keeps the provider default; False/True sends enable_thinking with each call.
    ask_key(name) is called for the key when the variable named in run.json is not set (live mode).
    package: the B package of the run; needed only when the round's data files were deleted.
    max_calls: call budget of the copy (calls already made in that round count against it); default as in run.json.
    """
    from team_ace.__main__ import bounded_worker
    from team_ace.backend import request_limits, thinking_body
    from team_ace.store import connect, exclusive, location, status

    search_dir = Path(search_dir).resolve()
    config = json.loads((search_dir / "run.json").read_text(encoding="utf-8"))
    ace = config.get("ace") or {}
    if ace.get("module") != "team_ace":
        raise ValueError(f"{search_dir} did not use team_ace")
    mode = ace.get("mode")
    if mode not in ("fixture", "live"):
        raise ValueError(f"ACE mode of that round is {mode!r}; only fixture and live call a model")
    if mode == "live" and not approve_live:
        raise ValueError("live ACE calls spend model quota; add --ace-approve-live")
    if max_calls is not None and not 1 <= max_calls <= 20:
        raise ValueError("--ace-max-calls must be within 1..20")
    if not 1 <= max_jobs <= 6:
        raise ValueError("--ace-max-jobs must be within 1..6")
    if worker_seconds < 2 * request_timeout + 30:
        raise ValueError(f"--ace-worker-seconds must be at least {2 * request_timeout + 30}: one job is a "
                         f"Reflector and a Curator call of up to --ace-request-timeout={request_timeout}s each")
    new = copy.deepcopy(config)
    new["ace"].update(request_timeout_s=request_timeout, max_tokens=max_tokens, live_calls_approved=bool(approve_live))
    new["ace"].pop("enable_thinking", None)
    if max_calls is not None:
        new["ace"]["max_total_calls"] = max_calls
    if thinking is not None:
        new["ace"]["enable_thinking"] = thinking
    request_limits(new["ace"])
    thinking_body(new["ace"])
    source = location(config)
    if not (source / "memory.sqlite3").is_file():
        raise FileNotFoundError(f"no ACE state at {source}")
    if not Path(config["parent_checkpoint"]).is_file():
        raise FileNotFoundError(f"the round's checkpoint is gone: {config['parent_checkpoint']}")
    missing = not (Path(config["train_path"]).is_file() and Path(config["validation_path"]).is_file())
    if missing and not package:
        raise ValueError("the round's data files were deleted after the run (no --keep-data); "
                         "pass --package <the B package of that run> to rebuild them")
    key = new["ace"].get("api_key_env")
    if mode == "live" and not os.environ.get(key or ""):
        if not key or ask_key is None:
            raise ValueError(f"set the model key in ${key} first (the variable named in run.json)")
        os.environ[key] = ask_key(key)   # this process only; the worker gets exactly this variable
        if not os.environ[key]:
            raise ValueError("no key given")

    tag = hashlib.sha256(str(search_dir).encode("utf-8")).hexdigest()[:6]
    first = root = PROJECT_ROOT / "state" / "sim" / f"retry-{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
    for n in itertools.count(1):
        if not root.exists():
            break
        root = first.with_name(f"{first.name}-{n}")
    target = root / config["task_name"] / source.name
    try:
        if missing:
            new.update(rebuild_round_data(config, search_dir, package, root / "data"))
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("worker-*", "worker.lock"))
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise
    new["ace"]["storage_root"] = str(root)
    new["playbook_path"] = str(target / "playbook.md")
    path = root / "run.json"
    path.write_text(json.dumps(new, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if location(new) != target:
        raise RuntimeError(f"copied ACE state is not where team_ace expects it: {target}")
    if include_failed:
        with exclusive(new):
            db = connect(new)
            try:
                with db:
                    db.execute("UPDATE jobs SET status='pending' WHERE status='failed' AND attempts<2")
            finally:
                db.close()
    before = _snapshot(target)
    workers = []
    for _ in range(max_jobs):
        if not status(new)["jobs"].get("pending", 0):
            break
        workers.append(bounded_worker(str(path), seconds=worker_seconds))
    return {"search_dir": str(search_dir), "source": str(source), "copy": str(target), "config": str(path),
            "before": before, "workers": workers, "status": status(new)}


def describe(result):
    """Readable lines about what this retry did: workers, changed jobs, new calls with tokens, new entries."""
    lines = [f"原状态（未改动）：{result['source']}", f"副本：{result['copy']}"]
    for w in result["workers"]:
        tail = ""
        for suffix in (".stdout", ".stderr"):
            p = Path(w["log"] + suffix)
            text = [l.strip() for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()] \
                if p.exists() else []
            tail += (text[-1][:300] + " ") if text else ""
        lines.append(f"worker：退出码 {w['exit_code']}，停止原因 {w['stop_reason']}，{w['seconds']:.0f} 秒，"
                     f"输出 {tail.strip() or '（空）'}")
    if not result["workers"]:
        lines.append("没有排队的任务（可加 --include-failed 重试失败过一次的任务）")
    s, before = result["status"], result["before"]
    lines.append(f"副本状态：任务 {s['jobs']}，调用 {s['calls']}，条目 {s['entries']}")
    db = sqlite3.connect(f"file:{Path(result['copy']) / 'memory.sqlite3'}?mode=ro", uri=True)
    try:
        for job, state, error in db.execute("SELECT id, status, error FROM jobs ORDER BY rowid"):
            if before["jobs"].get(job) != state:
                lines.append(f"任务 {job[:12]}：{state}" + (f"：{error[:300]}" if error else ""))
        total = 0
        for role, state, usage in db.execute("SELECT role, status, usage FROM calls WHERE id>? ORDER BY id",
                                             (before["last_call"],)):
            u = (json.loads(usage) if usage else None) or {}
            total += u.get("total_tokens") or 0
            lines.append(f"调用 {role}：{state}，输入 {u.get('prompt_tokens')} / 输出 {u.get('completion_tokens')} token")
        lines.append(f"本次调用 token 合计 {total}")
        replies = Path(result["copy"]) / "replies"
        if replies.is_dir():
            lines.append(f"每次调用的完整提示词和回复：{replies}/")
        for entry_id, body in db.execute("SELECT id, body FROM entries ORDER BY id"):
            if entry_id in before["entries"]:
                continue
            e = json.loads(body)
            lines.append(f"新经验 [{entry_id}]\n  症状：{e.get('symptom')}\n  做法：{e.get('action')}\n"
                         f"  适用：{e.get('scope_note')}\n  证据 {len(e.get('evidence_ids') or [])} 条，结果 {e.get('outcomes')}")
    finally:
        db.close()
    return lines
