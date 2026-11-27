#!/usr/bin/env python
"""Read-only status of a team_sim run with search and ACE (standard library only).

  python scripts/sim_status.py runs/B-live-02

For every search: model calls and tokens, finished iterations, best score against the start and how it
stopped; the ACE workers, failed jobs, call tokens, how many learned entries were carried into the round
and how many the search prompt actually showed (read from the playbook version the search records
name); and the full text of the entries ACE wrote. Then the rounds that asked for a search, and totals.
"""
import glob
import json
import os
import re
import sqlite3
import sys

TOKENS = re.compile(r"LLM Token Usage \([^)]*\): prompt_tokens=(\d+), completion_tokens=(\d+), total_tokens=(\d+)")


def read(path):
    return open(path, encoding="utf-8", errors="replace").read() if os.path.exists(path) else ""


def load(path):
    try:
        with open(path, encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def last_line(path):
    lines = [line.strip() for line in read(path).splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""


def search_part(d):
    log = read(f"{d}/evolve.log")
    calls = [int(m[2]) for m in TOKENS.findall(log)]
    iterations = sum("- Iteration " in line and "completed in" in line for line in log.splitlines())
    score = ""
    done = next((line for line in log.splitlines() if line.startswith("[harness] done:")), None)
    if done:
        try:
            summary = json.loads(done.split(":", 1)[1])
            best = (summary.get("best_metrics") or {}).get("combined_score")
            if best is not None:
                score = f"，最佳分 {best:.4g}（起点 {summary.get('reference_combined_score'):.4g}）"
        except (ValueError, TypeError):
            pass
    return calls, f"调用模型 {len(calls)} 次，{sum(calls)} token，完成迭代 {iterations}{score}"


def ace_part(d, result, con):
    seeded = load(f"{d}/seeded.json") or {}
    base, carried = set(seeded.get("seeded") or []), seeded.get("carried") or []
    lines, tokens = [], 0
    for job, error in con.execute("SELECT id, error FROM jobs WHERE status='failed' AND error IS NOT NULL"):
        lines.append(f"   ACE 任务失败 {job[:12]}：{error[:300]}")
    for (usage,) in con.execute("SELECT usage FROM calls WHERE usage IS NOT NULL"):
        tokens += ((json.loads(usage) if usage else None) or {}).get("total_tokens") or 0
    first = read(f"{d}/work/records/iterations.jsonl").split("\n", 1)[0]
    version = (json.loads(first).get("playbook_version") if first.strip() else None)
    row = con.execute("SELECT text FROM versions WHERE version=?", (version,)).fetchone() if version else None
    shown = [i for i in re.findall(r"^- \[([^\]]+)\]", row[0], re.M) if i not in base] if row else None
    lines.insert(0, f"   ACE：带入上一轮经验 {len(carried)} 条，本轮搜索提示词里的 ACE 经验 "
                    f"{len(shown) if shown is not None else '?'} 条，ACE 调用 token {tokens}")
    for (body,) in con.execute("SELECT body FROM entries ORDER BY id"):
        entry = json.loads(body)
        if entry.get("base_experience") or entry.get("carried_from") or entry["id"] in base:
            continue
        lines.append(f"   新经验 [{entry['id']}] 证据 {len(entry.get('evidence_ids') or [])} 条，结果 {entry.get('outcomes')}\n"
                     f"     症状：{entry.get('symptom')}\n     做法：{entry.get('action')}\n     适用：{entry.get('scope_note')}")
    return tokens, lines


def main(out):
    search_total = ace_total = 0
    for d in sorted(glob.glob(f"{out}/f*/p*/*/round-*/search")):
        name = "/".join(d.split("/")[-5:-1])
        calls, text = search_part(d)
        search_total += sum(calls)
        print(f"\n== {name}  搜索：{text}")
        result = load(f"{d}/search_result.json")
        if result is None:
            print("   没有 search_result.json（搜索或 ACE 中途出错，或被打断）")
            continue
        print(f"   退出码 {result.get('returncode')}，停止方式 {result.get('stopped')}，{result.get('seconds')} 秒，"
              f"有最佳策略 {result.get('best_program')}，和起点相同 {result.get('best_is_initial')}")
        if result.get("ace_error"):
            print(f"   ACE 出错：{result['ace_error']}")
        for w in result.get("ace_worker") or []:
            tail = " ".join(x for x in (last_line((w.get("log") or "") + ".stdout"),
                                        last_line((w.get("log") or "") + ".stderr")) if x)
            print(f"   ACE worker：退出码 {w.get('exit_code')}，停止原因 {w.get('stop_reason')}，"
                  f"{round(w.get('seconds') or 0)} 秒，输出 {tail or '（空）'}")
        status = result.get("ace_status") or {}
        print(f"   ACE 状态：任务 {status.get('jobs')}，调用 {status.get('calls')}，条目 {status.get('entries')}")
        db = os.path.join(status.get("folder") or "", "memory.sqlite3")
        if os.path.exists(db):
            con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                tokens, lines = ace_part(d, result, con)
            finally:
                con.close()
            ace_total += tokens
            print("\n".join(lines))
    print()
    for path in sorted(glob.glob(f"{out}/f*/p*/*/rounds.jsonl")):
        run = "/".join(path.split("/")[-4:-1])
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            d = r["decision"]
            if d.get("search_requested"):
                s = d.get("search") or {}
                act = "上线了搜索候选" if d.get("action") == "search" else f"上线 {d.get('action')}"
                print(f"{run} 第 {r['round']} 轮：搜出新候选 {d.get('search_ran')}（{s.get('stopped')}，"
                      f"{s.get('seconds')} 秒），{act}，测试优势 {r['test']['advantage']:+.1%}")
    print(f"\n搜索 token 合计 {search_total}，ACE token 合计 {ace_total}")
    print("report.md：", "已生成" if os.path.exists(f"{out}/report.md") else "没有（还在跑，或中途出错）")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/B-live-02")
