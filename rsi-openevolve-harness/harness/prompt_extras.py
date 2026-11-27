"""Extra prompt variables supplied to OpenEvolve in every worker, every iteration.

  {playbook}         the ACE playbook file, read fresh each time (so ACE can update it mid-run)
  {recent_failures}  the latest attempts that failed or made things worse; OpenEvolve
                     itself forgets attempts that never became a program

Configured with OPENEVOLVE_PROMPT_EXTRAS=harness.prompt_extras:get_extras.
This function must never raise and must always return both keys, because the prompt
template refers to them.
"""
import hashlib
from pathlib import Path

from .records import read_records
from .runcfg import load_run_config

EMPTY_PLAYBOOK = "（暂无经验条目）"
NO_FAILURES = "（暂无）"


def playbook_section(config):
    path = Path(config["playbook_path"])
    text = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
    version = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] if text else "none"
    limit = int(config["prompt"]["playbook_max_chars"])
    if len(text) > limit:
        text = text[:limit].rsplit("\n", 1)[0] + "\n（后面的条目因长度限制被省略）"
    return f"PLAYBOOK_VERSION: {version}\n{text or EMPTY_PLAYBOOK}"


def _short(text, limit=160):
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def recent_failures_section(config):
    limit = int(config["prompt"]["recent_failures"])
    if limit <= 0:
        return NO_FAILURES
    lines = []
    for record in reversed(read_records(config, last=200)):
        if record["outcome"] == "failed":
            what = record.get("hypothesis") or record.get("changes") or "（未说明）"
            lines.append(f"- 第 {record['iteration']} 轮，未得到有效结果：{_short(record.get('error'))}｜当时的想法：{_short(what)}")
        elif record["outcome"] == "regressed":
            what = record.get("hypothesis") or record.get("changes") or "（未说明）"
            flags = "、".join(record.get("flags", [])[:3]) or "无"
            lines.append(f"- 第 {record['iteration']} 轮，比父版本更差（{record['parent_combined_score']:.4f} → "
                         f"{record['combined_score']:.4f}），症状：{flags}｜当时的想法：{_short(what)}")
        if len(lines) >= limit:
            break
    return "\n".join(lines) or NO_FAILURES


def get_extras(context=None):
    try:
        config = load_run_config()
    except Exception:
        return {"playbook": f"PLAYBOOK_VERSION: none\n{EMPTY_PLAYBOOK}", "recent_failures": NO_FAILURES}
    try:
        playbook = playbook_section(config)
    except Exception:
        playbook = f"PLAYBOOK_VERSION: none\n{EMPTY_PLAYBOOK}"
    try:
        failures = recent_failures_section(config)
    except Exception:
        failures = NO_FAILURES
    return {"playbook": playbook, "recent_failures": failures}
