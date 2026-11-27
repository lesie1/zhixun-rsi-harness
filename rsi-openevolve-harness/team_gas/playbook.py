"""Base experiences: loaded into the playbook a stage actually reads, and verified in the real prompt.

The prompt reads `playbook_path` (harness.prompt_extras). With team_ace that file is
re-rendered from the scope's database on every publish, so a hand edit would be lost; the
base entry is therefore written into the scope database with team_ace's own functions
(under its lock) and published, with a stable id that sorts before team_ace's generated ids.
Existing entries are kept. Every stage gets a new team_ace scope, so each eligible stage is
seeded again; the id makes it idempotent and it appears once per prompt.

Without team_ace (e.g. harness.ace_stub) the playbook file is written directly, keeping any
lines already there.
"""
import json
import os
import re
from pathlib import Path

from harness import PROJECT_ROOT


def load_base(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["entries"]


def eligible(entry, context):
    """(ok, reasons). context: task_name, task_type, pair, trainable_new_labels."""
    cond = entry.get("conditions") or {}
    reasons = []
    if cond.get("task_names") and context.get("task_name") not in cond["task_names"]:
        reasons.append(f"任务 {context.get('task_name')} 不在 {cond['task_names']}")
    if cond.get("task_type") and context.get("task_type") != cond["task_type"]:
        reasons.append(f"任务类型 {context.get('task_type')} 不是 {cond['task_type']}")
    if cond.get("needs_confusion_pair") and not context.get("pair"):
        reasons.append("没有识别出“退化类→混淆类”的方向")
    if cond.get("needs_trainable_new_labels") and not context.get("trainable_new_labels"):
        reasons.append("没有允许训练的新标签数据")
    return not reasons, reasons


def _render_line(entry, task):
    """Same line format team_ace.store.publish uses."""
    return (f"- [{entry['id']}] 待验证｜{task}｜症状假设：{entry['symptom']}；建议：{entry['action']}；"
            f"范围：{entry['scope_note']}；证据：{','.join(x[:10] for x in entry['evidence_ids'])}；"
            f"结果关联：{entry['outcomes']}（不代表因果）")


def _uses_team_ace(config):
    ace = config.get("ace")
    return isinstance(ace, dict) and ace.get("module") == "team_ace"


def seed(run_config, entries):
    """Put `entries` into the playbook of this stage; return what the playbook now holds."""
    if not entries:
        return {"seeded": [], "mode": None}
    if _uses_team_ace(run_config):
        from team_ace.store import connect, digest, encoded, exclusive, publish
        with exclusive(run_config):
            db = connect(run_config)
            try:
                with db:
                    for entry in entries:
                        body = {"id": entry["id"], "symptom": entry["symptom"], "action": entry["action"],
                                "scope_note": entry["scope_note"], "evidence_ids": list(entry["evidence_ids"]),
                                "outcomes": entry["outcomes"], "scope_id": digest(run_config["ace"]["scope"]),
                                "status": "candidate", "mode": "base", "independent_validation": False,
                                "base_experience": True, "source": entry["source"], "added": entry["added"],
                                "conditions": entry.get("conditions"), "text": entry["text"]}
                        db.execute("INSERT OR REPLACE INTO entries VALUES (?,?)", (entry["id"], encoded(body)))
                version = publish(run_config, db)
            finally:
                db.close()
        return {"seeded": [e["id"] for e in entries], "mode": "team_ace", "version": version,
                "playbook_path": run_config["playbook_path"]}
    path = Path(run_config["playbook_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else [f"# {run_config['task_name']} 任务经验"]
    for entry in entries:
        if not any(f"[{entry['id']}]" in line for line in lines):
            lines.insert(1, _render_line(entry, run_config["task_name"]))
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return {"seeded": [e["id"] for e in entries], "mode": "file", "playbook_path": str(path)}


def render_prompt(run_config_path, oe_config=None, policy=None):
    """Build the user/system prompt exactly as an OpenEvolve worker does for this stage."""
    import sys
    vendored = PROJECT_ROOT / "third_party" / "openevolve"
    if str(vendored) not in sys.path:
        sys.path.insert(0, str(vendored))
    from openevolve.config import load_config
    from openevolve.prompt.sampler import PromptSampler

    from harness import prompt_extras
    from harness.runcfg import ENV_VAR, load_run_config, write_resolved

    previous = os.environ.get(ENV_VAR)
    os.environ[ENV_VAR] = write_resolved(load_run_config(run_config_path))
    try:
        cfg = load_config(str(oe_config or PROJECT_ROOT / "configs" / "openevolve.yaml"))
        sampler = PromptSampler(cfg.prompt)
        extras = prompt_extras.get_extras({"iteration": 1})
        code = Path(policy or PROJECT_ROOT / "harness" / "policy_initial.py").read_text(encoding="utf-8")
        return sampler.build_prompt(current_program=code, parent_program=code,
                                    program_metrics={"combined_score": 1.0, "primary_ratio": 1.0},
                                    language="python", evolution_round=1, diff_based_evolution=True, **extras)
    finally:
        if previous is None:
            os.environ.pop(ENV_VAR, None)
        else:
            os.environ[ENV_VAR] = previous


def occurrences(prompt, entry_id):
    text = prompt.get("user", "") + "\n" + prompt.get("system", "")
    return len(re.findall(re.escape(f"[{entry_id}]"), text))
