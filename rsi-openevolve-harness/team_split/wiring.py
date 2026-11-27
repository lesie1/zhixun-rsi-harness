"""Derive a run config that uses a split manifest, including a fresh ACE scope.

The harness picks up a manifest only through run config key "split_manifest". Changing
the split changes what every candidate is scored on, so the derived config always gets a
new stage_id and a new, empty work_dir (cached reference results and reports of the old
split must not be reused).

When the config uses team_ace, its experience store is partitioned by a "scope" that
includes the manifest path and the manifest file's SHA-256 (team_ace.store.scope_for).
A new split therefore means a new scope and a new, empty playbook: experience gathered
under another split is not carried over. For the same reason a manifest must never be
edited once a config refers to it; write a new versioned file instead.
"""
import copy
import json
import re
from pathlib import Path

from harness.runcfg import PATH_KEYS, load_run_config

from .manifest import read_manifest
from .table import load_table
from .verify import has_errors, verify_manifest

SUFFIX = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def _absolute(value, base_dir):
    path = Path(value)
    return str(path if path.is_absolute() else (base_dir / path).resolve())


def _uses_team_ace(config):
    ace = config.get("ace")
    return isinstance(ace, dict) and ace.get("module") == "team_ace" and "scope" in ace


def derive_run_config(base_config_path, manifest_path, out_path, suffix, force=False):
    """Write the derived config to `out_path` and return it (as written)."""
    base_config_path, manifest_path, out_path = Path(base_config_path).resolve(), Path(manifest_path).resolve(), Path(out_path)
    if not SUFFIX.match(suffix or ""):
        raise ValueError("suffix 只能包含字母、数字、点、下划线和连字符（1–64 个字符）")
    if out_path.resolve() == base_config_path:
        raise ValueError("不能覆盖原配置；请把新配置写到另一个文件")
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} 已存在；确认要覆盖时加 --force")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到清单 {manifest_path}")

    raw = json.loads(base_config_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(raw)
    base_dir = base_config_path.parent
    for key in PATH_KEYS:  # the derived file may live elsewhere: keep every path valid
        if config.get(key):
            config[key] = _absolute(config[key], base_dir)
    if _uses_team_ace(config) and config["ace"].get("storage_root"):
        config["ace"]["storage_root"] = _absolute(config["ace"]["storage_root"], base_dir)

    # The manifest must be valid for this stage's validation data before anything is wired.
    table = load_table(load_run_config(base_config_path))
    manifest = read_manifest(manifest_path)
    problems = verify_manifest(manifest, table)
    if has_errors(problems):
        details = "；".join(f"{p.code}: {p.message}" for p in problems if p.severity == "error")
        raise ValueError(f"清单不能用于这个阶段：{details}")

    config["split_manifest"] = str(manifest_path)
    config["stage_id"] = f"{config.get('stage_id', 'round-01')}-{suffix}"
    work_dir = Path(config["work_dir"])
    config["work_dir"] = str(work_dir.with_name(f"{work_dir.name}-{suffix}"))
    new_work = Path(config["work_dir"])
    if new_work.exists() and any(new_work.iterdir()):
        raise FileExistsError(f"新的 work_dir {new_work} 已存在且不为空；换一个 suffix，或先确认这个目录可以删除")

    if _uses_team_ace(config):
        from team_ace.store import digest, initialize, scope_for
        config["ace"]["scope"] = scope_for(config)
        folder = Path(config["ace"]["storage_root"]) / config["task_name"] / digest(config["ace"]["scope"])[:16]
        config["playbook_path"] = str(folder / "playbook.md")
        initialize(config)  # creates the empty, versioned experience store of the new scope

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_run_config(out_path)  # the harness must be able to read what we wrote
    return config
