"""Turn one batch into a stage the existing harness can run.

<out>/batches/<id>/
  stage.json          base config with this batch's data and the deployed checkpoint as parent
  split.json          split manifest made by team_split (+ split.report.json)
  run.json            stage.json wired to the manifest by team_split.derive_run_config
                      (new stage_id "...-split", new work dir "work-split", fresh ACE scope)
  work-split/         the harness work dir: search, records, gate/decision.json, selected/
"""
import contextlib
import copy
import io
import json
from pathlib import Path

from team_split import derive_run_config
from team_split.__main__ import main as split_cli

SUFFIX = "split"


def batch_dir(out_dir, batch_id):
    return Path(out_dir) / "batches" / batch_id


def prepare(base_raw, state, batch, out_dir):
    """Write stage.json, split.json and run.json for `batch`; return the path of run.json."""
    folder = batch_dir(out_dir, batch["batch_id"])
    folder.mkdir(parents=True, exist_ok=True)
    run_path = folder / "run.json"
    if run_path.is_file():  # an interrupted batch: reuse what was already wired
        return run_path
    config = copy.deepcopy(base_raw)
    config.update({"train_path": batch["train_path"], "validation_path": batch["validation_path"],
                   "parent_checkpoint": state["deployed"]["checkpoint"],
                   "stage_id": f"{state['stage_prefix']}-{batch['batch_id']}", "work_dir": str(folder / "work")})
    for key in ("split_manifest", "playbook_path"):
        config.pop(key, None)  # each batch gets its own split; the playbook follows the new scope
    stage_path = folder / "stage.json"
    stage_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = folder / "split.json"
    log = io.StringIO()
    if not manifest.is_file():
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            code = split_cli(["make", "--config", str(stage_path), "--out", str(manifest), "--suffix", SUFFIX])
        (folder / "split.log").write_text(log.getvalue(), encoding="utf-8")
        if code != 0:
            raise RuntimeError(f"划分失败，见 {folder / 'split.log'}")
    derive_run_config(stage_path, manifest, run_path, SUFFIX)
    return run_path
