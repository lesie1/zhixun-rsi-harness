"""Process one arriving batch: decide whether to retrain, run a round if so, output the decision.

    newest batch ──► trigger.check(deployed model) ──► skip ──────────────────────────┐
                                    │ retrain                                          │
                                    ▼                                                  ▼
               stage.prepare (split + wire) ─► run_evolve.py ─► run_gate.py ─► decision.json
                                                                                       │
                     deploy = gate adopted a new checkpoint; otherwise keep the old one ◄┘

The search and the gate run as separate processes (exactly the documented commands), so
a crash or timeout in the search cannot take the loop down: the gate then decides
between the official continuation and the deployed model, and if even the gate fails
the deployed model stays. Every batch therefore ends with a decision.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from harness import PROJECT_ROOT
from harness.dataio import read_json, write_json

from . import stage as stage_mod
from . import state as state_mod
from . import trigger

RETRAIN_MODES = ("auto", "always", "never")
SEARCH_MODES = ("evolve", "continuation")


def _run(command, log_path, timeout):
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        try:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=timeout or None)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
            log.write(f"\n[team_loop] stopped after {timeout} s\n")
    return {"returncode": code, "seconds": round(time.time() - started, 2), "log": str(log_path)}


def _evolve_command(run_config, args, policy):
    command = [sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "run_evolve.py"), "--run-config", str(run_config),
               "--oe-config", str(args["oe_config"]), "--initial-policy", str(policy)]
    for flag, key in (("--iterations", "iterations"), ("--api-base", "api_base"), ("--model", "model"),
                      ("--workers", "workers")):
        if args.get(key) is not None:
            command += [flag, str(args[key])]
    return command


def process(out_dir, batch, retrain="auto", search="evolve", search_args=None, trigger_options=None, log=print):
    """Handle one batch and return its decision record (also written to disk)."""
    if retrain not in RETRAIN_MODES or search not in SEARCH_MODES:
        raise ValueError("bad retrain or search mode")
    out_dir = Path(out_dir)
    state = state_mod.load(out_dir)
    batch_id = batch["batch_id"]
    folder = stage_mod.batch_dir(out_dir, batch_id)
    if state_mod.done(state, batch_id):
        decision = read_json(folder / "decision.json")
        log(f"[team_loop] batch {batch_id} 已经处理过，沿用当时的决定")
        return decision
    folder.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deployed = state["deployed"]
    base_raw = state_mod.read_base_config(state["base_config"])
    from baselines.model import Model
    spec = trigger.spec_for(Model.load(deployed["checkpoint"]), state["task_name"], state.get("spec_file"))

    # 1. should we retrain?
    report = trigger.check(deployed["checkpoint"], batch["validation_path"], deployed.get("reference"), spec,
                           trigger_options, cache_dir=out_dir / "cache")
    report["mode"] = retrain
    if retrain == "always" and not report["retrain"]:
        report["retrain"], report["reasons"] = True, ["retrain=always：不看指标，强制重训"] + report["reasons"]
    elif retrain == "never" and report["retrain"]:
        report["retrain"], report["reasons"] = False, ["retrain=never：只记录指标，不重训"] + report["reasons"]
    write_json(folder / "check.json", report)
    log(trigger.describe(report))

    decision = {"batch_id": batch_id, "train_path": batch["train_path"], "validation_path": batch["validation_path"],
                "retrain": report["retrain"], "trigger_reasons": report["reasons"],
                "metrics_before": report["metrics"], "previous_checkpoint": deployed["checkpoint"],
                "deploy": False, "action": "skip", "gate_decision": None, "search": None, "steps": {}}
    new_deployed = None
    if report["retrain"]:
        decision["action"] = "retrain"
        decision["search"] = search
        search_args = dict(search_args or {})
        if not search_args.get("oe_config"):
            search_args["oe_config"] = str(PROJECT_ROOT / "configs" / "openevolve.yaml")
        try:
            run_config = stage_mod.prepare(base_raw, state, batch, out_dir)
            decision["run_config"] = str(run_config)
            work = Path(json.loads(run_config.read_text(encoding="utf-8"))["work_dir"])
            if search == "evolve":
                log(f"[team_loop] OpenEvolve 搜索中（日志 {folder / 'evolve.log'}）…")
                step = _run(_evolve_command(run_config, search_args, deployed["policy"]), folder / "evolve.log",
                            search_args.get("timeout"))
                summary = work / "evolve_summary.json"
                if summary.is_file():
                    content = read_json(summary)
                    step.update({"iterations": content.get("iterations_recorded"),
                                 "best_score": (content.get("best_metrics") or {}).get("combined_score")})
                decision["steps"]["evolve"] = step
                if step["returncode"] != 0:
                    log(f"[team_loop] 搜索没有正常结束（{step['returncode']}），验收门只在官方续训和当前模型之间选")
            log("[team_loop] 验收门…")
            gate = _run([sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "run_gate.py"),
                         "--run-config", str(run_config)], folder / "gate.log", search_args.get("gate_timeout"))
            decision["steps"]["gate"] = gate
            gate_file = work / "gate" / "decision.json"
            if gate["returncode"] != 0 or not gate_file.is_file():
                raise RuntimeError(f"验收门失败，见 {folder / 'gate.log'}")
            result = read_json(gate_file)
            decision["gate_decision"] = result["decision"]
            decision["gate_reason"] = result["reason"]
            if result["decision"] != "keep_parent":
                target = out_dir / "deployed" / batch_id
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(work / "selected", target)
                new_deployed = {"checkpoint": str(target / "checkpoint.npz"), "policy": str(target / "policy.py"),
                                "since": batch_id}
                decision["deploy"] = True
        except Exception as error:  # unattended: a failed round keeps the deployed model
            decision["error"] = f"{type(error).__name__}: {error}"
            log(f"[team_loop] 本轮出错，保留当前模型：{decision['error']}")

    if decision["action"] == "retrain" and "error" not in decision:
        # re-baseline: the reference level is what the (possibly new) model achieves on this batch
        checkpoint = (new_deployed or deployed)["checkpoint"]
        reference = trigger.profile(checkpoint, batch["validation_path"], spec, cache_dir=out_dir / "cache")
        reference["measured_on"] = batch["validation_path"]
        if new_deployed:
            new_deployed["reference"] = reference
        else:
            deployed = dict(deployed, reference=reference)
        decision["metrics_after"] = reference["metrics"]
    state["deployed"] = new_deployed or deployed
    decision["checkpoint"] = state["deployed"]["checkpoint"]
    decision["policy"] = state["deployed"]["policy"]
    decision["seconds"] = round(time.time() - started, 2)

    write_json(folder / "decision.json", decision)
    with open(out_dir / "decisions.jsonl", "a", encoding="utf-8") as stream:
        stream.write(json.dumps(decision, ensure_ascii=False) + "\n")
    state["history"].append({"batch_id": batch_id, "retrain": decision["retrain"], "deploy": decision["deploy"],
                             "gate_decision": decision["gate_decision"], "checkpoint": decision["checkpoint"],
                             "seconds": decision["seconds"], "error": decision.get("error")})
    state_mod.save(out_dir, state)
    state_mod.write_current(out_dir, state)
    log(f"DECISION batch={batch_id} retrain={'yes' if decision['retrain'] else 'no'} "
        f"deploy={'yes' if decision['deploy'] else 'no'} checkpoint={decision['checkpoint']} "
        f"seconds={decision['seconds']}")
    return decision
