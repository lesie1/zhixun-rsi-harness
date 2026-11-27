"""Continual adaptation over time-ordered batches.

    stage k: prepare ──► search (OpenEvolve, stagnation-monitored) ──► gate ──► compare
                │                                                    │
                │   parent = selected checkpoint of stage k-1 ◄──────┘ (may be the unchanged parent)
                ▼
    advance to stage k+1 only if stage k ended for a reason in `advance_on` (default: the
    search stagnated) AND every batch of stage k+1 is declared, labelled and on disk.
    Otherwise stop ("stopped"), or wait without spending anything ("waiting").

All progress is in <out>/state.json and <out>/events.jsonl; running again resumes at the
first unfinished step. The reserved final batch is touched only by `final()`, once.
"""
import contextlib
import copy
import io
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from baselines.model import Model
from harness import PROJECT_ROOT
from harness.dataio import file_sha256, read_json, write_json
from harness.runcfg import PATH_KEYS
from harness.specs import resolve_spec
from team_split import derive_run_config
from team_split.__main__ import main as split_cli

from . import config as config_mod
from . import datasets, evaluate
from . import search as search_mod
from .ledger import Ledger, exposure_status

EXIT_CODES = {"finished": 0, "error": 1, "waiting": 3, "stopped": 4}
SPLIT_SUFFIX = "split"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def read_base_config(path):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in PATH_KEYS:
        if raw.get(key):
            value = Path(raw[key])
            raw[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    ace = raw.get("ace")
    if isinstance(ace, dict) and ace.get("storage_root"):
        value = Path(ace["storage_root"])
        ace["storage_root"] = str(value if value.is_absolute() else (path.parent / value).resolve())
    return raw


def copy_model(checkpoint, policy, target):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, target / "checkpoint.npz")
    shutil.copyfile(policy, target / "policy.py")
    Model.load(target / "checkpoint.npz")  # must be a loadable official checkpoint
    return {"checkpoint": str(target / "checkpoint.npz"), "policy": str(target / "policy.py"),
            "sha256": file_sha256(target / "checkpoint.npz")}


class Experiment:
    def __init__(self, config_path, out_dir, searcher=None, allow_llm=False, log=print, python=None):
        self.cfg = config_mod.load_config(config_path)
        self.out = Path(out_dir).resolve()
        self.log = log
        self.allow_llm = allow_llm
        self.python = python or sys.executable
        self.by_id = {b["id"]: b for b in self.cfg["batches"]}
        if searcher is None:
            searcher = (search_mod.EvolveSearch(python=self.python) if self.cfg["search"]["mode"] == "evolve"
                        else search_mod.ContinuationSearch())
        self.searcher = searcher
        self.ledger = Ledger(self.cfg.get("exposure_ledger") or self.out / "exposure_ledger.json")
        self.base_raw = read_base_config(self.cfg["base_config"])
        self.state = self._load_state()

    # ------------------------------------------------------------------ state
    def _save(self):
        write_json(self.out / "state.json", self.state)

    def event(self, kind, **fields):
        entry = {"at": now(), "event": kind, **fields}
        with open(self.out / "events.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    def _load_state(self):
        path = self.out / "state.json"
        if path.is_file():
            state = read_json(path)
            if state["fingerprint"] != config_mod.fingerprint(self.cfg):
                raise config_mod.ConfigError("实验的关键配置（任务、起点、训练方式、停滞参数等）和已开始的运行不一致；"
                                             "请换一个输出目录开始新实验")
            for bid, identity in state["used_batches"].items():
                if bid not in self.by_id or config_mod.batch_identity(self.by_id[bid]) != identity:
                    raise config_mod.ConfigError(f"已经用过的批次 {bid} 的声明被改动了（顺序、路径或用途），不能继续")
            return state
        self.out.mkdir(parents=True, exist_ok=True)
        initial = copy_model(self.cfg["initial_checkpoint"], self.cfg["initial_policy"], self.out / "models" / "initial")
        initial.update({"from": "initial", "lineage": {"trained_on": list(self.cfg["initial_train_batches"]),
                                                       "selected_on": []}})
        state = {"schema": "team_adapt.state.v1", "experiment_id": self.cfg["experiment_id"],
                 "task_name": self.cfg["task_name"], "fingerprint": config_mod.fingerprint(self.cfg),
                 "created": now(), "status": "running", "status_reason": None, "message": None,
                 "initial": initial, "current": dict(initial), "stages": [], "used_batches": {},
                 "total_iterations": 0, "final": None, "closed": False}
        self.state = state
        self._save()
        self.event("experiment_started", config=self.cfg["config_path"], plan=self.cfg["plan"],
                   initial_checkpoint_sha256=initial["sha256"], training_mode=self.cfg["training"]["mode"],
                   search_mode=self.cfg["search"]["mode"], advance_on=self.cfg["advance_on"])
        return state

    def _halt(self, kind, code, message):
        self.state.update({"status": kind, "status_reason": code, "message": message})
        self._save()
        self.event("halted", status=kind, reason=code, message=message)
        self.log(f"[team_adapt] {kind}: {message}")
        return EXIT_CODES[kind]

    # -------------------------------------------------------------------- run
    def run(self):
        if self.state.get("closed"):
            return self._halt("stopped", "closed", "最终验收已经做过，这个实验已关闭；新的适应请开新实验并预留新的最终批次")
        self.state.update({"status": "running", "status_reason": None, "message": None})
        self._save()
        try:
            while True:
                stage = self.state["stages"][-1] if self.state["stages"] else None
                if stage is None or stage["status"] == "done":
                    if stage is not None:
                        blocked = self._advance_blocked(stage)
                        if blocked:
                            return self._halt(*blocked)
                    index = 1 if stage is None else stage["index"] + 1
                    if index > len(self.cfg["plan"]):
                        return self._halt("finished", "plan_exhausted",
                                          "声明的批次已全部处理：没有更多可用于训练的新批次，停止（不再消耗预算）")
                    if index > self.cfg["budget"]["max_stages"]:
                        return self._halt("stopped", "max_stages", f"已达到最大阶段数 {self.cfg['budget']['max_stages']}")
                    plan = self.cfg["plan"][index - 1]
                    ready, reasons = config_mod.readiness(self.cfg, plan)
                    if not ready:
                        return self._halt("waiting", "data_not_ready",
                                          f"第 {index} 阶段需要的数据还不可用，等待（未消耗预算）：" + "；".join(reasons))
                    if stage is not None:
                        self.event("phase_switch", from_stage=stage["stage_id"], to_index=index,
                                   trigger=stage["search"]["end_reason"], new_train=plan["new_train"],
                                   train=plan["train"], dev=plan["dev"], training_mode=self.cfg["training"]["mode"],
                                   parent_sha256=self.state["current"]["sha256"],
                                   note="切换到新数据；停滞不等于已确认漂移")
                    stage = self._prepare(plan)
                if stage["status"] in ("prepared", "searching"):
                    self._search(stage)
                if stage["status"] == "searched":
                    self._gate(stage)
                if stage["status"] == "gated":
                    self._compare(stage)
        except Exception as error:
            self.state.update({"status": "error", "status_reason": type(error).__name__, "message": str(error)})
            self._save()
            self.event("error", error=f"{type(error).__name__}: {error}")
            raise

    def _advance_blocked(self, stage):
        reason = stage["search"]["end_reason"]
        if reason not in self.cfg["advance_on"]:
            messages = {
                "unhealthy": "搜索连续失败（API 或代码问题），这与数据无关；先排查，不切换数据",
                "process_error": "搜索进程异常结束；先排查，不切换数据",
                "budget_iterations": "本阶段预算用完但搜索没有停滞；按配置不切换新数据",
                "budget_seconds": "本阶段时间预算用完但搜索没有停滞；按配置不切换新数据",
                "budget_tokens": "本阶段 token 预算用完但搜索没有停滞；按配置不切换新数据",
                "stagnated": "搜索已停滞，但配置没有把 stagnated 列为切换条件",
                "no_search": "本阶段没有搜索（continuation 模式），配置没有把 no_search 列为切换条件",
            }
            return ("stopped", f"end_reason:{reason}", messages.get(reason, reason))
        limit = self.cfg["budget"].get("max_total_iterations")
        if limit and self.state["total_iterations"] >= limit:
            return ("stopped", "max_total_iterations", f"全实验迭代预算 {limit} 已用完")
        return None

    # ---------------------------------------------------------------- prepare
    def _prepare(self, plan):
        index = plan["index"]
        stage_id = f"{self.cfg['experiment_id']}-s{index:02d}"
        folder = self.out / "stages" / f"s{index:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        parent = copy.deepcopy(self.state["current"])
        data = datasets.materialize([(bid, self.by_id[bid]["path"]) for bid in plan["train"]], folder / "data" / "train.jsonl")
        dev_batch = self.by_id[plan["dev"]]
        dev_index, train_index = datasets.index(dev_batch["path"]), datasets.index(data["path"])
        for part in data["parts"]:
            config_mod.check_sha256(self.by_id[part["batch_id"]], part["sha256"])
        config_mod.check_sha256(dev_batch, dev_index["sha256"])
        overlap = datasets.overlap(train_index, dev_index)
        if overlap["ids"] or overlap["contents"]:
            raise ValueError(f"第 {index} 阶段训练数据与开发批次 {plan['dev']} 重叠：{overlap}")
        exposure = {}
        for part in data["parts"]:
            exposure[part["batch_id"]] = exposure_status(self.by_id[part["batch_id"]], self.ledger, part["sha256"],
                                                         self.cfg["experiment_id"])
        exposure[plan["dev"]] = exposure_status(dev_batch, self.ledger, dev_index["sha256"], self.cfg["experiment_id"])
        warnings = [f"开发批次 {plan['dev']} 此前已暴露（{'; '.join(exposure[plan['dev']]['declared'] + exposure[plan['dev']]['ledger'])}），"
                    "本阶段在它上面的结果只算开发证据"] if exposure[plan["dev"]]["exposed"] else []

        stage_config = copy.deepcopy(self.base_raw)
        for key in ("split_manifest", "playbook_path"):
            stage_config.pop(key, None)
        stage_config.update({"train_path": data["path"], "validation_path": dev_batch["path"],
                             "parent_checkpoint": parent["checkpoint"], "stage_id": stage_id,
                             "work_dir": str(folder / "work")})
        origin = (f"adapt|{self.cfg['experiment_id']}|{stage_id}|train={'+'.join(plan['train'])}|dev={plan['dev']}"
                  f"|mode={self.cfg['training']['mode']}|validation=dev_only")
        if isinstance(stage_config.get("ace"), dict):
            stage_config["ace"]["record_origin"] = origin
        stage_path = folder / "stage.json"
        run_path = folder / "run.json"
        if not run_path.is_file():
            stage_path.write_text(json.dumps(stage_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest = folder / "split.json"
            if not manifest.is_file():
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                    code = split_cli(["make", "--config", str(stage_path), "--out", str(manifest), "--suffix", SPLIT_SUFFIX])
                (folder / "split.log").write_text(captured.getvalue(), encoding="utf-8")
                if code != 0:
                    raise RuntimeError(f"划分失败，见 {folder / 'split.log'}")
            derive_run_config(stage_path, manifest, run_path, SPLIT_SUFFIX)
        run_config = read_json(run_path)
        for part in data["parts"]:
            self.ledger.record(part["sha256"], part["batch_id"], "train", self.cfg["experiment_id"], stage_id)
        self.ledger.record(dev_index["sha256"], plan["dev"], "dev", self.cfg["experiment_id"], stage_id)
        for bid in plan["train"] + [plan["dev"]]:
            self.state["used_batches"][bid] = config_mod.batch_identity(self.by_id[bid])
        stage = {"index": index, "stage_id": stage_id, "folder": str(folder), "status": "prepared", "plan": plan,
                 "training_mode": self.cfg["training"]["mode"],
                 "data": {"train": data, "dev": {"batch_id": plan["dev"], "path": dev_batch["path"],
                                                 "sha256": dev_index["sha256"], "rows": dev_index["rows"]},
                          "overlap_train_dev": overlap},
                 "exposure": exposure, "warnings": warnings, "parent": parent, "run_config": str(run_path),
                 "work_dir": run_config["work_dir"], "split_manifest": run_config["split_manifest"],
                 "ace_record_origin": origin if isinstance(stage_config.get("ace"), dict) else None,
                 "playbook_path": run_config.get("playbook_path"), "search": None, "gate": None,
                 "selected": None, "comparison": None, "prepared_at": now()}
        write_json(folder / "experience_provenance.json", {
            "experiment_id": self.cfg["experiment_id"], "stage_id": stage_id, "stage_index": index,
            "task_name": self.cfg["task_name"], "train_batches": plan["train"], "new_train_batches": plan["new_train"],
            "training_mode": self.cfg["training"]["mode"], "dev_batch": plan["dev"],
            "dev_exposed_before": exposure[plan["dev"]]["exposed"], "split_manifest": run_config["split_manifest"],
            "playbook_path": run_config.get("playbook_path"), "ace_record_origin": stage["ace_record_origin"],
            "validation_level": "development_only",
            "note": "本阶段 ACE 经验只来自挑选集反馈，只适用于上面的数据角色；没有经过最终验收，不能当作已验证的泛化经验"})
        self.state["stages"].append(stage)
        self._save()
        self.event("stage_prepared", stage_id=stage_id, new_train=plan["new_train"], train=plan["train"],
                   dev=plan["dev"], training_mode=self.cfg["training"]["mode"],
                   train_rows={p["batch_id"]: p["rows"] for p in data["parts"]}, train_sha256=data["sha256"],
                   dev_rows=dev_index["rows"], dev_sha256=dev_index["sha256"], parent_from=parent["from"],
                   parent_sha256=parent["sha256"], warnings=warnings, split_manifest=run_config["split_manifest"])
        for warning in warnings:
            self.log(f"[team_adapt] 注意：{warning}")
        self.log(f"[team_adapt] 第 {index} 阶段：训练 {'+'.join(plan['train'])}（{data['rows']} 行），开发 {plan['dev']}，"
                 f"起点 {parent['from']}")
        return stage

    # ----------------------------------------------------------------- search
    def _search(self, stage):
        if self.searcher.mode == "evolve" and not self.allow_llm:
            raise PermissionError("evolve 模式会调用大模型（可能产生费用）；确认预算后在命令行加 --allow-llm")
        stage["status"] = "searching"
        self._save()
        self.event("search_started", stage_id=stage["stage_id"], mode=self.searcher.mode,
                   budget=self.cfg["budget"], stagnation=self.cfg["stagnation"])
        settings = {"stagnation": self.cfg["stagnation"], "budget": self.cfg["budget"], "search": self.cfg["search"]}
        task = {"run_config": stage["run_config"], "work_dir": stage["work_dir"], "stage_id": stage["stage_id"],
                "initial_policy": stage["parent"]["policy"], "log_dir": stage["folder"]}
        previous_iterations = (stage.get("search") or {}).get("iterations", 0)
        summary = self.searcher.run(task, settings, previous=stage.get("search"), log=self.log)
        stage["search"] = summary
        stage["status"] = "searched"
        self.state["total_iterations"] += max(0, summary.get("iterations", 0) - previous_iterations)
        self._save()
        self.event("search_ended", stage_id=stage["stage_id"], end_reason=summary["end_reason"],
                   iterations=summary.get("iterations"), seconds=summary.get("seconds"), tokens=summary.get("tokens"),
                   description=summary.get("description"), counts=(summary.get("analysis") or {}).get("counts"))
        if summary["end_reason"] == "stagnated":
            self.event("stagnation_detected", stage_id=stage["stage_id"], analysis=summary["analysis"],
                       note="挑选集上搜索停滞，考虑新数据适应；这不等于已确认数据漂移")
        self.log(f"[team_adapt] 搜索结束：{summary['end_reason']}" +
                 (f"；{summary['description']}" if summary.get("description") else ""))

    # ------------------------------------------------------------------- gate
    def _gate(self, stage):
        work, folder = Path(stage["work_dir"]), Path(stage["folder"])
        command = [self.python, "-B", str(PROJECT_ROOT / "scripts" / "run_gate.py"), "--run-config", stage["run_config"]]
        best = work / "openevolve_output" / "best" / "best_program.py"
        if not best.is_file():
            checkpoint = search_mod.latest_openevolve_checkpoint(work)
            if checkpoint is not None and (checkpoint / "best_program.py").is_file():
                command += ["--policy", str(checkpoint / "best_program.py")]
        with open(folder / "gate.log", "a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(command)}\n")
            log.flush()
            code = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        decision_path = work / "gate" / "decision.json"
        if code != 0 or not decision_path.is_file():
            raise RuntimeError(f"验收门失败，见 {folder / 'gate.log'}")
        decision = read_json(decision_path)
        parent = stage["parent"]
        target = self.out / "models" / f"s{stage['index']:02d}"
        if decision["decision"] == "keep_parent":
            # the gate kept the starting checkpoint: there may be no selected/ directory at all
            selected = copy_model(parent["checkpoint"], parent["policy"], target)
            source = "parent"
            if selected["sha256"] != parent["sha256"]:
                raise RuntimeError("保留起点时复制出的检查点与起点不一致")
            trained_on = list(parent["lineage"]["trained_on"])
        else:
            chosen = work / "selected"
            selected = copy_model(chosen / "checkpoint.npz", chosen / "policy.py", target)
            source = "candidate" if decision["decision"] == "adopt_candidate" else "official_continuation"
            if selected["sha256"] != file_sha256(decision["selected_checkpoint"]):
                raise RuntimeError("复制出的检查点与验收决定里的 selected_checkpoint 不一致")
            trained_on = sorted(set(parent["lineage"]["trained_on"]) | set(stage["plan"]["train"]))
        selected.update({"from": f"{stage['stage_id']}:{source}", "lineage": {
            "trained_on": trained_on,
            "selected_on": sorted(set(parent["lineage"]["selected_on"]) | {stage["plan"]["dev"]})}})
        stage["gate"] = {"decision": decision["decision"], "reason": decision["reason"],
                         "threshold": decision.get("threshold"), "independent_gate": decision.get("independent_gate"),
                         "comparisons": {k: {"probability_better": v.get("probability_better"),
                                             "guard_ok": v.get("guard_ok"), "passes": v.get("passes")}
                                         for k, v in (decision.get("comparisons") or {}).items()},
                         "selected_checkpoint": decision.get("selected_checkpoint"),
                         "candidate_dir": decision.get("candidate_dir"), "decision_path": str(decision_path)}
        stage["selected"] = dict(selected, source=source)
        stage["status"] = "gated"
        self.state["current"] = dict(selected)
        self._save()
        write_json(self.out / "current.json", {"checkpoint": selected["checkpoint"], "policy": selected["policy"],
                                               "sha256": selected["sha256"], "from": selected["from"]})
        self.event("gate_decided", stage_id=stage["stage_id"], decision=decision["decision"], reason=decision["reason"],
                   selected_from=selected["from"], selected_sha256=selected["sha256"],
                   independent_gate=decision.get("independent_gate"))
        self.log(f"[team_adapt] 验收门：{decision['decision']}（{decision['reason']}）→ 下一阶段起点 {selected['from']}")

    # ---------------------------------------------------------------- compare
    def _spec(self):
        model = Model.load(self.state["initial"]["checkpoint"])
        return resolve_spec(model.meta["task"], self.cfg["task_name"], self.base_raw.get("spec_file"))

    def _compare(self, stage):
        work = Path(stage["work_dir"])
        reference = work / "reference" / "checkpoint.npz"
        candidate = stage["gate"].get("candidate_dir")
        arms = {"pre_update": stage["parent"]["checkpoint"],
                "official_continuation": str(reference) if reference.is_file() else None,
                "evolved": str(Path(candidate) / "checkpoint.npz") if candidate and (Path(candidate) / "checkpoint.npz").is_file() else None,
                "selected": stage["selected"]["checkpoint"]}
        sets = {"train": {"path": stage["data"]["train"]["path"], "role": "train", "batch": "+".join(stage["plan"]["train"])},
                "dev_select": {"path": str(work / "splits" / "select.jsonl"), "role": "dev_select", "batch": stage["plan"]["dev"]},
                "dev_gate": {"path": str(work / "splits" / "gate.jsonl"), "role": "dev_gate", "batch": stage["plan"]["dev"]}}
        for earlier in self.state["stages"]:
            if earlier["index"] < stage["index"]:
                sets[f"old:s{earlier['index']:02d}"] = {"path": str(Path(earlier["work_dir"]) / "splits" / "gate.jsonl"),
                                                        "role": "old", "batch": earlier["plan"]["dev"]}
        table = evaluate.compare(arms, sets, self._spec(), cache_dir=self.out / "cache",
                                 forgetting_tolerance=self.cfg["forgetting_tolerance"], focus=self.cfg["focus_classes"])
        table["missing_arms"] = {k: ("没有有效的进化候选（continuation 模式或搜索没有产出）" if k == "evolved" else "缺文件")
                                 for k, v in arms.items() if not v}
        table["selected_source"] = stage["selected"]["source"]
        table["verdict"] = evaluate.verdict(table)
        path = Path(stage["folder"]) / "comparison.json"
        write_json(path, table)
        stage["comparison"] = str(path)
        stage["status"] = "done"
        stage["finished_at"] = now()
        self._save()
        regressed = [name for name, entry in table["sets"].items()
                     if entry["arms"].get("selected", {}).get("regressed")]
        self.event("stage_compared", stage_id=stage["stage_id"], comparison=str(path), verdict=table["verdict"],
                   old_scenarios_regressed=regressed)

    # ------------------------------------------------------------------ final
    def final(self, allow_reuse=False):
        final = self.cfg.get("final")
        if not final:
            raise config_mod.ConfigError("配置没有声明 final 批次")
        batch = self.by_id[final["batch"]]
        if not batch["labels_available"] or not Path(batch["path"]).is_file():
            return self._halt("waiting", "final_not_ready", f"最终批次 {batch['id']} 还不可用")
        done = [s for s in self.state["stages"] if s["status"] == "done"]
        if not done:
            raise RuntimeError("还没有完成任何阶段，不能做最终验收")
        if self.state.get("final") and not allow_reuse:
            raise RuntimeError("最终验收已经做过一次；最终批次不能反复使用（确需复查请加 --allow-reuse，结果会标为非独立）")
        sha = file_sha256(batch["path"])
        config_mod.check_sha256(batch, sha)
        previous_final =[e for e in self.ledger.final_used(sha) if e["experiment_id"] != self.cfg["experiment_id"]]
        exposure = exposure_status(batch, self.ledger, sha, self.cfg["experiment_id"])
        if exposure["exposed"] and not final.get("allow_exposed") and not allow_reuse:
            raise RuntimeError(f"最终批次 {batch['id']} 此前已暴露（{exposure}），不能当独立验收集；"
                               "如只作参考，在配置 final.allow_exposed=true，结果会标为非独立")
        final_index = datasets.index(batch["path"])
        for stage in self.state["stages"]:
            for part in stage["data"]["train"]["parts"] + [stage["data"]["dev"]]:
                found = datasets.overlap(final_index, datasets.index(part["path"]))
                if found["ids"] or found["contents"]:
                    raise RuntimeError(f"最终批次与批次 {part['batch_id']} 重叠：{found}")
        last = done[-1]
        last_table = read_json(last["comparison"])
        arms = {"initial": self.state["initial"]["checkpoint"]}
        for stage in done:
            arms[f"selected_s{stage['index']:02d}"] = stage["selected"]["checkpoint"]
        work = Path(last["work_dir"])
        arms["last_pre_update"] = last["parent"]["checkpoint"]
        if (work / "reference" / "checkpoint.npz").is_file():
            arms["last_official_continuation"] = str(work / "reference" / "checkpoint.npz")
        if last["gate"].get("candidate_dir"):
            arms["last_evolved"] = str(Path(last["gate"]["candidate_dir"]) / "checkpoint.npz")
        sets = {"final": {"path": batch["path"], "role": "final", "batch": batch["id"]}}
        compare_arms = dict(arms, pre_update=arms["last_pre_update"])
        table = evaluate.compare(compare_arms, sets, self._spec(), cache_dir=self.out / "cache", focus=self.cfg["focus_classes"])
        table["sets"]["final"]["arms"].pop("pre_update", None)
        independent = not exposure["exposed"] and not self.state.get("final") and not previous_final
        result = {"batch": batch["id"], "sha256": sha, "rows": final_index["rows"], "independent": independent,
                  "exposure": exposure, "reused": bool(self.state.get("final") or previous_final),
                  "table": table, "at": now(), "last_stage": last["stage_id"],
                  "last_stage_selected_source": last_table.get("selected_source"),
                  "note": "比较只在同一个最终批次内进行；不同数据集上的分数不能直接相比"}
        self.ledger.record(sha, batch["id"], "final", self.cfg["experiment_id"], note="final acceptance")
        path = self.out / "final" / ("report.json" if not self.state.get("final") else f"report-reuse-{int(time.time())}.json")
        write_json(path, result)
        self.state["final"] = {"path": str(path), "independent": independent, "at": result["at"]}
        self.state["closed"] = True
        self._save()
        write_json(self.out / "final" / "experience_validation.json", {
            "experiment_id": self.cfg["experiment_id"], "final_batch": batch["id"], "independent": independent,
            "stages": [{"stage_id": s["stage_id"], "playbook_path": s.get("playbook_path"),
                        "ace_record_origin": s.get("ace_record_origin")} for s in done],
            "note": "这里只记录最终验收结果与各阶段经验来源的对应关系；ACE 条目本身保持“待验证”，不自动升级为已验证技能"})
        self.event("final_acceptance", batch=batch["id"], independent=independent, report=str(path))
        return result
