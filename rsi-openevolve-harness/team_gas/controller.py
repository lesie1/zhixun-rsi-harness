"""A-type gas classification: automatic adaptation, one controller per round.

    round r:  diagnose (team_loop check + per-class confusion)            [alarm batch]
                │ stable / insufficient evidence → done, nothing trained
                ▼
              repair on the current training data (OpenEvolve, alarm-aware stagnation,
              early stop when the alarm class is fixed) → harness gate      [alarm batch]
                │ repaired → deploy (gate rule), done; no data switch
                │ not stagnated / no trainable new samples → done (keep, no switch)
                ▼
              data mixing: candidates over (new-data share ≤ 20 %, degraded/confusion split),
              same checkpoint, same update steps → choose on dev, freeze  [selection batch]
                ▼
              final acceptance once: pre-update vs 0 % control vs chosen   [final batch]
                accepted → deploy;  rejected → keep the pre-update model (rollback)

State in <out>/state.json, events in <out>/events.jsonl, a readable record per round in
<out>/rounds/rNN/record.md. Running again resumes at the first unfinished step.
"""
import contextlib
import copy
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np

from baselines.model import Model
from harness import PROJECT_ROOT
from harness.dataio import file_sha256, read_json, write_json
from harness.gate import probability_better
from harness.specs import resolve_spec
from harness.training import run_training
from team_adapt import datasets as adapt_datasets
from team_adapt import search as adapt_search
from team_adapt.controller import copy_model, read_base_config
from team_adapt.ledger import Ledger, exposure_status
from team_split import derive_run_config
from team_split.__main__ import main as split_cli

from . import config as config_mod
from . import criteria, diagnose, mixture, playbook, repair
from . import report as report_mod

EXIT_CODES = {"finished": 0, "error": 1, "waiting": 3, "stopped": 4}


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def public(profile):
    return {k: v for k, v in profile.items() if not k.startswith("_")}


class GasFlow:
    def __init__(self, config_path, out_dir, allow_llm=False, searcher=None, log=print):
        self.cfg = config_mod.load(config_path)
        self.out = Path(out_dir).resolve()
        self.allow_llm, self.log = allow_llm, log
        self.searcher = searcher
        self.base_raw = read_base_config(self.cfg["base_config"])
        self.ledger = Ledger(self.cfg.get("exposure_ledger") or self.out / "exposure_ledger.json")
        self.base_entries = playbook.load_base(self.cfg["playbook"]["base_experiences"])
        model = Model.load(self.cfg["initial_checkpoint"])
        if model.meta["task"] != "classification":
            raise config_mod.ConfigError("A 流程只适用于分类任务")
        self.classes = model.meta["classes"]
        self.spec = resolve_spec("classification", self.cfg["task_name"], self.base_raw.get("spec_file"))
        self.cache = self.out / "cache"
        self.state = self._load_state()

    # ------------------------------------------------------------------ state
    def display(self, cls):
        return config_mod.name(self.cfg, cls)

    def _save(self):
        write_json(self.out / "state.json", self.state)

    def event(self, kind, **fields):
        with open(self.out / "events.jsonl", "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": now(), "event": kind, **fields}, ensure_ascii=False) + "\n")

    def _load_state(self):
        if (self.out / "state.json").is_file():
            return read_json(self.out / "state.json")
        self.out.mkdir(parents=True, exist_ok=True)
        initial = copy_model(self.cfg["initial_checkpoint"], self.cfg["initial_policy"], self.out / "models" / "initial")
        history = adapt_datasets.materialize([(b, config_mod.batch(self.cfg, b)["path"]) for b in self.cfg["initial_train_batches"]],
                                             self.out / "data" / "history-initial.jsonl")
        initial.update({"from": "initial", "train_file": history["path"], "train_parts": history["parts"]})
        self.state = {"schema": "team_gas.state.v1", "experiment_id": self.cfg["experiment_id"], "created": now(),
                      "status": "running", "message": None, "deployed": initial, "rounds": []}
        self._save()
        self.event("flow_started", config=self.cfg["config_path"], rounds=self.cfg["rounds"])
        return self.state

    def _halt(self, kind, message, rnd=None):
        self.state.update({"status": kind, "message": message})
        self._save()
        self.event("halted", status=kind, message=message, round=rnd["index"] if rnd else None)
        self.log(f"[team_gas] {kind}: {message}")
        return EXIT_CODES[kind]

    def _deploy(self, model, rnd, why):
        self.state["deployed"] = model
        write_json(self.out / "current.json", {k: model[k] for k in ("checkpoint", "policy", "sha256", "from")})
        self.event("deployed", round=rnd["index"], source=model["from"], sha256=model["sha256"], reason=why)

    # -------------------------------------------------------------------- run
    def run(self):
        self.state.update({"status": "running", "message": None})
        self._save()
        try:
            for index, spec in enumerate(self.cfg["rounds"], 1):
                rnd = next((r for r in self.state["rounds"] if r["index"] == index), None)
                if rnd is None:
                    rnd = {"index": index, **spec, "status": "pending", "folder": str(self.out / "rounds" / f"r{index:02d}")}
                    self.state["rounds"].append(rnd)
                    self._save()
                while rnd["status"] != "done":
                    code = self._step(rnd)
                    if code is not None:
                        return code
                report_mod.write_round(self, rnd)
            return self._halt("finished", "所有声明的轮次都已处理")
        except Exception as error:
            self.state.update({"status": "error", "message": f"{type(error).__name__}: {error}"})
            self._save()
            self.event("error", error=f"{type(error).__name__}: {error}")
            raise

    def _step(self, rnd):
        handler = {"pending": self._diagnose, "diagnosed": self._prepare_repair, "repair_prepared": self._search,
                   "repair_searching": self._search, "repair_searched": self._repair_gate,
                   "repair_gated": self._after_repair, "mixture_pending": self._mixture,
                   "mixture_frozen": self._final}[rnd["status"]]
        return handler(rnd)

    def _finish(self, rnd, outcome, why):
        rnd.update({"status": "done", "outcome": outcome, "outcome_text": why, "finished_at": now()})
        self._save()
        self.event("round_done", round=rnd["index"], outcome=outcome, why=why)
        self.log(f"[team_gas] 第 {rnd['index']} 轮结束：{outcome}（{why}）")

    # --------------------------------------------------------------- diagnose
    def _diagnose(self, rnd):
        ok, reasons = config_mod.ready(self.cfg, rnd["alarm"])
        if not ok:
            return self._halt("waiting", f"第 {rnd['index']} 轮的报警批次还不可用：" + "；".join(reasons), rnd)
        alarm = config_mod.batch(self.cfg, rnd["alarm"])
        sha = config_mod.check_sha256(alarm)
        reference = rnd.get("reference") or self.state["deployed"]["train_file"]
        if isinstance(reference, list):
            reference = adapt_datasets.materialize([(b, config_mod.batch(self.cfg, b)["path"]) for b in reference],
                                                   Path(rnd["folder"]) / "reference.jsonl")["path"]
        result = diagnose.diagnose(self.state["deployed"]["checkpoint"], reference, alarm["path"], self.cfg["diagnosis"],
                                   self.spec, self.display, cache_dir=self.cache)
        self.ledger.record(sha, alarm["id"], "alarm", self.cfg["experiment_id"], f"r{rnd['index']:02d}")
        result.update({"reference_path": str(reference), "alarm_batch": alarm["id"], "alarm_sha256": sha,
                       "deployed_from": self.state["deployed"]["from"]})
        write_json(Path(rnd["folder"]) / "diagnosis.json", result)
        rnd["diagnosis"] = {k: result[k] for k in ("status", "status_text", "pair", "alarms", "insufficient", "macro_f1",
                                                   "overall_check")}
        self.event("diagnosed", round=rnd["index"], status=result["status"], pair=result["pair"],
                   alarms=[a["text"] for a in result["alarms"]], insufficient=[i["text"] for i in result["insufficient"]])
        for line in [a["text"] for a in result["alarms"]] + [i["text"] for i in result["insufficient"]]:
            self.log(f"[team_gas] 诊断：{line}")
        if result["status"] in ("stable", "insufficient_evidence"):
            self._finish(rnd, "no_action", result["status_text"] + "；不训练")
        else:
            rnd["status"] = "diagnosed"
            self._save()

    # ----------------------------------------------------------------- repair
    def _stage_config(self, rnd, stage_name, train_file, validation_path):
        folder = Path(rnd["folder"]) / stage_name
        folder.mkdir(parents=True, exist_ok=True)
        stage_id = f"{self.cfg['experiment_id']}-r{rnd['index']:02d}-{stage_name}"
        run_path = folder / "run.json"
        if run_path.is_file():
            return folder, run_path
        config = copy.deepcopy(self.base_raw)
        for key in ("split_manifest", "playbook_path"):
            config.pop(key, None)
        config.update({"train_path": str(train_file), "validation_path": str(validation_path),
                       "parent_checkpoint": self.state["deployed"]["checkpoint"], "stage_id": stage_id,
                       "work_dir": str(folder / "work")})
        if isinstance(config.get("ace"), dict):
            config["ace"]["record_origin"] = (f"gas|{self.cfg['experiment_id']}|{stage_id}|train=deployed_history"
                                              f"|alarm={rnd['alarm']}|validation=dev_only")
        stage_path = folder / "stage.json"
        stage_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest = folder / "split.json"
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
            code = split_cli(["make", "--config", str(stage_path), "--out", str(manifest), "--suffix", "split"])
        (folder / "split.log").write_text(captured.getvalue(), encoding="utf-8")
        if code != 0:
            raise RuntimeError(f"划分失败，见 {folder / 'split.log'}")
        derive_run_config(stage_path, manifest, run_path, "split")
        return folder, run_path

    def _prepare_repair(self, rnd):
        alarm = config_mod.batch(self.cfg, rnd["alarm"])
        folder, run_path = self._stage_config(rnd, "repair", self.state["deployed"]["train_file"], alarm["path"])
        run_config = read_json(run_path)
        pair = rnd["diagnosis"]["pair"]
        context = {"task_name": self.cfg["task_name"], "task_type": "classification", "pair": pair,
                   "trainable_new_labels": "train" in alarm["uses"] and alarm["labels_available"]}
        chosen, skipped = [], []
        for entry in self.base_entries:
            ok, why = playbook.eligible(entry, context)
            (chosen if ok else skipped).append(entry if ok else {"id": entry["id"], "reasons": why})
        from harness.runcfg import load_run_config
        seeded = playbook.seed(load_run_config(run_path), chosen)
        prompt = playbook.render_prompt(run_path, self.cfg["repair"]["search"].get("oe_config"),
                                        self.state["deployed"]["policy"])
        check = {e["id"]: playbook.occurrences(prompt, e["id"]) for e in self.base_entries}
        (folder / "prompt_preview.txt").write_text(prompt["user"], encoding="utf-8")
        version_line = next((l for l in prompt["user"].splitlines() if l.startswith("PLAYBOOK_VERSION")), None)
        rnd["repair"] = {"folder": str(folder), "run_config": str(run_path), "work_dir": run_config["work_dir"],
                         "stage_id": run_config["stage_id"], "playbook_path": run_config.get("playbook_path"),
                         "playbook": {"seeded": seeded.get("seeded"), "skipped": skipped, "mode": seeded.get("mode"),
                                      "version": seeded.get("version"), "prompt_version_line": version_line,
                                      "occurrences_in_prompt": check},
                         "parent": copy.deepcopy(self.state["deployed"])}
        for entry in chosen:
            if check.get(entry["id"]) != 1:
                raise RuntimeError(f"基础经验 {entry['id']} 在真实提示里出现 {check.get(entry['id'])} 次（应为 1 次）")
        rnd["status"] = "repair_prepared"
        self._save()
        self.event("repair_prepared", round=rnd["index"], stage_id=run_config["stage_id"],
                   playbook_seeded=seeded.get("seeded"), prompt_occurrences=check, playbook_path=run_config.get("playbook_path"))

    def _search(self, rnd):
        stage = rnd["repair"]
        search_cfg = self.cfg["repair"]["search"]
        pair = rnd["diagnosis"]["pair"]
        analyzer = (repair.AlarmAnalyzer(pair["degraded"], pair["confusion"], stage["parent"]["checkpoint"],
                                         self.cfg["repair"]["stagnation"], self.cfg["repair"]["success"], self.cache)
                    if pair else None)
        if self.searcher is not None:
            searcher = self.searcher(analyzer)
        elif search_cfg["mode"] == "evolve":
            searcher = adapt_search.EvolveSearch(analyzer=analyzer)
        else:
            searcher = adapt_search.ContinuationSearch()
        if searcher.mode == "evolve" and not self.allow_llm:
            raise PermissionError("repair.search.mode=evolve 会调用大模型（可能产生费用）；确认预算后加 --allow-llm")
        rnd["status"] = "repair_searching"
        self._save()
        settings = {"stagnation": self.cfg["repair"]["stagnation"], "budget": self.cfg["repair"]["budget"], "search": search_cfg}
        task = {"run_config": stage["run_config"], "work_dir": stage["work_dir"], "stage_id": stage["stage_id"],
                "initial_policy": stage["parent"]["policy"], "log_dir": stage["folder"]}
        summary = searcher.run(task, settings, previous=stage.get("search"), log=self.log)
        stage["search"] = summary
        stage["prompts_seen"] = self._prompt_evidence(stage["work_dir"])
        rnd["status"] = "repair_searched"
        self._save()
        self.event("repair_search_ended", round=rnd["index"], end_reason=summary["end_reason"],
                   iterations=summary.get("iterations"), description=summary.get("description"),
                   counts=(summary.get("analysis") or {}).get("counts"))
        self.log(f"[team_gas] 修复搜索结束：{summary['end_reason']}" + (f"；{summary['description']}" if summary.get("description") else ""))

    def _prompt_evidence(self, work_dir):
        """How often each base entry appeared in the prompts actually sent during the search."""
        prompts = [r["prompt"] for r in adapt_search.read_records(work_dir) if isinstance(r.get("prompt"), dict)]
        out = {"prompts": len(prompts)}
        for entry in self.base_entries:
            counts = [playbook.occurrences(p, entry["id"]) for p in prompts]
            out[entry["id"]] = {"prompts_containing": sum(1 for c in counts if c), "max_per_prompt": max(counts, default=0)}
        return out

    def _repair_gate(self, rnd):
        stage = rnd["repair"]
        preferred = ((stage["search"] or {}).get("analysis") or {}).get("repaired_candidate")
        result = repair.gate(stage["run_config"], preferred, self.cache)
        parent = stage["parent"]
        target = Path(stage["folder"]) / "selected"
        if result["decision"] == "keep_parent":
            selected = dict(parent)
        else:
            source = Path(result["selected_dir"])
            policy = result.get("candidate_policy") if result["decision"] == "adopt_candidate" else None
            policy = policy or (str(PROJECT_ROOT / "harness" / "policy_initial.py") if result["decision"] == "adopt_reference"
                                else parent["policy"])
            selected = copy_model(source / "checkpoint.npz", policy, target)
            selected.update({"from": f"r{rnd['index']:02d}-repair:{result['decision']}", "train_file": parent["train_file"],
                             "train_parts": parent.get("train_parts")})
        gate_split = Path(stage["work_dir"]) / "splits" / "gate.jsonl"
        pair = rnd["diagnosis"]["pair"]
        check = None
        if pair and result["decision"] != "keep_parent":
            base = criteria.per_class_from_profile(diagnose.profile(parent["checkpoint"], gate_split, self.cache))
            cand = criteria.per_class_from_profile(diagnose.profile(selected["checkpoint"], gate_split, self.cache))
            s = self.cfg["repair"]["success"]
            check = criteria.targeted(base, cand, pair["degraded"], pair["confusion"], s["target_recall_gain"],
                                      s["confusion_guard"], s["other_guard"])
        stage["gate"] = {"decision": result["decision"], "reason": result["reason"], "candidate_source": result.get("candidate_source"),
                         "description": result["description"], "targeted_check_on_gate_split": check}
        stage["repaired"] = bool(check and check["ok"])
        if result["decision"] != "keep_parent":
            self._deploy(selected, rnd, "修复阶段验收门采纳（原验收规则）")
        rnd["status"] = "repair_gated"
        self._save()
        self.event("repair_gated", round=rnd["index"], decision=result["decision"], repaired=stage["repaired"],
                   check=check)

    def _after_repair(self, rnd):
        stage, pair = rnd["repair"], rnd["diagnosis"]["pair"]
        reason = stage["search"]["end_reason"]
        if stage["repaired"]:
            return self._finish(rnd, "repaired_on_old_data", "用现有训练集就修复了报警类别，且混淆类和其他类别通过护栏；不换数据")
        if pair is None:
            return self._finish(rnd, "no_targeted_pair", "没有定位到“退化类→混淆类”，不做针对性补样")
        if reason in ("unhealthy", "process_error"):
            return self._halt("stopped", f"修复搜索 {reason}（API/代码问题，与数据无关），先排查，不换数据", rnd)
        if reason not in self.cfg["repair"]["advance_on"]:
            return self._finish(rnd, f"no_switch:{reason}", f"修复搜索以 {reason} 结束，不是配置的切换条件，不换数据")
        alarm = config_mod.batch(self.cfg, rnd["alarm"])
        counts = {"degraded": 0, "confusion": 0}
        if "train" in alarm["uses"] and alarm["labels_available"]:
            for _, row in adapt_datasets.read_rows(alarm["path"]):
                if row["answer"] == pair["degraded"]:
                    counts["degraded"] += 1
                elif row["answer"] == pair["confusion"]:
                    counts["confusion"] += 1
        need = self.cfg["mixture"]["min_new_samples"]
        if "train" not in alarm["uses"] or counts["degraded"] < need["degraded"] or counts["confusion"] < need["confusion"]:
            return self._finish(rnd, "no_trainable_new_data",
                                f"没有足够的、允许训练的新样本（{counts}，至少需要 {need}）；停止，不消耗预算")
        searched = (f"旧数据上的策略搜索已停滞（{stage['search'].get('description')}）" if reason == "stagnated" else
                    f"修复阶段以 {reason} 结束（continuation 模式没有策略搜索，按配置 advance_on={self.cfg['repair']['advance_on']} 视为可切换），"
                    "且官方续训没有修复报警类别")
        rnd["switch"] = {"why": f"{searched}；报警批次 {alarm['id']} "
                                f"有允许训练的真实标签新样本：{self.display(pair['degraded'])} {counts['degraded']} 条、"
                                f"{self.display(pair['confusion'])} {counts['confusion']} 条",
                         "new_samples": counts}
        rnd["status"] = "mixture_pending"
        self._save()
        self.event("data_switch", round=rnd["index"], why=rnd["switch"]["why"], note="停滞不等于已确认漂移")

    # ---------------------------------------------------------------- mixture
    def _train_candidate(self, folder, train_path, selection_path, stage_id):
        out = folder / "model"
        if out.exists():
            shutil.rmtree(out)
        limits = self.base_raw.get("limits", {})
        outcome = run_training(self.cfg["mixture"]["policy"], self.state["deployed"]["checkpoint"], train_path,
                               selection_path, out, stage_id, max_epochs=limits.get("max_epochs", 50),
                               timeout_s=limits.get("train_timeout_s", 600), cpu_threads=limits.get("cpu_threads", 1),
                               cache_dir=self.cache)
        if outcome["timed_out"] or outcome["returncode"] != 0:
            return None, (outcome["stderr"] or "")[-1000:]
        return out, None

    def _mixture(self, rnd):
        ok, reasons = config_mod.ready(self.cfg, rnd["selection"])
        if not ok:
            return self._halt("waiting", f"第 {rnd['index']} 轮的开发批次还不可用：" + "；".join(reasons), rnd)
        m = self.cfg["mixture"]
        pair = rnd["diagnosis"]["pair"]
        D, C = pair["degraded"], pair["confusion"]
        folder = Path(rnd["folder"]) / "mixture"
        folder.mkdir(parents=True, exist_ok=True)
        selection = config_mod.batch(self.cfg, rnd["selection"])
        sel_sha = config_mod.check_sha256(selection)
        alarm = config_mod.batch(self.cfg, rnd["alarm"])
        deployed = copy.deepcopy(self.state["deployed"])
        pools = mixture.make_pools(deployed["train_file"], alarm["path"], D, C, m["seed"])
        sel_index = adapt_datasets.index(selection["path"])
        for name, path in (("历史训练数据", deployed["train_file"]), ("新样本批次", alarm["path"])):
            found = adapt_datasets.overlap(adapt_datasets.index(path), sel_index)
            if found["ids"] or found["contents"]:
                raise ValueError(f"开发批次 {selection['id']} 与{name}重叠：{found}")
        write_json(folder / "pools.json", {"history_order": [r["id"] for _, r in pools["history"]],
                                           "new": {k: [r["id"] for _, r in v] for k, v in pools["new"].items()},
                                           "degraded": D, "confusion": C, "seed": m["seed"],
                                           "history_sha256": pools["history_sha256"], "new_sha256": pools["new_sha256"]})
        self.ledger.record(pools["new_sha256"], alarm["id"], "train", self.cfg["experiment_id"], f"r{rnd['index']:02d}-mixture")
        self.ledger.record(sel_sha, selection["id"], "dev", self.cfg["experiment_id"], f"r{rnd['index']:02d}-mixture")
        pre = diagnose.profile(deployed["checkpoint"], selection["path"], self.cache)
        candidates, control, best, stale, stopped_early = [], None, None, 0, None
        stage_id = f"{self.cfg['experiment_id']}-r{rnd['index']:02d}-mixture"
        for level, fraction in enumerate(m["fractions"]):
            shares = [0.0] if fraction == 0 else m["degraded_shares"]
            improved = False
            for share in shares:
                cid = f"f{fraction:.2f}" + ("" if fraction == 0 else f"-d{share:.2f}")
                planned = mixture.plan(pools, fraction, share, m["cap"])
                if planned is None:
                    candidates.append({"id": cid, "fraction": fraction, "degraded_share": share, "status": "infeasible"})
                    continue
                cfolder = folder / cid
                train_path = cfolder / "train.jsonl"
                train_sha = mixture.write(planned, train_path)
                out, error = self._train_candidate(cfolder, train_path, selection["path"], stage_id)
                entry = {"id": cid, "fraction": fraction, "degraded_share": share, "train_file": str(train_path),
                         "train_sha256": train_sha, "accounting": {k: v for k, v in planned["accounting"].items()
                                                                   if k != "new_sample_ids"}}
                write_json(cfolder / "accounting.json", planned["accounting"])
                if out is None:
                    entry.update(status="train_failed", error=error)
                    candidates.append(entry)
                    continue
                entry["realised"] = mixture.verify_presentations(read_json(out / "runner_report.json"), read_json(out / "training.json"),
                                                                planned["accounting"], train_sha)
                prof = diagnose.profile(out / "checkpoint.npz", selection["path"], self.cache)
                entry.update(status="ok", checkpoint=str(out / "checkpoint.npz"), checkpoint_sha256=file_sha256(out / "checkpoint.npz"),
                             dev={"macro_f1": prof["macro_f1"], "accuracy": prof["accuracy"],
                                  "per_class": criteria.per_class_from_profile(prof)})
                if fraction == 0:
                    control = entry
                    entry["eligible"] = False
                    entry["role"] = "同预算旧数据对照（0% 新数据）"
                else:
                    check = criteria.targeted(control["dev"]["per_class"] if control else criteria.per_class_from_profile(pre),
                                              entry["dev"]["per_class"], D, C, m["min_gain"], m["confusion_guard"], m["other_guard"])
                    entry["check_vs_control"] = check
                    entry["eligible"] = check["ok"]
                    if check["ok"] and (best is None or entry["dev"]["macro_f1"] > best["dev"]["macro_f1"] + m["tie_tolerance"]):
                        best, improved = entry, True
                candidates.append(entry)
                self.log(f"[team_gas] 配比 {cid}：新数据 {entry['accounting']['new_presentations_per_epoch']}/"
                         f"{entry['accounting']['rows_per_epoch']}（{entry['accounting']['actual_fraction']:.1%}），"
                         f"开发集 macro_f1={prof['macro_f1']:.4f}，{self.display(D)} 召回 "
                         f"{entry['dev']['per_class'][D]['recall']:.2f}，{self.display(C)} 召回 {entry['dev']['per_class'][C]['recall']:.2f}"
                         + ("" if fraction == 0 else f"，{'通过' if entry['eligible'] else '未通过'}护栏"))
            if fraction > 0 and best is not None:
                stale = 0 if improved else stale + 1
                if stale >= m["patience_levels"] and level < len(m["fractions"]) - 1:
                    stopped_early = {"after_fraction": fraction, "skipped": m["fractions"][level + 1:]}
                    break
        eligible = [c for c in candidates if c.get("eligible")]
        chosen = None
        if eligible:
            top = max(c["dev"]["macro_f1"] for c in eligible)
            near = [c for c in eligible if c["dev"]["macro_f1"] >= top - m["tie_tolerance"]]
            chosen = min(near, key=lambda c: (c["fraction"], -c["dev"]["macro_f1"]))
        rnd["mixture"] = {"folder": str(folder), "pair": pair, "pre_update_dev": {"macro_f1": pre["macro_f1"],
                          "per_class": criteria.per_class_from_profile(pre)}, "candidates": candidates,
                          "control": control["id"] if control else None, "stopped_early": stopped_early,
                          "chosen": chosen["id"] if chosen else None, "parent": deployed,
                          "selection_batch": selection["id"]}
        if chosen is None:
            write_json(folder / "candidates.json", rnd["mixture"])
            return self._finish(rnd, "no_viable_mixture", "开发集上没有任何配比同时满足“退化类改善 + 混淆类和其他类别护栏”；保留原模型")
        frozen = {"candidate": chosen["id"], "fraction": chosen["fraction"], "degraded_share": chosen["degraded_share"],
                  "train_sha256": chosen["train_sha256"], "checkpoint_sha256": chosen["checkpoint_sha256"],
                  "policy": self.cfg["mixture"]["policy"], "frozen_at": now(),
                  "rule": "开发集护栏通过者中 macro_f1 最高；相差不超过 tie_tolerance 时取新数据更少的"}
        write_json(folder / "frozen.json", frozen)
        rnd["mixture"]["frozen"] = frozen
        write_json(folder / "candidates.json", rnd["mixture"])
        rnd["status"] = "mixture_frozen"
        self._save()
        self.event("mixture_frozen", round=rnd["index"], chosen=chosen["id"], tried=[c["id"] for c in candidates],
                   stopped_early=stopped_early)

    # ------------------------------------------------------------------ final
    def _final(self, rnd):
        ok, reasons = config_mod.ready(self.cfg, rnd["final"])
        if not ok:
            return self._halt("waiting", f"第 {rnd['index']} 轮的最终验收批次还不可用：" + "；".join(reasons), rnd)
        final = config_mod.batch(self.cfg, rnd["final"])
        sha = config_mod.check_sha256(final)
        exposure = exposure_status(final, self.ledger, sha, self.cfg["experiment_id"])
        used_here = [e for e in self.ledger.final_used(sha)
                     if not (e["experiment_id"] == self.cfg["experiment_id"] and e["stage_id"] == f"r{rnd['index']:02d}")]
        if exposure["exposed"] or used_here:
            raise RuntimeError(f"最终批次 {final['id']} 已暴露或已用过（{exposure}；{used_here}），不能作为独立验收")
        mix = rnd["mixture"]
        final_index = adapt_datasets.index(final["path"])
        for name, path in (("历史训练数据", mix["parent"]["train_file"]), ("新样本批次", config_mod.batch(self.cfg, rnd["alarm"])["path"]),
                           ("开发批次", config_mod.batch(self.cfg, rnd["selection"])["path"])):
            found = adapt_datasets.overlap(adapt_datasets.index(path), final_index)
            if found["ids"] or found["contents"]:
                raise RuntimeError(f"最终批次与{name}重叠：{found}")
        chosen = next(c for c in mix["candidates"] if c["id"] == mix["chosen"])
        control = next(c for c in mix["candidates"] if c["id"] == mix["control"])
        arms = {"pre_update": mix["parent"]["checkpoint"], "control_0pct": control["checkpoint"], "selected": chosen["checkpoint"]}
        profiles = {arm: diagnose.profile(path, final["path"], self.cache) for arm, path in arms.items()}
        D, C = mix["pair"]["degraded"], mix["pair"]["confusion"]
        a = self.cfg["acceptance"]
        pcs = {arm: criteria.per_class_from_profile(p) for arm, p in profiles.items()}
        check = criteria.targeted(pcs["pre_update"], pcs["selected"], D, C, a["target_recall_gain"], a["confusion_guard"], a["other_guard"])
        expected = profiles["selected"]["_expected"]
        p_better = probability_better("classification", "macro_f1", expected, profiles["selected"]["_predicted"],
                                      profiles["pre_update"]["_predicted"], len(self.classes), mode="sample",
                                      resamples=a["resamples"], seed=a["seed"])
        vs_control = criteria.targeted(pcs["control_0pct"], pcs["selected"], D, C, a["target_recall_gain"],
                                       a["confusion_guard"], a["other_guard"])
        accepted = check["ok"] and p_better >= a["threshold"]
        self.ledger.record(sha, final["id"], "final", self.cfg["experiment_id"], f"r{rnd['index']:02d}")
        result = {"batch": final["id"], "sha256": sha, "rows": final_index["rows"],
                  "profiles": {arm: public(p) for arm, p in profiles.items()}, "check_vs_pre_update": check,
                  "check_vs_control": vs_control, "p_macro_f1_better_than_pre_update": p_better,
                  "threshold": a["threshold"], "accepted": accepted}
        write_json(Path(rnd["folder"]) / "final_acceptance.json", result)
        rnd["final_acceptance"] = {k: result[k] for k in ("batch", "rows", "accepted", "p_macro_f1_better_than_pre_update",
                                                          "threshold", "check_vs_pre_update", "check_vs_control")}
        rnd["final_acceptance"]["macro_f1"] = {arm: p["macro_f1"] for arm, p in profiles.items()}
        if accepted:
            model = copy_model(chosen["checkpoint"], self.cfg["mixture"]["policy"], Path(rnd["folder"]) / "deployed")
            model.update({"from": f"r{rnd['index']:02d}-mixture:{chosen['id']}", "train_file": chosen["train_file"],
                          "train_parts": None})
            self._deploy(model, rnd, "最终验收通过")
            self._finish(rnd, "mixture_accepted", f"采用配比 {chosen['id']}：最终批次上通过验收")
        else:
            why = "；".join(check["notes"]) or ("宏平均 F1 提升不显著" if check["ok"] else "报警类改善不足或护栏未通过")
            self.event("rollback", round=rnd["index"], kept=self.state["deployed"]["from"], why=why)
            self._finish(rnd, "mixture_rejected", f"最终验收未通过（{why}）；保留更新前模型")
