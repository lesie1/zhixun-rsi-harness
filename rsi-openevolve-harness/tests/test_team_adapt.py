"""Tests for team_adapt (continual adaptation over time-ordered batches).

  python -m unittest tests.test_team_adapt -v

Everything runs on small synthetic batches with a real reference checkpoint; the real
split, gate and reference training run, but no model API is ever called.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402,F401
import numpy as np  # noqa: E402

from baselines.data import read_dataset  # noqa: E402
from baselines.model import Model  # noqa: E402
from harness.dataio import file_sha256, read_json  # noqa: E402
from team_adapt import config as config_mod  # noqa: E402
from team_adapt import controller as controller_mod  # noqa: E402
from team_adapt import datasets, stagnation  # noqa: E402
from team_adapt.controller import Experiment  # noqa: E402
from team_adapt.search import EvolveSearch, read_records  # noqa: E402

CLASSES = ["x", "y", "z"]
MEANS = {"x": [0, 0, 0, 0], "y": [3, 0, 0, 0], "z": [0, 3, 0, 0]}


def quiet(*_args, **_kwargs):
    pass


def rec(status="ok", score=1.0, iteration=1, error=None, tokens=0):
    return {"status": status, "combined_score": score, "iteration": iteration, "error": error, "run_id": "t",
            "token_usage": {"total_tokens": tokens}}


# --------------------------------------------------------------------------- pure logic

class StagnationTest(unittest.TestCase):
    def test_patience_counts_only_valid_candidates(self):
        records = [rec(score=1.0), rec("policy_rejected", 0), rec(score=1.001),
                   rec("rejected_before_eval", 0, error="LLM generation failed: timeout"), rec("train_failed", 0)]
        state = stagnation.analyze(records, patience=3, min_delta=0.01, initial_score=1.0)
        self.assertFalse(state["stagnated"])
        self.assertEqual(state["valid_since_last_improvement"], 2)
        self.assertEqual(state["counts"], {"valid": 2, "invalid_code": 1, "api_error": 1, "train_failure": 1,
                                           "internal_error": 0})
        state = stagnation.analyze(records + [rec(score=1.005)], patience=3, min_delta=0.01, initial_score=1.0)
        self.assertTrue(state["stagnated"])
        self.assertIn("不代表已确认数据漂移", state["note"])

    def test_improvement_must_reach_min_delta(self):
        records = [rec(score=1.004), rec(score=1.008), rec(score=1.02), rec(score=1.0), rec(score=1.0)]
        state = stagnation.analyze(records, patience=3, min_delta=0.01, initial_score=1.0)
        self.assertEqual(state["best_score"], 1.02)  # 1.004 / 1.008 were below min_delta
        self.assertEqual(state["valid_since_last_improvement"], 2)
        self.assertFalse(state["stagnated"])

    def test_first_valid_candidate_sets_baseline_without_initial_score(self):
        state = stagnation.analyze([rec(score=0.9)], patience=1, min_delta=0.0)
        self.assertEqual(state["best_score"], 0.9)
        self.assertFalse(state["stagnated"])

    def test_failure_streak_is_unhealthy_not_stagnation(self):
        records = [rec("rejected_before_eval", 0, error="LLM generation failed: Request timed out")] * 5
        state = stagnation.analyze(records, patience=2, min_delta=0.0, initial_score=1.0, max_consecutive_failures=4)
        self.assertTrue(state["unhealthy"])
        self.assertFalse(state["stagnated"])
        self.assertEqual(state["counts"]["api_error"], 5)

    def test_classification(self):
        self.assertEqual(stagnation.classify(rec("constraint_violation")), "invalid_code")
        self.assertEqual(stagnation.classify(rec("timeout")), "train_failure")
        self.assertEqual(stagnation.classify(rec("rejected_before_eval", error="No valid diffs found")), "invalid_code")
        self.assertEqual(stagnation.classify(rec("internal_error")), "internal_error")


# ---------------------------------------------------------------------- synthetic batches

def write_batch(path, k, per_class=12, drift=0.0, seed=0, extra_rows=()):
    rng = np.random.default_rng(seed + k)
    rows = []
    for c in CLASSES:  # class blocks in file order, like the gas data
        for i in range(per_class):
            v = np.array(MEANS[c], float) + drift * (k - 1) + rng.normal(size=4)
            rows.append({"id": f"b{k}-{c}-{i}", "inputs": {"features": [float(a) for a in v]}, "answer": c,
                         "meta": {"source_batch": k, "source_index": len(rows) + 1}})
    rows += list(extra_rows)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return rows


class Synthetic(unittest.TestCase):
    """A temporary directory with batches b1..b4, a checkpoint trained on b1 and a base config."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="team_adapt_"))
        self.rows = {f"b{k}": write_batch(self.tmp / f"b{k}.jsonl", k, drift=0.4) for k in range(1, 5)}
        train = read_dataset(self.tmp / "b1.jsonl", "classification", CLASSES)
        model = Model("classification", train["paths"], CLASSES, train["x"], train["y"], hidden=4, seed=1)
        model.fit(train["x"], train["y"], 2)
        model.save(self.tmp / "checkpoint.npz")
        self.base = self.tmp / "base.json"
        self.write_base()
        self.out = self.tmp / "out"
        self.cleanup = []

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        for path in self.cleanup:
            shutil.rmtree(path, ignore_errors=True)

    def write_base(self, **extra):
        base = {"task_name": "A", "stage_id": "syn", "seed": 42, "train_path": str(self.tmp / "b1.jsonl"),
                "validation_path": str(self.tmp / "b2.jsonl"), "parent_checkpoint": str(self.tmp / "checkpoint.npz"),
                "work_dir": str(self.tmp / "unused"), "split": {"min_groups": 4},
                "limits": {"max_epochs": 5, "train_timeout_s": 60, "cpu_threads": 1}, **extra}
        self.base.write_text(json.dumps(base), encoding="utf-8")

    def experiment_config(self, name="exp.json", labels=None, exposure=None, **overrides):
        labels, exposure = labels or {}, exposure or {}
        uses = {"b1": ["train"], "b2": ["train", "dev"], "b3": ["train", "dev"], "b4": ["final"]}
        cfg = {"experiment_id": "syn", "task_name": "A", "base_config": str(self.base),
               "initial_train_batches": ["b1"], "focus_classes": ["x"],
               "batches": [{"id": b, "order": k, "path": str(self.tmp / f"{b}.jsonl"),
                            "labels_available": labels.get(b, True), "uses": uses[b], "exposure": exposure.get(b, [])}
                           for k, b in enumerate(["b1", "b2", "b3", "b4"], 1)],
               "training": {"mode": "replay"}, "search": {"mode": "continuation"}, "final": {"batch": "b4"},
               "budget": {"stage_max_iterations": 10, "max_stages": 5}}
        for key, value in overrides.items():
            cfg[key] = value
        path = self.tmp / name
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return path


# ------------------------------------------------------------------------- plan / order

class PlanTest(Synthetic):
    def test_derived_plan_follows_time_order(self):
        cfg = config_mod.load_config(self.experiment_config())
        self.assertEqual([(s["new_train"], s["train"], s["dev"]) for s in cfg["plan"]],
                         [(["b1"], ["b1"], "b2"), (["b2"], ["b1", "b2"], "b3")])
        cfg = config_mod.load_config(self.experiment_config(training={"mode": "new_only"}))
        self.assertEqual([s["train"] for s in cfg["plan"]], [["b1"], ["b2"]])

    def test_batches_listed_out_of_order_are_sorted_by_order(self):
        path = self.experiment_config()
        cfg = json.loads(path.read_text())
        cfg["batches"].reverse()
        path.write_text(json.dumps(cfg))
        self.assertEqual([b["id"] for b in config_mod.load_config(path)["batches"]], ["b1", "b2", "b3", "b4"])

    def test_invalid_declarations_are_refused(self):
        bad = {
            "dev before its training data": {"stages": [{"new_train": ["b1"], "dev": "b2"}, {"new_train": ["b3"], "dev": "b2"}]},
            "stage going back in time": {"stages": [{"new_train": ["b2"], "dev": "b3"}, {"new_train": ["b1"], "dev": "b3"}]},
            "final used inside a stage": {"stages": [{"new_train": ["b1"], "dev": "b4"}]},
        }
        for label, override in bad.items():
            with self.subTest(label), self.assertRaises(config_mod.ConfigError):
                config_mod.load_config(self.experiment_config(**override))
        path = self.experiment_config()
        cfg = json.loads(path.read_text())
        cfg["batches"][3]["uses"] = ["final", "dev"]
        path.write_text(json.dumps(cfg))
        with self.assertRaises(config_mod.ConfigError):
            config_mod.load_config(path)
        cfg["batches"][3]["uses"] = ["final"]
        cfg["batches"][3]["order"] = 1
        path.write_text(json.dumps(cfg))
        with self.assertRaises(config_mod.ConfigError):
            config_mod.load_config(path)


# ----------------------------------------------------------------------- full pipeline

class PipelineTest(Synthetic):
    def test_waits_without_new_labels_then_continues(self):
        exp = Experiment(self.experiment_config(labels={"b3": False}), self.out, log=quiet)
        self.assertEqual(exp.run(), 3)
        state = read_json(self.out / "state.json")
        self.assertEqual(state["status"], "waiting")
        self.assertEqual(len(state["stages"]), 1)
        self.assertFalse((self.out / "stages" / "s02").exists())
        self.assertEqual(Experiment(self.experiment_config(labels={"b3": False}), self.out, log=quiet).run(), 3)
        self.assertEqual(len(read_json(self.out / "state.json")["stages"]), 1)  # waiting spends nothing
        exp = Experiment(self.experiment_config(labels={"b3": True}), self.out, log=quiet)
        self.assertEqual(exp.run(), 0)
        state = read_json(self.out / "state.json")
        self.assertEqual([s["status"] for s in state["stages"]], ["done", "done"])
        events = [json.loads(l)["event"] for l in (self.out / "events.jsonl").read_text().splitlines()]
        self.assertIn("phase_switch", events)

    def test_model_inheritance_and_recorded_data(self):
        self.assertEqual(Experiment(self.experiment_config(), self.out, log=quiet).run(), 0)
        state = read_json(self.out / "state.json")
        s1, s2 = state["stages"]
        decision = read_json(Path(s1["work_dir"]) / "gate" / "decision.json")
        expected = s1["parent"]["sha256"] if decision["decision"] == "keep_parent" else file_sha256(decision["selected_checkpoint"])
        self.assertEqual(s1["selected"]["sha256"], expected)
        self.assertEqual(s2["parent"]["sha256"], s1["selected"]["sha256"])
        self.assertEqual(read_json(s2["run_config"])["parent_checkpoint"], s2["parent"]["checkpoint"])
        self.assertNotEqual(s1["stage_id"], s2["stage_id"])
        self.assertNotEqual(s1["work_dir"], s2["work_dir"])
        self.assertNotEqual(s1["split_manifest"], s2["split_manifest"])
        self.assertEqual({p["batch_id"]: p["rows"] for p in s2["data"]["train"]["parts"]}, {"b1": 36, "b2": 36})
        self.assertEqual(s2["data"]["overlap_train_dev"], {"ids": 0, "contents": 0})
        table = read_json(s2["comparison"])
        self.assertEqual(table["sets"]["old:s01"]["role"], "old")
        self.assertIn("official_continuation", table["sets"]["dev_gate"]["arms"])
        self.assertIn("x", table["sets"]["dev_gate"]["arms"]["pre_update"]["per_class"])
        ledger = read_json(self.out / "exposure_ledger.json")["entries"]
        self.assertEqual(sorted({(e["batch_id"], e["use"]) for e in ledger}),
                         [("b1", "train"), ("b2", "dev"), ("b2", "train"), ("b3", "dev")])

    def test_keep_parent_without_selected_directory(self):
        exp = Experiment(self.experiment_config(), self.out, log=quiet)
        stage = exp._prepare(exp.cfg["plan"][0])
        work = Path(stage["work_dir"])

        def fake_gate(command, **kwargs):
            (work / "gate").mkdir(parents=True, exist_ok=True)
            (work / "gate" / "decision.json").write_text(json.dumps({
                "decision": "keep_parent", "reason": "nothing passed", "selected_checkpoint": stage["parent"]["checkpoint"],
                "candidate_dir": None, "comparisons": {}}))
            return subprocess.CompletedProcess(command, 0)
        original = controller_mod.subprocess.run
        controller_mod.subprocess.run = fake_gate
        try:
            stage["status"] = "searched"
            exp._gate(stage)
        finally:
            controller_mod.subprocess.run = original
        self.assertFalse((work / "selected").exists())
        self.assertEqual(stage["selected"]["sha256"], stage["parent"]["sha256"])
        self.assertEqual(exp.state["current"]["sha256"], stage["parent"]["sha256"])
        self.assertEqual(stage["selected"]["source"], "parent")

    def test_overlapping_dev_data_is_refused(self):
        dup = dict(self.rows["b1"][0], id="b2-copy-of-b1")
        write_batch(self.tmp / "b2.jsonl", 2, drift=0.4, extra_rows=[dup])
        with self.assertRaisesRegex(ValueError, "重叠"):
            Experiment(self.experiment_config(), self.out, log=quiet).run()

    def test_declared_sha256_must_match_the_file(self):
        path = self.experiment_config()
        cfg = json.loads(path.read_text())
        cfg["batches"][1]["sha256"] = "sha256:" + "0" * 64
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "摘要与声明不一致"):
            Experiment(path, self.out, log=quiet).run()

    def test_duplicate_ids_across_replayed_batches_are_refused(self):
        rows = [dict(r, id=r["id"].replace("b2-", "b1-")) for r in self.rows["b2"]]
        target = self.tmp / "dup.jsonl"
        target.write_text("".join(json.dumps(r) + "\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "同时出现"):
            datasets.materialize([("b1", self.tmp / "b1.jsonl"), ("b2", target)], self.tmp / "m.jsonl")

    def test_final_acceptance_runs_once(self):
        self.assertEqual(Experiment(self.experiment_config(), self.out, log=quiet).run(), 0)
        exp = Experiment(self.experiment_config(), self.out, log=quiet)
        result = exp.final()
        self.assertTrue(result["independent"])
        arms = result["table"]["sets"]["final"]["arms"]
        self.assertTrue({"initial", "selected_s01", "selected_s02", "last_pre_update",
                         "last_official_continuation"} <= set(arms))
        with self.assertRaisesRegex(RuntimeError, "已经做过一次"):
            Experiment(self.experiment_config(), self.out, log=quiet).final()
        self.assertEqual(Experiment(self.experiment_config(), self.out, log=quiet).run(), 4)  # closed

    def test_exposed_final_batch_is_refused(self):
        path = self.experiment_config(exposure={"b4": ["checked in an earlier study"]})
        self.assertEqual(Experiment(path, self.out, log=quiet).run(), 0)
        with self.assertRaisesRegex(RuntimeError, "已暴露"):
            Experiment(path, self.out, log=quiet).final()

    def test_changed_declaration_of_used_batch_is_refused(self):
        self.assertEqual(Experiment(self.experiment_config(), self.out, log=quiet).run(), 0)
        path = self.experiment_config()
        cfg = json.loads(path.read_text())
        cfg["batches"][1]["uses"] = ["dev"]
        path.write_text(json.dumps(cfg))
        with self.assertRaises(config_mod.ConfigError):
            Experiment(path, self.out, log=quiet)

    def test_evolve_mode_needs_explicit_permission(self):
        path = self.experiment_config(search={"mode": "evolve"})
        with self.assertRaises(PermissionError):
            Experiment(path, self.out, log=quiet).run()
        self.assertEqual(read_json(self.out / "state.json")["status"], "error")


# -------------------------------------------------------------- stagnation-driven stages

class ScriptedSearch(EvolveSearch):
    """Stands in for OpenEvolve: appends scripted records, optionally crashes once, writes a best program."""

    def __init__(self, scores, crash_after=None):
        super().__init__()
        self.scores, self.crash_after = list(scores), crash_after

    def run(self, task, settings, previous=None, log=print):
        work = Path(task["work_dir"])
        records = work / "records" / "iterations.jsonl"
        records.parent.mkdir(parents=True, exist_ok=True)
        done = len(read_records(work))
        best = work / "openevolve_output" / "best" / "best_program.py"
        best.parent.mkdir(parents=True, exist_ok=True)
        best.write_text((ROOT / "harness" / "policy_initial.py").read_text().replace('"epochs": 5', '"epochs": 2'))
        for i, score in enumerate(self.scores[done:], done + 1):
            with open(records, "a") as stream:
                stream.write(json.dumps(rec(score=score, iteration=i)) + "\n")
            if self.crash_after is not None and i >= self.crash_after:
                self.crash_after = None
                raise RuntimeError("simulated interruption")
            state = stagnation.analyze(read_records(work), settings["stagnation"]["patience"],
                                       settings["stagnation"]["min_delta"], 1.0)
            if state["stagnated"]:
                return self._summary("stagnated", [{"run_id": "scripted", "seconds": 0.0}], state)
        state = stagnation.analyze(read_records(work), settings["stagnation"]["patience"],
                                   settings["stagnation"]["min_delta"], 1.0)
        return self._summary("budget_iterations", [{"run_id": "scripted", "seconds": 0.0}], state)


class StagnationDrivenTest(Synthetic):
    def config(self):
        return self.experiment_config(search={"mode": "evolve"}, stagnation={"patience": 3, "min_delta": 0.01})

    def test_resume_after_interruption_keeps_budget_and_order(self):
        scores = [1.0, 1.05, 1.05, 1.04, 1.05, 1.0, 1.0]  # improves once, then 3 without improvement
        exp = Experiment(self.config(), self.out, searcher=ScriptedSearch(scores, crash_after=2), allow_llm=True, log=quiet)
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            exp.run()
        state = read_json(self.out / "state.json")
        self.assertEqual(state["stages"][0]["status"], "searching")
        exp = Experiment(self.config(), self.out, searcher=ScriptedSearch(scores), allow_llm=True, log=quiet)
        self.assertEqual(exp.run(), 0)
        state = read_json(self.out / "state.json")
        self.assertEqual([s["stage_id"] for s in state["stages"]], ["syn-s01", "syn-s02"])
        s1 = state["stages"][0]
        self.assertEqual(s1["search"]["end_reason"], "stagnated")
        self.assertEqual(s1["search"]["iterations"], 5)  # 2 before the crash + 3 after, nothing repeated
        self.assertEqual(len(read_records(s1["work_dir"])), 5)
        events = [json.loads(l) for l in (self.out / "events.jsonl").read_text().splitlines()]
        self.assertEqual(events[[e["event"] for e in events].index("phase_switch")]["trigger"], "stagnated")
        self.assertTrue(any(e["event"] == "stagnation_detected" for e in events))
        self.assertIn("evolved", read_json(s1["comparison"])["sets"]["dev_gate"]["arms"])

    def test_budget_end_does_not_switch_data_by_default(self):
        exp = Experiment(self.config(), self.out, searcher=ScriptedSearch([1.0, 1.1, 1.2]), allow_llm=True, log=quiet)
        self.assertEqual(exp.run(), 4)
        state = read_json(self.out / "state.json")
        self.assertEqual(len(state["stages"]), 1)
        self.assertEqual(state["status_reason"], "end_reason:budget_iterations")


# ----------------------------------------------------------------- the real monitor

FAKE_EVOLVE = textwrap.dedent("""
    import json, os, signal, sys, time
    from pathlib import Path
    args = sys.argv[1:]
    work = Path(json.loads(Path(args[args.index("--run-config") + 1]).read_text())["work_dir"])
    (work / "records").mkdir(parents=True, exist_ok=True)
    def stop(*_):
        (work / "terminated.txt").write_text("SIGTERM")
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    for i, item in enumerate(json.loads(os.environ["FAKE_RECORDS"]), 1):
        item.update(iteration=i, run_id=os.environ.get("HARNESS_RUN_ID"))
        with open(work / "records" / "iterations.jsonl", "a") as stream:
            stream.write(json.dumps(item) + "\\n")
        time.sleep(0.15)
    time.sleep(10)
""")


class MonitorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="team_adapt_monitor_"))
        self.script = self.tmp / "fake_run_evolve.py"
        self.script.write_text(FAKE_EVOLVE)
        (self.tmp / "work").mkdir()
        (self.tmp / "run.json").write_text(json.dumps({"work_dir": str(self.tmp / "work")}))
        self.task = {"run_config": str(self.tmp / "run.json"), "work_dir": str(self.tmp / "work"), "stage_id": "m",
                     "initial_policy": str(ROOT / "harness" / "policy_initial.py"), "log_dir": str(self.tmp)}
        self.settings = {"stagnation": {"patience": 3, "min_delta": 0.01, "max_consecutive_failures": 4},
                         "budget": {"stage_max_iterations": 40, "stage_max_seconds": 60},
                         "search": {"poll_seconds": 0.05, "stop_grace_seconds": 5}}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        os.environ.pop("FAKE_RECORDS", None)

    def run_search(self, records):
        os.environ["FAKE_RECORDS"] = json.dumps(records)
        return EvolveSearch(evolve_script=self.script).run(self.task, self.settings, log=quiet)

    def test_sigterm_on_stagnation(self):
        summary = self.run_search([{"status": "ok", "combined_score": 1.0}] * 20)
        self.assertEqual(summary["end_reason"], "stagnated")
        self.assertEqual(summary["attempts"][0]["stop_signal_reason"], "stagnated")
        self.assertTrue((self.tmp / "work" / "terminated.txt").is_file())
        self.assertLess(summary["iterations"], 20)

    def test_failures_stop_as_unhealthy(self):
        summary = self.run_search([{"status": "rejected_before_eval", "combined_score": 0,
                                    "error": "LLM generation failed: Request timed out"}] * 20)
        self.assertEqual(summary["end_reason"], "unhealthy")
        self.assertEqual(summary["analysis"]["counts"]["valid"], 0)


# --------------------------------------------------------------------------- ACE scope

class AceScopeTest(Synthetic):
    def test_record_origin_and_fresh_scope_per_stage(self):
        storage = ROOT / "state" / "tests" / f"team-adapt-{uuid.uuid4().hex[:8]}"
        self.cleanup.append(storage)
        self.write_base(ace={"module": "team_ace", "batch_size": 1, "mode": "disabled", "storage_root": str(storage),
                             "scope": {}})
        self.assertEqual(Experiment(self.experiment_config(), self.out, log=quiet).run(), 0)
        state = read_json(self.out / "state.json")
        origins = [read_json(s["run_config"])["ace"]["record_origin"] for s in state["stages"]]
        self.assertTrue(all(o.startswith("adapt|syn|syn-s0") and "validation=dev_only" in o for o in origins))
        playbooks = [read_json(s["run_config"])["playbook_path"] for s in state["stages"]]
        self.assertEqual(len(set(playbooks)), 2)
        self.assertTrue(all(str(storage) in p for p in playbooks))
        provenance = read_json(Path(state["stages"][1]["folder"]) / "experience_provenance.json")
        self.assertEqual(provenance["validation_level"], "development_only")


if __name__ == "__main__":
    unittest.main()
