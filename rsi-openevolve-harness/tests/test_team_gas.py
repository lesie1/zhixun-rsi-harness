"""Tests for team_gas (A-type gas classification adaptation flow).

  python -m unittest tests.test_team_gas -v

Synthetic gas batches only; no model API is called (repair search is either skipped or scripted).
"""
import json
import shutil
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402,F401
import numpy as np  # noqa: E402

from harness.dataio import read_json, write_json  # noqa: E402
from harness.runcfg import load_run_config  # noqa: E402
from harness.training import run_training  # noqa: E402
from team_adapt.search import EvolveSearch  # noqa: E402
from team_gas import criteria, diagnose, mixture, playbook, repair, synth  # noqa: E402
from team_gas.controller import GasFlow  # noqa: E402

CLASSES = synth.CLASSES
BASE = playbook.load_base(ROOT / "team_gas" / "base_experiences.json")[0]


def quiet(*_a, **_k):
    pass


def profile_from(labels, predictions):
    index = {c: i for i, c in enumerate(CLASSES)}
    return diagnose._from_predictions(CLASSES, np.array([index[x] for x in labels]), np.array([index[x] for x in predictions]))


PARAMS = {"min_support": 15, "recall_drop": 0.3, "precision_drop": 0.3, "confidence": 0.9, "resamples": 500,
          "min_confusion_share": 0.3, "seed": 1}


# ------------------------------------------------------------------------------- diagnosis

class DiagnosisTest(unittest.TestCase):
    def test_single_class_degradation_while_overall_is_stable(self):
        # reference: every class 80 % right except Acetaldehyde (100 %); new: the others 100 %, Acetaldehyde half -> Ethanol
        ref_l, ref_p, new_l, new_p = [], [], [], []
        for c in CLASSES:
            for i in range(40):
                ref_l.append(c)
                ref_p.append(c if c == "Acetaldehyde" or i < 34 else "Ammonia" if c != "Ammonia" else "Ethylene")
                new_l.append(c)
                new_p.append("Ethanol" if c == "Acetaldehyde" and i < 26 else c)
        ref, new = profile_from(ref_l, ref_p), profile_from(new_l, new_p)
        self.assertLess(abs(ref["macro_f1"] - new["macro_f1"]), 0.01)  # the overall score hardly moves
        result = diagnose.compare(ref, new, PARAMS)
        self.assertEqual(result["status"], "degraded")
        self.assertEqual(result["pair"]["degraded"], "Acetaldehyde")
        self.assertEqual(result["pair"]["confusion"], "Ethanol")
        alarm = next(a for a in result["alarms"] if a["kind"] == "recall")
        self.assertEqual((alarm["confused_count"], alarm["support_new"]), (26, 40))
        self.assertIn("真实 Acetaldehyde 在新批次中 26/40（65.0%）被判为 Ethanol", alarm["text"])

    def test_confusion_direction_and_precision_alarm(self):
        labels = [c for c in CLASSES for _ in range(30)]
        new_p = ["Acetone" if c == "Toluene" else c for c in labels]
        result = diagnose.compare(profile_from(labels, labels), profile_from(labels, new_p), PARAMS)
        self.assertEqual(result["pair"], {"degraded": "Toluene", "confusion": "Acetone", "count": 30, "support": 30,
                                          "share_of_class": 1.0})
        precision = next(a for a in result["alarms"] if a["kind"] == "precision")
        self.assertEqual((precision["class"], precision["mostly_from"]), ("Acetone", "Toluene"))

    def test_too_few_samples_is_not_called_drift(self):
        labels = [c for c in CLASSES for _ in range(30) if c != "Toluene"] + ["Toluene"] * 6
        new_p = ["Acetone" if c == "Toluene" else c for c in labels]
        result = diagnose.compare(profile_from(labels, labels), profile_from(labels, new_p), PARAMS)
        self.assertEqual(result["classes"]["Toluene"]["status"], "insufficient_evidence")
        self.assertIsNone(result["pair"])
        self.assertTrue(any("证据不足" in i["text"] for i in result["insufficient"]))


# ---------------------------------------------------------------------------- stagnation

class AlarmStagnationTest(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix="gas_alarm_"))
        (self.work / "reports").mkdir()
        self.analyzer = repair.AlarmAnalyzer("Acetaldehyde", "Ethanol", None,
                                             {"patience": 3, "min_delta": 0.002, "alarm_min_delta": 0.02,
                                              "max_consecutive_failures": 5},
                                             {"target_recall_gain": 0.3, "confusion_guard": 0.1, "other_guard": 0.1})
        self.base = {c: {"recall": 1.0, "precision": 1.0, "support": 20} for c in CLASSES}
        self.base["Acetaldehyde"] = {"recall": 0.0, "precision": None, "support": 20}
        self.analyzer._baseline = self.base
        self.n = 0

    def tearDown(self):
        shutil.rmtree(self.work, ignore_errors=True)

    def valid(self, score, alarm, ethanol=1.0):
        self.n += 1
        digest = f"{self.n:064d}"
        per_class = [{"class": c, "recall": 1.0, "precision": 1.0, "support": 20} for c in CLASSES]
        for row in per_class:
            if row["class"] == "Acetaldehyde":
                row["recall"] = alarm
            if row["class"] == "Ethanol":
                row["recall"] = ethanol
        write_json(self.work / "reports" / f"{digest}.json", {"status": "ok", "slices": {"per_class": per_class},
                                                               "candidate_dir": str(self.work / digest)})
        return {"status": "ok", "combined_score": score, "iteration": self.n, "code_sha256": digest}

    def test_failures_are_counted_separately_and_never_stagnate(self):
        failures = [{"status": "rejected_before_eval", "error": "LLM generation failed: Request timed out"},
                    {"status": "policy_rejected"}, {"status": "train_failed"}, {"status": "timeout"}]
        state = self.analyzer(self.work, [self.valid(1.0, 0.0)] + failures)
        self.assertFalse(state["stagnated"])
        self.assertEqual(state["counts"], {"valid": 1, "invalid_code": 1, "api_error": 1, "train_failure": 2,
                                           "internal_error": 0})

    def test_small_overall_gain_cannot_hide_a_worsening_alarm_class(self):
        records = [self.valid(1.00, 0.10), self.valid(1.01, 0.05), self.valid(1.02, 0.02), self.valid(1.03, 0.0)]
        state = self.analyzer(self.work, records)
        self.assertEqual([t["progress"] for t in state["trace"]], [True, False, False, False])
        self.assertTrue(state["stagnated"])

    def test_targeted_fix_stops_as_repaired(self):
        state = self.analyzer(self.work, [self.valid(1.0, 0.0), self.valid(1.2, 0.6, ethanol=0.95)])
        self.assertTrue(state["repaired"])
        self.assertFalse(state["stagnated"])
        state = self.analyzer(self.work, [self.valid(1.2, 0.6, ethanol=0.5)])  # fixed D but broke C
        self.assertFalse(state["repaired"])


# ------------------------------------------------------------------------------- mixture

def jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


class MixtureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gas_mix_"))
        rng = np.random.default_rng(0)
        hist = [{"id": f"h{i}", "inputs": {"features": rng.normal(size=4).tolist()}, "answer": CLASSES[i % 6]} for i in range(37)]
        new = [{"id": f"n{i}", "inputs": {"features": rng.normal(size=4).tolist()}, "answer": "Acetaldehyde" if i < 2 else "Ethanol"}
               for i in range(5)]
        jsonl(self.tmp / "hist.jsonl", hist)
        jsonl(self.tmp / "new.jsonl", new)
        self.pools = mixture.make_pools(self.tmp / "hist.jsonl", self.tmp / "new.jsonl", "Acetaldehyde", "Ethanol", 3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cap_holds_with_repeats_and_odd_sizes(self):
        cand = mixture.plan(self.pools, 0.20, 0.5, 0.20)
        acc = cand["accounting"]
        self.assertEqual(acc["rows_per_epoch"], 37)
        self.assertEqual(acc["new_presentations_per_epoch"], 7)  # floor(0.2 * 37)
        self.assertLessEqual(acc["actual_fraction"], 0.20)
        self.assertEqual(acc["new_unique"], {"degraded": 2, "confusion": 3})
        self.assertGreater(sum(acc["new_repeats"].values()), 0)  # 4 Acetaldehyde presentations from 2 samples
        self.assertEqual(len({r[3]["id"] for r in cand["rows"]}), 37)

    def test_the_cap_cannot_be_raised(self):
        with self.assertRaises(ValueError):
            mixture.plan(self.pools, 0.25, 0.5, 0.25)
        with self.assertRaises(ValueError):
            mixture.plan(self.pools, 0.21, 0.5, 0.20)

    def test_class_split_is_explored_independently_of_the_share(self):
        splits = {s: mixture.plan(self.pools, 0.20, s, 0.20)["accounting"]["new_presentations"] for s in (1.0, 0.5, 0.25)}
        self.assertEqual(splits[1.0], {"degraded": 7, "confusion": 0})
        self.assertEqual(splits[0.25], {"degraded": 2, "confusion": 5})
        self.assertEqual(sum(splits[0.5].values()), 7)

    def test_identity_is_fixed_and_history_rows_are_replaced(self):
        small, large = mixture.plan(self.pools, 0.05, 0.5, 0.2), mixture.plan(self.pools, 0.20, 0.5, 0.2)
        hist_small = {r[3]["id"] for r in small["rows"] if r[0] == "history"}
        hist_large = {r[3]["id"] for r in large["rows"] if r[0] == "history"}
        self.assertTrue(hist_large < hist_small)  # more new data = fewer history rows, same total

    def test_realised_share_is_certified_from_the_runner(self):
        made = synth.make(self.tmp / "syn", history_per_class=10, batch_per_class=10, batches=3)
        pools = mixture.make_pools(made["batches"]["b1"], made["batches"]["b2"], "Acetaldehyde", "Ethanol", 1)
        cand = mixture.plan(pools, 0.20, 0.5, 0.20)
        sha = mixture.write(cand, self.tmp / "train.jsonl")
        out = self.tmp / "model"
        result = run_training(ROOT / "harness" / "policy_initial.py", made["checkpoint"], self.tmp / "train.jsonl",
                              made["batches"]["b3"], out, "t", max_epochs=5, timeout_s=120)
        self.assertEqual(result["returncode"], 0, result["stderr"][-500:])
        report, training = read_json(out / "runner_report.json"), read_json(out / "training.json")
        realised = mixture.verify_presentations(report, training, cand["accounting"], sha)
        self.assertEqual(realised["new_presentations_total"], 12 * realised["epochs_run"])  # floor(0.2*60)=12 per epoch
        self.assertLessEqual(realised["whole_run_fraction"], 0.20)
        report["per_epoch"][1]["step"] += 1  # an extra (padded) update would break the count
        with self.assertRaises(AssertionError):
            mixture.verify_presentations(report, training, cand["accounting"], sha)


# --------------------------------------------------------------------------- flow tests

class FlowCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="gas_flow_"))
        self.storage = ROOT / "state" / "tests" / f"team-gas-{uuid.uuid4().hex[:8]}"
        self.made = synth.make(self.tmp / "syn", history_per_class=40, batch_per_class=24, ace_storage=self.storage)
        self.out = self.tmp / "out"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(self.storage, ignore_errors=True)

    def flow_config(self, **overrides):
        cfg = json.loads(self.made["flow"].read_text(encoding="utf-8"))
        cfg["mixture"] = {"fractions": [0.0, 0.05, 0.10, 0.20], "degraded_shares": [1.0, 0.5, 0.25]}
        for key, value in overrides.items():
            cfg[key] = value
        path = self.made["flow"].with_name("flow.test.json")
        path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
        return path


class FlowTest(FlowCase):
    def test_end_to_end_two_rounds_and_playbook_in_both_stages(self):
        flow = GasFlow(self.flow_config(), self.out, log=quiet)
        self.assertEqual(flow.run(), 0)
        rounds = flow.state["rounds"]
        self.assertEqual([r["diagnosis"]["pair"]["degraded"] for r in rounds], ["Acetaldehyde", "Toluene"])
        self.assertEqual([r["diagnosis"]["pair"]["confusion"] for r in rounds], ["Ethanol", "Acetone"])
        scopes = set()
        for rnd in rounds:
            pb = rnd["repair"]["playbook"]
            self.assertEqual(pb["seeded"], [BASE["id"]])
            self.assertEqual(pb["occurrences_in_prompt"][BASE["id"]], 1)
            scopes.add(rnd["repair"]["playbook_path"])
            prompt = playbook.render_prompt(rnd["repair"]["run_config"])  # re-render from the stage's own config
            self.assertEqual(playbook.occurrences(prompt, BASE["id"]), 1)
            mix = rnd["mixture"]
            tried = [c for c in mix["candidates"] if c.get("status") == "ok"]
            self.assertEqual(len({c["degraded_share"] for c in tried if c["fraction"] > 0}), 3)  # class split explored
            for c in tried:
                self.assertLessEqual(c["realised"]["whole_run_fraction"], 0.20)
                self.assertLessEqual(c["accounting"]["new_presentations_per_epoch"], int(0.2 * c["accounting"]["rows_per_epoch"]))
            self.assertIn(rnd["outcome"], ("mixture_accepted", "mixture_rejected", "no_viable_mixture"))
            self.assertTrue((Path(rnd["folder"]) / "record.md").is_file())
        self.assertEqual(len(scopes), 2)  # a new ACE scope per stage, both carry the entry
        ledger = read_json(self.out / "exposure_ledger.json")["entries"]
        self.assertEqual(sorted(e["batch_id"] for e in ledger if e["use"] == "final"), ["b4", "b7"])

    def test_rejected_final_acceptance_keeps_the_pre_update_model(self):
        cfg = self.flow_config(rounds=[{"alarm": "b2", "selection": "b3", "final": "b4"}],
                               acceptance={"target_recall_gain": 1.01})
        flow = GasFlow(cfg, self.out, log=quiet)
        before = flow.state["deployed"]["sha256"]
        self.assertEqual(flow.run(), 0)
        rnd = flow.state["rounds"][0]
        if rnd["outcome"] == "no_viable_mixture":
            self.skipTest("no mixture candidate passed on this synthetic draw")
        self.assertEqual(rnd["outcome"], "mixture_rejected")
        self.assertEqual(flow.state["deployed"]["sha256"], before)
        events = [json.loads(l)["event"] for l in (self.out / "events.jsonl").read_text().splitlines()]
        self.assertIn("rollback", events)

    def test_repair_on_old_data_does_not_switch_data(self):
        # A scripted search "finds" a candidate that fixes Acetaldehyde (trained here with the alarm data, to stand in
        # for a policy that works); the real harness gate then judges it on the alarm batch's gate split.
        b1, b2 = self.made["batches"]["b1"], self.made["batches"]["b2"]
        fix_train = self.tmp / "fix.jsonl"
        fix_train.write_text(b1.read_text() + b2.read_text())
        fixed = self.tmp / "fixed_model"
        result = run_training(ROOT / "harness" / "policy_initial.py", self.made["checkpoint"], fix_train,
                              self.made["batches"]["b3"], fixed, "fix", max_epochs=5, timeout_s=120)
        self.assertEqual(result["returncode"], 0, result["stderr"][-500:])

        class Repaired(EvolveSearch):
            def run(self_, task, settings, previous=None, log=print):
                return {"mode": "evolve", "end_reason": "repaired", "attempts": [], "iterations": 1, "tokens": 0,
                        "seconds": 0.0, "description": "scripted",
                        "analysis": {"repaired_candidate": {"code_sha256": "0" * 64, "candidate_dir": str(fixed)}}}
        cfg = self.flow_config(rounds=[{"alarm": "b2", "selection": "b3", "final": "b4"}],
                               repair={"search": {"mode": "evolve"}})
        flow = GasFlow(cfg, self.out, allow_llm=True, searcher=lambda analyzer: Repaired(analyzer=analyzer), log=quiet)
        self.assertEqual(flow.run(), 0)
        rnd = flow.state["rounds"][0]
        self.assertEqual(rnd["repair"]["gate"]["decision"], "adopt_candidate")
        self.assertEqual(rnd["outcome"], "repaired_on_old_data")
        self.assertNotIn("mixture", rnd)
        self.assertFalse((Path(rnd["folder"]) / "mixture").exists())

    def test_evolve_mode_needs_permission(self):
        cfg = self.flow_config(repair={"search": {"mode": "evolve"}})
        with self.assertRaises(PermissionError):
            GasFlow(cfg, self.out, log=quiet).run()


class PlaybookTest(FlowCase):
    def test_existing_entries_are_kept_and_the_base_entry_appears_once(self):
        flow = GasFlow(self.flow_config(rounds=[{"alarm": "b2", "selection": "b3", "final": "b4"}]), self.out, log=quiet)
        rnd = {"index": 1, "alarm": "b2", "selection": "b3", "final": "b4", "status": "pending",
               "folder": str(self.out / "rounds" / "r01")}
        flow.state["rounds"].append(rnd)
        flow._diagnose(rnd)
        flow._prepare_repair(rnd)
        run_config = load_run_config(rnd["repair"]["run_config"])
        from team_ace.store import connect, digest, encoded, publish
        db = connect(run_config)
        with db:
            db.execute("INSERT INTO entries VALUES (?,?)", ("A-feedbeefcafe", encoded({
                "id": "A-feedbeefcafe", "symptom": "existing", "action": "keep me", "scope_note": "s", "evidence_ids": ["x"],
                "outcomes": "o", "scope_id": digest(run_config["ace"]["scope"]), "status": "candidate", "mode": "disabled"})))
        publish(run_config, db)
        db.close()
        playbook.seed(run_config, [BASE])
        playbook.seed(run_config, [BASE])  # seeding twice is idempotent
        prompt = playbook.render_prompt(rnd["repair"]["run_config"])
        self.assertEqual(playbook.occurrences(prompt, BASE["id"]), 1)
        self.assertEqual(playbook.occurrences(prompt, "A-feedbeefcafe"), 1)
        text = Path(run_config["playbook_path"]).read_text(encoding="utf-8")
        self.assertLess(text.index(BASE["id"]), text.index("A-feedbeefcafe"))

    def test_entry_is_not_loaded_where_conditions_fail(self):
        ok, reasons = playbook.eligible(BASE, {"task_name": "B", "task_type": "regression", "pair": None,
                                               "trainable_new_labels": False})
        self.assertFalse(ok)
        self.assertEqual(len(reasons), 3)       # task type, confusion pair, trainable labels (no task-name limit since 11-27)
        self.assertTrue(playbook.eligible(BASE, {"task_name": "A", "task_type": "classification",
                                                 "pair": {"degraded": "x", "confusion": "y"}, "trainable_new_labels": True})[0])
        # The card is generic: any classification task with a degraded -> confused pair qualifies, not only gases.
        self.assertTrue(playbook.eligible(BASE, {"task_name": "D", "task_type": "classification",
                                                 "pair": {"degraded": "p", "confusion": "q"}, "trainable_new_labels": True})[0])


if __name__ == "__main__":
    unittest.main()
