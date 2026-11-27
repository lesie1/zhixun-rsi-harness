"""Tests for team_loop (batch loop + metric-based retraining trigger).

  HARNESS_DATA_A=<data>/A python -m unittest tests.test_team_loop -v

Tests that need the A package are skipped without HARNESS_DATA_A. Everything is written
to temporary directories; no model is called (search mode "continuation").
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402,F401

from baselines.model import Model  # noqa: E402
from team_loop import batches as batches_mod  # noqa: E402
from team_loop import stage as stage_mod  # noqa: E402
from team_loop import state as state_mod  # noqa: E402
from team_loop import step as step_mod  # noqa: E402
from team_loop import trigger  # noqa: E402
from team_loop.__main__ import main as cli  # noqa: E402

DATA_A = os.environ.get("HARNESS_DATA_A")
needs_a = unittest.skipUnless(DATA_A, "set HARNESS_DATA_A to run this test")


def quiet(*_args, **_kwargs):
    pass


class TriggerOptionsTest(unittest.TestCase):
    def test_defaults_and_overrides(self):
        opts = trigger.options({"primary_drop": 0.1, "confidence": None})
        self.assertEqual(opts["primary_drop"], 0.1)
        self.assertEqual(opts["confidence"], trigger.DEFAULTS["confidence"])

    def test_rejects_bad_values(self):
        for bad in ({"primary_drop": 1.5}, {"confidence": 0}, {"resamples": 10}, {"min_rows": 0}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                trigger.options(bad)


@needs_a
class LoopOnPackageA(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix="team_loop_test_"))
        data = Path(DATA_A)
        self.base = self.dir / "base_A.json"
        self.base.write_text(json.dumps({
            "task_name": "A", "stage_id": "test-loop-A",
            "train_path": str(data / "public" / "train.jsonl"),
            "validation_path": str(data / "public" / "validation.jsonl"),
            "parent_checkpoint": str(data / "checkpoint.npz"), "work_dir": str(self.dir / "unused"),
            "seed": 42, "limits": {"max_epochs": 2, "train_timeout_s": 60, "cpu_threads": 1},
            "split": {"min_groups": 4}}), encoding="utf-8")
        self.batches = batches_mod.simulate(self.base, 2, self.dir / "data")
        self.out = self.dir / "loop"

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def spec(self, checkpoint):
        return trigger.spec_for(Model.load(checkpoint), "A")

    # ------------------------------------------------------------ simulate
    def test_simulate_chunks_are_contiguous_and_history_grows(self):
        source = Path(DATA_A) / "public" / "validation.jsonl"
        lines = source.read_text(encoding="utf-8").splitlines()
        new = [Path(b["validation_path"]).read_text(encoding="utf-8").splitlines() for b in self.batches]
        self.assertEqual(new[0] + new[1], lines)
        train = [Path(b["train_path"]).read_text(encoding="utf-8").splitlines() for b in self.batches]
        self.assertEqual(train[1][len(train[0]):], new[0])

    # ------------------------------------------------------------- trigger
    def test_no_reference_always_retrains(self):
        checkpoint = Path(DATA_A) / "checkpoint.npz"
        report = trigger.check(checkpoint, self.batches[0]["validation_path"], None, self.spec(checkpoint))
        self.assertTrue(report["retrain"])

    def test_same_level_does_not_retrain(self):
        checkpoint = Path(DATA_A) / "checkpoint.npz"
        spec = self.spec(checkpoint)
        reference = trigger.profile(checkpoint, self.batches[0]["validation_path"], spec)
        report = trigger.check(checkpoint, self.batches[0]["validation_path"], reference, spec)
        self.assertFalse(report["retrain"])
        self.assertTrue(all(abs(c["relative_worsening"]) < 1e-12 for c in report["checks"]))

    def test_clear_drop_retrains_and_small_batch_does_not(self):
        checkpoint = Path(DATA_A) / "checkpoint.npz"
        spec = self.spec(checkpoint)
        reference = trigger.profile(checkpoint, self.batches[0]["validation_path"], spec)
        reference["metrics"] = dict(reference["metrics"], macro_f1=1.0)
        report = trigger.check(checkpoint, self.batches[0]["validation_path"], reference, spec)
        self.assertTrue(report["retrain"])
        self.assertTrue(any(c["metric"] == "macro_f1" and c["fires"] for c in report["checks"]))
        small = trigger.check(checkpoint, self.batches[0]["validation_path"], reference, spec,
                              {"min_rows": 100000})
        self.assertFalse(small["retrain"])

    # ---------------------------------------------------------------- loop
    def run_loop(self, retrain="auto"):
        state_mod.init(self.base, self.out)
        return [step_mod.process(self.out, b, retrain=retrain, search="continuation", log=quiet) for b in self.batches]

    def test_loop_writes_one_decision_per_batch_and_chains_checkpoints(self):
        decisions = self.run_loop()
        lines = (self.out / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["batch_id"] for line in lines], ["b01", "b02"])
        state = state_mod.load(self.out)
        self.assertEqual([h["batch_id"] for h in state["history"]], ["b01", "b02"])
        self.assertTrue(decisions[0]["retrain"])  # reference measured in-sample on train: the first batch drops
        current = json.loads((self.out / "current.json").read_text(encoding="utf-8"))
        self.assertEqual(current["checkpoint"], decisions[-1]["checkpoint"])
        for decision in decisions:
            self.assertEqual(decision["deploy"], decision["gate_decision"] not in (None, "keep_parent"))
            Model.load(decision["checkpoint"])
        if decisions[0]["deploy"] and decisions[1]["retrain"]:
            stage = json.loads((stage_mod.batch_dir(self.out, "b02") / "stage.json").read_text(encoding="utf-8"))
            self.assertEqual(stage["parent_checkpoint"], decisions[0]["checkpoint"])

    def test_processing_a_batch_twice_reuses_the_decision(self):
        first = self.run_loop()
        again = step_mod.process(self.out, self.batches[0], search="continuation", log=quiet)
        self.assertEqual(again, first[0])
        self.assertEqual(len(state_mod.load(self.out)["history"]), 2)

    def test_never_mode_trains_nothing(self):
        decisions = self.run_loop(retrain="never")
        self.assertTrue(all(not d["retrain"] and not d["deploy"] for d in decisions))
        self.assertFalse(any((stage_mod.batch_dir(self.out, b["batch_id"]) / "run.json").exists() for b in self.batches))

    def test_failed_round_keeps_the_deployed_model(self):
        state_mod.init(self.base, self.out)
        before = state_mod.load(self.out)["deployed"]

        def broken(*_args, **_kwargs):
            raise RuntimeError("simulated failure")
        original = stage_mod.prepare
        stage_mod.prepare = broken
        try:
            decision = step_mod.process(self.out, self.batches[0], retrain="always", search="continuation", log=quiet)
        finally:
            stage_mod.prepare = original
        self.assertIn("simulated failure", decision["error"])
        self.assertFalse(decision["deploy"])
        after = state_mod.load(self.out)["deployed"]
        self.assertEqual(after["checkpoint"], before["checkpoint"])
        self.assertEqual(after["reference"], before["reference"])

    def test_cli_run_and_status(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli(["run", "--out", str(self.out), "--base-config", str(self.base),
                        "--batches", str(self.dir / "data" / "batches.json"), "--search", "continuation"])
        self.assertEqual(code, 0)
        self.assertIn("DECISION batch=b01", out.getvalue())
        self.assertIn("b02:", out.getvalue())


if __name__ == "__main__":
    unittest.main()
