"""Tests for the harness.

  python -m unittest discover -s tests -v

Tests that need the competition packages are skipped unless these environment variables
point at the extracted packages (the directories that contain checkpoint.npz and public/):
  HARNESS_DATA_A, HARNESS_DATA_B, HARNESS_DATA_C
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402,F401  (pins BLAS threads before numpy)
import numpy as np  # noqa: E402

from harness import constraints, groups, metrics, records, scoring, specs, splits  # noqa: E402

INITIAL = (ROOT / "harness" / "policy_initial.py").read_text(encoding="utf-8")
PACKAGES = {name: os.environ.get(f"HARNESS_DATA_{name}") for name in "ABC"}


def package(name):
    return unittest.skipUnless(PACKAGES[name], f"set HARNESS_DATA_{name} to run this test")


class ScoringTest(unittest.TestCase):
    spec_c = specs.resolve_spec("regression", "C")

    def test_parity_is_one(self):
        base = {"rmse": 50.0, "mae": 40.0, "nasa_score": 1e9}
        self.assertAlmostEqual(scoring.score(self.spec_c, base, base)["combined_score"], 1.0)

    def test_auxiliary_gain_cannot_buy_primary_loss(self):
        parent = {"rmse": 52.87, "mae": 38.9, "nasa_score": 5.82e9}
        worse_primary = {"rmse": 54.5, "mae": 38.9, "nasa_score": 3.9e9}
        result = scoring.score(self.spec_c, worse_primary, parent)
        self.assertLess(result["combined_score"], 1.0)
        self.assertTrue(result["guard_ok"])

    def test_guard_violation_is_penalised(self):
        parent = {"rmse": 52.87, "mae": 38.9, "nasa_score": 5.82e9}
        candidate = {"rmse": 49.1, "mae": 37.0, "nasa_score": 8.0e9}
        result = scoring.score(self.spec_c, candidate, parent)
        self.assertFalse(result["guard_ok"])
        self.assertAlmostEqual(result["primary_ratio"], 52.87 / 49.1)
        self.assertLess(result["combined_score"], 1.0)
        self.assertEqual(result["violations"][0]["guard"], "nasa_score")

    def test_direction_of_higher_is_better_metric(self):
        spec = specs.resolve_spec("classification", "A")
        better = scoring.score(spec, {"macro_f1": 0.76, "accuracy": 0.88}, {"macro_f1": 0.714, "accuracy": 0.88})
        self.assertAlmostEqual(better["primary_ratio"], 0.76 / 0.714)
        dropped = scoring.score(spec, {"macro_f1": 0.76, "accuracy": 0.80}, {"macro_f1": 0.714, "accuracy": 0.88})
        self.assertFalse(dropped["guard_ok"])

    def test_group_regression_guard(self):
        spec = specs.resolve_spec("regression", "B")
        base = {"rmse": 100.0, "mae": 80.0}
        result = scoring.score(spec, base, base, group_regressions=[("g1", 0.30), ("g2", -0.10)])
        self.assertFalse(result["guard_ok"])
        self.assertEqual(result["violations"][0]["guard"], "group:g1")


class SpecTest(unittest.TestCase):
    def test_resolution_order(self):
        self.assertEqual(specs.resolve_spec("classification")["source"], "default:classification")
        self.assertEqual(specs.resolve_spec("regression", "C")["source"], "known_task:C")
        self.assertTrue(specs.resolve_spec("regression", "C")["with_nasa"])
        self.assertEqual(specs.resolve_spec("regression", "unseen-task")["primary"], "rmse")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spec.json"
            path.write_text(json.dumps({"primary": "weighted_f1", "guards": {"accuracy": 0.01}}))
            spec = specs.resolve_spec("classification", "A", str(path))
            self.assertEqual(spec["primary"], "weighted_f1")
            self.assertTrue(spec["higher_is_better"])

    def test_rejects_unknown_or_mismatched_metric(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "spec.json"
            path.write_text(json.dumps({"primary": "bleu"}))
            with self.assertRaises(ValueError):
                specs.resolve_spec("classification", None, str(path))
            path.write_text(json.dumps({"primary": "rmse"}))
            with self.assertRaises(ValueError):
                specs.resolve_spec("classification", None, str(path))
        with self.assertRaises(ValueError):
            specs.resolve_spec("ranking")


class MetricsTest(unittest.TestCase):
    def test_macro_f1_by_hand(self):
        expected = np.array([0, 0, 1, 1, 2])
        labels = np.array([0, 1, 1, 1, 1])
        result = metrics.classification_metrics(expected, labels, 3)
        # class 0: 2*1/(1+2); class 1: 2*2/(4+2); class 2: 0
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 4 / 6 + 0) / 3)
        self.assertAlmostEqual(result["accuracy"], 3 / 5)

    def test_macro_f1_ignores_absent_classes_but_all_variant_does_not(self):
        result = metrics.classification_metrics(np.array([0, 0, 1]), np.array([0, 0, 1]), 3)
        self.assertAlmostEqual(result["macro_f1"], 1.0)
        self.assertAlmostEqual(result["macro_f1_all"], 2 / 3)

    def test_regression_and_nasa(self):
        result = metrics.regression_metrics(np.array([10.0, 20.0]), np.array([13.0, 16.0]), with_nasa=True)
        self.assertAlmostEqual(result["rmse"], (12.5) ** 0.5)
        self.assertAlmostEqual(result["mae"], 3.5)
        self.assertAlmostEqual(result["nasa_score"], (np.exp(3 / 10) - 1) + (np.exp(4 / 13) - 1))
        late = metrics.nasa_score(np.array([10.0]), np.array([20.0]))
        early = metrics.nasa_score(np.array([20.0]), np.array([10.0]))
        self.assertGreater(late, early)


class GroupTest(unittest.TestCase):
    def test_constant_unique_contiguous_and_composite(self):
        rows = []
        for subset in ("FD001", "FD003"):
            for unit in (1, 2):
                for cycle in (1, 2, 3):
                    rows.append({"subset": subset, "unit": unit, "cycle": cycle, "split": "train",
                                 "row": len(rows)})
        found = groups.discover_groups(rows)
        self.assertIn(["subset"], found["keys"])
        self.assertIn(["subset", "unit"], found["keys"])
        self.assertEqual(found["split_key"], ["subset", "unit"])
        self.assertEqual(found["sizes"]["subset+unit"], 4)
        self.assertNotIn(["split"], found["keys"])
        self.assertNotIn(["row"], found["keys"])
        self.assertNotIn(["unit"], found["keys"])  # unit 1 exists in both subsets

    def test_no_meta(self):
        found = groups.discover_groups([{} for _ in range(5)])
        self.assertEqual(found["keys"], [])
        self.assertIsNone(found["split_key"])


class SplitTest(unittest.TestCase):
    def test_classification_without_groups_keeps_every_class_on_both_sides(self):
        targets = [0] * 10 + [1] * 6 + [2] * 5  # measured class by class, like the gas data
        ids = [f"s{i}" for i in range(len(targets))]
        manifest = splits.make_split(ids, targets, [{} for _ in ids], "classification", "r1", "sha256:x",
                                     class_names=["a", "b", "c"])
        self.assertEqual(manifest["method"], "class_time")
        self.assertEqual(manifest["counts"]["select"]["labels"], {"a": 5, "b": 3, "c": 3})
        self.assertEqual(manifest["counts"]["gate"]["labels"], {"a": 5, "b": 3, "c": 2})
        self.assertEqual(manifest["select_ids"][:5], ["s0", "s1", "s2", "s3", "s4"])  # earlier half
        splits.validate_manifest(manifest, ids)

    def test_groups_never_straddle_and_split_is_deterministic(self):
        meta = [{"engine": f"e{i // 5}"} for i in range(40)]
        ids = [f"s{i}" for i in range(40)]
        first = splits.make_split(ids, list(range(40)), meta, "regression", "r1", "sha256:x")
        second = splits.make_split(ids, list(range(40)), meta, "regression", "r1", "sha256:x")
        self.assertEqual(first, second)
        self.assertEqual(first["method"], "group")
        engine = {i: m["engine"] for i, m in zip(ids, meta)}
        self.assertFalse({engine[i] for i in first["select_ids"]} & {engine[i] for i in first["gate_ids"]})
        self.assertEqual(sorted(first["select_ids"] + first["gate_ids"]), sorted(ids))

    def test_too_few_groups_means_no_split(self):
        meta = [{"bearing": f"b{i // 10}"} for i in range(30)]
        ids = [f"s{i}" for i in range(30)]
        manifest = splits.make_split(ids, list(range(30)), meta, "regression", "r1", "sha256:x")
        self.assertEqual(manifest["method"], "no_split")
        self.assertTrue(manifest["warnings"])
        self.assertEqual(manifest["select_ids"], manifest["gate_ids"])

    def test_manifest_validation(self):
        ids = ["a", "b", "c", "d"]
        manifest = splits.make_split(ids, [0.0, 1.0, 2.0, 3.0], [{} for _ in ids], "regression", "r1", "sha256:x",
                                     time_blocks=2)
        broken = dict(manifest, gate_ids=manifest["gate_ids"] + [manifest["select_ids"][0]])
        with self.assertRaises(ValueError):
            splits.validate_manifest(broken, ids)


class ConstraintTest(unittest.TestCase):
    def test_initial_policy_is_clean(self):
        self.assertEqual(constraints.check_policy_source(INITIAL), [])

    def test_rejections(self):
        cases = {
            "import": INITIAL.replace("def pick_epoch(history):", "import os\n\n\ndef pick_epoch(history):"),
            "open": INITIAL.replace("return len(history) - 1", "return len(open('/etc/passwd').read())"),
            "dunder": INITIAL.replace("return len(history) - 1", "return history.__class__"),
            "missing hook": INITIAL.replace("def pick_epoch(history):", "def choose_epoch(history):"),
            "marker": INITIAL.replace("# EVOLVE-BLOCK-END", ""),
            "numpy io": INITIAL.replace("return base_lr", "return float(np.load('x.npy'))"),
            "syntax": INITIAL.replace("return base_lr", "return base_lr +"),
        }
        for name, code in cases.items():
            with self.subTest(name):
                self.assertTrue(constraints.check_policy_source(code))


class RecordParsingTest(unittest.TestCase):
    def test_used_bullets_and_hypothesis(self):
        response = "HYPOTHESIS: 乙醛召回率为 0，提高它的权重\nUSED_BULLETS: [str-00003, 'mis-00008']\n<<<<<<< SEARCH"
        self.assertEqual(records.parse_used_bullets(response), ["str-00003", "mis-00008"])
        self.assertEqual(records.parse_hypothesis(response), "乙醛召回率为 0，提高它的权重")
        self.assertEqual(records.parse_used_bullets("USED_BULLETS: []"), [])
        self.assertEqual(records.parse_used_bullets("no such line"), [])


def _run_config(directory, name, extra=None):
    source = Path(PACKAGES[name])
    config = {"task_name": name, "stage_id": "test",
              "train_path": str(source / "public" / "train.jsonl"),
              "validation_path": str(source / "public" / "validation.jsonl"),
              "parent_checkpoint": str(source / "checkpoint.npz"), "work_dir": str(Path(directory) / "work"),
              **(extra or {})}
    path = Path(directory) / "run.json"
    path.write_text(json.dumps(config))
    return str(path)


class PackageTest(unittest.TestCase):
    """End-to-end checks against the real competition packages."""

    @package("A")
    def test_runner_reproduces_official_trainer_byte_for_byte(self):
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_equivalence.py"),
                               "--package", PACKAGES["A"], "--profile", "A"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("IDENTICAL", done.stdout)

    def _metrics_match_report(self, name):
        from baselines.model import Model
        from harness.dataio import load_dataset
        source = Path(PACKAGES[name])
        model = Model.load(source / "checkpoint.npz")
        data = load_dataset(source / "public" / "validation.jsonl", model.meta["task"], model.meta["classes"])
        ours = metrics.overall_metrics(model, data)
        official = json.loads((source / "training.json").read_text())["validation_after"]
        for key, value in official.items():
            self.assertAlmostEqual(ours[key], value, places=12, msg=key)

    @package("A")
    def test_metrics_match_official_report_a(self):
        self._metrics_match_report("A")

    @package("B")
    def test_metrics_match_official_report_b(self):
        self._metrics_match_report("B")

    @package("C")
    def test_metrics_match_official_report_c(self):
        self._metrics_match_report("C")

    @package("A")
    def test_evaluator_scores_and_rejects(self):
        from harness.evaluator import evaluate_policy
        from harness.runcfg import load_run_config
        from harness.stage import StageContext
        with tempfile.TemporaryDirectory() as temporary:
            context = StageContext(load_run_config(_run_config(temporary, "A")))
            self.assertEqual(context.manifest["method"], "class_time")
            self.assertEqual(len(context.manifest["counts"]["gate"]["labels"]), 6)
            policy = Path(temporary) / "policy.py"
            policy.write_text(INITIAL)
            good = evaluate_policy(policy, context)
            self.assertEqual(good["report"]["status"], "ok")
            self.assertAlmostEqual(good["metrics"]["combined_score"], context.reference_score["combined_score"])
            self.assertIn("CLASS_NEVER_PREDICTED", [flag["code"] for flag in good["report"]["flags"]])
            self.assertIn("[按类别]", good["artifacts"]["feedback"])
            cases = {
                "policy_rejected": INITIAL.replace("def pick_epoch(history):", "import os\n\n\ndef pick_epoch(history):"),
                "train_failed": INITIAL.replace("return base_lr", "return -1.0"),
            }
            for status, code in cases.items():
                with self.subTest(status):
                    policy.write_text(code)
                    bad = evaluate_policy(policy, context)
                    self.assertEqual(bad["report"]["status"], status)
                    self.assertEqual(bad["metrics"]["combined_score"], 0.0)
            policy.write_text(INITIAL.replace("return np.ones(len(y))", "return np.ones(len(y) + 1)"))
            wrong_shape = evaluate_policy(policy, context)
            self.assertEqual(wrong_shape["report"]["status"], "train_failed")
            self.assertIn("shape", wrong_shape["artifacts"]["feedback"])

    @package("A")
    def test_training_timeout_is_reported(self):
        from harness.evaluator import evaluate_policy
        from harness.runcfg import load_run_config
        from harness.stage import StageContext
        with tempfile.TemporaryDirectory() as temporary:
            context = StageContext(load_run_config(_run_config(temporary, "A")))
            context.config["limits"]["train_timeout_s"] = 2
            policy = Path(temporary) / "policy.py"
            policy.write_text(INITIAL.replace("    return base_lr", "    while True:\n        pass\n    return base_lr"))
            result = evaluate_policy(policy, context)
            self.assertEqual(result["report"]["status"], "timeout")

    @package("B")
    def test_b_has_no_independent_gate(self):
        from harness.runcfg import load_run_config
        from harness.stage import StageContext
        with tempfile.TemporaryDirectory() as temporary:
            context = StageContext(load_run_config(_run_config(temporary, "B")))
            self.assertEqual(context.manifest["method"], "no_split")
            self.assertEqual(context.aggregate, "group_mean")
            self.assertEqual(context.guard_key, "bearing")

    @package("C")
    def test_c_splits_by_engine(self):
        from harness.dataio import load_meta
        from harness.runcfg import load_run_config
        from harness.stage import StageContext
        with tempfile.TemporaryDirectory() as temporary:
            context = StageContext(load_run_config(_run_config(temporary, "C")))
            manifest = context.manifest
            self.assertEqual(manifest["method"], "group_stratified")
            self.assertEqual(manifest["group_key"], ["source_subset", "unit"])
            ids, meta = load_meta(Path(PACKAGES["C"]) / "public" / "validation.jsonl")
            engine = {i: (m["source_subset"], m["unit"]) for i, m in zip(ids, meta)}
            select = {engine[i] for i in manifest["select_ids"]}
            gate = {engine[i] for i in manifest["gate_ids"]}
            self.assertFalse(select & gate)
            self.assertEqual(len(select | gate), 106)
            self.assertEqual({e[0] for e in select}, {e[0] for e in gate})


if __name__ == "__main__":
    unittest.main()
