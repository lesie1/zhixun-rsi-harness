"""swanlab_backfill: discovery and record planning (no SwanLab needed, nothing is uploaded)."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("swanlab_backfill",
                                              Path(__file__).resolve().parents[1] / "scripts" / "swanlab_backfill.py")
backfill = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(backfill)


def write(path, obj, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if lines:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in obj), encoding="utf-8")
    else:
        path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


class BackfillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name)
        run = root / "B-live-01" / "f0" / "p1" / "v3" / "round-02" / "v3-1.0"
        write(run / "training.json", {"profile": "B", "task": "regression", "stage_id": "sim-v3-1.0-r02",
                                      "created_at": "2026-11-24T01:00:00+00:00", "learning_rate": 0.003,
                                      "validation_before": {"rmse": 500.0}, "validation_after": {"rmse": 480.0},
                                      "training_seconds": 0.4, "configuration": {"hidden": 24}})
        write(run / "history.jsonl", [{"epoch": 16, "loss": 0.5, "validation": {"rmse": 490.0, "mae": 400.0}},
                                      {"epoch": 17, "loss": 0.4, "validation": {"rmse": 480.0, "mae": 390.0},
                                       "selected": True}], lines=True)
        (run.parent / "v3-1.0.policy.py").write_text("CONFIG = {}\n", encoding="utf-8")
        write(root / "A-split" / "gate" / "decision.json",
              {"decision": "keep_parent", "reason": "没有候选通过", "threshold": 0.9,
               "comparisons": {"candidate_vs_parent": {"probability_better": 0.55, "passes": False}}})
        write(root / "B-live-01" / "f0" / "p1" / "v3" / "rounds.jsonl",
              [{"round": 1, "trainings": 4, "test": {"rmse": 300.0, "advantage": 0.1},
                "decision": {"red": False, "action": "incumbent"}}], lines=True)
        write(root / "A-split" / "records" / "iterations.jsonl",
              [{"iteration": 1, "status": "ok", "outcome": "better", "combined_score": 1.02,
                "hypothesis": "降低学习率", "task_name": "A"}], lines=True)
        write(root / "L" / "v24" / "run" / "results.json", {"rmse": 450.0, "note": "复核"})
        write(root / "tests" / "x" / "training.json", {"profile": "A"})
        write(root / "cache" / "y" / "training.json", {"profile": "A"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_discover_kinds_and_skips(self):
        items = backfill.discover([self.root], extra_names=["results.json"])
        kinds = sorted(kind for kind, _, _ in items)
        self.assertEqual(kinds, ["gate", "record", "search", "sim", "train"])
        self.assertFalse(any("tests" in str(p) or "cache" in str(p) for _, _, p in items))
        with_tests = backfill.discover([self.root], include_tests=True)
        self.assertEqual(sum(kind == "train" for kind, _, _ in with_tests), 2)

    def test_train_plan_keeps_curves_and_marks_backfill(self):
        path = next(p for kind, _, p in backfill.discover([self.root]) if kind == "train")
        task, rel, original, config, steps = backfill.plan_train(path, self.root)
        self.assertEqual(task, "B")
        self.assertEqual(original, "2026-11-24T01:00:00+00:00")
        self.assertTrue(config["backfilled"])
        self.assertEqual([s for s, _ in steps[:2]], [16, 17])
        self.assertEqual(steps[1][1]["val/rmse"], 480.0)
        merged = {k: v for _, values in steps for k, v in values.items()}
        self.assertEqual(merged["summary/val_after_rmse"], 480.0)
        self.assertIn("CONFIG", merged["policy/source"])

    def test_other_planners(self):
        found = {kind: (root, p) for kind, root, p in backfill.discover([self.root], extra_names=["results.json"])}
        _, _, _, config, steps = backfill.plan_gate(found["gate"][1], self.root)
        self.assertEqual(config["decision"], "keep_parent")
        self.assertEqual(steps[0][1]["comparisons/candidate_vs_parent/passes"], 0)
        _, _, _, _, steps = backfill.plan_sim(found["sim"][1], self.root)
        self.assertEqual(steps[0][0], 1)
        self.assertEqual(steps[0][1]["test/advantage"], 0.1)
        task, _, _, _, steps = backfill.plan_search(found["search"][1], self.root)
        self.assertEqual(task, "A")
        self.assertEqual(steps[0][1]["iteration/hypothesis"], "降低学习率")
        _, _, _, _, steps = backfill.plan_record(found["record"][1], self.root)
        self.assertEqual(steps[0][1]["rmse"], 450.0)

    def test_dry_run_uploads_nothing(self):
        ledger = self.root / "ledger.json"
        code = backfill.main(["--root", str(self.root), "--dry-run", "--ledger", str(ledger)])
        self.assertEqual(code, 0)
        self.assertFalse(ledger.exists())


if __name__ == "__main__":
    unittest.main()
