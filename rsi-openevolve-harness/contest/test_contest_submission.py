"""End-to-end check of the contest image's app/ directory with the SDK's own host double.

Env: APP_DIR (contest/build/app), SDK_DIR (unpacked contest-harness-sdk-1.0.4), DATA_DIR (cpu-baselines-20261110-v1),
TASKS (default "A,B,C"). Runs describe, one full train round per task and predict on a few validation rows.
No real model call: the host double answers the one Qwen request with a fixed JSON verdict.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

APP = Path(os.environ["APP_DIR"]).resolve()
SDK = Path(os.environ["SDK_DIR"]).resolve()
DATA = Path(os.environ["DATA_DIR"]).resolve()
TASKS = os.environ.get("TASKS", "A,B,C").split(",")
sys.path.insert(0, str(SDK / "tests"))
sys.path.insert(0, str(SDK))
from support import Host  # noqa: E402

CONTRACT = {"A": ("classification", "macro_f1", "higher"), "B": ("regression", "rmse", "lower"),
            "C": ("regression", "rmse", "lower")}


def sha(path):
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def start_frame(key, previous, verdict_ok=True):
    task, metric, direction = CONTRACT[key]
    ckpt = previous / "checkpoint.npz"
    digest = sha(ckpt)
    parent = {"artifact_id": f"baseline-{key}", "artifact_sha256": digest,
              "artifact": {"checkpoint_paths": ["checkpoint.npz"], "code_paths": [],
                           "predict_argv": ["python3", "/app/harness.py", "predict"]},
              "files": [{"path": "checkpoint.npz", "sha256": digest, "size_bytes": ckpt.stat().st_size}]}
    return {"protocol": "contest.harness.workspace.v2", "run_id": "run-local", "dataset_key": key, "round_index": 1,
            "stage_id": "local-test", "prompt": "官方提示（测试替身）", "prompt_version": "industrial-workspace-flexible-v3",
            "rounds_per_dataset": 10, "parent_checkpoint_sha256": digest,
            "task_contract": {"dataset_key": key, "task": task, "metric": metric, "metric_direction": direction,
                              "public_projection": {"train_path": str(DATA / key / "public" / "train.jsonl"),
                                                    "validation_path": str(DATA / key / "public" / "validation.jsonl"),
                                                    "row_fields": ["id", "inputs", "answer"], "source_metadata": False}},
            "model_name": "base:test-model", "parent_artifact": parent, "artifacts": [parent], "state_sha256": "",
            "state_path": None, "max_output_tokens": 512,
            "resource_limits": {"artifact_upload_protocol": "chunked-v1", "artifact_bytes": 512 * 1024 * 1024,
                                "state_bytes": 512 * 1024 * 1024, "candidate_upload_bytes_per_round": 1024 ** 3,
                                "state_upload_bytes_per_round": 512 * 1024 * 1024},
            "current_validation": {}, "wall_seconds": 1800}


def fake_model(agree):
    counter = {"n": 0}

    def answer(messages, options):
        counter["n"] += 1
        assert options.get("max_tokens", 1) <= 512
        body = json.dumps({"agree": agree, "reason": "测试替身的固定回复"}, ensure_ascii=False)
        return {"request_id": f"call-{counter['n']}", "choices": [{"message": {"role": "assistant", "content": body},
                                                                    "finish_reason": "stop"}]}
    return answer


class ContestImageTest(unittest.TestCase):
    def env(self, tmp):
        extra = os.environ.get("EXTRA_PYTHONPATH", "")      # e.g. the image layer's site-packages
        env = dict(os.environ, PYTHONPATH=":".join(p for p in (extra, str(SDK), str(APP)) if p),
                   CONTEST_SUBMISSION="contest_submission",
                   CONTEST_WORK=str(tmp / "work"), CONTEST_PREVIOUS=str(tmp / "previous"),
                   OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        return env

    def test_describe(self):
        with tempfile.TemporaryDirectory() as t:
            out = subprocess.run([sys.executable, str(APP / "harness.py"), "describe"], cwd=APP, env=self.env(Path(t)),
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(out.returncode, 0, out.stderr)
            info = json.loads(out.stdout.strip().splitlines()[-1])
            self.assertEqual(info["protocol"], "contest.harness.workspace.v2")
            self.assertEqual(sorted(info["tasks"]), ["classification", "regression"])

    def run_round(self, key, agree=True):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            (tmp / "previous").mkdir()
            (tmp / "previous" / "checkpoint.npz").write_bytes((DATA / key / "checkpoint.npz").read_bytes())
            host = Host(start_frame(key, tmp / "previous"), fake_model(agree))
            began = time.time()
            err_path = tmp / "stderr.txt"
            with open(err_path, "w", encoding="utf-8") as err:
                process = subprocess.Popen([sys.executable, str(APP / "harness.py"), "train"], cwd=APP,
                                           env=self.env(tmp), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=err, text=True)
                thread = host.serve(process)
                process.wait(timeout=900)
                thread.join(timeout=30)
            stderr = err_path.read_text(encoding="utf-8")
            print(f"\n--- {key} ({time.time() - began:.1f}s)\n" + "\n".join(l for l in stderr.splitlines() if "[rsi]" in l),
                  file=sys.stderr)
            self.assertIsNone(host.failure, stderr[-3000:])
            self.assertEqual(process.returncode, 0, stderr[-3000:])
            record = host.record
            self.assertIsNotNone(record)
            self.assertTrue(record["model_call_ids"])
            self.assertIn(record["selected_artifact_id"], (f"baseline-{key}", "model-test-1"))
            self.assertTrue(record["reason"])
            return record, host

    def test_rounds(self):
        for key in TASKS:
            with self.subTest(task=key):
                record, host = self.run_round(key)
                self.assertEqual(len(host.raw_messages), 1)        # exactly one model call
                print(f"{key}: selected {record['selected_artifact_id']}; {record['reason'][:300]}", file=sys.stderr)

    def test_second_round_trains_new_data_candidates(self):
        """Round 1 leaves the seen training ids in its state; round 2 (training = old rows + half of the public
        validation rows as "new data") must find exactly those rows as new and train the new-data candidate."""
        for key in [k for k in ("A", "B") if k in TASKS]:
            with self.subTest(task=key), tempfile.TemporaryDirectory() as t:
                tmp = Path(t)
                (tmp / "previous").mkdir()
                (tmp / "previous" / "checkpoint.npz").write_bytes((DATA / key / "checkpoint.npz").read_bytes())
                train = [l for l in (DATA / key / "public" / "train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
                val = [l for l in (DATA / key / "public" / "validation.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
                half = len(val) // 2
                (tmp / "train2.jsonl").write_text("\n".join(train + val[:half]) + "\n", encoding="utf-8")
                (tmp / "val2.jsonl").write_text("\n".join(val[half:]) + "\n", encoding="utf-8")
                records = []
                for round_index in (1, 2):
                    frame = start_frame(key, tmp / "previous")
                    frame["round_index"] = round_index
                    if round_index == 2:
                        frame["state_path"] = str(tmp / "work" / "rsi-round-01" / "state")
                        projection = frame["task_contract"]["public_projection"]
                        projection["train_path"], projection["validation_path"] = str(tmp / "train2.jsonl"), str(tmp / "val2.jsonl")
                    host = Host(frame, fake_model(True))
                    with open(tmp / f"stderr{round_index}.txt", "w", encoding="utf-8") as err:
                        process = subprocess.Popen([sys.executable, str(APP / "harness.py"), "train"], cwd=APP,
                                                   env=self.env(tmp), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                   stderr=err, text=True)
                        thread = host.serve(process)
                        process.wait(timeout=900)
                        thread.join(timeout=30)
                    stderr = (tmp / f"stderr{round_index}.txt").read_text(encoding="utf-8")
                    self.assertIsNone(host.failure, stderr[-3000:])
                    self.assertEqual(process.returncode, 0, stderr[-3000:])
                    records.append((host.record, stderr))
                first, second = records[0][0]["configuration"], records[1][0]["configuration"]
                self.assertIsNone(first["data"]["train_new_rows"])            # nothing to compare with in round 1
                self.assertFalse(first["data"]["recent_candidates"])
                self.assertEqual(second["data"]["train_rows"], len(train) + half)
                self.assertEqual(second["data"]["train_new_rows"], half)       # exactly the added rows are new
                self.assertTrue(second["data"]["recent_candidates"])
                recent = "recent+balanced" if key == "A" else "recent+pick"
                self.assertIn(recent, second["candidates"])
                if key == "A":
                    self.assertIn("balanced", first["candidates"])
                    self.assertIn("pick+balanced", first["candidates"])
                print(f"\n{key} round 2: choice {second['choice']} among {second['candidates']}; data {second['data']}",
                      file=sys.stderr)

    def test_model_veto_keeps_parent(self):
        record, _ = self.run_round(TASKS[0], agree=False)
        self.assertEqual(record["selected_artifact_id"], f"baseline-{TASKS[0]}")

    def test_predict(self):
        for key in TASKS:
            task = CONTRACT[key][0]
            rows = []
            with open(DATA / key / "public" / "validation.jsonl", encoding="utf-8") as stream:
                for line in stream:
                    rows.append(json.loads(line))
                    if len(rows) == 5:
                        break
            payload = "".join(json.dumps({"task": task, "inputs": r["inputs"]}) + "\n" for r in rows)
            with tempfile.TemporaryDirectory() as t:
                env = self.env(Path(t))
                env["CONTEST_ARTIFACT_PATH"] = str(DATA / key / "checkpoint.npz")
                out = subprocess.run([sys.executable, str(APP / "harness.py"), "predict"], cwd=APP, env=env, input=payload,
                                     capture_output=True, text=True, timeout=120)
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            lines = [json.loads(l) for l in out.stdout.strip().splitlines()]
            self.assertEqual(len(lines), 5)
            sys.path.insert(0, str(APP / "rsi"))
            import numpy as np
            from baselines.data import flatten
            from baselines.model import Model
            model = Model.load(str(DATA / key / "checkpoint.npz"))
            for row, got in zip(rows, lines):
                names, values = flatten(row["inputs"])
                x = np.array([[dict(zip(names, values))[p] for p in model.meta["paths"]]])
                want = model.predict(x)
                if task == "classification":
                    labels = [o["label"] for o in got["outputs"]]
                    self.assertEqual(labels, model.meta["classes"])
                    best = max(got["outputs"], key=lambda o: o["score"])["label"]
                    self.assertEqual(best, model.meta["classes"][int(want[0].argmax())])
                else:
                    self.assertAlmostEqual(got["outputs"]["value"], float(want[0]), places=6)


class HelperTest(unittest.TestCase):
    """The new-data and class-weighting helpers, imported from the app directory."""

    def setUp(self):
        for path in (str(SDK), str(APP), str(APP / "rsi")):     # rsi/ first: /app/harness.py must not shadow rsi/harness
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
        import contest_submission
        self.cs = contest_submission

    def test_class_weighted_policy_passes_the_policy_checks(self):
        from harness.constraints import check_policy_source
        for name in ("policy_pick.py",):
            source = self.cs.class_weighted((APP / "rsi" / "policies" / name).read_text(encoding="utf-8"))
            self.assertIn("np.bincount", source)
            self.assertEqual(check_policy_source(source), [])
        with self.assertRaises(ValueError):
            self.cs.class_weighted("def sample_weights(y, task, num_classes):\n    return y\n")

    def test_seen_ids_and_recent_file(self):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            self.assertEqual(self.cs.load_seen(tmp), set())
            rows = [{"id": f"r{i}", "inputs": {"x": i}, "answer": "a"} for i in range(5)]
            (tmp / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            self.cs.save_seen(tmp, {self.cs.row_key(f"r{i}") for i in range(3)})
            seen = self.cs.load_seen(tmp)
            new = {self.cs.row_key(r["id"]) for r in rows} - seen
            path = self.cs.write_recent(tmp / "train.jsonl", new, tmp / "recent.jsonl", copies=2)
            ids = [json.loads(l)["id"] for l in Path(path).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(ids), 5 + 2 * 2)
            self.assertEqual(len(ids), len(set(ids)))                     # copies get unique ids
            self.assertEqual(sorted(i for i in ids if "#" in i), ["r3#recent1", "r3#recent2", "r4#recent1", "r4#recent2"])
            old = self.cs.MAX_TRAIN_BYTES
            try:
                self.cs.MAX_TRAIN_BYTES = 10                              # too large -> no file, no candidate
                self.assertIsNone(self.cs.write_recent(tmp / "train.jsonl", new, tmp / "recent2.jsonl"))
                self.assertFalse((tmp / "recent2.jsonl").exists())
            finally:
                self.cs.MAX_TRAIN_BYTES = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
