"""Tests for team_sim (public-data stream simulator), harness/policy_rul_cap.py and the preset skills.

  HARNESS_DATA_B=<data>/B python -m unittest tests.test_team_sim -v

Tests that need the B package are skipped without HARNESS_DATA_B. Everything is written to
temporary directories; no model is called.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import harness  # noqa: E402,F401

from harness import runner  # noqa: E402
from harness.constraints import check_policy_source  # noqa: E402
from team_sim import data as simdata  # noqa: E402
from team_sim import decide  # noqa: E402
from team_sim import seeds as preset  # noqa: E402
from team_sim.policy import TEMPLATE, render, write  # noqa: E402

DATA_B = os.environ.get("HARNESS_DATA_B")
needs_b = unittest.skipUnless(DATA_B, "set HARNESS_DATA_B to run this test")


def fake_units(per_group=4, groups=3, life=30):
    rows = []
    for g in range(1, groups + 1):
        for u in range(1, per_group + 1):
            n = life + 7 * u
            for i in range(n):
                rows.append({"id": f"Bearing{g}_{u}_s{i:04d}", "inputs": {"features": {"x": float(i)}},
                             "answer": float(n - 1 - i), "unit": f"Bearing{g}_{u}", "group": str(g), "source": "train"})
    return simdata.by_unit(rows)


class PolicyTest(unittest.TestCase):
    def test_template_passes_static_checks_and_renders(self):
        self.assertEqual(check_policy_source(TEMPLATE.read_text(encoding="utf-8")), [])
        self.assertEqual(render(None), TEMPLATE.read_text(encoding="utf-8"))
        source = render(1.0, 954.0, 954.2, 785.2, epochs=7)
        self.assertEqual(check_policy_source(source), [])
        self.assertIn('"cap_multiplier": 1.0,', source)
        self.assertIn('"epochs": 7,', source)
        with self.assertRaises(ValueError):
            render(1.0)                       # a cap needs the label statistics

    def test_cap_math(self):
        namespace = {}
        exec(compile(render(1.0, 954.0, 954.0, 785.0), "policy", "exec"), namespace)
        out = np.zeros((3, 1))
        target = np.array([-1.0, 0.0, 2.0])           # normalised; cap at the mean = 0
        _, grad = namespace["output_grad"](out, target, np.ones(3), "regression")
        np.testing.assert_allclose(grad[:, 0], 2 * (out[:, 0] - np.minimum(target, 0.0)) / 3)

    @needs_b
    def test_uncapped_template_is_byte_identical_to_policy_initial(self):
        package = Path(DATA_B)
        files = (str(package / "checkpoint.npz"), str(package / "public/train.jsonl"),
                 str(package / "public/validation.jsonl"))
        with tempfile.TemporaryDirectory() as folder:
            a = runner.train(str(ROOT / "harness" / "policy_initial.py"), *files, str(Path(folder) / "a"), max_epochs=30)
            policy = write(Path(folder) / "none.py", render(None))
            b = runner.train(policy, *files, str(Path(folder) / "b"), max_epochs=30)
        self.assertEqual(a["checkpoint_sha256"], b["checkpoint_sha256"])


class DataTest(unittest.TestCase):
    def test_folds_rotate_every_unit_through_hidden(self):
        folds = simdata.make_folds(fake_units())
        self.assertEqual(len(folds), 4)
        hidden = sorted(u for f in folds for u in f["hidden"])
        self.assertEqual(hidden, sorted(fake_units()))
        for fold in folds:
            self.assertEqual(len(fold["train"]), 6)
            self.assertEqual(len(fold["train_2c"]), 3)
            roles = [set(fold[k]) for k in ("hidden", "stream", "stream_val", "train_2c")]
            self.assertEqual(sum(len(r) for r in roles), len(set().union(*roles)))   # disjoint

    def test_segments_partition_in_time_order(self):
        rows = fake_units()["Bearing1_1"]
        parts = [simdata.segment(rows, k) for k in range(1, 11)]
        self.assertEqual([r["id"] for p in parts for r in p], [r["id"] for r in rows])
        self.assertTrue(all(parts))

    def test_protocol_invariants(self):
        units = fake_units()
        fold = simdata.make_folds(units)[0]
        hidden_ids = {r["id"] for u in fold["hidden"] for r in units[u]}
        for protocol in simdata.PROTOCOLS:
            seen_tests = []
            for k in range(1, 11):
                rr = simdata.round_rows(units, fold, protocol, k)
                train = {r["id"] for r in rr["train"]}
                validation = {r["id"] for r in rr["validation"]}
                test = {r["id"] for r in rr["test"]}
                self.assertFalse(train & validation, protocol)
                self.assertFalse(train & test, protocol)
                if protocol == "1":
                    self.assertTrue(train & hidden_ids >= set().union(*seen_tests[:-1]) if len(seen_tests) > 1 else True)
                    if seen_tests:
                        self.assertEqual({r["id"] for r in rr["newest"]}, seen_tests[-1])
                else:
                    self.assertFalse(train & hidden_ids, protocol)        # tested data never trains
                if protocol == "2c":
                    val_units = {r["unit"] for r in rr["validation"]}
                    self.assertFalse(val_units & {r["unit"] for r in rr["train"]})   # unit-disjoint
                seen_tests.append(test)


class StatisticsTest(unittest.TestCase):
    def test_advantage_and_paired_bootstrap(self):
        y = np.linspace(0, 100, 200)
        good, bad = y + 1.0, y + 10.0
        index = decide.resample_index(len(y), 300, 0)
        self.assertGreater(decide.p_better(good, bad, y, index), 0.99)
        self.assertAlmostEqual(decide.advantage(good, bad, y), 0.9, places=6)
        samples = decide.advantage_samples(bad, good, y, index)
        self.assertTrue(np.all(samples < 0))

    def test_unit_rmse_weights_units_equally(self):
        y = np.array([0.0] * 9 + [0.0])
        pred = np.array([0.0] * 9 + [10.0])
        units = np.array(["a"] * 9 + ["b"])
        self.assertAlmostEqual(decide.unit_rmse(pred, y, units), 5.0)


class SeedTest(unittest.TestCase):
    def test_eligibility(self):
        entries = preset.load()
        ids = {e["id"] for e in entries}
        self.assertIn("R-00-BASE-CAP-01", ids)
        b2 = {e["id"] for e in preset.select({"task_name": "B", "task_type": "regression", "protocol": "2b"})}
        b1 = {e["id"] for e in preset.select({"task_name": "B", "task_type": "regression", "protocol": "1"})}
        c2 = {e["id"] for e in preset.select({"task_name": "C", "task_type": "regression", "protocol": "2b"})}
        a = preset.select({"task_name": "A", "task_type": "classification", "protocol": "1"})
        self.assertNotIn("R-00-BASE-NOCAP-06", b2)
        self.assertIn("R-00-BASE-NOCAP-06", b1)
        self.assertNotIn("R-00-BASE-FEATURES-03", c2)
        self.assertNotIn("R-00-BASE-NOCOND-07", c2)      # B only: C's inputs do carry the operating settings
        self.assertIn("R-00-BASE-NOCOND-07", b2)
        self.assertEqual(a, [])

    def test_seed_file_playbook_within_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            config = {"task_name": "B", "ace": {"module": "harness.ace_stub"},
                      "playbook_path": str(Path(folder) / "playbook.md")}
            result = preset.seed(config, {"task_name": "B", "task_type": "regression", "protocol": "1"})
            text = Path(config["playbook_path"]).read_text(encoding="utf-8")
        self.assertEqual(len(result["seeded"]), 7)
        self.assertLess(len(text), 1700)                   # leaves room for learned entries in the 2400 view
        for entry_id in result["seeded"]:
            self.assertIn(f"[{entry_id}]", text)


@needs_b
class SimulationSmokeTest(unittest.TestCase):
    def test_one_fold_three_rounds(self):
        from team_sim.run import Simulation
        with tempfile.TemporaryDirectory() as folder:
            sim = Simulation(DATA_B, folder, protocols=("2c",), policies=("never", "v3"), folds=[0], rounds=3,
                             resamples=200)
            result = sim.run()
            self.assertEqual(result["runs"], 2)
            records = [json.loads(line) for line in
                       (Path(folder) / "f0" / "p2c" / "v3" / "rounds.jsonl").read_text().splitlines()]
            self.assertEqual([r["round"] for r in records], [1, 2, 3])
            self.assertTrue(all("advantage" in r["test"] for r in records))
            never = json.loads((Path(folder) / "f0" / "p2c" / "never" / "summary.json").read_text())
            self.assertEqual(never["mean_advantage"], 0.0)
            self.assertFalse((Path(folder) / "f0" / "p2c" / "data").exists())   # round files removed
            self.assertTrue((Path(folder) / "report.md").is_file())


# A stand-in for run_evolve.py: starts a "pool worker" that ignores SIGTERM (as OpenEvolve's forked
# workers do), writes its pid, then waits. MODE=graceful exits 0 on SIGTERM, MODE=stubborn ignores it.
FAKE_SEARCH = r"""
import os, signal, subprocess, sys, time
worker = subprocess.Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(120)"])
open(sys.argv[1], "w").write(str(worker.pid))
if sys.argv[2] == "graceful":
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))
else:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(120)
"""


def _gone(pid, wait=5.0):
    """True once `pid` no longer runs (absent or a zombie waiting to be reaped by init)."""
    import time
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            return True
        if state == "Z":
            return True
        time.sleep(0.05)
    return False


class SearchControlTest(unittest.TestCase):
    def _run(self, mode, seconds, grace):
        from team_sim.search import run_in_group
        with tempfile.TemporaryDirectory() as folder:
            script, pid_file = Path(folder) / "fake.py", Path(folder) / "worker.pid"
            script.write_text(FAKE_SEARCH, encoding="utf-8")
            result = run_in_group([sys.executable, str(script), str(pid_file), mode], seconds, grace=grace,
                                  stdout=subprocess_devnull(), stderr=subprocess_devnull())
            return result, int(pid_file.read_text())

    def test_time_limit_stops_gracefully_and_kills_leftover_workers(self):
        (code, stopped), worker = self._run("graceful", seconds=1, grace=10)
        self.assertEqual((code, stopped), (0, "time_limit"))
        self.assertTrue(_gone(worker), "pool worker survived the search")

    def test_unresponsive_search_is_killed_after_grace(self):
        (code, stopped), worker = self._run("stubborn", seconds=1, grace=1)
        self.assertEqual(stopped, "killed")
        self.assertEqual(code, -9)
        self.assertTrue(_gone(worker))

    def test_finished_search_is_untouched(self):
        from team_sim.search import run_in_group
        code, stopped = run_in_group([sys.executable, "-c", "import sys; sys.exit(3)"], 30)
        self.assertEqual((code, stopped), (3, None))

    def test_ace_state_is_separate_per_run_and_policy(self):
        from types import SimpleNamespace
        from team_sim.search import Search
        def root(out, policy):
            return Search.ace_root(SimpleNamespace(out=Path(out), _run_dir=Path(out) / "f0" / "p2c" / policy))
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / "runs" / "B-live-01", Path(folder) / "runs" / "B-live-02"
            roots = {root(a, "v3"), root(a, "v3_cap1"), root(b, "v3")}
            self.assertEqual(len(roots), 3)
            self.assertEqual(root(a, "v3"), root(a, "v3"))
            for r in roots:
                self.assertIn(ROOT / "state" / "sim", r.parents)
                self.assertEqual(r.name.split("-")[0], "f0")

    def test_ace_worker_time_must_cover_two_calls(self):
        from team_sim.search import Search
        with self.assertRaises(ValueError):
            Search("http://127.0.0.1:9/v1", "m", ace_mode="live", ace_approve_live=True, ace_max_calls=6,
                   ace_worker_seconds=120, ace_request_timeout=300)


class AceTransportSettingsTest(unittest.TestCase):
    def test_defaults_unchanged_and_overrides_checked(self):
        from team_ace.backend import request_limits, strip_reasoning
        self.assertEqual(request_limits({}), (20, 1024))
        self.assertEqual(request_limits({"request_timeout_s": 300, "max_tokens": 4096}), (300, 4096))
        for bad in ({"request_timeout_s": 0}, {"request_timeout_s": "300"}, {"max_tokens": 10 ** 6}):
            with self.assertRaises(ValueError):
                request_limits(bad)
        self.assertEqual(strip_reasoning("<think>\nlong\nreasoning</think>\n{\"a\": 1}"), "{\"a\": 1}")
        self.assertEqual(strip_reasoning("{\"a\": \"<think>x</think>\"}"), "{\"a\": \"<think>x</think>\"}")
        self.assertIsNone(strip_reasoning(None))


class AceThinkingSwitchTest(unittest.TestCase):
    def test_switch_is_sent_only_when_set(self):
        from team_ace.backend import thinking_body
        self.assertIsNone(thinking_body({}))
        self.assertEqual(thinking_body({"enable_thinking": False}), {"enable_thinking": False})
        with self.assertRaises(ValueError):
            thinking_body({"enable_thinking": "false"})

    def test_search_config_carries_the_switch(self):
        from types import SimpleNamespace
        from unittest import mock
        from team_sim.search import Search
        with tempfile.TemporaryDirectory() as folder, mock.patch.dict(os.environ, {"OPENEVOLVE_API_KEY": "x"}):
            folder = Path(folder)
            for name in ("train.jsonl", "validation.jsonl", "checkpoint.npz"):
                (folder / name).write_text(name, encoding="utf-8")
            out = folder / "runs" / "X"
            sim = SimpleNamespace(task="B", files={"train": str(folder / "train.jsonl"),
                                                   "validation": str(folder / "validation.jsonl")},
                                  max_epochs=5, out=out, _run_dir=out / "f0" / "p2c" / "v3")
            work = sim._run_dir / "round-01" / "search"
            for thinking, expected in ((None, "absent"), (False, False)):
                searcher = Search("http://127.0.0.1:9/v1", "m", ace_mode="fixture", ace_max_calls=6, ace_thinking=thinking)
                config = searcher._run_config(sim, 1, folder / "checkpoint.npz", work)
                self.assertEqual(config["ace"].get("enable_thinking", "absent"), expected)
                self.assertEqual(config["ace"]["render"], {"base_first": True, "max_learned": 3, "max_chars": 2400})
            config = Search("http://127.0.0.1:9/v1", "m", ace_max_shown=1)._run_config(sim, 1, folder / "checkpoint.npz", work)
            self.assertEqual(config["ace"]["render"]["max_learned"], 1)
            with self.assertRaises(ValueError):
                Search("http://127.0.0.1:9/v1", "m", ace_thinking="off")
            with self.assertRaises(ValueError):
                Search("http://127.0.0.1:9/v1", "m", ace_max_shown=50)


def _openai_available():
    import importlib.util
    return importlib.util.find_spec("openai") is not None


@unittest.skipUnless(_openai_available(), "the openai package is not installed")
class AceLiveTransportTest(unittest.TestCase):
    """team_ace's live transport against a local fake endpoint (no real model, no network)."""

    def setUp(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.server.requests.append(body)
                reply = self.server.reply.pop(0) if isinstance(self.server.reply, list) else self.server.reply
                data = json.dumps({"id": "fake", "object": "chat.completion", "created": 0, "model": body.get("model"),
                                   "choices": [{"index": 0, "finish_reason": reply["finish_reason"],
                                                "message": {"role": "assistant", "content": reply["content"],
                                                            "reasoning_content": "fake thinking"}}],
                                   "usage": {"prompt_tokens": 10, "completion_tokens": reply["completion_tokens"],
                                             "total_tokens": 10 + reply["completion_tokens"]}}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.requests, self.server.reply = [], {}
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    ENV = {"TEAM_SIM_FAKE_KEY": "not-a-real-key", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}

    def setting(self, **ace):
        import sqlite3
        db = sqlite3.connect(":memory:")
        db.execute("CREATE TABLE calls(id INTEGER PRIMARY KEY, job TEXT, role TEXT, mode TEXT, status TEXT, "
                   "usage TEXT, created REAL)")
        settings = {"mode": "live", "live_calls_approved": True, "max_total_calls": 5, "model": "fake",
                    "api_base": f"http://127.0.0.1:{self.server.server_address[1]}/v1",
                    "api_key_env": "TEAM_SIM_FAKE_KEY", "request_timeout_s": 10, "max_tokens": 512}
        return db, {"task_name": "B", "ace": dict(settings, **ace)}

    def call(self, reply, **ace):
        from unittest import mock
        from team_ace.backend import Transport
        self.server.reply = reply
        db, config = self.setting(**ace)
        transport = Transport(db, config, "job-1", [])
        with mock.patch.dict(os.environ, self.ENV):
            try:
                return transport(None, "adapter", "fake", "prompt", "reflector", "c1", max_tokens=4096), db
            except Exception as error:   # returned so the test can look at the failure and the calls table
                return error, db

    def test_no_thinking_is_sent_and_the_reply_accepted(self):
        (response, info), db = self.call({"content": '{"ok": true}', "finish_reason": "stop", "completion_tokens": 6},
                                         enable_thinking=False)
        self.assertEqual(response, '{"ok": true}')
        self.assertIs(self.server.requests[-1]["enable_thinking"], False)
        self.assertEqual(self.server.requests[-1]["max_tokens"], 512)
        self.assertEqual(db.execute("SELECT status FROM calls").fetchone()[0], "complete")

    def test_default_request_has_no_switch(self):
        self.call({"content": '{"ok": true}', "finish_reason": "stop", "completion_tokens": 6})
        self.assertNotIn("enable_thinking", self.server.requests[-1])

    def propose(self, *texts):
        """Reflector + Curator through the upstream roles; each reply comes from the fake endpoint in turn."""
        from unittest import mock
        from team_ace.backend import propose
        self.server.reply = [{"content": text, "finish_reason": "stop", "completion_tokens": 100} for text in texts]
        db, config = self.setting(enable_thinking=False)
        with mock.patch.dict(os.environ, self.ENV):
            return propose(db, config, "job-1", [{"evidence_id": "e1", "task_name": "B", "outcome": "improved"}],
                           "# B 任务候选经验\n")

    REFLECTOR = json.dumps({"reflection": "模拟反思", "bullet_tags": []}, ensure_ascii=False)
    CURATOR = json.dumps({"reasoning": "模拟", "operations": [{"type": "ADD", "section": "STRATEGIES", "content": json.dumps(
        {"task": "B", "symptom": "s", "action": "a", "scope_note": "n", "evidence_ids": ["e1"]})}]}, ensure_ascii=False)

    def test_replies_wrapped_in_a_code_block_or_text_are_accepted_like_upstream(self):
        for reflector, curator in ((self.REFLECTOR, "```json\n" + self.CURATOR + "\n```"),
                                   ("```json\n" + self.REFLECTOR + "\n```", "下面是操作：\n" + self.CURATOR)):
            with self.subTest(curator=curator[:12]):
                operations, artifact = self.propose(reflector, curator)
                self.assertEqual(len(operations), 1)

    # What Qwen3.8-27B wrote as ADD content on captain (2026-11-24 ace-retry, ②c round 10), copying the
    # playbook's style; the second one is the 200-character preview of the error message.
    QWEN_1 = ("待验证｜B｜症状假设：模型可能因‘假设非事实’约束而拒绝输出或混淆反馈与现有条目；建议：无论反馈是否匹配现有条目，"
              "必须生成结构化JSON；将‘未证明’状态体现在scope_note或action措辞中（如‘候选建议’），而非拒绝回答；直接从反馈JSON提取"
              "task和evidence_ids；范围：所有基于反馈生成建议的任务；证据：reflection-01；结果关联：防止因过度谨慎导致的格式错误（")
    QWEN_2 = ("待验证｜B｜症状假设：收到包含具体实验结果（如 epochs 5->10, RMSE 改善）的 'Environment Feedback' 时，模型可能因 "
              "'Hypotheses are not facts' 约束而错误拒绝生成建议；建议：必须将提供的 Feedback JSON 视为唯一事实来源，提取 "
              "task_name, evidence_id, changes, outcome 和 fla")

    def test_reply_without_operations_fails_the_job_but_prose_reflection_is_fine(self):
        with self.assertRaises(ValueError):
            self.propose(self.REFLECTOR, "I cannot produce any operations.")
        operations, artifact = self.propose("这次改动有效：提高 epochs 后 RMSE 下降。", self.CURATOR)
        self.assertEqual(len(operations), 1)

    def test_memory_content_reads_the_ways_models_write(self):
        from team_ace.backend import memory_content
        one = memory_content(self.QWEN_1)
        self.assertEqual(one["task"], "B")
        self.assertTrue(one["symptom"].startswith("模型可能因"))
        self.assertIn("而非拒绝回答；直接从反馈JSON提取task和evidence_ids", one["action"])   # '；' inside a part stays
        self.assertEqual(one["scope_note"], "所有基于反馈生成建议的任务")
        self.assertTrue(memory_content(self.QWEN_2)["action"].startswith("必须将提供的 Feedback JSON"))
        same = {"symptom": "验证误差上升", "action": "先降学习率", "scope_note": "仅 B"}
        for written in (same, json.dumps(same, ensure_ascii=False), "```json\n" + json.dumps(same) + "\n```",
                        {"症状假设": "验证误差上升", "建议": "先降学习率", "范围": "仅 B"},
                        "症状假设：验证误差上升；建议：先降学习率；范围：仅 B"):
            with self.subTest(written=str(written)[:30]):
                self.assertEqual({k: memory_content(written)[k] for k in same}, same)
        self.assertEqual(memory_content("增加训练轮数可以提高效果")["action"], "增加训练轮数可以提高效果")
        for useless in ("", "症状假设：只有现象没有做法", {"symptom": "x"}, None, ["a"]):
            with self.subTest(useless=useless), self.assertRaises(ValueError):
                memory_content(useless)

    def test_worker_writes_memory_from_qwen_style_replies(self):
        """process_one end to end in live mode against the fake endpoint, with the content Qwen wrote on
        captain, an object with a made-up evidence ID and an extra key, a cross-task bullet, and none."""
        import shutil
        import uuid
        from unittest import mock
        from team_ace.store import BASE, connect, digest, initialize, on_batch, scope_for, status
        from team_ace.worker import process_one
        root = BASE / "state" / "tests" / ("ace-live-" + uuid.uuid4().hex[:8])

        def record(i):
            return {"schema": "harness.iteration.v1", "run_id": "live-test", "stage_id": config["stage_id"],
                    "task_name": "B", "iteration": i, "kind": "evaluated", "status": "ok", "outcome": "improved",
                    "program_id": str(i), "hypothesis": f"假设{i}：把 epochs 从 5 提到 10 能降低 RMSE",
                    "changes": "Change 1: epochs 5 -> 10",
                    "primary": {"name": "rmse", "candidate": 150.0, "parent_program": 160.0},
                    "used_bullets": [], "flags": ["LATE_EPOCH_DEGRADATION"], "guard_violations": []}

        def curator(*contents):
            return {"content": json.dumps({"reasoning": "r", "operations": [
                {"type": "ADD", "section": "STRATEGIES", "content": c} for c in contents]}, ensure_ascii=False),
                "finish_reason": "stop", "completion_tokens": 50}

        def run(i, reply):
            on_batch([record(i)], config)
            self.server.reply = [{"content": "反思：提高 epochs 有效。", "finish_reason": "stop", "completion_tokens": 50},
                                 reply]
            with mock.patch.dict(os.environ, self.ENV):
                return process_one(config)["status"]

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name in ("train.jsonl", "validation.jsonl", "checkpoint.npz"):
                (folder / name).write_text(name, encoding="utf-8")
            _, config = self.setting(enable_thinking=False, max_total_calls=20)
            config.update(stage_id="sim-B-live-r01", train_path=str(folder / "train.jsonl"),
                          validation_path=str(folder / "validation.jsonl"),
                          parent_checkpoint=str(folder / "checkpoint.npz"), work_dir=str(folder / "work"),
                          seed=42, split={"min_groups": 4})
            config["ace"].update(module="team_ace", batch_size=1, storage_root=str(root), record_origin="simulation")
            config["ace"]["scope"] = scope_for(config)
            config["playbook_path"] = str(root / "B" / digest(config["ace"]["scope"])[:16] / "playbook.md")
            try:
                initialize(config)
                self.assertEqual(run(0, curator(self.QWEN_1, self.QWEN_2)), "done")
                reflector_prompt = self.server.requests[-2]["messages"][0]["content"]
                self.assertIn("假设0：把 epochs 从 5 提到 10", reflector_prompt)      # hypothesis = reasoning trace
                self.assertIn("Change 1: epochs 5 -> 10", reflector_prompt)        # code change = answer
                self.assertIn("outcome=improved", reflector_prompt)                # measured result = feedback
                self.assertNotIn("No supervised answer", reflector_prompt)
                self.assertIn("Keep each part under 100 characters", self.server.requests[-1]["messages"][0]["content"])
                db = connect(config)
                first = db.execute("SELECT id FROM records").fetchone()[0]
                entries = [json.loads(b) for (b,) in db.execute("SELECT body FROM entries")]
                db.close()
                self.assertEqual(len(entries), 2)
                self.assertTrue(all(e["evidence_ids"] == [first] for e in entries))   # never "reflection-01"
                self.assertTrue(all(len(e[k]) <= 250 for e in entries for k in ("symptom", "action", "scope_note")))
                self.assertTrue(all(e["updated"] >= e["created"] > 0 for e in entries))
                replies = sorted((Path(config["playbook_path"]).parent / "replies").glob("*.json"))
                self.assertEqual([p.name.split("-", 1)[1] for p in replies], ["reflector.json", "curator.json"])
                self.assertEqual(run(1, curator({"task": "B", "symptom": "验证误差上升", "action": "先降学习率",
                                                 "evidence_ids": ["reflection-01"], "confidence": "high"})), "done")
                db = connect(config)
                newest = [json.loads(b) for (b,) in db.execute("SELECT body FROM entries")
                          if json.loads(b)["action"] == "先降学习率"][0]
                db.close()
                self.assertNotIn("confidence", newest)
                self.assertNotIn("reflection-01", newest["evidence_ids"])
                self.assertEqual(run(2, curator("待验证｜A｜症状假设：x；建议：y；范围：z")), "failed")
                self.assertEqual(run(3, curator("症状假设：只有现象")), "failed")
                db = connect(config)
                errors = [e for (e,) in db.execute("SELECT error FROM jobs WHERE status='failed' ORDER BY rowid")]
                db.close()
                self.assertIn("cross-task memory rejected", errors[0])
                self.assertIn("no usable advice", errors[1])
                self.assertIn("只有现象", errors[1])
                self.assertEqual(status(config)["entries"], 3)
            finally:
                shutil.rmtree(root, ignore_errors=True)

    def test_answer_cut_off_while_thinking_is_reported_and_its_tokens_kept(self):
        error, db = self.call({"content": "", "finish_reason": "length", "completion_tokens": 512})
        self.assertIsInstance(error, ValueError)
        self.assertIn("cut off", str(error))
        state, usage = db.execute("SELECT status, usage FROM calls").fetchone()
        self.assertEqual(state, "ValueError")
        self.assertEqual(json.loads(usage)["completion_tokens"], 512)


class AceRetryTest(unittest.TestCase):
    def test_retry_processes_a_copy_and_leaves_the_round_alone(self):
        import shutil
        import uuid
        from team_ace.store import BASE, digest, initialize, on_batch, scope_for, status
        from team_sim.ace_retry import describe, retry
        root = BASE / "state" / "tests" / ("ace-retry-" + uuid.uuid4().hex[:8])
        result = None
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            for name in ("train.jsonl", "validation.jsonl", "checkpoint.npz"):
                (folder / name).write_text(name, encoding="utf-8")
            config = {"task_name": "B", "stage_id": "sim-B-test-r01", "train_path": str(folder / "train.jsonl"),
                      "validation_path": str(folder / "validation.jsonl"),
                      "parent_checkpoint": str(folder / "checkpoint.npz"), "work_dir": str(folder / "work"),
                      "seed": 42, "split": {"min_groups": 4},
                      "ace": {"module": "team_ace", "batch_size": 1, "mode": "fixture", "storage_root": str(root),
                              "record_origin": "simulation", "max_total_calls": 6, "live_calls_approved": False,
                              "model": None, "api_base": None, "api_key_env": None}}
            config["ace"]["scope"] = scope_for(config)
            config["playbook_path"] = str(root / "B" / digest(config["ace"]["scope"])[:16] / "playbook.md")
            try:
                initialize(config)
                on_batch([{"schema": "harness.iteration.v1", "run_id": "retry-test", "stage_id": config["stage_id"],
                           "task_name": "B", "iteration": 0, "kind": "evaluated", "status": "ok",
                           "outcome": "equal", "program_id": "0", "hypothesis": "模拟输入。",
                           "primary": {"name": "rmse"}, "used_bullets": [], "flags": [], "guard_violations": []}],
                         config)
                search = folder / "round-01" / "search"
                search.mkdir(parents=True)
                (search / "run.json").write_text(json.dumps(config), encoding="utf-8")
                live = dict(config, ace=dict(config["ace"], mode="live"))
                (folder / "live").mkdir()
                (folder / "live" / "run.json").write_text(json.dumps(live), encoding="utf-8")
                with self.assertRaises(ValueError):          # live needs explicit approval
                    retry(folder / "live", request_timeout=5, worker_seconds=60)
                with self.assertRaises(ValueError):
                    retry(search, request_timeout=5, worker_seconds=60, max_calls=50)
                result = retry(search, thinking=False, request_timeout=5, worker_seconds=60, max_jobs=2, max_calls=8)
                if any(w["stop_reason"] == "memory_limit" for w in result["workers"]):
                    self.skipTest("too little free memory for the bounded ACE worker on this machine")
                self.assertEqual([w["exit_code"] for w in result["workers"]], [0])
                copied = json.loads(Path(result["config"]).read_text(encoding="utf-8"))
                self.assertIs(copied["ace"]["enable_thinking"], False)
                self.assertEqual(copied["ace"]["max_total_calls"], 8)
                self.assertEqual(status(copied)["jobs"], {"done": 1})
                self.assertEqual(status(copied)["entries"], 1)
                self.assertEqual(status(config)["jobs"], {"pending": 1})   # the round itself is untouched
                self.assertEqual(status(config)["entries"], 0)
                self.assertEqual(status(config)["calls"], {})
                self.assertTrue(any(line.startswith("新经验") for line in describe(result)))
            finally:
                shutil.rmtree(root, ignore_errors=True)
                if result:
                    shutil.rmtree(Path(result["config"]).parent, ignore_errors=True)

    @needs_b
    def test_deleted_round_data_are_rebuilt_and_checked(self):
        from team_sim.ace_retry import _sha256, rebuild_round_data
        units = simdata.by_unit(simdata.load_rows(DATA_B, "B"))
        fold = next(f for f in simdata.make_folds(units) if f["fold"] == 0)
        rows = simdata.round_rows(units, fold, "2c", 3)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            original = {"train_path": simdata.write_jsonl(folder / "run" / "train.jsonl", rows["train"]),
                        "validation_path": simdata.write_jsonl(folder / "run" / "validation.jsonl", rows["validation"])}
            config = {"task_name": "B", "ace": {"scope": {"source_hashes": {k: _sha256(v) for k, v in original.items()}}}}
            runs = folder / "runs" / "X" / "f0" / "p2c" / "v3"
            paths = rebuild_round_data(config, runs / "round-03" / "search", DATA_B, folder / "rebuilt")
            self.assertEqual({k: _sha256(v) for k, v in paths.items()}, config["ace"]["scope"]["source_hashes"])
            with self.assertRaises(ValueError):       # another round's data do not match the recorded hashes
                rebuild_round_data(config, runs / "round-04" / "search", DATA_B, folder / "other")
            with self.assertRaises(ValueError):
                rebuild_round_data(config, folder / "not-a-round", DATA_B, folder / "other")


class PlaybookRenderTest(unittest.TestCase):
    """team_ace's rendered playbook: preset skills first, then the newest learned entries (ace.render),
    and the old ID order up to 1900 characters when ace.render is absent (A/B/C runs)."""

    def scope(self, folder, render):
        import uuid
        from team_ace.store import BASE, digest, initialize, scope_for
        for name in ("train.jsonl", "validation.jsonl", "checkpoint.npz"):
            (folder / name).write_text(name, encoding="utf-8")
        root = BASE / "state" / "tests" / ("render-" + uuid.uuid4().hex[:8])
        config = {"task_name": "B", "stage_id": "render", "train_path": str(folder / "train.jsonl"),
                  "validation_path": str(folder / "validation.jsonl"), "parent_checkpoint": str(folder / "checkpoint.npz"),
                  "work_dir": str(folder / "w"), "seed": 42, "split": {"min_groups": 4},
                  "ace": {"module": "team_ace", "batch_size": 1, "mode": "live", "storage_root": str(root),
                          "record_origin": "simulation", "max_total_calls": 6, "live_calls_approved": True}}
        if render:
            config["ace"]["render"] = render
        config["ace"]["scope"] = scope_for(config)
        config["playbook_path"] = str(root / "B" / digest(config["ace"]["scope"])[:16] / "playbook.md")
        initialize(config)
        preset.seed(config, {"task_name": "B", "task_type": "regression", "protocol": "2c"})
        return root, config

    def learn(self, config, count):
        from team_ace.store import connect, digest, encoded, publish
        db = connect(config)
        try:
            with db:
                for i in range(count):
                    eid = f"B-{i:012x}"
                    db.execute("INSERT OR REPLACE INTO entries VALUES (?,?)", (eid, encoded({
                        "id": eid, "task": "B", "symptom": f"现象{i}" * 20, "action": f"做法{i}" * 30, "scope_note": "回归",
                        "evidence_ids": ["e" * 64], "outcomes": {"improved": 1}, "scope_id": digest(config["ace"]["scope"]),
                        "status": "candidate", "mode": "live", "independent_validation": False, "updated": 1000.0 + i})))
            publish(config, db)
        finally:
            db.close()
        text = Path(config["playbook_path"]).read_text(encoding="utf-8")
        return text, [line.split("]")[0][3:] for line in text.splitlines() if line.startswith("- [")]

    def test_preset_skills_first_then_newest_learned(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            root, config = self.scope(Path(folder), {"base_first": True, "max_learned": 3, "max_chars": 2400})
            try:
                text, ids = self.learn(config, 5)
                presets = [i for i in ids if i.startswith("R-00-BASE")]
                self.assertEqual(len(presets), len(preset.select({"task_name": "B", "task_type": "regression",
                                                                  "protocol": "2c"})))       # every preset is shown
                self.assertEqual(ids[:len(presets)], sorted(presets))                         # presets first
                learned = ids[len(presets):]
                self.assertEqual(learned, ["B-000000000004", "B-000000000003", "B-000000000002"][:len(learned)])
                self.assertGreaterEqual(len(learned), 2)
                self.assertLessEqual(len(text), 2400)
            finally:
                shutil.rmtree(root, ignore_errors=True)

    def test_default_rendering_is_unchanged(self):
        import shutil
        from team_ace.store import render_settings
        with tempfile.TemporaryDirectory() as folder:
            root, config = self.scope(Path(folder), None)
            try:
                text, ids = self.learn(config, 5)
                self.assertEqual(ids, sorted(ids))            # plain ID order: learned 'B-...' before the presets
                self.assertTrue(ids[0].startswith("B-"))
                self.assertLessEqual(len(text), 1900)
            finally:
                shutil.rmtree(root, ignore_errors=True)
        self.assertIsNone(render_settings({}))
        for bad in ({"base_first": False}, {"base_first": True, "max_learned": -1},
                    {"base_first": True, "max_chars": 100000}, {"base_first": True, "max_learned": "3"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                render_settings({"render": bad})


class ReportTest(unittest.TestCase):
    def test_old_summaries_without_new_columns(self):
        from team_sim.report import COLUMNS, build
        with tempfile.TemporaryDirectory() as folder:
            for fold, extra in ((0, {}), (1, {"search_adopted": 1})):
                summary = {key: 0.0 for key, _, _ in COLUMNS if key not in ("search_adopted", "search_time_limited")}
                summary.update(extra, fold=fold, protocol="1", policy="v3")
                path = Path(folder) / f"f{fold}" / "p1" / "v3" / "summary.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(summary), encoding="utf-8")
            build(folder)
            text = (Path(folder) / "report.md").read_text(encoding="utf-8")
            self.assertIn("搜索候选上线", text)
            self.assertIn("—", text)


def subprocess_devnull():
    import subprocess
    return subprocess.DEVNULL


if __name__ == "__main__":
    unittest.main()
