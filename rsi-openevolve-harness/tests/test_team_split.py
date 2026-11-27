"""Tests for team_split (the formal data-split module).

  python -m unittest tests.test_team_split -v

Tests marked with a package need HARNESS_DATA_A / _B / _C (see tests/test_harness.py);
everything else runs on small synthetic data and synthetic checkpoints.
"""
import contextlib
import copy
import io
import json
import os
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

from baselines.data import read_dataset  # noqa: E402
from baselines.model import Model  # noqa: E402
from harness.dataio import file_sha256  # noqa: E402
from harness.splits import MANIFEST_KEYS, make_split, manifest_digest  # noqa: E402
from team_split import (LabeledTable, SplitOptions, build_manifest, derive_run_config, load_table,  # noqa: E402
                        read_manifest, split, verify_manifest)
from team_split.__main__ import main as cli  # noqa: E402

PACKAGES = {name: os.environ.get(f"HARNESS_DATA_{name}") for name in "ABC"}


def package(name):
    return unittest.skipUnless(PACKAGES[name], f"set HARNESS_DATA_{name} to run this test")


# ------------------------------------------------------------------ synthetic data

def table_of(task, meta_rows, targets, class_names=None, sha="sha256:" + "0" * 64):
    ids = [f"s{i:04d}" for i in range(len(meta_rows))]
    dtype = np.int64 if task == "classification" else np.float64
    return LabeledTable(ids=ids, targets=np.asarray(targets, dtype=dtype), meta_rows=meta_rows, task_type=task,
                        class_names=list(class_names or []), source_path="synthetic", source_sha256=sha)


def classes_in_time():
    """Classification measured class by class, like the gas data; no usable meta."""
    targets = [0] * 10 + [1] * 6 + [2] * 5 + [0] * 4
    meta = [{"batch": 1, "row": i} for i in range(len(targets))]
    return table_of("classification", meta, targets, ["a", "b", "c"])


def engines(subsets=("FD001", "FD003"), units=(1, 2, 3, 4), cycles=5, sizes=None):
    """Regression with a composite engine key: unit numbers repeat across subsets."""
    meta, targets = [], []
    for s, subset in enumerate(subsets):
        for u, unit in enumerate(units):
            n = sizes[(s * len(units) + u) % len(sizes)] if sizes else cycles
            for cycle in range(n):
                meta.append({"subset": subset, "unit": unit, "cycle": cycle})
                targets.append(float(n - cycle - 1))
    return table_of("regression", meta, targets)


def plain_regression(n=40):
    return table_of("regression", [{} for _ in range(n)], [float(i % 7) for i in range(n)])


def few_groups():
    meta = [{"bearing": f"b{i // 10}"} for i in range(30)]
    return table_of("regression", meta, [float(i) for i in range(30)])


def redigest(manifest):
    manifest = copy.deepcopy(manifest)
    manifest["manifest_sha256"] = manifest_digest(manifest)
    return manifest


def codes(problems):
    return {p.code for p in problems}


class StrategyTest(unittest.TestCase):
    def assert_same_as_provisional(self, table, options=SplitOptions()):
        ours = build_manifest(table, split(table, options), "r1", options)
        reference = make_split(table.ids, table.targets, table.meta_rows, table.task_type, "r1", table.source_sha256,
                               seed=options.seed, min_groups=options.min_groups,
                               class_names=table.class_names or None, time_blocks=options.time_blocks)
        self.assertEqual(ours["select_ids"], reference["select_ids"])
        self.assertEqual(ours["gate_ids"], reference["gate_ids"])
        self.assertEqual(ours["method"], reference["method"])
        self.assertEqual(ours["counts"], reference["counts"])
        self.assertEqual(tuple(ours), MANIFEST_KEYS)
        self.assertFalse(ours["provisional"])
        return ours

    def test_class_time(self):
        manifest = self.assert_same_as_provisional(classes_in_time())
        self.assertEqual(manifest["method"], "class_time")
        self.assertEqual(manifest["counts"]["select"]["labels"], {"a": 7, "b": 3, "c": 3})
        self.assertEqual(verify_manifest(manifest, classes_in_time(), "r1", SplitOptions()), [])

    def test_time_blocks(self):
        manifest = self.assert_same_as_provisional(plain_regression())
        self.assertEqual(manifest["method"], "time_blocks")
        self.assertEqual(manifest["select_ids"][:4], ["s0000", "s0001", "s0002", "s0003"])

    def test_group_with_composite_key_and_coarse_key(self):
        table = engines()
        manifest = self.assert_same_as_provisional(table)
        self.assertEqual(manifest["method"], "group_stratified")
        self.assertEqual(manifest["group_key"], ["subset", "unit"])
        self.assertEqual(manifest["counts"]["select"]["groups"] + manifest["counts"]["gate"]["groups"], 8)

    def test_group_without_coarse_key(self):
        table = table_of("regression", [{"engine": f"e{i // 5}"} for i in range(40)], [float(i) for i in range(40)])
        manifest = self.assert_same_as_provisional(table)
        self.assertEqual(manifest["method"], "group")

    def test_too_few_groups(self):
        manifest = self.assert_same_as_provisional(few_groups())
        self.assertEqual(manifest["method"], "no_split")
        self.assertEqual(manifest["select_ids"], manifest["gate_ids"])
        self.assertTrue(manifest["warnings"])

    def test_forced_methods(self):
        with self.assertRaises(ValueError):
            split(plain_regression(), SplitOptions(method="class_time"))
        with self.assertRaises(ValueError):
            split(plain_regression(), SplitOptions(method="group"))
        with self.assertRaises(ValueError):
            split(few_groups(), SplitOptions(method="group"))
        forced = split(engines(), SplitOptions(method="no_split"))
        self.assertEqual(forced.method, "no_split")
        self.assertTrue(forced.warnings)
        self.assertEqual(split(engines(), SplitOptions(method="group")).method, "group")
        with self.assertRaises(ValueError):
            SplitOptions(method="kfold").validate()

    def test_deterministic_and_seed_dependent(self):
        table = engines(units=tuple(range(1, 9)))
        first = build_manifest(table, split(table, SplitOptions()), "r1", SplitOptions())
        second = build_manifest(table, split(table, SplitOptions()), "r1", SplitOptions())
        self.assertEqual(first, second)
        other = build_manifest(table, split(table, SplitOptions(seed=7)), "r1", SplitOptions(seed=7))
        self.assertNotEqual(first["select_ids"], other["select_ids"])

    def test_row_balancing(self):
        table = engines(units=tuple(range(1, 9)), sizes=[30, 2, 25, 3, 20, 4, 15, 5, 12, 6])
        default = build_manifest(table, split(table, SplitOptions()), "r1", SplitOptions())
        options = SplitOptions(balance="rows")
        balanced = build_manifest(table, split(table, options), "r1", options)
        gap = lambda m: abs(m["counts"]["select"]["rows"] - m["counts"]["gate"]["rows"])  # noqa: E731
        self.assertLessEqual(gap(balanced), gap(default))
        self.assertLess(gap(balanced), gap(default))
        self.assertEqual(balanced["method"], "group_stratified_rows")
        self.assertEqual(balanced["warnings"], [])  # recorded in the method, see strategies.py
        self.assertEqual(verify_manifest(balanced, table, "r1", options), [])
        self.assertIn("NOT_REPRODUCIBLE", codes(verify_manifest(balanced, table, "r1", SplitOptions())))


class VerifyTest(unittest.TestCase):
    def setUp(self):
        self.table = engines()
        self.manifest = build_manifest(self.table, split(self.table), "r1", SplitOptions())
        self.ctable = classes_in_time()
        self.cmanifest = build_manifest(self.ctable, split(self.ctable), "r1", SplitOptions())

    def check(self, mutate, code, table=None, base=None, digest=True, **kwargs):
        manifest = copy.deepcopy(base or self.manifest)
        mutate(manifest)
        if digest:
            manifest = redigest(manifest)
        found = codes(verify_manifest(manifest, table or self.table, **kwargs))
        self.assertIn(code, found)

    def test_valid_manifest_has_no_problems(self):
        self.assertEqual(verify_manifest(self.manifest, self.table, "r1", SplitOptions()), [])

    def test_every_problem_code(self):
        select, gate = self.manifest["select_ids"], self.manifest["gate_ids"]
        self.check(lambda m: m.pop("counts"), "KEYS_MISSING")
        self.check(lambda m: m.update(method="kfold"), "TYPE_INVALID")
        self.check(lambda m: m.update(select_ids="s0000"), "TYPE_INVALID")
        self.check(lambda m: m.update(seed=43), "DIGEST_MISMATCH", digest=False)
        self.check(lambda m: m.update(source_sha256="sha256:" + "1" * 64), "SOURCE_MISMATCH")
        self.check(lambda m: m.update(task_type="classification"), "TASK_MISMATCH")
        self.check(lambda m: None, "STAGE_MISMATCH", stage_id="r2")
        self.check(lambda m: m["select_ids"].append(select[0]), "DUPLICATE_ID")
        self.check(lambda m: m["gate_ids"].append("ghost"), "UNKNOWN_ID")
        self.check(lambda m: m.update(gate_ids=[]), "EMPTY_SIDE")
        self.check(lambda m: m["gate_ids"].append(select[0]), "OVERLAP")
        self.check(lambda m: m["gate_ids"].remove(gate[0]), "NOT_COVERING")
        self.check(lambda m: m["counts"]["select"].update(rows=1), "COUNTS_MISMATCH")
        self.check(lambda m: m.update(group_key=["subset", "nope"]), "GROUP_KEY_INVALID")
        self.check(lambda m: m.update(group_key=None), "GROUP_KEY_INVALID")

        def straddle(m):
            moved = m["select_ids"].pop()  # last row of an engine goes to the other side
            m["gate_ids"].append(moved)
        self.check(straddle, "GROUP_STRADDLES")
        self.check(lambda m: None, "NOT_REPRODUCIBLE", options=SplitOptions(seed=7))

    def test_no_split_rules(self):
        table = few_groups()
        manifest = build_manifest(table, split(table), "r1", SplitOptions())
        self.assertEqual(verify_manifest(manifest, table, "r1", SplitOptions()), [])
        self.check(lambda m: m["gate_ids"].pop(), "NO_SPLIT_INCOMPLETE", table=table, base=manifest)
        self.check(lambda m: m.update(warnings=[]), "NO_SPLIT_INCOMPLETE", table=table, base=manifest)

    def test_class_rules(self):
        select, gate = self.cmanifest["select_ids"], self.cmanifest["gate_ids"]

        def swap(m):  # an early sample of class "a" moves to the gate, a late one to selection
            m["select_ids"].remove(select[0])
            m["gate_ids"].remove(gate[0])
            m["select_ids"].append(gate[0])
            m["gate_ids"].append(select[0])
            m["counts"] = {"select": dict(m["counts"]["select"]), "gate": dict(m["counts"]["gate"])}
        self.check(swap, "CLASS_ORDER_VIOLATED", table=self.ctable, base=self.cmanifest)

        def drop_class_c(m):
            moving = [i for i in m["gate_ids"] if self.ctable.labels()[self.ctable.ids.index(i)] == "c"]
            m["gate_ids"] = [i for i in m["gate_ids"] if i not in moving]
            m["select_ids"] += moving
        manifest = copy.deepcopy(self.cmanifest)
        drop_class_c(manifest)
        problems = verify_manifest(redigest(manifest), self.ctable)
        missing = [p for p in problems if p.code == "CLASS_MISSING_ON_SIDE"]
        self.assertTrue(missing)
        self.assertEqual(missing[0].severity, "warning")

    def test_garbage_does_not_raise(self):
        self.assertIn("TYPE_INVALID", codes(verify_manifest(["not", "a", "dict"], self.table)))
        self.assertTrue(verify_manifest({}, self.table))


# ------------------------------------------------------- synthetic files on disk

def write_synthetic_stage(directory, n=60, n_classes=3):
    """Train/validation JSONL plus a real checkpoint built with the reference Model."""
    directory = Path(directory)
    classes = ["x", "y", "z"][:n_classes]
    rng = np.random.default_rng(0)
    for name, count in (("train", n), ("validation", n)):
        with open(directory / f"{name}.jsonl", "w") as stream:
            for i in range(count):
                label = classes[(i * n_classes) // count]  # class blocks in file order
                row = {"id": f"{name}-{i}", "inputs": {"features": [float(v) for v in rng.normal(size=3)]},
                       "answer": label, "meta": {"batch": 1, "row": i}}
                stream.write(json.dumps(row) + "\n")
    train = read_dataset(directory / "train.jsonl", "classification", classes)
    model = Model("classification", train["paths"], classes, train["x"], train["y"], hidden=4, seed=1)
    model.save(directory / "checkpoint.npz")
    return directory


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = write_synthetic_stage(self.tmp.name)
        self.name = f"team-split-{self._testMethodName}-{uuid.uuid4().hex[:8]}"
        self.cleanup = [ROOT / "state" / "tests" / self.name, ROOT / "runs" / "tests" / self.name]

    def tearDown(self):
        self.tmp.cleanup()
        for path in self.cleanup:
            shutil.rmtree(path, ignore_errors=True)

    def base_config(self, with_ace=False):
        config = {"task_name": "A", "stage_id": "syn", "seed": 42, "split": {"min_groups": 4},
                  "train_path": "train.jsonl", "validation_path": "validation.jsonl",
                  "parent_checkpoint": "checkpoint.npz", "work_dir": str(self.cleanup[1] / "work")}
        if with_ace:
            from team_ace.store import digest, initialize, scope_for
            absolute = dict(config, **{k: str(self.dir / config[k]) for k in ("train_path", "validation_path",
                                                                              "parent_checkpoint")})
            config["ace"] = {"module": "team_ace", "batch_size": 1, "mode": "disabled",
                             "storage_root": str(self.cleanup[0]), "scope": scope_for(absolute)}
            config["playbook_path"] = str(self.cleanup[0] / "A" / digest(config["ace"]["scope"])[:16] / "playbook.md")
            initialize(dict(absolute, ace=config["ace"], playbook_path=config["playbook_path"]))
        path = self.dir / "run.json"
        path.write_text(json.dumps(config, indent=2))
        return path

    def run_cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli(list(args))
        return code, out.getvalue() + err.getvalue()

    def test_load_table_uses_checkpoint_and_file_digest(self):
        from harness.runcfg import load_run_config
        table = load_table(load_run_config(self.base_config()))
        self.assertEqual(table.task_type, "classification")
        self.assertEqual(table.class_names, ["x", "y", "z"])
        self.assertEqual(table.source_sha256, file_sha256(self.dir / "validation.jsonl"))
        self.assertEqual(len(table), 60)

    def test_cli_make_verify_report_and_refusals(self):
        config = str(self.base_config())
        manifest = str(self.dir / "splits" / "syn-v1.json")
        code, output = self.run_cli("make", "--config", config, "--out", manifest)
        self.assertEqual(code, 0, output)
        self.assertTrue((self.dir / "splits" / "syn-v1.report.json").is_file())
        self.assertEqual(read_manifest(manifest)["method"], "class_time")
        self.assertEqual(self.run_cli("verify", "--config", config, "--manifest", manifest)[0], 0)
        self.assertEqual(self.run_cli("report", "--config", config, "--manifest", manifest)[0], 0)
        self.assertEqual(self.run_cli("make", "--config", config, "--out", manifest)[0], 0)  # same content: fine
        code, output = self.run_cli("make", "--config", config, "--out", manifest, "--method", "time_blocks")
        self.assertEqual(code, 1, output)  # different content: refused
        broken = read_manifest(manifest)
        broken["gate_ids"].append(broken["select_ids"][0])
        Path(manifest).write_text(json.dumps(broken))
        self.assertEqual(self.run_cli("verify", "--config", config, "--manifest", manifest)[0], 1)

    def test_wire_with_team_ace(self):
        base = self.base_config(with_ace=True)
        before = base.read_bytes()
        manifest = self.dir / "splits" / "syn-v1.json"
        self.assertEqual(self.run_cli("make", "--config", str(base), "--out", str(manifest), "--suffix", "split-v1")[0], 0)
        out = self.dir / "run.split-v1.json"
        config = derive_run_config(base, manifest, out, "split-v1")
        self.assertEqual(base.read_bytes(), before)
        self.assertEqual(config["split_manifest"], str(manifest.resolve()))
        self.assertEqual(config["stage_id"], "syn-split-v1")
        self.assertTrue(config["work_dir"].endswith("work-split-v1"))
        self.assertEqual(config["train_path"], str((self.dir / "train.jsonl").resolve()))
        old_scope = json.loads(before)["ace"]["scope"]
        self.assertNotEqual(config["ace"]["scope"], old_scope)
        self.assertIn("split_manifest", config["ace"]["scope"]["source_hashes"])
        from team_ace.store import digest, location
        self.assertEqual(Path(config["playbook_path"]).parent.name, digest(config["ace"]["scope"])[:16])
        self.assertTrue(Path(config["playbook_path"]).is_file())
        self.assertEqual(location(config), Path(config["playbook_path"]).parent)
        self.assertEqual(json.loads(out.read_text()), config)
        with self.assertRaises(FileExistsError):
            derive_run_config(base, manifest, out, "split-v1")
        broken = read_manifest(manifest)
        broken["source_sha256"] = "sha256:" + "1" * 64
        bad = self.dir / "splits" / "syn-bad.json"
        bad.write_text(json.dumps(redigest(broken)))
        with self.assertRaises(ValueError):
            derive_run_config(base, bad, self.dir / "run.bad.json", "bad")

    def test_wire_without_ace_keeps_other_keys(self):
        base = self.base_config()
        manifest = self.dir / "splits" / "syn-v1.json"
        self.assertEqual(self.run_cli("make", "--config", str(base), "--out", str(manifest))[0], 0)
        config = derive_run_config(base, manifest, self.dir / "run.v1.json", "v1")
        self.assertNotIn("ace", config)
        self.assertEqual(config["split"], {"min_groups": 4})


# ------------------------------------------------------------- real packages

def _package_config(directory, name, extra=None):
    source = Path(PACKAGES[name])
    config = {"task_name": name, "stage_id": "test", "seed": 42,
              "train_path": str(source / "public" / "train.jsonl"),
              "validation_path": str(source / "public" / "validation.jsonl"),
              "parent_checkpoint": str(source / "checkpoint.npz"), "work_dir": str(Path(directory) / "work"),
              **(extra or {})}
    path = Path(directory) / f"run-{uuid.uuid4().hex[:6]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config))
    return path


class PackageTest(unittest.TestCase):
    def manifest_for(self, name):
        from harness.runcfg import load_run_config
        with tempfile.TemporaryDirectory() as temporary:
            table = load_table(load_run_config(_package_config(temporary, name)))
        options = SplitOptions()
        return table, build_manifest(table, split(table, options), "test", options)

    def assert_matches_provisional(self, name):
        table, manifest = self.manifest_for(name)
        reference = make_split(table.ids, table.targets, table.meta_rows, table.task_type, "test",
                               table.source_sha256, class_names=table.class_names or None)
        self.assertEqual(manifest["select_ids"], reference["select_ids"])
        self.assertEqual(manifest["gate_ids"], reference["gate_ids"])
        self.assertEqual(verify_manifest(manifest, table, "test", SplitOptions()), [])
        return table, manifest

    @package("A")
    def test_a(self):
        table, manifest = self.assert_matches_provisional("A")
        self.assertEqual(manifest["method"], "class_time")
        self.assertEqual((manifest["counts"]["select"]["rows"], manifest["counts"]["gate"]["rows"]), (623, 621))
        self.assertEqual(len(manifest["counts"]["select"]["labels"]), 6)
        self.assertEqual(len(manifest["counts"]["gate"]["labels"]), 6)
        index = {identity: row for row, identity in enumerate(table.ids)}
        labels = table.labels()
        for name in set(labels):
            select = [index[i] for i in manifest["select_ids"] if labels[index[i]] == name]
            gate = [index[i] for i in manifest["gate_ids"] if labels[index[i]] == name]
            self.assertLess(max(select), min(gate), name)

    @package("B")
    def test_b(self):
        _, manifest = self.assert_matches_provisional("B")
        self.assertEqual(manifest["method"], "no_split")
        self.assertEqual(manifest["group_key"], ["bearing"])
        self.assertEqual(manifest["select_ids"], manifest["gate_ids"])
        self.assertTrue(manifest["warnings"])

    @package("C")
    def test_c(self):
        table, manifest = self.assert_matches_provisional("C")
        self.assertEqual(manifest["method"], "group_stratified")
        self.assertEqual(manifest["group_key"], ["source_subset", "unit"])
        engine = {i: (m["source_subset"], m["unit"]) for i, m in zip(table.ids, table.meta_rows)}
        select = {engine[i] for i in manifest["select_ids"]}
        gate = {engine[i] for i in manifest["gate_ids"]}
        self.assertFalse(select & gate)
        self.assertEqual(len(select | gate), 106)
        self.assertEqual((len(select), len(gate)), (55, 51))
        self.assertEqual((manifest["counts"]["select"]["rows"], manifest["counts"]["gate"]["rows"]), (11817, 11232))
        self.assertEqual({e[0] for e in select}, {e[0] for e in gate})
        self.assertEqual(len({e[0] for e in select}), 4)

    @package("A")
    def test_end_to_end_with_harness(self):
        from harness.evaluator import evaluate_policy
        from harness.runcfg import load_run_config
        from harness.stage import StageContext
        from team_split.manifest import write_manifest
        policy = ROOT / "harness" / "policy_initial.py"
        with tempfile.TemporaryDirectory() as temporary:
            provisional = StageContext(load_run_config(_package_config(Path(temporary) / "p", "A")))
            baseline = evaluate_policy(policy, provisional)
            table, manifest = self.manifest_for("A")
            manifest_path = Path(temporary) / "A-v1.json"
            write_manifest(manifest, manifest_path)
            config_path = _package_config(Path(temporary) / "m", "A", {"split_manifest": str(manifest_path)})
            context = StageContext(load_run_config(config_path))
            self.assertFalse(context.manifest["provisional"])
            self.assertEqual(context.manifest["manifest_sha256"], manifest["manifest_sha256"])
            result = evaluate_policy(policy, context)
            self.assertEqual(result["report"]["status"], "ok")
            self.assertAlmostEqual(result["metrics"]["combined_score"], baseline["metrics"]["combined_score"], places=12)


if __name__ == "__main__":
    unittest.main()
