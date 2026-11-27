"""Run the simulation: folds x protocols x update policies, 10 rounds each.

<out>/
  f<fold>/static/checkpoint.npz          pseudo-baseline: official recipe, trained from scratch on the
                                         fold's train units only (the official checkpoint has seen _1.._3)
  f<fold>/p<protocol>/<policy>/rounds.jsonl   one record per round: data sizes, decision, test scores
  f<fold>/p<protocol>/<policy>/summary.json   scores in the shape of the objective score items
  f<fold>/p<protocol>/<policy>/round-XX/...   candidate checkpoints and their policies
  f<fold>/p<protocol>/data/round-XX/*.jsonl   round files, deleted at the end unless --keep-data
  summary.json, report.md                 all runs (python -m team_sim report rebuilds them)
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from baselines.data import flatten
from baselines.model import Model
from harness import PROJECT_ROOT
from harness import runner

from . import data as simdata
from .decide import advantage, make, resample_index, rmse, unit_rmse


def _log(message):
    print(message, file=sys.stderr, flush=True)


class Simulation:
    def __init__(self, package, out_dir, task="B", protocols=simdata.PROTOCOLS,
                 policies=("never", "always", "v1", "v3"), folds=None, rounds=simdata.SEGMENTS,
                 epochs=5, max_epochs=30, resamples=1000, seed=0, keep_data=False, search=None, force_search=False):
        self.package, self.out, self.task = Path(package), Path(out_dir), task
        self.protocols, self.policies, self.fold_ids = tuple(protocols), tuple(policies), folds
        self.rounds, self.epochs, self.max_epochs = int(rounds), int(epochs), int(max_epochs)
        self.resamples, self.seed, self.keep_data, self.search = int(resamples), int(seed), keep_data, search
        self.force_search = bool(force_search)   # testing aid: v3 asks for a search every round
        self.cache_dir = str(self.out / "cache")
        recipe = json.loads((PROJECT_ROOT / "baselines" / "configs" / f"{task}.json").read_text(encoding="utf-8"))
        self.recipe = {k: recipe[k] for k in ("hidden", "seed", "learning_rate", "batch_size", "epochs")}
        self.paths = list(Model.load(str(self.package / "checkpoint.npz")).meta["paths"])
        self._sets, self._index, self._pred = {}, {}, {}
        self._run_dir = None
        self.trainings = 0

    # ------------------------------------------------------------ data of the current round
    def _arrays(self, rows):
        x = np.asarray([flatten(r["inputs"])[1] for r in rows], dtype=np.float64)
        return (x, np.asarray([r["answer"] for r in rows], dtype=np.float64),
                np.asarray([r["unit"] for r in rows]))

    def _set_round(self, rows_by_key, files, k, salt):
        self._sets = {key: self._arrays(rows) for key, rows in rows_by_key.items() if rows}
        self._index = {key: resample_index(len(v[1]), self.resamples, self.seed + 7919 * k + salt + i)
                       for i, (key, v) in enumerate(sorted(self._sets.items()))}
        self._pred = {}
        self.files, self.k = files, k

    def has(self, key):
        return key in self._sets

    def y(self, key):
        return self._sets[key][1]

    def index(self, key):
        return self._index[key]

    def predict(self, checkpoint, key):
        cache = (str(checkpoint), key)
        if cache not in self._pred:
            self._pred[cache] = Model.load(str(checkpoint)).predict(self._sets[key][0])
        return self._pred[cache]

    # ------------------------------------------------------------ training
    def pseudo_baseline(self, fold, units, protocol):
        names = simdata.init_units(fold, protocol)
        tag = "static-2c" if protocol == "2c" else "static"
        path = self.out / f"f{fold['fold']}" / tag / "checkpoint.npz"
        if path.is_file():
            return str(path)
        rows = [row for unit in names for row in units[unit]]
        x, y, _ = self._arrays(rows)
        r = self.recipe
        model = Model("regression", self.paths, [], x, y, hidden=r["hidden"], seed=r["seed"],
                      learning_rate=r["learning_rate"], batch_size=r["batch_size"])
        model.fit(x, y, r["epochs"])
        model.meta["profile"] = simdata.TASKS[self.task]["profile"]
        model.meta["target_unit"] = "minutes"
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(path))
        (path.parent / "fold.json").write_text(json.dumps(dict(fold, init_units=names), ensure_ascii=False, indent=1) + "\n",
                                               encoding="utf-8")
        return str(path)

    def train(self, parent, policy_source, k, label):
        folder = self._run_dir / f"round-{k:02d}" / label
        policy_path = folder.parent / f"{label}.policy.py"
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(policy_source, encoding="utf-8")
        runner.train(str(policy_path), str(parent), self.files["train"], self.files["validation"], str(folder),
                     stage_id=f"sim-{label}-r{k:02d}", max_epochs=self.max_epochs, cache_dir=self.cache_dir)
        self.trainings += 1
        return str(folder / "checkpoint.npz")

    # ------------------------------------------------------------ one fold x protocol
    def run_fold_protocol(self, fold, protocol, units, static):
        base = self.out / f"f{fold['fold']}" / f"p{protocol}"
        meta = Model.load(static).meta
        self.static, self.target_mean, self.target_scale = static, meta["target_mean"], meta["target_scale"]
        self.protocol = protocol
        plan, previous = [], {row["id"] for unit in simdata.init_units(fold, protocol) for row in units[unit]}
        for k in range(1, self.rounds + 1):
            rows = simdata.round_rows(units, fold, protocol, k)
            folder = base / "data" / f"round-{k:02d}"
            files = {"train": simdata.write_jsonl(folder / "train.jsonl", rows["train"]),
                     "validation": simdata.write_jsonl(folder / "validation.jsonl", rows["validation"])}
            if rows["newest"]:
                files["newest"] = simdata.write_jsonl(folder / "newest.jsonl", rows["newest"])
            ids = {row["id"] for row in rows["train"]}
            plan.append({"k": k, "rows": rows, "files": files, "new_train": bool(ids - previous),
                         "label_mean": float(np.mean([row["answer"] for row in rows["train"]]))})
            previous = ids
        salt = 104729 * fold["fold"] + 1299709 * simdata.PROTOCOLS.index(protocol)
        for name in self.policies:
            policy, state = make(name), {"deployed": static, "deployed_label": "static"}
            policy.start(self, state)
            self._run_dir = base / name
            self._run_dir.mkdir(parents=True, exist_ok=True)
            started, trainings_before, records = time.time(), self.trainings, []
            for step in plan:
                k, rows = step["k"], step["rows"]
                self._set_round({"validation": rows["validation"], "newest": rows["newest"], "test": rows["test"]},
                                step["files"], k, salt)
                ctx = {"files": step["files"], "new_train": step["new_train"], "label_mean": step["label_mean"]}
                before = self.trainings
                decision = policy.round(self, k, ctx, state)
                p_dep, p_sta, y_t = self.predict(state["deployed"], "test"), self.predict(static, "test"), self.y("test")
                units_t = self._sets["test"][2]
                records.append({
                    "fold": fold["fold"], "protocol": protocol, "policy": name, "round": k,
                    "rows": {key: len(rows[key]) for key in ("train", "validation", "newest", "test")},
                    "new_train": step["new_train"], "label_mean": step["label_mean"],
                    "trainings": self.trainings - before, "deployed": state["deployed_label"],
                    "decision": decision,
                    "test": {"rmse": rmse(p_dep, y_t), "static_rmse": rmse(p_sta, y_t),
                             "advantage": advantage(p_dep, p_sta, y_t),
                             "unit_rmse": unit_rmse(p_dep, y_t, units_t), "static_unit_rmse": unit_rmse(p_sta, y_t, units_t),
                             "sse": float(np.sum((p_dep - y_t) ** 2)), "static_sse": float(np.sum((p_sta - y_t) ** 2)),
                             "n": int(len(y_t))}})
            with (self._run_dir / "rounds.jsonl").open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=_plain) + "\n")
            summary = summarize(records)
            summary.update(fold=fold["fold"], protocol=protocol, policy=name, seconds=time.time() - started,
                           trainings=self.trainings - trainings_before, hidden=fold["hidden"], stream=fold["stream"])
            (self._run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=_plain) + "\n",
                                                        encoding="utf-8")
            _log(f"[team_sim] fold {fold['fold']} 给法 {protocol} {name}: 平均相对优势 {summary['mean_advantage']:+.1%}，"
                 f"最差一轮 {summary['worst_advantage']:+.1%}，训练 {summary['trainings']} 次")
        if not self.keep_data:
            target = (base / "data").resolve()
            if target.parent == base.resolve() and target.name == "data":
                shutil.rmtree(target, ignore_errors=True)

    def run(self):
        rows = simdata.load_rows(self.package, self.task)
        units = simdata.by_unit(rows)
        folds = simdata.make_folds(units)
        if self.fold_ids is not None:
            folds = [f for f in folds if f["fold"] in set(self.fold_ids)]
        self.out.mkdir(parents=True, exist_ok=True)
        for fold in folds:
            _log(f"[team_sim] fold {fold['fold']}: 隐藏 {fold['hidden']}，数据流 {fold['stream']}，训练 {fold['train']}"
                 f"（②c：验证设备 {fold['stream_val']}，训练 {fold['train_2c']}）")
            for protocol in self.protocols:
                self.run_fold_protocol(fold, protocol, units, self.pseudo_baseline(fold, units, protocol))
        from .report import build
        return build(self.out)


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def summarize(records):
    """Scores shaped like the objective items: average gain, worst stage and recovery, stability, resources."""
    adv = np.array([r["test"]["advantage"] for r in records])
    unit_adv = np.array([(r["test"]["static_unit_rmse"] - r["test"]["unit_rmse"]) / r["test"]["static_unit_rmse"]
                         if r["test"]["static_unit_rmse"] > 0 else 0.0 for r in records])
    sse, static_sse = sum(r["test"]["sse"] for r in records), sum(r["test"]["static_sse"] for r in records)
    n = sum(r["test"]["n"] for r in records)
    overall, static_overall = (sse / n) ** 0.5, (static_sse / n) ** 0.5
    runs, current = [], 0
    for value in adv:
        if value < 0:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    decisions = [r["decision"] for r in records]
    return {
        "rounds": len(records),
        "mean_advantage": float(adv.mean()), "mean_unit_advantage": float(unit_adv.mean()),
        "overall_rmse": overall, "static_overall_rmse": static_overall,
        "overall_advantage": (static_overall - overall) / static_overall if static_overall > 0 else 0.0,
        "worst_advantage": float(adv.min()), "worst_round": int(np.argmin(adv)) + 1,
        "rounds_below_static": int((adv < 0).sum()),
        "mean_rounds_to_recover": float(np.mean(runs)) if runs else 0.0,
        "advantage_std": float(adv.std()),
        "search_requests": int(sum(bool(d.get("search_requested")) for d in decisions)),
        "searches_run": int(sum(bool(d.get("search_ran")) for d in decisions)),
        "search_adopted": int(sum(d.get("action") == "search" for d in decisions)),
        "search_time_limited": int(sum(bool((d.get("search") or {}).get("stopped")) for d in decisions)),
        "rollbacks": int(sum(bool(d.get("rollback")) or bool(d.get("safety_rollback")) for d in decisions)),
        "per_round_advantage": adv.tolist(),
    }
