"""Frozen training loop: continue a parent checkpoint under a training policy.

This is the reference `Model.fit` (baselines/model.py) with its hard-coded choices
replaced by calls into a policy module. Everything else is kept operation for operation,
so the unmodified policy (harness/policy_initial.py) produces a checkpoint that is
byte-identical to `python -m baselines.train --from-baseline ... --epochs 5`.

Never evolved. The policy cannot reach the model weights except through gradients,
cannot change normalization, target scaling, architecture or the checkpoint format.

Usage:
  python -m harness.runner --policy POLICY.py --parent checkpoint.npz \
      --train train.jsonl --validation select.jsonl --output NEW_DIR --stage-id round-01
Exit codes: 0 ok, 3 policy error (message on stderr as "POLICY_ERROR: ..."), 1 other error.
"""
import argparse
import importlib.util
import json
import math
import os
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from baselines.model import Model, checkpoint_digest, _THREADS  # must precede numpy (thread env)
import numpy as np

from baselines.data import sha256 as bytes_sha256
from .dataio import load_dataset
from .metrics import overall_metrics

BASELINES = Path(__file__).resolve().parents[1] / "baselines"
PARAMETERS = ("w1", "b1", "w2", "b2")
HEADLINE = {"classification": ("accuracy", "macro_f1"), "regression": ("rmse", "mae")}


class PolicyError(ValueError):
    """The policy returned something outside its contract."""


def load_policy(path):
    spec = importlib.util.spec_from_file_location("evolved_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _call(name, function, *args):
    try:
        return function(*args)
    except Exception as error:
        raise PolicyError(f"{name} raised {type(error).__name__}: {error}") from error


def _finite_array(name, value, shape):
    try:
        array = np.asarray(value, dtype=np.float64)
    except Exception as error:
        raise PolicyError(f"{name} must be numeric: {error}") from error
    if array.shape != shape:
        raise PolicyError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise PolicyError(f"{name} contains NaN or infinity")
    return array


def _read_config(policy, model, max_epochs):
    config = getattr(policy, "CONFIG", None)
    if not isinstance(config, dict):
        raise PolicyError("CONFIG must be a dict")
    epochs = config.get("epochs")
    if type(epochs) is not int or not 1 <= epochs <= max_epochs:
        raise PolicyError(f"CONFIG['epochs'] must be an int in 1..{max_epochs}, got {epochs!r}")
    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = model.meta["batch_size"]
    if type(batch_size) is not int or not 1 <= batch_size <= 65536:
        raise PolicyError(f"CONFIG['batch_size'] must be None or an int in 1..65536, got {batch_size!r}")
    clip = config.get("grad_clip")
    if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not math.isfinite(clip) or clip <= 0:
        raise PolicyError(f"CONFIG['grad_clip'] must be a positive finite number, got {clip!r}")
    return epochs, batch_size, float(clip)


def _readonly(arrays):
    views = {}
    for name, value in arrays.items():
        view = value.view()
        view.flags.writeable = False
        views[name] = view
    return views


def _headline(model, data):
    metrics = overall_metrics(model, data)
    return {name: metrics[name] for name in HEADLINE[model.meta["task"]]}


def _saturation(model, x_normalized):
    hidden, _ = model.forward(x_normalized)
    return float(np.mean(np.abs(hidden) > 0.99))


def _weight_summary(weights, y, task, num_classes):
    mean = float(weights.mean())
    summary = {"min": float(weights.min()), "max": float(weights.max()), "mean": mean,
               "cv": float(weights.std() / mean) if mean > 0 else 0.0}
    if task == "classification":
        summary["per_class_mean"] = [float(weights[y == label].mean()) if np.any(y == label) else None
                                     for label in range(num_classes)]
    else:
        edges = np.quantile(y, [0, .25, .5, .75, 1])
        summary["per_target_quartile_mean"] = [
            float(weights[(y >= low) & (y <= high)].mean()) for low, high in zip(edges[:-1], edges[1:])]
    return summary


def train(policy_path, parent_path, train_path, validation_path, output, stage_id="round",
          max_epochs=50, cache_dir=None):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("output must be a new directory; previous rounds are never overwritten")
    if not isinstance(stage_id, str) or not stage_id.strip() or len(stage_id) > 128:
        raise ValueError("stage ID must be nonempty and bounded")
    policy_source = Path(policy_path).read_bytes()
    parent_digest = checkpoint_digest(parent_path)
    model = Model.load(parent_path)
    meta = model.meta
    task, classes = meta["task"], meta["classes"]
    train_data = load_dataset(train_path, task, classes, cache_dir=cache_dir)
    validation = load_dataset(validation_path, task, classes, train_data["paths"], cache_dir=cache_dir)
    if set(train_data["ids"]) & set(validation["ids"]):
        raise ValueError("training and validation sample IDs overlap")
    if meta["paths"] != train_data["paths"]:
        raise ValueError("checkpoint feature schema does not match the training data")

    try:
        policy = load_policy(policy_path)
    except Exception as error:
        raise PolicyError(f"policy failed to import: {type(error).__name__}: {error}") from error
    for name in ("learning_rate", "sample_weights", "output_grad", "extra_grad", "pick_epoch"):
        if not callable(getattr(policy, name, None)):
            raise PolicyError(f"policy must define a function named {name}")
    epochs, batch_size, clip = _read_config(policy, model, max_epochs)
    meta["batch_size"] = batch_size
    model.validate()

    x = model.normalized(train_data["x"])
    y = train_data["y"]
    if y.shape != (len(x),) or not len(x):
        raise ValueError("target shape mismatch")
    if task == "regression":
        target = (y - meta["target_mean"]) / meta["target_scale"]
    else:
        if np.any(y < 0) or np.any(y >= len(classes)):
            raise ValueError("unknown class")
        target = y
    weights = _finite_array("sample_weights(...)", _call("sample_weights", policy.sample_weights, y.copy(), task,
                                                         len(classes)), (len(x),))
    if np.any(weights < 0) or weights.sum() <= 0:
        raise PolicyError("sample_weights(...) must be non-negative with a positive sum")

    base_lr = meta["learning_rate"]
    initial_steps, initial_epochs = meta["steps"], meta["epochs"]
    parent_params = {name: model.params[name].copy() for name in PARAMETERS}
    parent_norm = math.sqrt(sum(float(np.sum(value ** 2)) for value in parent_params.values()))
    x_validation = model.normalized(validation["x"])
    validation_before = _headline(model, validation)
    saturation_before = _saturation(model, x_validation)

    history, snapshots, per_epoch, rates = [], [], [], []
    started = time.monotonic()
    for index in range(epochs):
        rate = _call("learning_rate", policy.learning_rate, index, epochs, base_lr)
        if isinstance(rate, bool) or not isinstance(rate, (int, float, np.floating)) or not math.isfinite(rate) \
                or not 0 < rate <= 1:
            raise PolicyError(f"learning_rate(...) must return a float in (0, 1], got {rate!r}")
        rate = float(rate)
        rates.append(rate)
        # Epoch-indexed shuffling, identical to the reference trainer.
        rng = np.random.default_rng(np.random.SeedSequence([meta["seed"], meta["epochs"]]))
        order = rng.permutation(len(x))
        loss_sum, norms, clipped = 0.0, [], 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            batch, expected = x[indices], target[indices]
            hidden, out = model.forward(batch)
            result = _call("output_grad", policy.output_grad, out, expected, weights[indices], task)
            if not isinstance(result, tuple) or len(result) != 2:
                raise PolicyError("output_grad(...) must return (loss, grad)")
            loss, delta = result
            try:
                loss = float(loss)
            except Exception as error:
                raise PolicyError(f"output_grad(...) loss must be a number: {error}") from error
            if not math.isfinite(loss):
                raise PolicyError("output_grad(...) returned a non-finite loss")
            delta = _finite_array("output_grad(...) grad", delta, (len(indices), model.params["w2"].shape[1]))
            loss_sum += loss
            gradients = {"w2": hidden.T @ delta, "b2": delta.sum(axis=0)}
            hidden_delta = (delta @ model.params["w2"].T) * (1 - hidden ** 2)
            gradients.update(w1=batch.T @ hidden_delta, b1=hidden_delta.sum(axis=0))
            extra = _call("extra_grad", policy.extra_grad, _readonly(model.params))
            if not isinstance(extra, dict) or set(extra) != set(PARAMETERS):
                raise PolicyError("extra_grad(...) must return a dict with keys w1, b1, w2, b2")
            for name in gradients:
                addition = _finite_array(f"extra_grad(...)['{name}']", extra[name], model.params[name].shape)
                if np.any(addition):
                    gradients[name] = gradients[name] + addition
            norm = math.sqrt(sum(float(np.sum(g * g)) for g in gradients.values()))
            if not math.isfinite(norm):
                raise PolicyError("non-finite gradient (check output_grad / extra_grad / learning rate)")
            norms.append(norm)
            clipped += norm > clip
            meta["steps"] += 1
            for name, gradient in gradients.items():
                gradient *= min(1.0, clip / max(norm, 1e-12))
                model.first[name] = .9 * model.first[name] + .1 * gradient
                model.second[name] = .999 * model.second[name] + .001 * gradient ** 2
                first = model.first[name] / (1 - .9 ** meta["steps"])
                second = model.second[name] / (1 - .999 ** meta["steps"])
                model.params[name] -= rate * first / (np.sqrt(second) + 1e-8)
            if not all(np.all(np.isfinite(p)) for p in model.params.values()):
                raise PolicyError("model weights became non-finite (learning rate or loss too aggressive)")
        meta["epochs"] += 1
        model.validate()
        train_loss = loss_sum / len(x)
        headline = _headline(model, validation)
        history.append({"epoch": meta["epochs"], "train_loss": train_loss, "val": dict(headline)})
        change = math.sqrt(sum(float(np.sum((model.params[n] - parent_params[n]) ** 2)) for n in PARAMETERS))
        per_epoch.append({"epoch": meta["epochs"], "step": meta["steps"], "train_loss": train_loss,
                          "val": headline, "learning_rate": rate,
                          "grad_norm_mean": float(np.mean(norms)), "grad_norm_max": float(np.max(norms)),
                          "clip_rate": clipped / len(norms), "saturation": _saturation(model, x_validation),
                          "weight_change": change / parent_norm if parent_norm > 0 else 0.0})
        snapshots.append(({n: model.params[n].copy() for n in PARAMETERS},
                          {n: model.first[n].copy() for n in PARAMETERS},
                          {n: model.second[n].copy() for n in PARAMETERS}, dict(meta)))
    duration = time.monotonic() - started

    chosen = _call("pick_epoch", policy.pick_epoch, [dict(entry, val=dict(entry["val"])) for entry in history])
    if isinstance(chosen, bool) or not isinstance(chosen, (int, np.integer)) or not 0 <= chosen < len(history):
        raise PolicyError(f"pick_epoch(...) must return an int in 0..{len(history) - 1}, got {chosen!r}")
    chosen = int(chosen)
    params, first, second, chosen_meta = snapshots[chosen]
    model.params, model.first, model.second, model.meta = params, first, second, chosen_meta
    model.validate()
    meta = model.meta

    profile_file = BASELINES / "configs" / f"{meta.get('profile')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".runner-", dir=output.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir(mode=0o700)
        checkpoint = staging / "checkpoint.npz"
        model.save(checkpoint)
        report = {
            "format": "contest.cpu-baseline-report.v1", "profile": meta.get("profile"), "task": task,
            "purpose": "development_reference", "official_scoring_ready": False, "device": "cpu",
            "cpu_threads_requested": _THREADS, "numpy_version": np.__version__,
            "python_version": platform.python_version(), "machine": platform.machine(), "stage_id": stage_id,
            "seed": meta["seed"], "parameter_count": meta["parameter_count"], "input_features": len(meta["paths"]),
            "train_rows": len(train_data["ids"]), "validation_rows": len(validation["ids"]),
            "train_sha256": train_data["sha256"], "validation_sha256": validation["sha256"],
            "profile_sha256": bytes_sha256(profile_file.read_bytes()) if profile_file.is_file() else None,
            "parent_checkpoint_sha256": parent_digest, "checkpoint_sha256": checkpoint_digest(checkpoint),
            "initial_steps": initial_steps, "initial_epochs": initial_epochs,
            "final_steps": meta["steps"], "final_epochs": meta["epochs"], "training_seconds": duration,
            "learning_rate": meta["learning_rate"], "batch_size": meta["batch_size"],
            "validation_before": validation_before, "validation_after": history[chosen]["val"],
            "configuration": {
                "hidden": meta["hidden"], "activation": "tanh", "optimizer": "Adam", "adam_beta1": .9,
                "adam_beta2": .999, "adam_epsilon": 1e-8, "gradient_clip_norm": clip,
                "feature_clip": [-10.0, 10.0], "normalization": "initial_training_mean_std",
                "learning_rate": meta["learning_rate"], "batch_size": meta["batch_size"],
                "requested_epochs": epochs, "classes": meta["classes"]},
            "source_sha256": {name: bytes_sha256((BASELINES / name).read_bytes())
                              for name in ("model.py", "data.py", "train.py", "predict.py")},
            "label_provenance": None, "created_at": datetime.now(timezone.utc).isoformat(),
            # Additions of this harness (the reference report has no such keys):
            "trainer": "harness.runner", "policy_sha256": bytes_sha256(policy_source),
            "runner_sha256": bytes_sha256(Path(__file__).read_bytes()),
            "trained_epochs": epochs, "selected_epoch_index": chosen,
            "learning_rate_schedule": rates,
        }
        for key in ("target_unit",):
            if key in meta:
                report[key] = meta[key]
        (staging / "training.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        with (staging / "history.jsonl").open("w") as stream:
            for position, entry in enumerate(per_epoch):
                record = {"stage_id": stage_id, "seed": meta["seed"], "epoch": entry["epoch"], "step": entry["step"],
                          "loss": entry["train_loss"], "validation": entry["val"], "selected": position == chosen}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        internals = {
            "config": {"epochs": epochs, "batch_size": batch_size, "grad_clip": clip},
            "base_learning_rate": base_lr, "selected_epoch_index": chosen, "per_epoch": per_epoch,
            "validation_before": validation_before, "saturation_before": saturation_before,
            "sample_weights": _weight_summary(weights, y, task, len(classes)),
            "initial_steps": initial_steps, "initial_epochs": initial_epochs,
            "final_steps": meta["steps"], "final_epochs": meta["epochs"], "training_seconds": duration,
        }
        (staging / "runner_report.json").write_text(json.dumps(internals, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
        os.replace(staging, output)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--parent", required=True, help="checkpoint to continue from (weights, Adam, normalization)")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True, help="selection split; never used for gradient updates")
    parser.add_argument("--output", required=True, help="new directory for checkpoint.npz, training.json, history.jsonl")
    parser.add_argument("--stage-id", default="round")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--cache-dir")
    args = parser.parse_args(argv)
    try:
        report = train(args.policy, args.parent, args.train, args.validation, args.output,
                       stage_id=args.stage_id, max_epochs=args.max_epochs, cache_dir=args.cache_dir)
    except PolicyError as error:
        print(f"POLICY_ERROR: {error}", file=sys.stderr)
        return 3
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
