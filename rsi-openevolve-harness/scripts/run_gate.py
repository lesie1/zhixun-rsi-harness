#!/usr/bin/env python
"""Acceptance gate for one stage: adopt the evolved policy, the official continuation, or
keep the parent checkpoint (rollback).

  python scripts/run_gate.py --run-config configs/run_A.json
  python scripts/run_gate.py --run-config configs/run_A.json --policy some_policy.py

Writes <work_dir>/gate/decision.json and, when a checkpoint is adopted, copies its
directory to <work_dir>/selected/ (checkpoint.npz, training.json, history.jsonl, policy.py).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--policy", help="default: <work_dir>/openevolve_output/best/best_program.py")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)

    from baselines.model import Model
    from harness.evaluator import evaluate_policy
    from harness.gate import decide, describe
    from harness.runcfg import load_run_config
    from harness.stage import get_context

    config = load_run_config(args.run_config)
    context = get_context(config)
    work = Path(config["work_dir"])
    policy = Path(args.policy) if args.policy else work / "openevolve_output" / "best" / "best_program.py"
    candidate_model, candidate_dir, evaluation = None, None, None
    if policy.is_file():
        evaluation = evaluate_policy(policy, context)
        if evaluation["report"]["status"] == "ok":
            candidate_dir = Path(evaluation["report"]["candidate_dir"])
            candidate_model = Model.load(candidate_dir / "checkpoint.npz")
        else:
            print(f"[gate] candidate policy is invalid ({evaluation['report']['status']}); it cannot be adopted")
    else:
        print(f"[gate] no candidate policy at {policy}; deciding between official continuation and parent")

    result = decide(context, candidate_model, resamples=args.resamples, seed=config["seed"])
    result.update({"stage_id": config["stage_id"], "task_name": config.get("task_name"),
                   "policy": str(policy) if policy.is_file() else None,
                   "candidate_dir": str(candidate_dir) if candidate_dir else None,
                   "selection_score": evaluation["metrics"]["combined_score"] if evaluation else None,
                   "parent_checkpoint": config["parent_checkpoint"]})
    chosen = {"adopt_candidate": candidate_dir, "adopt_reference": context.reference_dir}.get(result["decision"])
    selected = work / "selected"
    if selected.exists():
        shutil.rmtree(selected)
    if chosen:
        shutil.copytree(chosen, selected)
        source = policy if result["decision"] == "adopt_candidate" else ROOT / "harness" / "policy_initial.py"
        shutil.copyfile(source, selected / "policy.py")
        result["selected_checkpoint"] = str(selected / "checkpoint.npz")
    else:
        result["selected_checkpoint"] = config["parent_checkpoint"]
    (work / "gate").mkdir(exist_ok=True)
    (work / "gate" / "decision.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(describe(result))
    print(f"[gate] next round continues from: {result['selected_checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
