#!/usr/bin/env python
"""Run one OpenEvolve search for one task/stage.

Example:
  python scripts/run_evolve.py --run-config configs/run_A.json --iterations 40 \
      --api-base http://127.0.0.1:8000/v1 --model qwen

What it does:
  1. resolves the run config and prepares the stage (split, parent and reference results);
  2. builds the system message for this task;
  3. starts OpenEvolve (vendored in third_party/) with the policy as initial program and
     harness/evaluator.py as evaluator, the prompt extension and the iteration hook enabled;
  4. writes <work_dir>/evolve_summary.json.
The acceptance gate is a separate step: scripts/run_gate.py.
"""
import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "third_party" / "openevolve"
for entry in (str(VENDORED), str(ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(ROOT), str(VENDORED)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-config", required=True, help="stage run config (JSON), see configs/run_A.json")
    parser.add_argument("--oe-config", default=str(ROOT / "configs" / "openevolve.yaml"))
    parser.add_argument("--iterations", type=int, help="override max_iterations")
    parser.add_argument("--api-base", help="OpenAI-compatible endpoint of the driver model")
    parser.add_argument("--api-key")
    parser.add_argument("--model", help="model name at the endpoint")
    parser.add_argument("--workers", type=int, help="parallel evaluations")
    parser.add_argument("--initial-policy", default=str(ROOT / "harness" / "policy_initial.py"))
    parser.add_argument("--resume", help="OpenEvolve checkpoint directory to resume from")
    parser.add_argument("--admit-invalid", action="store_true",
                        help="let candidates that scored 0 enter the population (OpenEvolve's default behaviour)")
    args = parser.parse_args(argv)

    from harness.briefing import system_message
    from harness.records import read_records
    from harness.runcfg import ENV_VAR, load_run_config, write_resolved
    from harness.stage import get_context

    config = load_run_config(args.run_config)
    os.environ[ENV_VAR] = write_resolved(config)
    os.environ["OPENEVOLVE_PROMPT_EXTRAS"] = "harness.prompt_extras:get_extras"
    os.environ["OPENEVOLVE_ITERATION_HOOK"] = "harness.records:on_iteration"
    os.environ.setdefault("HARNESS_RUN_ID", time.strftime("%Y%m%dT%H%M%S"))
    config = load_run_config()

    print(f"[harness] preparing stage in {config['work_dir']} ...", flush=True)
    context = get_context(config)
    summary = context.summary()
    primary = context.spec["primary"]
    print(f"[harness] task={summary['task_name']} type={summary['task_type']} primary={primary} "
          f"split={summary['split']['method']} select/gate rows="
          f"{summary['split']['counts']['select']['rows']}/{summary['split']['counts']['gate']['rows']}")
    print(f"[harness] parent {primary}={summary['parent_metrics'][primary]:.6g}  "
          f"official continuation {primary}={summary['reference_metrics'][primary]:.6g} "
          f"(combined_score {summary['reference_score']['combined_score']:.4f})", flush=True)
    for warning in summary["split"]["warnings"]:
        print(f"[harness] WARNING: {warning}")

    from openevolve.config import load_config
    from openevolve.controller import OpenEvolve

    oe_config = load_config(args.oe_config)
    oe_config.prompt.system_message = system_message(context)
    if args.iterations is not None:
        oe_config.max_iterations = args.iterations
    if args.workers is not None:
        oe_config.evaluator.parallel_evaluations = args.workers
    overrides = {key: value for key, value in (("api_base", args.api_base), ("api_key", args.api_key)) if value}
    if overrides:
        for key, value in overrides.items():
            setattr(oe_config.llm, key, value)
        oe_config.llm.update_model_params(overrides, overwrite=True)
    if args.model:
        for model in oe_config.llm.models + oe_config.llm.evaluator_models:
            model.name = args.model

    work = Path(config["work_dir"])
    initial = work / "initial_policy.py"
    shutil.copyfile(args.initial_policy, initial)
    (work / "system_message.txt").write_text(oe_config.prompt.system_message, encoding="utf-8")
    output_dir = work / "openevolve_output"
    from harness.population import strategy
    controller = OpenEvolve(initial_program_path=str(initial),
                            evaluation_file=str(ROOT / "harness" / "evaluator.py"),
                            config=oe_config, output_dir=str(output_dir),
                            population_strategy=None if args.admit_invalid else strategy())
    started = time.time()
    best = asyncio.run(controller.run(iterations=oe_config.max_iterations, checkpoint_path=args.resume))

    records = read_records(config)
    result = {
        "run_id": os.environ["HARNESS_RUN_ID"], "task_name": config.get("task_name"), "stage_id": config["stage_id"],
        "iterations_recorded": len(records), "seconds": time.time() - started,
        "status_counts": dict(Counter(record["status"] for record in records)),
        "outcome_counts": dict(Counter(record["outcome"] for record in records)),
        "best_program": str(output_dir / "best" / "best_program.py") if best else None,
        "best_metrics": best.metrics if best else None,
        "reference_combined_score": summary["reference_score"]["combined_score"],
        "records": str(work / "records" / "iterations.jsonl"),
    }
    (work / "evolve_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("[harness] done:", json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
