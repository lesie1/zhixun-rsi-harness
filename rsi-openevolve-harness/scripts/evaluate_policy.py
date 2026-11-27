#!/usr/bin/env python
"""Evaluate one policy file under a run config and print the feedback the LLM would see.

  python scripts/evaluate_policy.py --run-config configs/run_A.json
  python scripts/evaluate_policy.py --run-config configs/run_A.json --policy my_policy.py --json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--policy", default=str(ROOT / "harness" / "policy_initial.py"))
    parser.add_argument("--json", action="store_true", help="print the full report as JSON instead of the text")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    from harness.evaluator import evaluate_policy
    from harness.runcfg import load_run_config
    from harness.stage import get_context

    context = get_context(load_run_config(args.run_config))
    result = evaluate_policy(args.policy, context, use_cache=not args.no_cache)
    if args.json:
        print(json.dumps(result["report"], ensure_ascii=False, indent=2))
    else:
        print("metrics:", json.dumps(result["metrics"]))
        print(result["artifacts"]["feedback"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
