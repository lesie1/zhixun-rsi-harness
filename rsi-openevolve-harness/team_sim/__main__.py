"""Command line of the public-data stream simulator (docs/SIMULATOR.md).

  python -m team_sim run --package ../cpu-baselines-20261110-v1/B --out runs/sim/B
  python -m team_sim report --out runs/sim/B
  python -m team_sim seeds --protocol 2b          # what the preset skills look like in a prompt
  python -m team_sim ace-retry --search-dir runs/B-live-01/f0/p2c/v3/round-10/search \\
      --ace-approve-live --ace-no-thinking       # rerun queued ACE jobs of a finished round on a copy
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

from harness import PROJECT_ROOT


def _ints(text):
    return None if text in (None, "all") else [int(x) for x in text.split(",") if x.strip()]


def _list(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m team_sim", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="simulate folds x protocols x policies")
    run.add_argument("--package", required=True, help="unzipped cpu-b package (contains checkpoint.npz and public/)")
    run.add_argument("--out", default=str(PROJECT_ROOT / "runs" / "sim" / "B"))
    run.add_argument("--task", default="B", choices=["B"])
    run.add_argument("--protocols", default="1,2a,2b,2c", help="comma list of 1, 2a, 2b, 2c")
    run.add_argument("--policies", default="never,always,v1,v3",
                     help="comma list of never, always, v1, v3, v3_cap1, v3_nocap")
    run.add_argument("--folds", default="all", help="'all' or comma list such as 0,1")
    run.add_argument("--rounds", type=int, default=10)
    run.add_argument("--epochs", type=int, default=5, help="epochs per continuation (policy CONFIG)")
    run.add_argument("--max-epochs", type=int, default=30)
    run.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples per comparison")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--keep-data", action="store_true", help="keep the per-round jsonl files (large)")
    search = run.add_argument_group("search on red/yellow (off by default)")
    search.add_argument("--search", default="none", choices=["none", "evolve"])
    search.add_argument("--api-base", help="OpenAI-compatible endpoint of the driver model (key: $OPENEVOLVE_API_KEY)")
    search.add_argument("--model", help="model name at the endpoint")
    search.add_argument("--iterations", type=int, default=6)
    search.add_argument("--workers", type=int, default=1)
    search.add_argument("--oe-config", default=str(PROJECT_ROOT / "configs" / "openevolve.split-v1.yaml"))
    search.add_argument("--search-timeout", type=int, default=1800,
                        help="time budget per search in seconds; at the limit OpenEvolve stops and keeps the best "
                             "program of the finished iterations")
    search.add_argument("--ace", default="team_ace", choices=["team_ace", "stub"])
    search.add_argument("--ace-mode", default="disabled", choices=["disabled", "fixture", "live"])
    search.add_argument("--ace-max-calls", type=int, default=0)
    search.add_argument("--ace-approve-live", action="store_true")
    search.add_argument("--ace-model")
    search.add_argument("--ace-api-base")
    search.add_argument("--ace-api-key-env")
    search.add_argument("--ace-worker-seconds", type=int, default=0)
    search.add_argument("--ace-max-jobs", type=int, default=3, help="team_ace jobs processed after each search")
    search.add_argument("--ace-request-timeout", type=int, default=300,
                        help="seconds one ACE model call may take (team_ace's own default of 20 is too short "
                             "for a reasoning model); --ace-worker-seconds must be at least twice this plus 30")
    search.add_argument("--ace-max-tokens", type=int, default=4096, help="output-token cap of one ACE model call")
    search.add_argument("--ace-no-thinking", action="store_true",
                        help="send enable_thinking=false with each ACE call (pytrio: Qwen otherwise thinks first "
                             "and can use up --ace-max-tokens before it answers)")
    search.add_argument("--ace-max-shown", type=int, default=3,
                        help="learned ACE entries shown in a round's playbook, newest first, after the preset skills")
    search.add_argument("--seeds", help="preset skills file (default team_sim/seeds.json)")
    search.add_argument("--force-search", action="store_true", help="testing aid: v3 requests a search every round")

    report = sub.add_parser("report", help="rebuild summary.json and report.md")
    report.add_argument("--out", default=str(PROJECT_ROOT / "runs" / "sim" / "B"))

    seeds = sub.add_parser("seeds", help="render the preset skills as a stage playbook")
    seeds.add_argument("--task", default="B")
    seeds.add_argument("--protocol", default="2b")
    seeds.add_argument("--file", help="preset skills file (default team_sim/seeds.json)")

    redo = sub.add_parser("ace-retry", help="process queued ACE jobs of a finished search round on a copy of its state")
    redo.add_argument("--search-dir", required=True, help="<out>/f<fold>/p<protocol>/<policy>/round-XX/search")
    redo.add_argument("--ace-approve-live", action="store_true", help="needed when that round used live ACE (spends quota)")
    redo.add_argument("--ace-no-thinking", action="store_true", help="send enable_thinking=false with each call")
    redo.add_argument("--ace-request-timeout", type=int, default=300)
    redo.add_argument("--ace-max-tokens", type=int, default=4096)
    redo.add_argument("--ace-worker-seconds", type=int, default=630)
    redo.add_argument("--ace-max-jobs", type=int, default=1)
    redo.add_argument("--ace-max-calls", type=int, help="call budget of the copy (default: as in that run; "
                      "calls already made there count against it)")
    redo.add_argument("--include-failed", action="store_true", help="also retry jobs that failed once")
    redo.add_argument("--package", help="B package of that run; needed when its per-round data were deleted")

    args = parser.parse_args(argv)
    if args.command == "report":
        from .report import build
        print(json.dumps(build(args.out), ensure_ascii=False, indent=1))
        return 0
    if args.command == "seeds":
        from . import seeds as preset
        context = {"task_name": args.task, "task_type": "regression", "protocol": args.protocol}
        with tempfile.TemporaryDirectory() as folder:
            config = {"task_name": args.task, "ace": {"module": "harness.ace_stub"},
                      "playbook_path": str(Path(folder) / "playbook.md")}
            result = preset.seed(config, context, args.file)
            text = Path(config["playbook_path"]).read_text(encoding="utf-8")
        print(text)
        print(f"[team_sim] 写入 {len(result['seeded'])} 条：{', '.join(result['seeded'])}；共 {len(text)} 个字符"
              f"（team_ace 渲染上限 1900，提示词截断上限 2500）", file=sys.stderr)
        return 0
    if args.command == "ace-retry":
        from .ace_retry import describe, retry
        ask = None
        if sys.stdin.isatty():
            import getpass
            ask = lambda name: getpass.getpass(f"${name} 没有设置，请输入模型密钥（输入时不显示）: ")  # noqa: E731
        try:
            result = retry(args.search_dir, approve_live=args.ace_approve_live,
                           thinking=False if args.ace_no_thinking else None,
                           request_timeout=args.ace_request_timeout, max_tokens=args.ace_max_tokens,
                           worker_seconds=args.ace_worker_seconds, max_jobs=args.ace_max_jobs,
                           include_failed=args.include_failed, ask_key=ask, package=args.package,
                           max_calls=args.ace_max_calls)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
        print("\n".join(describe(result)))
        return 0

    from .data import PROTOCOLS
    from .decide import POLICIES
    from .run import Simulation
    protocols, policies = _list(args.protocols), _list(args.policies)
    for p in protocols:
        if p not in PROTOCOLS:
            parser.error(f"unknown protocol {p!r}")
    for p in policies:
        if p not in POLICIES:
            parser.error(f"unknown policy {p!r}")
    searcher = None
    if args.search == "evolve":
        if not args.api_base or not args.model:
            parser.error("--search evolve needs --api-base and --model")
        from .search import Search
        try:
            searcher = Search(args.api_base, args.model, iterations=args.iterations, workers=args.workers,
                              oe_config=args.oe_config, timeout=args.search_timeout,
                              ace="team_ace" if args.ace == "team_ace" else "stub", ace_mode=args.ace_mode,
                              ace_max_calls=args.ace_max_calls, ace_approve_live=args.ace_approve_live,
                              ace_model=args.ace_model, ace_api_base=args.ace_api_base,
                              ace_api_key_env=args.ace_api_key_env, ace_worker_seconds=args.ace_worker_seconds,
                              ace_max_jobs=args.ace_max_jobs, ace_request_timeout=args.ace_request_timeout,
                              ace_max_tokens=args.ace_max_tokens,
                              ace_thinking=False if args.ace_no_thinking else None,
                              ace_max_shown=args.ace_max_shown, seeds_path=args.seeds)
        except ValueError as error:
            parser.error(str(error))
    sim = Simulation(args.package, args.out, task=args.task, protocols=protocols, policies=policies,
                     folds=_ints(args.folds), rounds=args.rounds, epochs=args.epochs, max_epochs=args.max_epochs,
                     resamples=args.resamples, seed=args.seed, keep_data=args.keep_data, search=searcher,
                     force_search=args.force_search)
    result = sim.run()
    print(json.dumps({"runs": result["runs"], "report": result["report"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
