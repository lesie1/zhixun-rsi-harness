"""Batch loop: one decision per arriving batch ("deploy a new model" or "keep the current one").

  python -m team_loop init     --base-config configs/run_A.json --out runs/loop/A
  python -m team_loop check    --out runs/loop/A --validation new_batch.jsonl
  python -m team_loop step     --out runs/loop/A --batch-id b03 --train history.jsonl --validation new_batch.jsonl
  python -m team_loop run      --out runs/loop/A --batches batches.json [--base-config configs/run_A.json]
  python -m team_loop simulate --base-config configs/run_A.json --chunks 3 --out runs/loop/A-sim/data
  python -m team_loop status   --out runs/loop/A

Search options (step / run): --search evolve|continuation, --oe-config, --iterations,
--api-base, --model, --workers, --timeout. The API key is read from the environment by
the OpenEvolve config, never passed on the command line.
Trigger options (check / step / run): --retrain auto|always|never, --primary-drop,
--confidence, --min-rows, --no-guards.

Exit codes: 0 ok, 1 refused or failed, 2 bad arguments.
"""
import argparse
import json
import sys
from pathlib import Path

import harness  # noqa: F401  (pins BLAS threads before numpy, like the reference trainer)

from harness.dataio import write_json

from . import batches as batches_mod
from . import state as state_mod
from . import step as step_mod
from . import trigger


def _trigger_args(parser):
    parser.add_argument("--primary-drop", type=float, help="主指标相对变差超过多少触发重训，默认 0.05")
    parser.add_argument("--confidence", type=float, help="自助法判定“确实变差”的概率门槛，默认 0.90")
    parser.add_argument("--min-rows", type=int, help="新批次少于多少行不重训，默认 50")
    parser.add_argument("--no-guards", action="store_true", help="只看主指标，不看辅助指标")


def _trigger_options(args):
    return {"primary_drop": args.primary_drop, "confidence": args.confidence, "min_rows": args.min_rows,
            "use_guards": False if args.no_guards else None}


def _search_args(parser):
    parser.add_argument("--retrain", choices=step_mod.RETRAIN_MODES, default="auto",
                        help="auto 按指标判断（默认）；always 每批都训；never 从不训（用来算“不训练”的对照）")
    parser.add_argument("--search", choices=step_mod.SEARCH_MODES, default="evolve",
                        help="evolve 用 OpenEvolve 搜索（默认）；continuation 不调大模型，只比较官方续训和当前模型")
    parser.add_argument("--oe-config")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--api-base")
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--timeout", type=float, help="搜索最多运行多少秒，超时后照常进入验收门")
    _trigger_args(parser)


def _search_options(args):
    return {"oe_config": args.oe_config and str(Path(args.oe_config).resolve()), "iterations": args.iterations,
            "api_base": args.api_base, "model": args.model, "workers": args.workers, "timeout": args.timeout}


def cmd_init(args):
    state = state_mod.init(args.base_config, args.out, args.reference_data)
    ref = state["deployed"]["reference"]
    print(f"已初始化 {args.out}：当前模型 {state['deployed']['checkpoint']}")
    print(f"  参照水平（在 {ref['measured_on']} 上，{ref['rows']} 行）: "
          + "，".join(f"{k}={v:.4g}" for k, v in ref["metrics"].items()))
    return 0


def cmd_check(args):
    state = state_mod.load(args.out)
    from baselines.model import Model
    deployed = state["deployed"]
    spec = trigger.spec_for(Model.load(deployed["checkpoint"]), state["task_name"], state.get("spec_file"))
    report = trigger.check(deployed["checkpoint"], args.validation, deployed.get("reference"), spec,
                           _trigger_options(args), cache_dir=Path(args.out) / "cache")
    print(trigger.describe(report))
    if args.json:
        write_json(args.json, report)
    return 0


def _process(args, batch):
    return step_mod.process(args.out, batch, retrain=args.retrain, search=args.search,
                            search_args=_search_options(args), trigger_options=_trigger_options(args))


def cmd_step(args):
    batch = batches_mod.normalize({"batch_id": args.batch_id, "train_path": args.train,
                                   "validation_path": args.validation})
    _process(args, batch)
    return 0


def cmd_run(args):
    if not state_mod.state_path(args.out).is_file():
        if not args.base_config:
            raise ValueError("循环目录还没初始化：加 --base-config，或先运行 init")
        state_mod.init(args.base_config, args.out)
    for batch in batches_mod.read_batches(args.batches):
        print(f"\n===== batch {batch['batch_id']} =====", flush=True)
        _process(args, batch)
    return cmd_status(args)


def cmd_simulate(args):
    for batch in batches_mod.simulate(args.base_config, args.chunks, args.out):
        print(f"{batch['batch_id']}: 训练 {batch['train_path']}  新批次 {batch['validation_path']}")
    print(f"已写入 {Path(args.out) / 'batches.json'}")
    return 0


def cmd_status(args):
    state = state_mod.load(args.out)
    print(f"当前模型: {state['deployed']['checkpoint']}（自 {state['deployed']['since']} 起）")
    for entry in state["history"]:
        print(f"  {entry['batch_id']}: 重训={'是' if entry['retrain'] else '否'}  上线={'是' if entry['deploy'] else '否'}"
              f"  验收门={entry['gate_decision'] or '-'}  用时 {entry['seconds']} s"
              + (f"  出错: {entry['error']}" if entry.get("error") else ""))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m team_loop", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("init", help="建立循环目录，把基础配置的检查点作为第一个上线模型")
    p.add_argument("--base-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--reference-data", help="在哪份数据上量参照水平，默认基础配置的 train_path")
    p = sub.add_parser("check", help="只做重训判断，不训练")
    p.add_argument("--out", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--json", help="把判断报告写到这个文件")
    _trigger_args(p)
    p = sub.add_parser("step", help="处理一批数据并输出决定")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-id", required=True)
    p.add_argument("--train", required=True, help="可用于训练的全部历史数据")
    p.add_argument("--validation", required=True, help="最新一批有标签数据")
    _search_args(p)
    p = sub.add_parser("run", help="按 batches.json 依次处理每一批")
    p.add_argument("--out", required=True)
    p.add_argument("--batches", required=True)
    p.add_argument("--base-config", help="循环目录还没初始化时用它初始化")
    _search_args(p)
    p = sub.add_parser("simulate", help="把公开验证集按文件顺序切成几批，用来本地测试")
    p.add_argument("--base-config", required=True)
    p.add_argument("--chunks", type=int, default=3)
    p.add_argument("--out", required=True)
    p = sub.add_parser("status", help="查看每一批的决定")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    handler = {"init": cmd_init, "check": cmd_check, "step": cmd_step, "run": cmd_run,
               "simulate": cmd_simulate, "status": cmd_status}[args.command]
    try:
        return handler(args)
    except (ValueError, FileExistsError, FileNotFoundError, RuntimeError, json.JSONDecodeError) as error:
        print(f"失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
