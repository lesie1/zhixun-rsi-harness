"""Continual adaptation over time-ordered batches (see docs/ADAPTATION.md).

  python -m team_adapt check    --config configs/adapt_A_local.json [--out runs/adapt/A]
  python -m team_adapt run      --config configs/adapt_A_local.json --out runs/adapt/A [--allow-llm]
  python -m team_adapt status   --out runs/adapt/A
  python -m team_adapt final    --config configs/adapt_A_local.json --out runs/adapt/A
  python -m team_adapt report   --out runs/adapt/A
  python -m team_adapt simulate --base-config configs/run_A.json --chunks 4 --out runs/adapt/A-sim

`run` never calls a model unless the config's search.mode is "evolve" AND --allow-llm is given.
Exit codes: 0 finished, 1 error, 2 bad arguments, 3 waiting for data, 4 stopped.
"""
import argparse
import json
import sys
from pathlib import Path

import harness  # noqa: F401  (pins BLAS threads before numpy, like the reference trainer)

from harness.dataio import file_sha256, read_json, read_rows, write_json

from . import config as config_mod
from . import report as report_mod
from .controller import EXIT_CODES, Experiment
from .ledger import Ledger


def cmd_check(args):
    cfg = config_mod.load_config(args.config)
    ledger_path = cfg.get("exposure_ledger") or (Path(args.out) / "exposure_ledger.json" if args.out else None)
    ledger = Ledger(ledger_path) if ledger_path else None
    final_id = (cfg.get("final") or {}).get("batch")
    print(f"实验 {cfg['experiment_id']}（任务 {cfg['task_name']}），训练方式 {cfg['training']['mode']}，"
          f"搜索 {cfg['search']['mode']}，切换条件 {cfg['advance_on']}")
    print("批次（按时间顺序）：")
    for b in cfg["batches"]:
        exists = Path(b["path"]).is_file()
        detail = "文件不存在"
        if exists and b["id"] != final_id and b["labels_available"]:
            sha = file_sha256(b["path"])
            others = [f"{e['experiment_id']}:{e['use']}" for e in (ledger.uses(sha, cfg["experiment_id"]) if ledger else [])]
            detail = f"{sum(1 for _ in read_rows(b['path']))} 行，{sha[7:19]}…" + (f"，台账：{others}" if others else "")
            try:
                config_mod.check_sha256(b, sha)
            except ValueError as error:
                detail += f"  错误：{error}"
        elif exists and b["id"] == final_id:
            detail = "最终批次：final 命令之前不读取内容"
        elif exists:
            detail = "标签不可用：不读取"
        flag = "【已暴露】" if b["exposure"] else ""
        print(f"  {b['id']:>6} order={b['order']:<5} uses={'/'.join(b['uses']):<10} 标签{'可用' if b['labels_available'] else '不可用'}  "
              f"{detail} {flag}{'；'.join(b['exposure'])}")
    print("阶段计划（只有前一阶段满足切换条件、且数据就绪时才会推进）：")
    devs = {}
    for stage in cfg["plan"]:
        ready, reasons = config_mod.readiness(cfg, stage)
        devs.setdefault(stage["dev"], []).append(stage["index"])
        print(f"  第 {stage['index']} 阶段：训练 {'+'.join(stage['train'])}（新 {'+'.join(stage['new_train'])}），"
              f"开发 {stage['dev']} —— {'数据就绪' if ready else '等待：' + '；'.join(reasons)}")
    for dev, stages in devs.items():
        if len(stages) > 1:
            print(f"  注意：开发批次 {dev} 被第 {stages} 阶段重复使用")
    for stage in cfg["plan"]:
        if cfg_batch(cfg, stage["dev"])["exposure"]:
            print(f"  注意：第 {stage['index']} 阶段的开发批次 {stage['dev']} 已暴露，在它上面的结果只算开发证据")
    if final_id:
        f = cfg_batch(cfg, final_id)
        if f["exposure"] and not cfg["final"].get("allow_exposed"):
            print(f"  错误：最终批次 {final_id} 已暴露，final 命令会拒绝运行（或设 final.allow_exposed=true 并接受“非独立”标记）")
        elif f["exposure"]:
            print(f"  注意：最终批次 {final_id} 已暴露，最终验收结果会标为非独立")
    else:
        print("  没有声明最终批次：只能得到开发证据")
    if cfg["search"]["mode"] == "evolve":
        print("  evolve 模式会调用大模型：run 需要显式加 --allow-llm；每阶段最多 "
              f"{cfg['budget']['stage_max_iterations']} 次迭代、{cfg['budget']['stage_max_seconds']} 秒")
    return 0


def cfg_batch(cfg, batch_id):
    return next(b for b in cfg["batches"] if b["id"] == batch_id)


def cmd_run(args):
    experiment = Experiment(args.config, args.out, allow_llm=args.allow_llm)
    code = experiment.run()
    report_mod.build(args.out)
    print(f"报告：{Path(args.out) / 'report.md'}")
    return code


def cmd_status(args):
    state = read_json(Path(args.out) / "state.json")
    print(f"{state['experiment_id']}：{state['status']}" + (f"（{state['message']}）" if state.get("message") else ""))
    print(f"当前模型：{state['current']['from']}  {state['current']['checkpoint']}")
    for stage in state["stages"]:
        search = stage.get("search") or {}
        gate = stage.get("gate") or {}
        print(f"  {stage['stage_id']}: {stage['status']}；训练 {'+'.join(stage['plan']['train'])} 开发 {stage['plan']['dev']}；"
              f"搜索 {search.get('end_reason', '-')}（{search.get('iterations', 0)} 次）；验收 {gate.get('decision', '-')}")
    if state.get("final"):
        print(f"  最终验收：{'独立' if state['final']['independent'] else '非独立'}，{state['final']['path']}")
    return EXIT_CODES.get(state["status"], 0)


def cmd_final(args):
    experiment = Experiment(args.config, args.out)
    result = experiment.final(allow_reuse=args.allow_reuse)
    if isinstance(result, int):
        return result
    report_mod.build(args.out)
    primary = result["table"]["primary"]
    print(f"最终验收（{result['batch']}，{'独立' if result['independent'] else '非独立'}）：")
    for arm, value in result["table"]["sets"]["final"]["arms"].items():
        print(f"  {arm:28s} {primary}={value['overall'][primary]:.4f}")
    return 0


def cmd_report(args):
    print(report_mod.build(args.out))
    return 0


def cmd_simulate(args):
    """Local public-data simulation: official train = b1, the public validation file cut into chunks."""
    base_path = Path(args.base_config).resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    resolve = lambda p: str(Path(p) if Path(p).is_absolute() else (base_path.parent / p).resolve())
    out = Path(args.out).resolve()
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    lines = [line for line, _ in read_rows(resolve(base["validation_path"]))]
    if args.chunks < 3 or len(lines) < args.chunks:
        raise ValueError("chunks 至少为 3，且不能多于验证集行数")
    bounds = [round(i * len(lines) / args.chunks) for i in range(args.chunks + 1)]
    exposure = ["来自官方公开验证集：此前的搜索、验收和诊断实验都用过它，不是独立数据"]
    batches = [{"id": "b1", "order": 1, "path": resolve(base["train_path"]), "labels_available": True,
                "uses": ["train"], "exposure": ["官方基线检查点就是用它训练的"]}]
    for k in range(args.chunks):
        path = data / f"c{k + 1}.jsonl"
        path.write_text("".join(line + "\n" for line in lines[bounds[k]:bounds[k + 1]]), encoding="utf-8")
        last = k == args.chunks - 1
        batches.append({"id": f"c{k + 1}", "order": k + 2, "path": str(path), "labels_available": True,
                        "uses": ["final"] if last else ["train", "dev"], "exposure": exposure})
    config = {"experiment_id": f"{base.get('task_name', 'X')}-sim-{args.chunks}", "task_name": base.get("task_name"),
              "base_config": str(base_path), "initial_train_batches": ["b1"], "batches": batches,
              "training": {"mode": args.mode}, "search": {"mode": "continuation"},
              "final": {"batch": f"c{args.chunks}", "allow_exposed": True},
              "focus_classes": args.focus or []}
    write_json(out / "experiment.json", config)
    print(f"已写入 {out / 'experiment.json'}（{args.chunks - 1} 个可训练/开发的分段 + 1 个最终分段）")
    print(f"下一步：python -m team_adapt check --config {out / 'experiment.json'}")
    print(f"        python -m team_adapt run   --config {out / 'experiment.json'} --out {out / 'run'}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m team_adapt", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="校验配置、列出批次用途和暴露情况、给出阶段计划（不训练）")
    p.add_argument("--config", required=True)
    p.add_argument("--out")
    p = sub.add_parser("run", help="按阶段推进，直到等待数据、停止或计划用完；可断点续跑")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-llm", action="store_true", help="允许 evolve 模式调用大模型（可能产生费用）")
    p = sub.add_parser("status", help="查看状态")
    p.add_argument("--out", required=True)
    p = sub.add_parser("final", help="在预留的最终批次上做一次性最终验收")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-reuse", action="store_true", help="允许再次使用最终批次（结果标为非独立）")
    p = sub.add_parser("report", help="重新生成 report.md")
    p.add_argument("--out", required=True)
    p = sub.add_parser("simulate", help="用官方训练集 + 切开的公开验证集生成本地模拟实验配置")
    p.add_argument("--base-config", required=True)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--mode", choices=config_mod.TRAINING_MODES, default="replay")
    p.add_argument("--focus", nargs="*", help="报告里重点列出的类别，例如 Acetaldehyde")
    p.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    handler = {"check": cmd_check, "run": cmd_run, "status": cmd_status, "final": cmd_final,
               "report": cmd_report, "simulate": cmd_simulate}[args.command]
    try:
        return handler(args)
    except (config_mod.ConfigError, ValueError, RuntimeError, PermissionError, FileNotFoundError, FileExistsError) as error:
        print(f"失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
