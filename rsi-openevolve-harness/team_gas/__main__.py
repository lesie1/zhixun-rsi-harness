"""A-type gas classification adaptation flow (see docs/GAS_FLOW.md).

  python -m team_gas check  --config configs/gas_A.example.json
  python -m team_gas run    --config <flow.json> --out <dir> [--allow-llm]
  python -m team_gas status --out <dir>
  python -m team_gas report --config <flow.json> --out <dir>
  python -m team_gas prompt --run-config <dir>/rounds/r01/repair/run.json
  python -m team_gas synth  --out <dir> [--ace-storage state/demo-gas]

Exit codes: 0 finished, 1 error, 2 bad arguments, 3 waiting for data, 4 stopped.
"""
import argparse
import sys
from pathlib import Path

import harness  # noqa: F401  (pins BLAS threads before numpy, like the reference trainer)

from harness.dataio import read_json

from . import config as config_mod
from . import playbook, synth
from .controller import EXIT_CODES, GasFlow


def cmd_check(args):
    cfg = config_mod.load(args.config)
    print(f"A 流程 {cfg['experiment_id']}：修复搜索 {cfg['repair']['search']['mode']}，切换条件 {cfg['repair']['advance_on']}，"
          f"新数据上限 {cfg['mixture']['cap']:.0%}（硬上限 20%），候选比例 {cfg['mixture']['fractions']}，"
          f"退化类份额 {cfg['mixture']['degraded_shares']}")
    for b in cfg["batches"]:
        exists = Path(b["path"]).is_file()
        reading = "（final：最终验收前不读取）" if "final" in b["uses"] else ""
        print(f"  {b['id']:>4} order={b['order']:<4} uses={'/'.join(b['uses']) or '（只登记）':<12} 标签{'可用' if b['labels_available'] else '不可用'} "
              f"{'存在' if exists else '文件不存在'}{reading}  {'【已暴露】' + '；'.join(b['exposure']) if b['exposure'] else ''}")
    for i, r in enumerate(cfg["rounds"], 1):
        ok, why = config_mod.ready(cfg, r["alarm"])
        print(f"  第 {i} 轮：报警 {r['alarm']} → 开发 {r['selection']} → 最终 {r['final']}  "
              f"{'报警批次就绪' if ok else '等待：' + '；'.join(why)}")
        final = config_mod.batch(cfg, r["final"])
        if final["exposure"]:
            print(f"    错误：最终批次 {final['id']} 已暴露，验收会被拒绝")
    for entry in playbook.load_base(cfg["playbook"]["base_experiences"]):
        print(f"  基础经验 {entry['id']}：{entry['title']}（条件 {entry['conditions']}）")
    if cfg["repair"]["search"]["mode"] == "evolve":
        print("  evolve 模式会调用大模型：run 需要显式加 --allow-llm")
    return 0


def cmd_run(args):
    flow = GasFlow(args.config, args.out, allow_llm=args.allow_llm)
    code = flow.run()
    for rnd in flow.state["rounds"]:
        if (Path(rnd["folder"]) / "record.md").is_file():
            print(f"记录：{Path(rnd['folder']) / 'record.md'}")
    return code


def cmd_status(args):
    state = read_json(Path(args.out) / "state.json")
    print(f"{state['experiment_id']}：{state['status']}" + (f"（{state['message']}）" if state.get("message") else ""))
    print(f"当前部署：{state['deployed']['from']}  {state['deployed']['checkpoint']}")
    for rnd in state["rounds"]:
        print(f"  第 {rnd['index']} 轮：{rnd['status']}  {rnd.get('outcome', '')} {rnd.get('outcome_text', '')}")
    return EXIT_CODES.get(state["status"], 0)


def cmd_report(args):
    from . import report
    flow = GasFlow(args.config, args.out)
    for rnd in flow.state["rounds"]:
        print(report.write_round(flow, rnd))
    return 0


def cmd_prompt(args):
    prompt = playbook.render_prompt(args.run_config, args.oe_config)
    text = prompt["user"]
    start = text.find("# 搜索经验（playbook）")
    print(text[start:start + 3000] if start >= 0 else text[:3000])
    for entry in playbook.load_base(config_mod.DEFAULTS["playbook"]["base_experiences"]):
        print(f"\n[{entry['id']}] 在这份真实提示中出现 {playbook.occurrences(prompt, entry['id'])} 次")
    return 0


def cmd_synth(args):
    made = synth.make(args.out, ace_storage=args.ace_storage)
    print(f"已生成合成数据和配置：{made['flow']}")
    print(f"下一步：python -m team_gas run --config {made['flow']} --out {Path(args.out) / 'run'}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m team_gas", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check", help="校验配置，列出批次用途、暴露和轮次（不训练）")
    p.add_argument("--config", required=True)
    p = sub.add_parser("run", help="按轮次运行；可断点续跑")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-llm", action="store_true", help="允许修复搜索调用大模型（可能产生费用）")
    p = sub.add_parser("status")
    p.add_argument("--out", required=True)
    p = sub.add_parser("report", help="重新生成每轮 record.md")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("prompt", help="用真实的提示词构造函数渲染某个阶段的提示，检查 playbook")
    p.add_argument("--run-config", required=True)
    p.add_argument("--oe-config")
    p = sub.add_parser("synth", help="生成合成的分批气体分类数据和流程配置")
    p.add_argument("--out", required=True)
    p.add_argument("--ace-storage", help="team_ace 经验库目录（必须在项目 state/ 下）；不给则不用 team_ace")
    args = parser.parse_args(argv)
    handler = {"check": cmd_check, "run": cmd_run, "status": cmd_status, "report": cmd_report,
               "prompt": cmd_prompt, "synth": cmd_synth}[args.command]
    try:
        return handler(args)
    except (config_mod.ConfigError, ValueError, RuntimeError, PermissionError, FileNotFoundError, AssertionError) as error:
        print(f"失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
