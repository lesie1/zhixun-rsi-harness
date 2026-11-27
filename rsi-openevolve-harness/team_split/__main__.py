"""Command line of the split module.

  python -m team_split make   --config configs/run_A.json --out splits/A-v1.json [--suffix v1]
  python -m team_split verify --config configs/run_A.json --manifest splits/A-v1.json
  python -m team_split report --config configs/run_A.json --manifest splits/A-v1.json
  python -m team_split wire   --config configs/run_A.json --manifest splits/A-v1.json \
                              --out configs/run_A.split-v1.json --suffix split-v1

Exit codes: 0 ok, 1 the manifest has errors or an operation was refused, 2 bad arguments.
"""
import argparse
import json
import sys
from pathlib import Path

import harness  # noqa: F401

from harness.dataio import write_json
from harness.runcfg import load_run_config

from .manifest import build_manifest, read_manifest, write_manifest
from .report import render_report, split_report
from .strategies import BASE_METHODS, SplitOptions, options_for_manifest, split
from .table import load_table
from .verify import has_errors, verify_manifest
from .wiring import derive_run_config


def sidecar_path(manifest_path):
    path = Path(manifest_path)
    return path.with_name(path.stem + ".report.json")


def _option_args(parser):
    parser.add_argument("--seed", type=int, help="默认取运行配置的 seed")
    parser.add_argument("--min-groups", type=int, help="默认取运行配置的 split.min_groups")
    parser.add_argument("--time-blocks", type=int, help="time_blocks 方法的块数，默认 10")
    parser.add_argument("--method", choices=("auto",) + BASE_METHODS, help="默认 auto（按数据自动选择）")
    parser.add_argument("--balance", choices=("groups", "rows"), help="分组方法的分配方式，默认 groups")


def _options(args, config, stored=None):
    base = dict(stored or {})
    base.setdefault("seed", config.get("seed", 42))
    base.setdefault("min_groups", (config.get("split") or {}).get("min_groups", 4))
    for name in ("seed", "min_groups", "time_blocks", "method", "balance"):
        value = getattr(args, name, None)
        if value is not None:
            base[name] = value
    return SplitOptions(**base).validate()


def _print_problems(problems):
    if not problems:
        print("校验：没有发现问题")
    for p in problems:
        print(f"[{'错误' if p.severity == 'error' else '提示'}] {p.code}: {p.message}")


def cmd_make(args):
    config = load_run_config(args.config)
    stage_id = args.stage_id or config["stage_id"] + (f"-{args.suffix}" if args.suffix else "")
    table = load_table(config)
    options = _options(args, config)
    manifest = build_manifest(table, split(table, options), stage_id, options)
    problems = verify_manifest(manifest, table, stage_id, options)
    _print_problems(problems)
    if has_errors(problems):
        print("清单有错误，没有写文件")
        return 1
    report = split_report(manifest, table)
    write_manifest(manifest, args.out, overwrite=args.force)
    write_json(sidecar_path(args.out), {
        "tool": "team_split", "config": str(Path(args.config).resolve()), "manifest": str(Path(args.out).resolve()),
        "manifest_sha256": manifest["manifest_sha256"], "options": options.to_dict(),
        "problems": [p.to_dict() for p in problems], "report": report})
    print(render_report(report))
    print(f"已写入 {args.out}（摘要 {manifest['manifest_sha256'][:23]}…）和 {sidecar_path(args.out)}")
    return 0


def _stored_options(manifest_path):
    sidecar = sidecar_path(manifest_path)
    if sidecar.is_file():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8")).get("options")
        except (OSError, json.JSONDecodeError):
            return None
    return None


def cmd_verify(args):
    config = load_run_config(args.config)
    table = load_table(config)
    manifest = read_manifest(args.manifest)
    stored = _stored_options(args.manifest)
    if stored is None:
        stored = options_for_manifest(manifest).to_dict()
        stored.pop("min_groups"), stored.pop("time_blocks")
    options = None if args.no_reproduce else _options(args, config, stored)
    stage_id = None if args.no_stage_check else config["stage_id"]
    problems = verify_manifest(manifest, table, stage_id, options)
    _print_problems(problems)
    return 1 if has_errors(problems) else 0


def cmd_report(args):
    config = load_run_config(args.config)
    table = load_table(config)
    print(render_report(split_report(read_manifest(args.manifest), table)))
    return 0


def cmd_wire(args):
    config = derive_run_config(args.config, args.manifest, args.out, args.suffix, force=args.force)
    print(f"已生成 {args.out}")
    print(f"  stage_id: {config['stage_id']}")
    print(f"  work_dir: {config['work_dir']}")
    print(f"  split_manifest: {config['split_manifest']}")
    if "playbook_path" in config and isinstance(config.get("ace"), dict) and config["ace"].get("module") == "team_ace":
        print(f"  ACE 经验库（新 scope，空库）: {Path(config['playbook_path']).parent}")
    manifest = read_manifest(args.manifest)
    if manifest.get("stage_id") != config["stage_id"]:
        print(f"  注意：清单的 stage_id 是 {manifest.get('stage_id')!r}，新配置是 {config['stage_id']!r}。"
              f"下次可在 make 时加 --suffix {args.suffix} 让两者一致")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m team_split", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("make", help="生成划分清单（有错误时不写文件）")
    make.add_argument("--config", required=True)
    make.add_argument("--out", required=True)
    make.add_argument("--stage-id", help="清单的 stage_id，默认取运行配置的 stage_id")
    make.add_argument("--suffix", help="与 wire 用同一个 suffix，清单的 stage_id 就和新配置一致")
    make.add_argument("--force", action="store_true", help="允许覆盖内容不同的同名清单（已被配置引用的清单不要覆盖）")
    _option_args(make)
    verify = sub.add_parser("verify", help="严格校验清单")
    verify.add_argument("--config", required=True)
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--no-stage-check", action="store_true", help="不检查 stage_id 是否与配置一致")
    verify.add_argument("--no-reproduce", action="store_true", help="不检查能否用同样参数重建")
    _option_args(verify)
    report = sub.add_parser("report", help="打印划分质量报告")
    report.add_argument("--config", required=True)
    report.add_argument("--manifest", required=True)
    wire = sub.add_parser("wire", help="生成引用该清单的新运行配置")
    wire.add_argument("--config", required=True)
    wire.add_argument("--manifest", required=True)
    wire.add_argument("--out", required=True)
    wire.add_argument("--suffix", required=True)
    wire.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    handler = {"make": cmd_make, "verify": cmd_verify, "report": cmd_report, "wire": cmd_wire}[args.command]
    try:
        return handler(args)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        print(f"失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
