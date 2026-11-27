# 划分清单存放目录

由 `python -m team_split make` 生成，用法见 [docs/SPLIT_MODULE.md](../docs/SPLIT_MODULE.md)。

命名规则：`<任务>-v<版本>.json`，例如 `C-v1.json`；同名的 `C-v1.report.json` 记录生成参数、校验结果和质量报告。

清单一旦被某个运行配置引用就不要再改（ACE 经验库的 scope 记着它的摘要）。要换划分，生成 `C-v2.json`，再用 `python -m team_split wire` 生成新配置。
