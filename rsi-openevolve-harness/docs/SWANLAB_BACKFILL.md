# 把已有实验补录到 SwanLab

2026-11-25 新增 `scripts/swanlab_backfill.py`。它把服务器上已经跑完的实验产物导入 SwanLab，每条记录一个 SwanLab 实验。不重新训练，不改任何数字，不上传检查点、数据文件或密钥。

## 收录什么

| 类型 | 找哪个文件 | 在 SwanLab 里是什么 |
| --- | --- | --- |
| train | 含 `training.json` 的目录（同目录 `history.jsonl`） | 一次工业模型（MLP）训练：逐 epoch 的训练损失和公开验证指标曲线，训练设置作为 config，旁边有策略文件时附上策略源码 |
| gate | `gate/decision.json`、`final_acceptance.json` | 一次验收决定：决定、理由、各模型指标、bootstrap 概率 |
| sim | `rounds.jsonl`（同目录 `summary.json`） | 模拟器的一条运行：每轮数据量、红黄灯决定、测试分数 |
| search | `records/iterations.jsonl` | 一次 OpenEvolve 搜索：每次迭代的分数、状态、假设 |
| record | 用 `--extra 文件名` 指定的其他结果文件，如 `results.json` | 文件里的数字作为指标，全文作为文本 |

默认跳过 `tests/`、`cache/`、`data/`、`locks/`、`openevolve_output/` 目录。

每条记录都标明是补录：描述里写原始文件路径和原始时间，config 里有 `backfilled: true`、`source_path`、`original_time`，标签里有 `backfill`。SwanLab 上显示的创建时间是上传时间，不是实验时间。

## 怎么用

```bash
python -m pip install swanlab
swanlab login            # 自己输入 API Key；脚本不读也不存密钥

# 1. 先看会上传什么
python scripts/swanlab_backfill.py --root runs --root ../别的项目/runs --extra results.json --dry-run

# 2. 上传；--public 在新建项目时设为公开（组委会要求实验记录公开）
python scripts/swanlab_backfill.py --root runs --root ../别的项目/runs --extra results.json \
    --project zhixun-industrial-rsi --public
```

- 台账 `state/swanlab_ledger.json` 记住已上传的文件和摘要；再次运行只传新增或变化的记录，所以以后每跑完一批实验再运行一次即可。
- 服务器连不上 SwanLab 时用 `--mode offline --log-dir swanlog`，把 `swanlog/` 拷到能联网的机器上运行 `swanlab sync swanlog/run-*`。
- `--include` / `--exclude` 用通配符筛路径，`--kinds` 选类型，`--max` 限制条数。
- 项目已经存在且是私有的，`--public` 不会改它，要在 SwanLab 网页的项目设置里改成公开。

## 已验证

- `tests/test_swanlab_backfill.py`：4 项通过（发现、跳过目录、各类记录的内容、dry-run 不写台账）。
- 用 SwanLab 0.10.6 离线模式导入本地 `runs/gas_demo` 的训练、验收、搜索记录和一条模拟器运行，共 13 条，无失败，日志无警告。
