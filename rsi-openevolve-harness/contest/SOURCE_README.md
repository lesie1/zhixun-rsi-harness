# 智训引擎 · 工业模型自进化系统 —— 源代码包说明

## 目录

| 路径 | 内容 |
| --- | --- |
| `rsi-openevolve-harness/contest/` | **比赛镜像入口（v1.4）**：`contest_submission.py`（SDK 的 `train(rnd)` / `predict`，含 A 同期校准候选 `same_period`、B 新样本按需训练与验收、C 有限动作智能体接入）、`c_agent/`（C 任务 12 动作智能体）、`Dockerfile`、`build_context.sh`、`build_image_nodocker.py`、端到端测试 `test_contest_submission.py`、候选策略 `policies/policy_pick.py` 及 `CHANGES_v1.1.md`–`CHANGES_v1.4.md` |
| `rsi-openevolve-harness/harness/` | 冻结的续训 runner（与官方续训逐字节一致）、评测与指标、验收门（配对 bootstrap）、详细反馈、划分、ACE 适配 |
| `rsi-openevolve-harness/baselines/` | 官方 NumPy MLP 基线代码（模型、数据读取、训练、预测） |
| `rsi-openevolve-harness/team_sim/` | B 任务数据流模拟器（12 根公开轴承轮换、10 轮、四种给法）、v3 更新规则、SwanLab 补录脚本见 `scripts/swanlab_backfill.py` |
| `rsi-openevolve-harness/team_ace/`、`team_loop/`、`team_gas/`、`team_adapt/`、`team_split/` | ACE 经验库适配、逐批循环、气体任务回放、划分工具 |
| `rsi-openevolve-harness/third_party/openevolve/` | OpenEvolve 0.4.1（commit 3e72b5b，Apache-2.0，原 LICENSE 保留），仅扩展 `process_parallel.py` |
| `rsi-openevolve-harness/docs/` | 钩子约定、反馈、ACE、划分、模拟器、补录、各次修改记录 |
| `experiments/L/` | A 任务（气体）ACE/规划器实验代码与协议（v19–v37、walk-forward） |
| `experiments/Y/` | C 任务（发动机）开发配置、种子策略、搜索脚本、框架检查 |

## 不包含的内容

- 数据集（官方数据包另行发放；镜像运行时由平台挂载 `/data`）、隐藏测试数据、模型权重与检查点、运行产物。
- 任何密钥：PyTRiO / SwanLab / OpenAI 兼容接口的密钥只放环境变量或平台密钥栏。打包时按文件名和内容（`sk-…`、`api_key=…` 等）扫描，命中的文件已剔除。
- 比赛 SDK（`contest_harness` 1.0.4，由组委会提供，构建镜像时从 SDK 目录复制）。

## 怎么运行

```bash
python3 -m pip install numpy==1.26.4
# harness 测试（设置官方数据包位置后运行依赖数据的测试）
HARNESS_DATA_A=… HARNESS_DATA_B=… HARNESS_DATA_C=… python3 -m unittest discover -s tests
# 比赛镜像：组装构建目录、端到端测试（SDK 自带平台替身，不调用真实模型）、构建镜像
contest/build_context.sh <SDK 目录>
APP_DIR=contest/build/app SDK_DIR=<SDK 目录> DATA_DIR=<cpu-baselines-20261110-v1> python3 contest/test_contest_submission.py
docker build -t zhixun-rsi:v1.4 contest/build && docker save zhixun-rsi:v1.4 -o zhixun-rsi-v1.4.tar
```

## 开源组件与许可证

见 `THIRD_PARTY_LICENSES.md`。
