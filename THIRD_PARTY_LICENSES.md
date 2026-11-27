# 开源组件与许可证

| 组件 | 版本 / 来源 | 许可证 | 用在哪里 | 是否进入比赛镜像 |
| --- | --- | --- | --- | --- |
| NumPy | 1.26.4（PyPI） | BSD-3-Clause | 工业模型 MLP 的训练与预测 | 是 |
| Python | 3.12（官方 `python:3.12-slim-bookworm` 镜像） | PSF License | 运行环境 | 是 |
| contest_harness SDK | 1.0.4（组委会提供） | 按组委会提供条款使用 | 与平台通信的 `train` / `predict` 协议 | 是 |
| OpenEvolve | 0.4.1，github.com/algorithmicsuperintelligence/openevolve，commit 3e72b5b | Apache-2.0（原 LICENSE 保留在 `third_party/openevolve/LICENSE`） | 训练策略搜索（开发阶段与离线实验） | 否（v1 镜像未启用搜索） |
| ACE（Agentic Context Engineering） | github.com/ace-agent/ace，commit 82709de | Apache-2.0 | 经验库的 Reflector / Curator 角色与提示模板（`team_ace` 适配层调用） | 否（v1 镜像未启用） |
| SwanLab | PyPI `swanlab` | Apache-2.0 | 实验记录上传（`scripts/swanlab_backfill.py`） | 否 |
| 官方基线代码 | cpu-baselines-20261110-v1（组委会随数据包发布） | 按组委会提供条款使用 | `baselines/`：MLP 模型、数据读取、训练 | 是 |

驱动 Harness 的大模型：PyTRiO 平台 Qwen3.8-27B，经平台网关调用；镜像内不含模型权重和推理服务。

方法参考（未复制代码）：Self-Harness（arXiv 2606.09498，验收门思路）、SHML Self-Healing ML（NeurIPS 2024，arXiv 2411.00186，监测—诊断—适配—测试闭环）、ACE（arXiv 2510.04618）。
