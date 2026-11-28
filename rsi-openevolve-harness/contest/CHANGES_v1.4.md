# 比赛镜像 v1.4 改动说明（2026-11-28）

镜像标签 `zhixun-rsi:v1.4`。直接上传 `zhixun-rsi-v1.4.tar`：193,331,200 字节（约 188,800 KB），SHA256：`3161fcd82279226689ec7fcc95ca0b986283622530d2978b5b7cd84eeb87799d`（见 `SHA256SUMS`）。

## 改了什么（只对 A 新增一个候选，外加全任务 UTF-8 文本长度保护；B 和 C 逻辑与 v1.3 相同）

1. **A 任务新增同期校准候选 `same_period`**（借鉴队友 `067e1af` 的 `a_same_period.py`）：
   - 去掉了原本永远为空的人工公告清单锁（`REVIEWED_RELEASES`），直接复用 `/state` 中已有的跨轮新训练样本识别结果（第 2 轮起、当本轮存在新训练样本且不全是新样本时触发）。
   - 按队友冻结的 Qwen 经验卡规则自动选择抽样池：
     - 若新样本覆盖类别数 `<= 2` 且占当前训练集比例 `<= 10%`：使用 `history_mix`（旧样本 + 新样本合并池均匀有放回抽样）；
     - 否则：使用 `recent_only`（仅从本轮新样本中有放回抽样）。
   - 从父模型检查点副本出发，以固定种子 `SeedSequence([20261127, round_index, seed])` 执行 `20` 步 Adam 更新（每步抽取 `128` 条）。
   - 生成的 `same_period` 与原有的 `continue`、`pick`、`balanced`、`pick+balanced`、`recent+balanced` 一起进入同一个配对 bootstrap 验收门（`P >= 90%`、准确率变差不超过 `2%`、点估计更优）及 Qwen 审阅，不通过则不上线，出错时自动跳过该候选不影响整轮。
2. **SDK 4000 字节 UTF-8 上限保护（`clip_utf8`）**：
   - SDK 对 `hypothesis`、`change_summary`、`reason` 按 **UTF-8 字节数 `<= 4000`** 校验（原代码 `[:3900]` 按字符截断，中文多字节文本极端情况下可能超过 4000 字节）。现统一改为按 UTF-8 字节安全截断到 `<= 3900` 字节。

## 验证结果
从打包生成的 `zhixun-rsi-v1.4.tar` 中解出镜像内的 `/app`（设为只读权限模拟容器 `65534:65534`）和镜像自带的 `numpy==1.26.4` + `contest_harness==1.0.4`，运行全部 16 项测试（含入口预检、`describe`、A/B/C 单轮与多轮、`same_period` 权重逐位一致性、`same_period` 胜出上线→下一轮续训→`predict` 全链路、UTF-8 字节截断）：16 项全部通过。
