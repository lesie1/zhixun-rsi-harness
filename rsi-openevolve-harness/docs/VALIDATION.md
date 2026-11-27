# 2026-11-15 收尾验证记录

结论：本次只补文档、依赖清单和忽略规则；现有代码、配置、提示模板和测试共 119 个文件与操作前 SHA-256 一致。项目 harness 首尾各运行一次完整测试，均通过。下面区分本次重跑和原有全链路证据，避免把假服务测试当作真实 Qwen 结果。

## 本次实际重跑

| 检查 | 结果 |
| --- | --- |
| 收尾前，设置三个 HARNESS_DATA 环境变量 | 27 项，31.688 秒，OK |
| 收尾后，设置三个 HARNESS_DATA 环境变量 | 27 项，30.660 秒，OK，无跳过 |
| 不设置 HARNESS_DATA_A/B/C | 27 项，OK，跳过 8 项数据相关测试 |
| 三任务初始策略与官方续训 5 轮 | A/B/C 均 IDENTICAL，退出码均为 0 |
| 三个基线检查点的指标对比 | 完整测试覆盖，与官方 training.json 对应数值一致到 12 位小数 |
| 受保护文件 SHA-256 | 119 个均与操作前一致 |
| OpenEvolve 补丁反向检查 | git apply --reverse --check 成功，当前版本已含此补丁 |
| 文档字段核对 | feedback._flags 的 18 种标签全部覆盖；build_record 初始字典的 35 个字段全部覆盖，另说明可选 prompt |
| 文档命令/片段 | 12 段 Python、Bash、JSON 语法检查通过；这是语法检查，不冒充真实模型服务执行 |

文件内容摘要前 16 位：

| 任务 | 官方续训检查点 | runner 初始策略检查点 |
| --- | --- | --- |
| A | 90150659753f6e58 | 90150659753f6e58 |
| B | a2926c0606cc2680 | a2926c0606cc2680 |
| C | c3367b6f8c8f6468 | c3367b6f8c8f6468 |

复测使用已有 Python 环境和项目外三份官方数据；没有把它们复制到最终包中。最终压缩包排除 runs、数据子目录、Python 字节码缓存；对归档内容检查 CRC、文件清单、文档链接和开发临时绝对路径残留。

## 已有 24 轮假 LLM 运行的证据

本次读取并核对了已有运行目录的 evolve_summary、每轮记录样例和 gate/decision；没有重新声称运行过真实 Qwen。为便于队友离线核查，补充脱敏后的摘要和 A/B 验收决定，C 决定沿用已有样例。

| 任务 | 轮数证据 | 验收证据 | 解释 |
| --- | --- | --- | --- |
| A | [24 轮摘要](samples/evolve_summary_A.json) | [adopt_reference](samples/gate_decision_A.json) | 候选对起点的 bootstrap 胜出频率 0.54775，未达到 0.90；官方续训通过。搜索最优不自动被采用 |
| B | [24 轮摘要](samples/evolve_summary_B.json) | [keep_parent](samples/gate_decision_B.json) | 候选对起点的胜出频率虽为 1.0，但 Bearing1_4 退化约 13.49%，超过 10% 分组护栏；官方续训也未通过 |
| C | [24 轮摘要](samples/evolve_summary_C.json) | [adopt_candidate](samples/gate_decision_C.json) | 候选通过独立验收；样例采用 51 台发动机的 cluster bootstrap |

假服务故意生成部分无效替换、禁用导入或非法参数，记录中的失败状态是覆盖失败路径的证据。上述结果只证明链路和当前规则下的决策，不证明真实大模型生成有效率或收益。

## 官方 baselines 测试的实际差异

| 位置 | 测试数 | 结果 |
| --- | --- | --- |
| 本项目 baselines/tests | 12 | 2 个 error，其余通过；缺 scripts.package_baselines |
| 原始 A 包 baselines/tests | 9 | 全部通过 |
| 原始 B 包 baselines/tests | 12 | 同样 2 个 error，在导入 scripts.package_baselines 时失败（原包连 scripts 包也未提供） |
| 原始 C 包 baselines/tests | 9 | 全部通过 |

报错的是 test_b_package_delivers_numeric_rul_code_and_weights_without_hidden_data 与 test_documented_edition_preserves_trained_code_weights_and_data。交接说明中“原始数据包同样失败”应具体限定为含这两个测试的 B 包。没有为让测试变绿而补造主办方的打包模块，也没有修改这些测试。

## 本次发现但按要求未修改的边界

- 外部清单校验未覆盖摘要、来源、分组隔离、时间顺序等全部验收标准；见 SPLIT_INTERFACE。
- ACE 同步执行且不自动处理尾批，也不自动重试失败批次；见 ACE_INTERFACE。
- gate 的实际选择条件允许“候选通过、官方续训不通过”时采用候选，但 reason 可能仍写“且不差于官方续训”；应读 comparisons 数值。
- 评分使用数值平局项而非严格字典序，极小主指标差异仍可能被平局项反转。
- SPLIT_WARNING 对所有划分 warning 都触发，固定建议文案不一定完全对应每种 warning。

真实 Qwen 27B、完整 ACE、SwanLab、Dockerfile、正式划分和主办方规则确认仍待完成，详见 [README](../README.md)。
