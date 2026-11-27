# ACE 接口：读记录、写经验、追踪引用

> 当前 A/B/C 配置使用已合并的 `team_ace`，不再使用 `ace_stub`。下文仍保留底层接口与原占位模块说明。ACE 后台处理、恢复与版本管理见 [接入说明](ACE_ADAPTER.txt)；真实 Qwen 尚未接通。

接入 ACE 不需要修改 OpenEvolve 或 evaluator：实现可导入模块的 `on_batch(records, config)`，在运行配置的 `ace.module` 填模块路径。ACE 可由 Reflector（解释实验现象）和 Curator（整理经验条目）组成；本项目只约定输入输出，没有实现完整 ACE。下面三个契约分别约束实验记录、playbook（可复用经验条目文件）和引用归因。

## 契约一：每轮记录 iterations.jsonl

路径为 `<work_dir>/records/iterations.jsonl`。JSONL 表示每行一个独立 JSON 对象；由 OpenEvolve **主进程单写、只追加**，写完记录后才交给 ACE。不要把 ACE 的总结回写成实验结果，不要从大模型措辞推断是否有效。字段以下列 [records.py 的 build_record](../harness/records.py) 为准；失败记录允许大量 null。

| 字段 | 含义和来源 |
| --- | --- |
| schema | 固定 `harness.iteration.v1` |
| run_id | 环境变量 HARNESS_RUN_ID；直接调用时默认 run，run_evolve 默认设时间字符串 |
| stage_id | 运行配置的阶段标识 |
| task_name | 配置任务名，通常 A/B/C，也可为空 |
| iteration | OpenEvolve 事件的迭代编号 |
| timestamp | 主进程生成的 UTC ISO 时间 |
| kind | 事件类型：failed、db_rejected、evaluated |
| status | failed 事件为 rejected_before_eval；否则来自评测报告，无报告为 internal_error |
| outcome | improved、regressed、equal、failed；定义见后文 |
| program_id | child_program.id，未生成子程序可为空 |
| parent_id | 搜索所选父程序 id，不是起点检查点 |
| island | OpenEvolve 的目标岛编号（岛是独立维护的候选子群） |
| evolution_mode | 事件给出的演化模式，例如 diff |
| code_sha256 | 子程序完整源码的 SHA-256；无代码为空 |
| parent_code_sha256 | 父程序完整源码的 SHA-256；父程序已不在数据库时可为空 |
| hypothesis | 从回复 HYPOTHESIS 行提取，去除首尾空白并截到 500 字符；没有则 null |
| used_bullets | 从 USED_BULLETS 方括号内容按逗号切出的条目编号列表；缺失或空列表时为 [] |
| playbook_version | 从 prompt.user 的 PLAYBOOK_VERSION 行读取版本；找不到为 null，空手册时通常是字符串 none |
| changes | 子程序 metadata.changes，通常是替换摘要；没有则 null |
| llm_response | 原始模型回复，缺省为空字符串 |
| combined_score | 子程序 metrics.combined_score；未评估时可为空 |
| parent_combined_score | 优先取子程序 metadata.parent_metrics，再取父程序 metrics 中的分数 |
| delta | combined_score−parent_combined_score；只有有效报告且两分数均存在时计算 |
| metrics | OpenEvolve 子程序的指标字典；harness 通常含 combined_score、primary_ratio |
| primary | 有效报告时含 name、candidate、parent_checkpoint、reference_policy；父程序报告也有效时另加 parent_program |
| flags | 报告 flags 中的 code 列表，可重复，不是含 detail/hint 的完整对象 |
| guard_violations | score.violations；含具体 guard、worsening、tolerance，指标护栏还可含 candidate/reference |
| feedback_text | 报告的中文反馈；缺报告时退用事件 artifacts.feedback |
| report_path | 当前报告绝对路径；失败/缺失报告时可能为空 |
| delta_vs_parent_program | 父程序与候选报告均有效时计算，含 overall、类别/分位切片及 by_group 的原值差 |
| error | 原始事件错误；非 ok 报告则拼接 problems；缺报告用反馈或缺报告提示，后两种截到 1000 字符 |
| token_usage | 原样保存事件用量，可为空；样例有 prompt_tokens、completion_tokens、total_tokens、model |
| iteration_time | 事件耗时；通常 evaluated 分支有值，其余可为空 |
| is_new_best | 事件布尔值，缺省 false；是搜索时新最佳，不是最终验收采用 |
| admitted | 仅 kind==evaluated 时为 true；表示进入候选种群，不表示验收通过 |
| prompt（可选） | 仅 records.include_prompt=true 时保存完整提示字典，一般含 system、user |

两种“父”必须区分：parent_checkpoint 是本阶段所有训练共同使用的起点；parent_program 是 OpenEvolve 本轮拿来改代码的父策略。outcome 比较**父策略的搜索分数**；报告 deltas 通常比较起点/官方参考。每个候选始终从阶段起点重新续训。

参考 [格式化单条记录](samples/iteration_record_pretty.json) 和 [JSONL 样例](samples/iteration_records_example.jsonl)。样例中的 <workspace> 为脱敏占位；时间、耗时和 token 用量只属于对应运行。假服务 token 数是估算，不能据此估算真实 Qwen 成本。

## 契约二：playbook 文件

默认路径 `<work_dir>/playbook/playbook.md`；运行配置顶层 `playbook_path` 可覆盖，相对路径以运行配置所在目录解析。**内容归 ACE 同学维护**；当前 ace_stub 不写此文件，只写统计。文件不存在或内容为空时，提示显示“暂无经验条目”，搜索仍可继续。

建议每条有稳定编号、适用任务/症状、建议改动和支持证据。例如：

~~~text
- str-00003｜A｜出现 CLASS_NEVER_PREDICTED 时，单独试验类别逆频率加权；同时检查 accuracy 护栏。
~~~

这是建议的条目样式，不是 parser 强制的格式。使用 UTF-8；建议在同目录写临时文件后原子替换，防止并行 worker 读到半份内容。

每次 worker 构建提示前，prompt_extras.get_extras 重新读取文件，经 `{playbook}` 注入 diff_user/full_rewrite_user 模板。默认 `prompt.playbook_max_chars=2500`：先 strip，再取前 limit 个字符并回退到换行边界，追加省略提示；版本前缀和省略提示不计入这个正文截取长度，因此总注入长度可略大于 limit。

`PLAYBOOK_VERSION` 自动生成：对**完整、strip 后、尚未截断的文件文本**做 SHA-256，取前 12 个十六进制字符；空手册为 none。ACE 不要手工伪造此行。若正文超长，不可根据版本号认定所有条目都被模型看见；引用分析需结合实际注入范围。提示版本写进每条记录。

同一扩展还注入 `{recent_failures}`：扫描最后 200 条记录，从新到旧挑 outcome 为 failed 或 regressed 的记录，默认最多 3 条，由 prompt.recent_failures 控制。并行 worker 已经生成的提示不会被后续更新追溯修改。

## 契约三：引用协议和 outcome

模型回复先给出两行，再给 SEARCH/REPLACE：

~~~text
HYPOTHESIS: 根据某条实际反馈，修改哪个钩子，预期哪个指标或切片改善
USED_BULLETS: [str-00003, mis-00008]
~~~

未使用经验时写 `USED_BULLETS: []`。解析器只按第一处匹配提取编号，不验证编号是否存在，不去重；不要在回复中重复列同一个编号。HYPOTHESIS 是一行说明，不是跨行分析全文。

| outcome | 精确定义 |
| --- | --- |
| improved | 报告 status=ok、两分数存在，delta>1e-9 |
| regressed | 同上，delta<−1e-9 |
| equal | 同上，−1e-9≤delta≤1e-9 |
| failed | 初始化默认值：提前失败、报告无效/缺失，或虽有效但无法计算父子分数差时保留 |

因此 failed 不一定等同“训练报错”，需结合 status、error；improved 也不等同“通过护栏”或“最终被采纳”，还要看 guard_violations 与单独的 gate/decision.json。

## 如何替换 on_batch

在项目根目录可导入的位置创建自己的 ACE 模块（下面的 team_ace 只是待实现的示例模块名），并在**自己的运行配置**加入：

~~~json
{
  "ace": {"module": "team_ace", "batch_size": 10},
  "records": {"include_prompt": true},
  "prompt": {"playbook_max_chars": 2500, "recent_failures": 3}
}
~~~

接口形状为：

~~~python
def on_batch(records, config):
    # records: 按主进程记录/完成顺序排列的列表，不保证并行迭代编号有序
    # config: 路径已经解析的运行配置；经验文件在 config["playbook_path"]
    # 读取证据，更新自己的 playbook；无须返回值
    pass
~~~

用绝对导入路径（如 team.ace_adapter）设置 ace.module，不带 .py 或 :on_batch。默认每累计 10 条记录调用一次，包括失败与未入种群记录；batch_size=0 关闭批处理调用，但仍写每轮 JSONL。批次累积在内存中；当前实现**不会在结束时自动冲刷不足一批的尾部记录**，重启也不会自动回放，应由接入方按记录文件补处理，并设计去重标识。

on_batch 在 OpenEvolve **主进程同步执行**，不要在里面长时间等待模型/网络，否则拖慢结果处理。如需异步 ACE，可在自己的适配器中快速落盘/排队，由独立处理器原子更新手册。当前实现未给 ACE 回调加超时。异常记录到 `<work_dir>/records/ace_errors.log`，不会中断搜索；批次已经从内存取走，不会自动重试。

## bullet_stats.json 表示什么

默认模块在 `Path(playbook_path).parent/bullet_stats.json` 保存 `条目编号 → task_name（缺省 unknown）→ helpful/harmful/neutral/failed 计数`。四项对应 improved/regressed/equal/failed；每条记录每引用一个编号就加一次。它是实验结果与自报引用的关联统计，不是因果贡献，也不是大模型给经验打的分。

可以复用 `harness.ace_stub.update_bullet_stats(records, stats_path)`。它不自动去重、不按 run_id 分层、不按版本分层，不加并发锁；重复回放、重复编号或多写者会影响计数。当前默认单写路径只处理已凑满的批次，所以最后不足一批不一定已计入。ACE 接入时应自己处理幂等性与历史重放。
