# OpenEvolve 的两个扩展点

项目携带 OpenEvolve 0.4.1（交接固定版本 commit 3e72b5b）。仅在 `third_party/openevolve/openevolve/process_parallel.py` 增加两个可选扩展点；补丁保存在 [openevolve_extension_points.patch](openevolve_extension_points.patch)。未设置环境变量时不调用扩展；hook 的导入或执行异常会记录日志并继续搜索。

这两个扩展解决两个问题：worker 构建提示时需要读到 ACE 刚更新的经验；主进程需要记录包括无效修改和未入种群候选在内的每轮结果。仅靠最佳程序或数据库无法保留所有失败尝试。

## 提示扩展 OPENEVOLVE_PROMPT_EXTRAS

运行脚本设为 `harness.prompt_extras:get_extras`。变量值格式为 `package.module:function`，由 _load_extension 动态导入并在每个进程缓存函数；变量值变化会重新加载。函数缓存不等于手册缓存：get_extras 每次仍读取 playbook 文件。

调用位置：工作进程为一次迭代采样父程序、决定 evolution_mode 后，紧邻 `_worker_prompt_sampler.build_prompt(...)` **之前**调用 `_prompt_extras(context)`。返回的字典经键值字符串化，以 `**prompt_extras` 传给 build_prompt。

context 的完整键集：

| 键 | 来源 |
| --- | --- |
| iteration | 本轮编号 |
| island | db_snapshot.sampling_island |
| parent_id | 父程序 id |
| parent_metrics | 父程序 metrics |
| evolution_mode | 当前演化模式 |

当前返回 `playbook` 和 `recent_failures`，模板使用同名花括号占位符。提供者应始终返回这两个字符串；错误时 harness 自己给空经验/空失败的安全默认文本。OpenEvolve 的扩展包装捕获异常则返回空字典，不会主动生成这些模板字段，因此不要依赖外层包装替代提供者的默认值。

## 迭代扩展 OPENEVOLVE_ITERATION_HOOK

运行脚本设为 `harness.records:on_iteration`。在主进程处理 worker 返回结果时同步调用 `_notify_iteration(event)`。不是在子进程训练中调用，也不是后台异步任务。当前有三处调用分支：

1. worker 返回 error：记录失败轨迹之后发出 kind=failed。
2. 已有 child_program，但数据库 admission（入种群规则）返回 None：在抛出内部 _CandidateRejected 之前发出 kind=db_rejected。
3. 子程序成功入库、相关记录处理完并计算 is_new_best 后：在“新最佳”日志分支之前发出 kind=evaluated。

这里的“每轮”指到达这些完成分支的结果；强制中断或进程崩溃不能保证补发记录。

| event 键 | failed | db_rejected | evaluated | 含义 |
| --- | --- | --- | --- | --- |
| kind | 有 | 有 | 有 | 上述三种事件名 |
| iteration | 有 | 有 | 有 | completed_iteration |
| parent_id | 有 | 有 | 有 | 所选父程序 ID |
| parent_program | 有 | 有 | 有 | 数据库中父程序的 to_dict；已淘汰时为 None |
| prompt | 有 | 有 | 有 | 原始提示字典 |
| llm_response | 有 | 有 | 有 | 原始模型回复 |
| island | 有 | 有 | 有 | result.target_island |
| evolution_mode | 有 | 有 | 有 | 演化模式 |
| token_usage | 有 | 有 | 有 | 模型用量 |
| error | 有 | 无 | 无 | worker 错误 |
| child_program | 无 | 有 | 有 | 子程序 to_dict |
| artifacts | 无 | 有 | 有 | evaluator 返回的反馈等产物 |
| iteration_time | 无 | 无 | 有 | worker 报告的耗时 |
| is_new_best | 无 | 无 | 有 | 此刻数据库最佳 ID 是否等于子程序 ID |

表中“有”指包含键，值仍可能为 None。records.build_record 会规范化缺省值，字段说明见 [ACE_INTERFACE.md](ACE_INTERFACE.md)。请用 .get() 读取分支可选键，不要假定失败事件也有子程序。

当前调用会先把 JSONL 写盘，再按 ace.batch_size 同步调用 ACE。回调不能长时间阻塞；异常仅记录并继续，并无自动重试、尾批冲刷或强制超时。

## 升级时如何重新应用

不要直接 pip 升级并以为扩展仍存在。升级应在新目录/开发分支进行，保留当前可用副本、LICENSE 和补丁；以下只说明后续开发流程，本次收尾没有升级代码。

1. 从 [上游 codelion/openevolve](https://github.com/codelion/openevolve) 获取目标版本并记录完整 commit，核对其 process_parallel.py 与事件处理流程。
2. 在**未打补丁的上游根目录**执行 `git apply --check <本项目>/docs/openevolve_extension_points.patch`；通过后执行同命令去掉 --check 应用。尖括号路径需替换为实际路径。
3. 若上下文不匹配，不要强行忽略失败：按本页三个完成分支和一个提示调用位置重新移植；确认 extras 传入正确 sampler，事件在主进程发出，拒绝入库的候选也能记录，异常不会中断搜索。已有补丁目录可用 `git apply --reverse --check ...` 检查是否已应用，避免重复打补丁。
4. 相对新版本原文件重新生成只包含 process_parallel.py 的补丁，并更新 VENDORED.md 的版本/commit；完整保留上游许可证和来源信息。
5. 跑 harness 测试及假 LLM 全链路，至少覆盖精确替换失败、块外修改恢复、静态拒绝、训练失败、0 分不入种群、成功入库、playbook 中途更新、ACE 批处理和验收；逐条核对 JSONL 没有漏掉预期失败事件，也没有重复写同一完成事件。

补丁上下文行号只适用于当前版本；升级时以语义位置和实际调用顺序为准。
