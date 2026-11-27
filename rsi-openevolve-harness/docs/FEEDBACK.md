# 细致反馈：从结果定位该改哪个钩子

先看 status 和护栏，再看主指标相对起点与官方续训是否改善，最后用曲线和切片解释原因。反馈全部由程序计算，只使用挑选集；验收集不进入搜索提示。完整 JSON 保存在 `<work_dir>/reports/<代码SHA-256>.json`，中文文本是便于模型阅读的压缩版本。

## 六层证据

| 层次 | 主要内容 | 能回答的问题 |
| --- | --- | --- |
| 1. 结果与硬约束 | status、problems、主指标、评分、护栏、步数和轮次 | 是否有效？是否真正续训？提高是否以牺牲其他部分为代价？ |
| 2. 训练动态 | 每轮损失、挑选集指标、学习率、最佳/选中轮次 | 后几轮退化、仍在改善，还是训练根本没有进展？ |
| 3. 数据切片 | 分类各类 P/R/F1 和混淆；回归目标分位区间；自动发现的分组 | 改善或退化集中在哪类、哪个目标区间、哪台设备？ |
| 4. 模型内部 | tanh 饱和、梯度范数和裁剪率、权重相对变化、样本权重、耗时 | 更新是否过猛、梯度是否饱和、权重是否过度偏斜？ |
| 5. 最差样本 | 分类中置信度最高的错误；回归中绝对误差最大的样本 | 是否有少量严重错误支配表现？ |
| 6. 相对参考的变化 | 相对本阶段起点、官方初始策略续训的总体与切片差值 | 比不训练好多少？比不改策略直接续训好多少？ |

P 是精确率，R 是召回率；bias=预测−真值，正数表示高估。饱和定义为隐藏层输出绝对值 >0.99。分位切片由真值的 0%、25%、50%、75%、100% 分位点划分，空区间省略。默认 top_k=5。NASA 分数贡献按非对称指数损失计算，“前 1%”实际取贡献最大的 max(1, n//100) 个样本。

**B 的两种口径要分开**：candidate 中用于评分的 rmse 默认是各轴承 RMSE 的等权平均；candidate_pooled 和训练曲线是合并全部样本计算的 RMSE，数值不应直接互相比较。

## 全部症状标签与精确触发条件

以下逐项对应 [feedback.py](../harness/feedback.py) 的 `_flags`，不是模型自由判断。记 W(m,c,r) 为“c 相对 r 变差的比例”：越大越好的指标为 1−c/r；越小越好的指标为 c/r−1。当 r≤1e-12 时，前者返回 0，后者返回 c−r。严格大于/小于的边界与代码一致。

| 标签 | 触发条件 | 主要检查方向 |
| --- | --- | --- |
| GUARD_VIOLATION | score.violations 中每个违规项各产生一条 | 检查具体指标/分组，缩小改动 |
| WORSE_THAN_START | primary_ratio < 1.0 | 学习率、轮数、pick_epoch |
| WORSE_THAN_REFERENCE | 有官方续训指标，且官方主指标严格优于候选 | 与初始策略对照，换改动方向 |
| LATE_EPOCH_DEGRADATION | 最佳轮不是最后轮，且 W(曲线指标,最后轮,最佳轮)>0.01；即使已选最佳轮也触发 | pick_epoch 或学习率衰减 |
| STILL_IMPROVING | 与上一条互斥；最佳轮是最后轮，至少两轮，且 W(曲线指标,倒数第二轮,最后轮)>0.005 | 考虑增加 epochs |
| NO_TRAIN_PROGRESS | 至少两轮，最后训练损失≥第一轮损失 | 学习率或损失/权重尺度 |
| TRAIN_VAL_DIVERGE | 与上一条互斥；至少两轮，末损失<0.95×首损失，且 W(曲线指标,末指标,首指标)>0.01 | 过拟合/漂移，pick_epoch 或 L2 |
| HIGH_SATURATION | 候选在挑选集的 tanh 饱和比例>0.30 | 减小学习率或抑制 w1 增长 |
| GRAD_CLIP_FREQUENT | 各训练轮 clip_rate 的算术平均>0.5 | 检查样本权重和损失量级；这是逐轮平均 |
| LARGE_WEIGHT_CHANGE | 所选轮参数相对起点的范数变化>0.5 | 减小学习率或轮数；分母是起点参数总范数 |
| CLASS_NEVER_PREDICTED | 分类某类 support>0 且 predicted=0 | 提高该类权重 |
| CLASS_LOW_RECALL | 同一类别未触发上一条，support≥5，recall 非空且<0.5 | 类别权重和混淆方向 |
| LABEL_SHIFT | 分类有训练类别统计，训练/挑选类别占比的总变差距离 0.5×Σ绝对差>0.2 | 不盲从训练集大类占比 |
| OVERESTIMATE_LOW_TARGET | 回归首个非空分位区间 bias>0.5×rmse | 抑制临近失效时高估 |
| UNDERESTIMATE_HIGH_TARGET | 回归最后非空分位区间 bias<−0.5×rmse | 大目标权重/欠训练 |
| NASA_DOMINATED_BY_FEW | 启用 NASA 分析，top1pct_share>0.9 | 压低高贡献严重误差，特别是高估 |
| GROUP_DOMINATES | 某个报告分组键下有多组，某组 share>0.7 | 检查小组是否被总体分数掩盖 |
| SPLIT_WARNING | split_warnings 每条警告产生一条 | 阅读原始 warning；不只凭标签判断是否 no_split |

同一个标签可以因多个类别/分组出现多次。曲线指标优先采用主指标；如果主指标不在 runner 的逐轮 val 中，则分类回退 macro_f1、回归回退 rmse。best_index 使用首次最佳的位置。

SPLIT_WARNING 的固定建议文本提到“挑选与验收相同”，但代码对所有划分警告都加此标签，例如单样本类别警告也可能触发；应以 method 和 warning 具体内容为准。

## 完整报告 JSON 的主要字段

| 字段 | 含义 |
| --- | --- |
| schema | 固定 `harness.feedback.v1` |
| status、problems | 状态与问题列表；正常状态为 ok |
| task | task_name、task_type、stage_id、primary、unit、split_method、aggregate、ranking_metrics |
| flags | 列表，每项 code、detail、hint；完整标签不受文本前 8 条限制 |
| stderr_tail | 可选的失败 stderr 尾部 |
| score | combined_score、primary_ratio、penalty、tie、violations、guard_ok |
| metrics | candidate、parent_checkpoint、reference_policy、candidate_pooled；原始指标值，不是统一方向后的比值 |
| constraints | passed；steps 和 epochs 分别为 [起点,所选结果] |
| curve | metric、epochs、train_loss、val、before、best_index、selected_index、learning_rate |
| slices | 按任务包含 per_class、top_confusions、true_share、pred_share、by_quantile、by_group、nasa、extras |
| deltas | vs_parent_checkpoint、可选 vs_reference_policy；overall、per_class 或 by_quantile、by_group。全部为候选−参考的原值差，不会自动把“变好”统一成正数 |
| internals | saturation、saturation_before、clip_rate、grad_norm_mean、weight_change、config、sample_weights、training_seconds |
| worst | 默认至多 5 个，带样本 id、真值、预测和 confidence 或 error |
| candidate_dir、checkpoint_sha256 | 成功候选目录和检查点摘要 |
| code_sha256 | 完整策略源码摘要 |
| result_metrics | 返回 OpenEvolve 的 combined_score 与 primary_ratio，均按越大越好解释 |
| feedback_text | 实际注入提示的中文文本 |

非 ok 报告不会具有成功报告的全部字段。失败状态包含 policy_rejected（静态拒绝）、train_failed（训练错误）、timeout、constraint_violation；OpenEvolve 入口捕获其他异常时返回 0 分和错误文本，可能没有落盘报告，迭代记录会将缺报告识别为 internal_error。

中文渲染默认最多 3500 字符（feedback.max_chars），成功报告先显示最多 8 条症状；超长时从末尾逐节删减，再截断。失败报告最多显示前 8 个 problems。文本缺少某层不代表 JSON 缺少数据。大于 12 组的分组表不会直接展开到中文文本中，完整信息仍在 JSON。

## 读样例

- [A 初始策略反馈](samples/feedback_initial_policy_A.txt)：Acetaldehyde 未被预测，伴随 LABEL_SHIFT；先分析类别权重。
- [B 初始策略反馈](samples/feedback_initial_policy_B.txt)：总体与分组口径不同，Bearing1_4 护栏违规，且有无独立验收警告。
- [C 初始策略反馈](samples/feedback_initial_policy_C.txt)：末轮退化、低目标高估及 NASA 贡献集中，可逐个检验假设。
- [真实提示](samples/prompt_example_A.txt)：查看精简反馈如何进入模型上下文。
- [完整迭代记录样例](samples/iteration_record_pretty.json)：含 feedback_text、主指标及相对父策略切片差值。
- [C 验收决定](samples/gate_decision_C.json)：这是最终验收输出，不能作为下一轮搜索反馈泄漏验收集。

样例中的 `<workspace>` 只是脱敏占位符，不能直接当作运行路径。
