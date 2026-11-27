# 数据划分接口：只在官方验证集内部划分

负责划分的同学可以直接产出一个清单文件，在运行配置中设置 `split_manifest`；不必改训练器或 evaluator。当前 [splits.py](../harness/splits.py) 是临时实现，正式实现应按本页验收。挑选集供策略搜索与选轮，验收集供搜索结束后的最终决策；官方训练集保持原样。

## 清单格式

格式为 UTF-8 JSON。以下字段逐项对应 `MANIFEST_KEYS`；ID 必须来自本阶段 validation_path，保留原始类型和值。

| 字段 | 类型与约定 |
| --- | --- |
| stage_id | 字符串，与运行配置阶段一致 |
| source_sha256 | 原 validation_path 文件字节摘要，格式 sha256:加完整十六进制摘要 |
| task_type | classification 或 regression |
| method | 划分方式；现实现为 group_stratified、group、class_time、time_blocks、no_split |
| group_key | 分组字段名列表，例如 ["source_subset", "unit"]；无组时 null |
| seed | 固定的随机种子，默认 42 |
| select_ids | 挑选集样本 ID 列表 |
| gate_ids | 验收集样本 ID 列表 |
| counts | {"select": {...}, "gate": {...}}；各有 rows，分类另有 labels（类别名→数目），有组时另有 groups |
| warnings | 警告字符串列表，正常可为空；no_split 必须解释无独立验收集 |
| provisional | 是否为临时实现的布尔值；make_split 输出 true。验证器允许此字段缺失，但交接时应明确提供 |
| manifest_sha256 | 用 manifest_digest 计算，格式 sha256:加完整摘要 |

manifest_digest 仅取 MANIFEST_KEYS 中除 manifest_sha256 外的已存在字段，用 sort_keys=True、ensure_ascii=False、separators=(",", ":") 序列化为 UTF-8 后计算 SHA-256。不要直接对带缩进的 JSON 文件求哈希当作清单摘要。source_sha256 则是原数据文件字节摘要，两者含义不同。

示例见 [C 清单片段](samples/split_manifest_C_truncated.json)。该文件为阅读方便将 ID 列表替换成省略号，**不能直接作为运行清单**，也不能以截短内容重新核验原摘要。

## 两种接入方式

**方式一，推荐：配置外部清单。** 运行配置增加 `"split_manifest": "../splits/round-01-C.json"`，实际文件由同学生成；相对路径以配置文件所在目录解析。配置指定文件不存在时直接报错，不会悄悄使用临时划分。

StageContext 读取清单、调用 validate_manifest，按原验证文件顺序生成 `<work_dir>/splits/select.jsonl`、`gate.jsonl`，另写 `materialized.json` 保存清单摘要和实际源文件摘要。外部清单本身不会自动复制为工作目录的 manifest.json。请随实验保存原清单。

**方式二：替换 make_split 的实现。** 保持如下签名和返回结构，在正式开发分支完成；本次文档收尾没有修改该函数：

~~~python
make_split(ids, targets, meta_rows, task_type, stage_id, source_sha256,
           seed=42, min_groups=4, class_names=None, time_blocks=10)
~~~

ids、targets、meta_rows 必须逐行对齐；分类 targets 是类别下标，class_names 提供类别名；回归 targets 是原单位目标。内部返回上述 manifest dict。无外部清单时，首次生成 `<work_dir>/splits/manifest.json`，之后优先复用。改划分应创建新 work_dir 和 stage_id，不要在搜索中途改清单或沿用旧 reference/reports 缓存。

## 当前临时算法

分组发现先忽略常量字段和逐行唯一字段，再识别文件中连续出现的组；必要时组合外层字段与重复编号。报告使用较粗分组，划分使用最细可识别分组。C 的 unit 会在不同 source_subset 中重复，必须用二者组合识别发动机。

- 有至少 min_groups（默认 4）组：固定种子打乱分组，按位置交替分给两边；有较粗报告键时在粗组内进行，method=group_stratified，否则 group。
- 有组但数量不足：no_split，两边均使用完整验证集，并警告。
- 无组的分类：class_time，按每类在文件中的顺序，前 ceil(n/2) 个给挑选，后半给验收；文件顺序被当成时间顺序。单样本类别仅进挑选并警告。
- 无组的回归：time_blocks，按连续块交替分给两边；默认 10 块参数。

## 交付验收标准

| 检查项 | 必须满足 |
| --- | --- |
| A 六分类 | 两边都含全部 6 类；对每个类别，挑选侧所有样本在原文件中的位置早于验收侧样本。不要跨类比较全局时间 |
| C 发动机分组 | group_key=["source_subset", "unit"]；验证集总共 106 台，同一台不能跨两边；固定输入、种子下结果一致 |
| C 当前参考结果 | seed=42 的现实现为挑选 55 台/11817 行，验收 51 台/11232 行；正式算法可以重划，但须说明变更，不必硬凑相同行数 |
| B 轴承 | 验证集只有 3 根轴承，method=no_split；select_ids 和 gate_ids 均等于完整验证集，必须输出警告 |
| 可复现 | 相同输入、字段顺序、种子、阶段信息产生相同清单与摘要 |
| 完整性 | ID 无重复、无未知 ID，两边非空，counts 与实际一致；独立划分两边不重叠，合并全覆盖 |
| no_split 特例 | 两边重叠是明确约定，但每一边自身都必须全覆盖，不能拿这个模式绕过完整性要求 |
| 来源与摘要 | 核对 stage_id、task_type、source_sha256 与本阶段一致，manifest_sha256 可重新计算一致 |

C 当前测试还检查两边都覆盖原数据的各个 source_subset。验收时应读取完整元数据核对分组隔离，不能只相信 counts。已有 tests/test_harness.py 包含 A/C/B 的关键检查，但不代表以上全部条件均被现有 validate_manifest 自动执行。

**当前校验器的边界**：它只检查所需字段存在、两边非空、无未知 ID，以及非 no_split 的不重叠/全覆盖。不会验证摘要正确性、来源一致、counts、重复 ID、组隔离或时间顺序；no_split 也不会自动检查每边全覆盖。接入方必须补足这些交付检查。此处只报告现状，不修改行为。

## 三条禁区

1. 不动官方训练/验证边界：只拆 validation_path，不把验证样本加入训练，也不从训练中拿样本补验收。
2. 不依赖任何模型预测、损失、候选得分或搜索结果来决定划分；否则验收已受模型选择影响。
3. 不使用未固定种子的随机过程；随机源、排序规则和种子均应稳定且可记录。

B 的 no_split 会使挑选与验收同源；默认评分使用轴承等权平均，验收阈值由 0.90 提升到 0.95 并警告，仍不能消除选择偏差。正式数据足够后再考虑独立划分。
