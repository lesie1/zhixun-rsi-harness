# 数据划分模块 team_split

这个模块把一个阶段的官方验证集拆成两份：**挑选集**给 OpenEvolve 搜索打分、给 `pick_epoch` 选轮；**验收集**只在搜索结束后的验收门里用，搜索过程从来看不到它。它是 [SPLIT_INTERFACE.md](SPLIT_INTERFACE.md) 所要求的正式实现，接入方式只有一种：运行配置里的 `split_manifest`。harness 的临时实现 `harness/splits.py` 和 `StageContext` 都没有改。

默认参数下，本模块的结果和临时实现**逐个 ID 相同**（A、B、C 三个数据包都由测试钉住），所以已经验证过的数字都不变。

## 三步用法

在项目根目录运行，`python` 指项目使用的解释器。

```bash
# 1. 生成清单（同时校验；有错误就不写文件，退出码 1）
python -m team_split make --config configs/run_C.json --out splits/C-v1.json --suffix split-v1

# 2. 需要时单独校验或查看报告
python -m team_split verify --config configs/run_C.json --manifest splits/C-v1.json --no-stage-check
python -m team_split report --config configs/run_C.json --manifest splits/C-v1.json

# 3. 生成一份引用该清单的新运行配置（原配置不动）
python -m team_split wire --config configs/run_C.json --manifest splits/C-v1.json \
    --out configs/run_C.split-v1.json --suffix split-v1
```

之后搜索和验收都用新配置：`scripts/run_evolve.py --run-config configs/run_C.split-v1.json`，`scripts/run_gate.py` 同理。

| 命令 | 产出 |
| --- | --- |
| make | `splits/C-v1.json`（清单，字段即 `MANIFEST_KEYS`）；`splits/C-v1.report.json`（使用的参数、校验结果、质量报告）；屏幕上打印报告 |
| verify | 逐条打印问题；有错误时退出码 1 |
| report | 打印质量报告 |
| wire | 新配置文件；如果配置用了 team_ace，还会在 `state/<命名空间>/<任务>/<新 scope>/` 建一个空经验库 |

`make` 的参数：`--seed`、`--min-groups` 默认取运行配置的 `seed` 和 `split.min_groups`；`--time-blocks` 默认 10；`--method` 默认 `auto`；`--balance` 默认 `groups`。`--stage-id` 直接指定清单的 stage_id；更方便的是 `--suffix`：它和 `wire` 用同一个后缀时，清单的 stage_id 正好等于新配置的 stage_id（`<原 stage_id>-<后缀>`）。

`verify` 默认会做两件额外检查：stage_id 是否等于 `--config` 里的 stage_id（对原配置校验时用 `--no-stage-check` 关掉），以及能否用同样的参数重建出同一份清单（参数从同名 `.report.json` 读取，`--no-reproduce` 关掉）。

## 三个数据包上的结果（默认参数）

| 任务 | 方法 | 分组键 | 挑选 / 验收 | 说明 |
| --- | --- | --- | --- | --- |
| A | class_time | 无 | 623 / 621 行，六类两边都有 | 甲苯只有 5 个，验收侧 2 个，报告会提示 |
| B | no_split | bearing（3 组） | 1679 / 1679 行 | 只有 3 根轴承，没有独立验收集，清单带警告 |
| C | group_stratified | source_subset + unit | 55 台 11817 行 / 51 台 11232 行 | 四个子集两边都有；`--balance rows` 时为 53 台 11525 行 / 53 台 11524 行 |

## 划分方法

`--method auto` 按下面的顺序自动选择；也可以强制指定，但不适用时会直接报错（例如对回归任务用 class_time）。

| 方法 | 自动选择的条件 | 规则 |
| --- | --- | --- |
| group_stratified | 找到分组字段、组数 ≥ min_groups、且有更粗的报告键 | 在每个粗分组里，组名排序后用固定种子打乱，按位置交替分给两边。同一组不会跨两边 |
| group | 同上但没有更粗的键 | 同上，不按粗分组分桶 |
| no_split | 有分组字段但组数不足 | 两边都是全部数据，清单写明警告 |
| class_time | 没有分组字段的分类任务 | 每个类别内部按文件顺序，前 ⌈n/2⌉ 个给挑选，后面给验收 |
| time_blocks | 没有分组字段的回归任务 | 按文件顺序切成 time_blocks 块，交替分配 |

分组字段由 `harness/groups.py` 自动发现，全项目只有这一份实现，划分和反馈报告里的"一组"含义完全相同。C 的 `unit` 在不同子集里会重复（例如 FD001 和 FD003 都有 71 号），所以发动机必须用 `source_subset + unit` 组合识别。

`--balance rows` 只影响两种分组方法：每个粗分组里，组按行数从大到小，依次放到当前行数较少的一边（种子只决定同样大小的组谁先放）。结果的方法名带 `_rows` 后缀（`group_stratified_rows`、`group_rows`），清单因此是自描述的，`verify` 能据此重建。没有把它写进 `warnings`，原因见下面的"注意"。

## 校验项

`verify_manifest` 从不抛异常，只返回问题列表。harness 自带的 `validate_manifest` 只查前几项，下面全部由本模块检查。

| 代码 | 级别 | 含义 |
| --- | --- | --- |
| KEYS_MISSING | 错误（缺 provisional 时为提示） | 缺少 MANIFEST_KEYS 中的字段 |
| TYPE_INVALID | 错误 | 字段类型或取值不对，或方法名未知 |
| DIGEST_MISMATCH | 错误 | manifest_sha256 与 manifest_digest 重算结果不同 |
| SOURCE_MISMATCH / TASK_MISMATCH / STAGE_MISMATCH | 错误 | 与本阶段验证文件摘要、任务类型、stage_id 不一致 |
| DUPLICATE_ID / UNKNOWN_ID / EMPTY_SIDE | 错误 | 一边内有重复 ID、有不认识的 ID、或一边为空 |
| OVERLAP / NOT_COVERING | 错误 | （非 no_split）两边有重叠、或合起来没覆盖全部样本 |
| NO_SPLIT_INCOMPLETE | 错误 | no_split 时某一边不等于全部样本，或没有写警告 |
| COUNTS_MISMATCH | 错误 | counts 与重新统计的结果不符 |
| GROUP_KEY_INVALID | 错误 | 分组方法没有 group_key，或其中字段不是每行都有 |
| GROUP_STRADDLES | 错误 | 有组同时出现在两边 |
| CLASS_ORDER_VIOLATED | 错误 | class_time 下，某类在挑选侧有样本排在验收侧样本之后 |
| CLASS_MISSING_ON_SIDE | 提示 | 某个类别只出现在一边 |
| NOT_REPRODUCIBLE | 错误 | 用给定参数重建出的清单摘要不同 |

## 质量报告怎么读

报告回答"验收集能不能代表挑选集"。内容包括两边的行数和占比、在文件中的行号范围、组数；分类任务给出各类别两边的数量和类别占比的总变差距离（0 表示完全相同）；回归任务给出两边目标的均值、四分位，以及均值相差几个标准差；有更粗分组键时（例如 C 的子集），给出每个粗分组在两边的组数和行数。

出现下列情况时报告会给出提示，但不阻止写文件：验收集占比不在 30%–70% 之间；某类别在验收侧少于 5 个；类别占比的总变差距离大于 0.15；回归目标均值相差超过 0.25 个标准差；no_split。

## 和 ACE 的关系

team_ace 的经验库按 scope 分区（`team_ace/store.py` 的 `scope_for`），scope 里记着清单的路径和**文件摘要**。因此：

- 换划分就是换 scope。`wire` 会重算 scope，把 `playbook_path` 指到新的空经验库，旧经验不沿用。这是有意的：不同划分下得到的反馈不能混成一份经验。
- **清单一旦被某个配置引用，就不能再修改**，否则经验库会报"storage scope changed"或经验和数据对不上。需要改划分时，换一个版本号的新文件（`C-v2.json`），再 `wire` 出新配置。`make` 对内容不同的同名文件默认拒绝覆盖。
- `wire` 也会给新配置换一个新的 `work_dir` 和 `stage_id`，要求新 work_dir 不存在或为空，避免沿用旧划分下缓存的官方续训结果和评测报告。

没有使用 team_ace 的配置（或用别的 ACE 模块），`wire` 只改 `split_manifest`、`stage_id`、`work_dir`，其余键原样保留；相对路径会改写成绝对路径，保证新配置放在哪里都能用。

## 和 SPLIT_INTERFACE.md 的关系

[SPLIT_INTERFACE.md](SPLIT_INTERFACE.md) 规定清单格式、接入方式和验收标准；本模块按它实现，并把其中"现有校验器不检查"的部分全部补上。清单格式没有改动，任何人手写的清单也可以用 `python -m team_split verify` 检查。

## 注意和已知限制

- **manifest 的 warnings 会进入大模型的反馈。** harness 把清单里每条警告都当作 SPLIT_WARNING 症状写进提示，并配上"挑选集与验收集相同"的建议（`harness/feedback.py`）。所以只有真正的问题（no_split、单样本类别）才写进 warnings；按行数均衡这类说明改为体现在方法名上。
- **B 没有独立验收集。** 验证集只有 3 根轴承，no_split 下挑选和验收是同一份数据；harness 会按轴承等权打分并把验收阈值提到 0.95，但选择偏差依然存在。
- **文件顺序被当作时间顺序。** class_time 和 time_blocks 都依赖这一点；数据如果不是按时间排列的，这两种方法的"挑选在前、验收在后"就没有意义。
- **只拆验证集。** 不碰训练集，不使用任何模型预测或搜索结果，随机只来自固定种子。交叉验证、多折划分不在本模块范围内。
