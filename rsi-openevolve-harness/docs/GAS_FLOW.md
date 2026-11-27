# A 类气体分类的自动适应流程 team_gas

这个流程自动回答四个问题：

1. 当前模型在哪一种气体上退化了？
2. 它最常被错认成哪一种？
3. 只用现有训练数据还能不能修复？
4. 修不好的话，怎样针对这两类气体，从小比例开始混入新数据？新数据占比最高 20%，由代码强制执行。

它把 `team_loop` 的新批次性能检查和 `team_adapt` 的搜索停滞判断、模型承接接在同一个入口里。每一轮由一个控制器负责训练、预算和验收。原有的 A/B/C 运行方式（`scripts/run_evolve.py`、`run_gate.py`、`team_loop`、`team_adapt`）都不变。

```
第 r 轮
 ① 诊断：同一个部署检查点，参照数据 vs 新到达的有标签批次（alarm）
      稳定 / 样本不足 → 本轮结束，不训练
 ② 修复：从部署检查点出发，在现有训练集上做有限预算的策略搜索，监视报警类别（alarm 批次挑选/验收）
      已修复（报警类改善，混淆类和其他类通过护栏）→ 按原验收规则部署，本轮结束，不换数据
      没有停滞 / 没有允许训练的新样本 → 本轮结束，不换数据
 ③ 换数据：探索新数据占比 × 退化类/混淆类配比，在开发批次上比较（selection），选定并冻结
 ④ 最终验收：预留批次（final）只用一次；更新前 vs 0% 对照 vs 选中方案 → 采用 / 拒绝（保留原模型）
```

## 配置

示例见 [configs/gas_A.example.json](../configs/gas_A.example.json)。

### 批次

批次字段和 team_adapt 一样：`id`、`order`、`path`、`labels_available`、`uses`、`exposure`、`sha256`。`uses` 只能取下面四种：

| 用途 | 含义 |
| --- | --- |
| `train` | 可以参与梯度训练（作为历史数据，或作为换数据阶段的新样本） |
| `alarm` | 可以用来诊断部署模型，并给修复搜索打分、验收 |
| `dev` | 可以用来比较候选配比 |
| `final` | 预留，只在方案冻结后用一次；不能兼任其他用途 |

`uses` 可以为空，表示只登记暴露情况、不使用。

### 轮次和参数

`rounds` 每轮写 `{"alarm", "selection", "final"}`，必须满足 alarm < selection < final 的时间顺序。最终批次不能在任何一轮兼作报警或开发批次，也不能被多轮使用。

| 配置项 | 关键参数 |
| --- | --- |
| `diagnosis` | `min_support`（每类至少多少样本才下结论）、`recall_drop` / `precision_drop`（退化阈值，绝对值）、`confidence`（自助法判定"确实变差"的概率）、`min_confusion_share`（主要混淆类至少占该类错误的比例） |
| `repair` | `search.mode`（`evolve` 调用大模型，`continuation` 不调用）、`stagnation`（`patience`、`min_delta`、`alarm_min_delta`、`max_consecutive_failures`）、`budget`（每阶段的迭代、秒数和 token 上限）、`success`（修复标准）、`advance_on` |
| `mixture` | `fractions`（默认 0、2%、5%、10%、15%、20%）、`degraded_shares`（新数据中退化类所占份额，默认 1、0.75、0.5、0.25）、`cap`（≤ 0.2）、`tie_tolerance`、`patience_levels`、`min_gain`、护栏、`policy`（默认官方策略） |
| `acceptance` | 最终验收的退化类最小增益、护栏和宏平均 F1 自助法门槛 |
| `display_names` | 只影响报告里的显示名；逻辑里的类别全部来自检查点 |

## ① 诊断

代码在 [team_gas/diagnose.py](../team_gas/diagnose.py)。参照数据默认是部署模型的训练集，每一轮也可以单独指定。对参照数据和新批次分别计算每一类的召回率、精确率、样本数和混淆矩阵：

| 判定 | 条件 |
| --- | --- |
| 性能退化，疑似漂移 | 指标下降超过阈值，两边样本都够，且自助法判定确实变差 |
| 样本太少，证据不足 | 指标下降了，但样本不够或统计上不显著 |
| 稳定 | 其他情况 |

退化类的混淆方向，是它的新错误里占比最高的那一类，例如"真实 乙醛 在新批次中 108/109（99.1%）被判为 乙醇"。精确率退化会写明误报主要来自哪一类。

`team_loop.trigger` 的整体检查同时运行。整体分数稳定时，单类退化照样会报出来。

## ② 先用现有训练集修复

代码在 [team_gas/repair.py](../team_gas/repair.py)。修复阶段建立新的 stage_id、工作目录和划分清单：训练数据是部署模型的训练集，挑选集和验收集来自报警批次。

**关注报警类别的停滞判断。** 一个有效候选只有满足下面任一条件才算进展：

- 报警类召回比此前最好值至少高 `alarm_min_delta`；
- 总分至少提高 `min_delta`，并且报警类召回不低于此前最好值。

这样，总分小涨不能掩盖目标类别持续变差。连续 `patience` 个有效候选都没有进展，就算停滞。

**单独统计的失败。** 只有成功训练并拿到有效评分的候选才计入停滞。API 失败或超时、无效代码、训练失败都单独计数；连续失败太多记为"不健康"，这时停止，不换数据。

**提前结束。** 只要某个候选在挑选集上做到报警类召回提高 `target_recall_gain`，且混淆类的召回和精确率、其他类的召回都在护栏内，就停止搜索。

**验收。** 修复阶段的验收沿用 harness 原有的验收门规则（同一个 `harness.gate.decide`）。验收门采纳的模型，还要在验收集上再按同一条标准核对一遍，才算"已修复"。已修复就结束本轮，不换数据。

## ③ 换数据：探索两个量，硬上限 20%

代码在 [team_gas/mixture.py](../team_gas/mixture.py)。

**进入条件。** 只有修复搜索停滞（或按配置的 `advance_on`），并且报警批次允许训练、退化类和混淆类都有足够的真实标签新样本时，才进入这一步。

**探索的两个量：**
- 新数据占全部训练次数的比例（`fractions`）；
- 新数据在退化类和混淆类之间的份额（`degraded_shares`）。

0% 是同预算的旧数据对照。

**20% 硬上限的计数规则：**
- runner 每个 epoch 把训练文件里的每一行恰好用一次。最后一个 mini-batch 不补齐；如果补齐，补的行也在文件里，同样计数。
- 每个候选的训练文件行数固定为 N（历史数据的行数），其中新数据 k 行，k = ⌊占比 × N⌋ 且不超过 ⌊0.2 × N⌋。新样本不够时循环重复，每次重复都算一次训练次数。
- 所以每个 epoch 和整个训练过程的实际新数据占比都是 k/N。
- `cap` 只能调低，大于 0.2 会被拒绝。
- 训练完成后，按 runner 自己记录的每轮步数核对：每轮步数必须等于 ⌈N/批大小⌉，训练文件摘要必须和计划写出的一致。对不上就报错，不采用。

**公平比较：**
- 新数据替换历史行，所以每个候选的总行数、更新步数和打乱顺序都相同（打乱的种子来自检查点）。
- 同一适应阶段内，新旧样本的身份固定：历史行按一个固定顺序去掉，新样本按固定顺序抽取。
- 所有候选从同一个部署检查点出发，默认用官方策略（统一权重、固定轮数），避免样本权重绕过上限。

**记录。** 每个候选的 `accounting.json` 记录：目标比例和实际比例、每轮新旧训练次数、各类别和各来源的训练次数、独立样本数、重复次数、所用新样本的 ID。

**选择规则：**
- 只看开发批次。候选相对 0% 对照必须让退化类召回至少提高 `min_gain`，混淆类和其他类通过护栏。
- 在合格者里取 macro_f1 最高的；相差在 `tie_tolerance` 以内时，取新数据更少的。
- 连续 `patience_levels` 个比例档位都没有更好的方案时，提前停止。
- 选定后写入 `frozen.json`，再做最终验收。

## ④ 最终验收和回退

在预留批次上比较三个模型：更新前模型、0% 新数据的同预算续训、选出的针对性混合方案。报告整体 macro_f1，以及六类气体各自的召回率、精确率和 F1。

采用需要同时满足：
- 相对更新前，退化类召回提高 `target_recall_gain`；
- 混淆类的召回和精确率、其他类的召回都在护栏内（专门检查"退化类恢复了，但混淆类被大量误判"）；
- 自助法判定宏平均 F1 更好的概率 ≥ `threshold`。

通过则部署；不通过则保留更新前模型，并记录一条 `rollback` 事件。

最终批次如果已暴露、在台账里被用过，或与训练、新样本、开发批次有重复，会被拒绝。

## 基础经验写入 playbook

[team_gas/base_experiences.json](../team_gas/base_experiences.json) 里有一条基础经验：

| 字段 | 内容 |
| --- | --- |
| 编号 | `A-00-BASE-CLASSMIX-01`（稳定，排在 team_ace 自动生成的条目前面） |
| 标题 | 某一类退化并主要被认成另一类时：同时加强这两类，保留其余类别 |
| 来源 | 2026-11-27 由 `A-00-BASE-GASMIX-01` 通用化改写：不写具体气体；证据改为官方数据包公开部分内的对照实验（原卡的 v16、v17 来自数据包之外的 UCI 后续批次，不再作为规则依据） |
| 适用条件 | 分类任务（不限 A）、识别出"退化类→混淆类"、新批次允许训练 |

比赛镜像 v1.1 把这张卡编译成了可执行的候选：按类别加权（每类在损失里的总份量相同）和新样本加权（本轮新出现的训练样本每条按 3 条计），见 `contest/contest_submission.py`。它们和其他候选一样要过验收门。

每个修复阶段建好 team_ace 分区后，控制器检查这些条件。符合条件就用 team_ace 自己的函数（在它的锁内）把条目写进该分区的数据库，再重新发布 playbook.md，已有条目全部保留。写入按编号去重。

每阶段的 ACE 分区不同，所以下一个符合条件的阶段会再写一次。不用 team_ace 时，直接写入阶段的 playbook 文件。

**验证方式。** 写入后，用真实的提示词构造函数渲染一次提示（OpenEvolve 的 `PromptSampler` 加 `harness.prompt_extras.get_extras`），要求这条经验恰好出现 1 次，否则报错。搜索结束后，再统计实际发给模型的每条提示中它出现的次数。

单独检查某个阶段：

```bash
python -m team_gas prompt --run-config <out>/rounds/r01/repair/run.json
```

## 命令

```bash
python -m team_gas check  --config configs/gas_A.example.json          # 只检查，不训练
python -m team_gas run    --config <flow.json> --out <dir>              # continuation 模式不调用大模型
python -m team_gas run    --config <flow.json> --out <dir> --allow-llm  # evolve 模式必须显式允许
python -m team_gas status --out <dir>
python -m team_gas report --config <flow.json> --out <dir>              # 重新生成每轮 record.md
python -m team_gas synth  --out <dir> --ace-storage state/demo-gas      # 合成的分批气体数据
```

**输出目录：**

| 路径 | 内容 |
| --- | --- |
| `state.json`、`events.jsonl` | 状态和事件，可断点续跑 |
| `current.json` | 当前应部署的模型 |
| `exposure_ledger.json` | 暴露台账 |
| `rounds/rNN/` | `diagnosis.json`、`repair/`（含 `prompt_preview.txt`）、`mixture/`（每个候选的训练文件、计数和模型，`candidates.json`、`frozen.json`）、`final_acceptance.json`、`record.md` |

## 合成例子

```bash
python -m team_gas synth --out runs/gas_demo --ace-storage $PWD/state/demo-gas
python -m team_gas run   --config runs/gas_demo/flow.json --out runs/gas_demo/run
```

合成数据有两次漂移：从 batch2 起，乙醛漂到乙醇旁边；从 batch5 起，甲苯漂到丙酮旁边。乙醇和丙酮也略向反方向移动，所以只补退化类会挤压混淆类。

修复搜索想用 OpenEvolve 跑、又不想调用收费 API 时，可以启动 `scripts/mock_llm_server.py`。它是本机的模拟服务，按脚本给出改动。启动后把 `repair.search` 设为 `evolve`，`api_base` 指向它，运行时加 `--allow-llm`。

一次完整运行的记录见 [docs/examples/gas_flow_record_r01.md](examples/gas_flow_record_r01.md) 和 [r02](examples/gas_flow_record_r02.md)。

## 示例配置里的数据角色

[configs/gas_A.example.json](../configs/gas_A.example.json) 按目前已知的使用情况写了暴露记录：

| 批次 | 暴露情况 |
| --- | --- |
| b2、b3 | 此前开发用过 |
| b4 | v14 检查用过；是否作为 v16/v17 的补样来源待核对 |
| b5 | v16、v17 用过 |
| b6 | v17 验收用过 |
| b7 | 尚未使用的保留批次，`labels_available: false` |

在确认可以打开 b7 之前，流程会停在最终验收前等待，不会读取它。请把 b4–b7 的路径和摘要按服务器上的实际文件补全。

## 已知限制

- 混合候选默认使用官方策略，没有同时搜索策略和数据配比。
- 换数据阶段只用报警批次里退化类和混淆类的样本；没有使用其他类别的新样本。
- 诊断的参照默认是部署模型的训练集，属于样本内，偏乐观；必要时按轮次指定参照数据。
- 最终验收只基于一个预留批次，统计力度有限。合成例子里的结果只说明流程能走通，不代表真实数据上的效果。
