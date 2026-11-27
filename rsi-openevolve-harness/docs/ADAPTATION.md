# 按时间批次推进的持续适应 team_adapt

现在的实验主要用 batch1 续训，再用后面的批次评估。`team_adapt` 在这之上加一层控制：在旧数据上的策略搜索停滞后，如果有已经到达、标签可用、允许用于训练的新批次，就切换到新数据继续学习，再用更晚的数据检查适应效果。

它只新增文件，通过原有入口工作：`team_split make/wire`、`scripts/run_evolve.py`、`scripts/run_gate.py`。训练、评分、验收门和 ACE 的行为都没有改。

```
第 k 阶段：准备数据和划分 ─► 搜索（监视停滞） ─► 验收门 ─► 三方对比
                                                     │
          第 k+1 阶段的起点 = 第 k 阶段验收决定选中的检查点（可能就是原起点）
满足切换条件且下一阶段数据就绪 → 进入第 k+1 阶段；数据未就绪 → 等待；不满足切换条件 → 停止
```

## 1. 声明批次和用途

实验配置是一个 JSON 文件，示例见 [configs/adapt_A_public.example.json](../configs/adapt_A_public.example.json)。每个批次的字段如下：

| 字段 | 含义 |
| --- | --- |
| `id`、`order` | 唯一；`order` 是时间顺序，也是唯一的处理顺序 |
| `path` | 官方格式的 JSONL；可写 `${环境变量}`，相对路径以配置文件所在目录为准 |
| `labels_available` | 现在能否读取标签。为 false 时控制器只等待，不读这个文件 |
| `uses` | `train`、`dev`、`final` 的子集；`final` 不能和另外两个同时出现 |
| `exposure` | 这一批此前被谁用过，必须如实填写。非空就不是独立数据 |
| `sha256`（可选） | 预期的文件摘要；使用前核对，不一致就拒绝运行 |

其他配置项：

| 配置项 | 说明 |
| --- | --- |
| `initial_train_batches` | 起点检查点是用哪些批次训练的 |
| `training.mode` | `replay`：新数据加上之前所有训练批次回放；`new_only`：只用新数据 |
| `stagnation` | `patience`、`min_delta`、`max_consecutive_failures` |
| `budget` | 每阶段的迭代、秒数和 token 上限，`max_stages`，全实验迭代上限 |
| `search.mode` | `evolve`：调用大模型；`continuation`：不调用大模型，只做官方续训和验收 |
| `advance_on` | 搜索以哪些原因结束时允许切换到新数据。evolve 模式默认 `["stagnated"]` |
| `final` | 预留的最终验收批次；`allow_exposed` 默认为 false |
| `focus_classes` | 报告里重点列出的类别，例如 `Acetaldehyde` |

`python -m team_adapt check --config ...` 会列出所有批次的顺序、用途、标签状态、行数、摘要、暴露情况，以及阶段计划和每个阶段的数据是否就绪。最终批次在 `final` 命令之前不会被读取；标签不可用的批次也不读取。

## 2. 阶段怎么排

如果配置没有写 `stages`，按时间顺序自动推导：

- 第 1 阶段用 `initial_train_batches` 训练，开发集是它们之后的第一个可作 dev 的批次。
- 之后每个阶段取上一阶段训练批次之后的下一个可训练批次作为新数据，开发集是新数据之后的第一个可作 dev 的批次。
- 一次只前进一个批次，不跳着读未来的批次。

也可以显式写 `"stages": [{"new_train": ["b1"], "dev": "b2"}, {"new_train": ["b3"], "dev": "b4"}]`。配置加载时会校验：

- 开发批次晚于训练批次；
- 新训练批次晚于上一阶段的训练批次；
- 最终批次晚于所有阶段，并且不出现在任何阶段里。

每个阶段都会生成新的 `stage_id`（`<实验>-sNN`）、新的工作目录和新的划分清单（`team_split make` 加 `derive_run_config`）。ACE 经验库也随之换到新的分区。

## 3. 停滞检测

停滞检测只看本阶段挑选集上的分数，数据来自 `records/iterations.jsonl`。代码在 [team_adapt/stagnation.py](../team_adapt/stagnation.py)。

- 当前最好分数的起点是本阶段初始程序的分数。只有当某个有效候选（评测状态为 ok）超过当前最好分数至少 `min_delta` 时，当前最好分数才更新。
- 连续 `patience` 个有效候选都没做到，就标记为"搜索停滞，考虑新数据适应"。
- 下面几类单独计数，既不算停滞，也不打断停滞计数：

  | 类别 | 内容 |
  | --- | --- |
  | 无效代码 | 静态检查不通过、改到 EVOLVE-BLOCK 外、diff 无法应用、违反续训规则 |
  | API 失败或超时 | 模型调用失败或超时 |
  | 训练失败 | 训练报错或超时 |
  | 内部错误 | 评测器没有产出报告 |

- 连续失败达到 `max_consecutive_failures` 时，记为"搜索不健康"（unhealthy）。这与数据无关，默认不切换数据，需要人工排查。
- 停滞只说明在这份挑选集上搜索找不到足够的改进，不等于已确认数据漂移。事件和报告里都这样写。

监视器每隔几秒读一次记录文件。遇到停滞、超出预算或持续失败时，它向 run_evolve 发 SIGTERM。OpenEvolve 自己的信号处理会停止循环、保存当前最优程序，run_evolve 照常写出总结，所以不会浪费剩余预算。

预算按阶段累计。中断后续跑时，已经用掉的迭代照样计入；如果有 OpenEvolve 检查点，就从检查点续跑。

## 4. 何时切换、等待或停止

一个阶段完成后（验收和对比都做完），按下面的顺序判断：

1. 搜索结束原因不在 `advance_on` 里 → **停止**，退出码 4。例如预算用完但没有停滞、搜索不健康。
2. 已达到 `max_stages`，或全实验迭代预算用完 → **停止**，退出码 4。
3. 没有更多已声明的阶段 → **结束**，退出码 0。没有合规的新数据就不再消耗预算。
4. 下一阶段需要的批次标签不可用或文件不存在 → **等待**，退出码 3。这时不创建任何东西，也不消耗预算；数据就绪后再运行 `run` 即可接着走。
5. 否则记录一条 `phase_switch` 事件，进入下一阶段。

## 5. 承接上一阶段的模型

新阶段的起点是上一阶段验收决定选中的检查点：

- 采纳候选或采纳官方续训时，复制验收门的 `selected/`，并核对它与 `decision.json` 里 `selected_checkpoint` 的摘要一致；
- 验收门保留原起点（`keep_parent`）时，**不假设** `selected/checkpoint.npz` 存在，而是直接复制原起点。

复制结果统一放在 `<out>/models/sNN/`，下一阶段从这里读。初始程序沿用产出该检查点的策略。在同一阶段内，所有候选都从同一个起点、用同一份训练数据训练，比较是公平的。`current.json` 始终指向当前选中的模型。

## 6. 训练和评估的边界

| 评估集 | 角色 |
| --- | --- |
| `train` | 本阶段训练数据，样本内，**不能**作为适应有效的证据 |
| `dev_select` | 本阶段挑选集，搜索打分和停滞判断用过 |
| `dev_gate` | 本阶段验收集，本阶段的验收门用过一次 |
| `old:sNN` | 早先阶段的验收集，用来检查旧场景是否退化；当时的验收门已经用过它 |
| `final` | 预留的最终批次，全流程只用一次（`final` 命令） |

验收集和最终批次从不参与停滞判断和调参。最终验收做过一次之后，实验即关闭，再次调用会被拒绝；`--allow-reuse` 只用于复查，结果会标为非独立。

最终批次如果声明了暴露、在台账里被别的实验用过，或与任何训练/开发批次有重复样本（按 ID 或按内容），都会被拒绝；设了 `allow_exposed` 时可以运行，但结果标为非独立。

暴露台账 `exposure_ledger.json` 按文件摘要记录每次训练、开发和最终验收的使用。某批次一旦参与训练或策略选择，就不再称为该模型的独立测试集。

## 7. 怎么证明新数据适应确实有效

每个阶段都在上面每个评估集上比较以下四个模型：

| 模型 | 说明 |
| --- | --- |
| 更新前模型 | 本阶段的起点 |
| 官方续训 | 相同新数据、相同起点、官方策略 |
| 进化策略续训 | 搜索出的最优程序，在相同数据上训练 |
| 验收门选中 | 验收门最终选的那个 |

报告的内容：

- 整体指标；
- 分类任务的各类别召回、精确率、F1，用 `focus_classes` 突出例如乙醛；回归任务的设备分组指标；
- 每个模型相对更新前模型的变化；
- 旧场景退化：相对变差超过 `forgetting_tolerance` 时标出"退化"。

只有 `dev_gate`、`old:*` 和 `final` 上的结果可以作为证据，`train` 上的分数上升不算。最终验收比较的是：实验起点、每个阶段选中的模型，以及最后一个阶段的更新前模型、官方续训和进化策略续训。

## 8. 记录和 ACE

`<out>/events.jsonl` 按时间记录以下事件：

| 事件 | 内容 |
| --- | --- |
| `experiment_started` | 实验开始 |
| `stage_prepared` | 各批次实际使用的行数和摘要、回放方式、起点检查点摘要、暴露警告 |
| `search_started` / `search_ended` | 预算，结束原因，各类失败计数，迭代、秒数和 token |
| `stagnation_detected` | 停滞详情，并注明不等于漂移 |
| `gate_decided` | 验收门的决定 |
| `stage_compared` | 三方对比结果 |
| `phase_switch` | 触发原因、新批次、起点 |
| `halted` | 停止或等待的原因 |
| `final_acceptance` | 最终验收结果 |

当前进度保存在 `state.json`，`report.md` 是人看的汇总。

ACE 方面不改 team_ace：

- 每个阶段的 scope 按 team_split 的规则重算（数据摘要、划分清单都在 scope 里），不同阶段的经验分开存放。
- 每个阶段的 `ace.record_origin` 写成 `adapt|<实验>|<阶段>|train=…|dev=…|mode=…|validation=dev_only`，team_ace 会把它和每条记录一起存进证据。
- `stages/sNN/experience_provenance.json` 写明这批经验的数据角色和验证级别（仅开发集）。
- 最终验收之后，`final/experience_validation.json` 只记录"哪个阶段的经验对应怎样的最终结果"，不会把 ACE 条目升级为已验证技能。
- 报告不跨数据集比较分数。

## 9. 命令

在项目根目录运行。

```bash
python -m team_adapt check  --config <实验配置> [--out <输出目录>]     # 只检查，不训练
python -m team_adapt run    --config <实验配置> --out <输出目录>        # continuation 模式不调用大模型
python -m team_adapt run    --config <实验配置> --out <输出目录> --allow-llm   # evolve 模式必须显式允许
python -m team_adapt status --out <输出目录>
python -m team_adapt final  --config <实验配置> --out <输出目录>        # 一次性最终验收
python -m team_adapt report --out <输出目录>
```

`run` 可以随时中断，再运行同一条命令就从中断处继续。evolve 模式不加 `--allow-llm` 会直接拒绝，不会自动调用收费 API。

## 10. 本地模拟（只验证流程）

本机只有 A 的 batch1（训练集）和 batch2（公开验证集）。可以把 batch2 按文件顺序切成几段，模拟后续批次：

```bash
python -m team_adapt simulate --base-config <不带 team_ace 的基础配置> --chunks 4 --focus Acetaldehyde --out runs/adapt/A-sim
python -m team_adapt run --config runs/adapt/A-sim/experiment.json --out runs/adapt/A-sim/run
```

这些分段都来自已经暴露过的公开验证集，最终验收会如实标为"非独立"。A 的 batch2 是按气体分段采集的，切开后每段只含几种气体（例如最后一段全是丙酮），所以这只能验证流程能走通，不能评估适应效果。

用 `configs/run_A.json` 这类带 team_ace live 存储的配置做模拟，会在 `state/live` 下建新的经验分区。本地试验请用不带 team_ace 的基础配置。

## 11. 公开数据实验（在服务器上）

示例配置 [configs/adapt_A_public.example.json](../configs/adapt_A_public.example.json) 按证据包如实标了暴露情况：

| 批次 | 暴露情况 |
| --- | --- |
| b1 | 官方训练集 |
| b2 | live 搜索和多次检查都用过 |
| b3 | v7–v14 开发选择用过 |
| b4 | v14 独立检查已用掉一次；是否在 v16/v17 中作为补样来源，待按 captain 记录核对 |
| b5 | v16、v17 中使用过（2026-11-18 更正：原示例把它当作未访问的最终批次，已过时） |
| b6 | v17 中用于验收（2026-11-18 更正） |
| b7 | 尚未使用的保留批次，作为最终批次；`labels_available` 默认为 false，确认可以打开前流程不会读取它 |

使用前需要：

1. 设置环境变量 `A_PUBLIC`（官方 A 包的 public 目录）和 `A_SERVER_BATCHES`（b3–b7 所在目录），并把 b4–b7 的文件名改成服务器上的真实文件名。
2. b5–b7 没有填 `sha256`：请按服务器上的文件补上；b7 还没有被读过，第一次使用时由台账记录。
3. 先运行 `check` 核对，再小预算运行 `run --allow-llm`。确认可以动用 b7 之后，再把它的 `labels_available` 改成 true。

在这个实验里，b2–b6 上的结果都只算开发证据；只有 b7 上的最终验收是独立的。

## 12. 对接将来的官方数据下发

官方的方式是：数据分批到达，标签随批次提供，之前的数据每次重新给一遍，每批必须先给出上线或不上线的决定。对应关系如下：

- 每到一批，适配层往实验配置的 `batches` 末尾追加一条：`order` 为到达序号，`labels_available: true`，`uses: ["train", "dev"]`，`exposure: []`。然后运行 `run`。
- 追加批次、把标签从不可用改成可用，都是允许的配置变化。改动已经用过的批次的声明会被拒绝。
- 新批次在下一阶段作为开发集，再下一个阶段作为训练数据（`replay` 模式下与历史数据一起），和平台"每次重给之前数据"的方式一致。
- 每批要输出的部署决定就是最新阶段的验收结果。`current.json` 指向应上线的检查点；验收门保留原起点就等于"不上线新模型"。
- 官方流程没有我们自己的最终批次（隐藏测试由平台做），所以不配置 `final`。为了每批都按时给出决定，官方模式宜用 `advance_on: ["stagnated", "budget_iterations"]`，并按批次间隔设定每阶段预算。
- `team_loop` 的指标触发（要不要重训）可以放在适配层前面：新批次上指标没有变差时，不追加训练阶段，直接沿用当前模型。

## 已知限制

- 停滞判断基于挑选集分数，对噪声敏感。`patience` 和 `min_delta` 需要按任务调。
- 每个阶段的经验库都从空开始，因为 scope 随数据变化。跨阶段迁移经验需要另行设计，并标明来源。
- 回放模式把所有历史批次等权拼接，没有按时间衰减加权。钩子里的 `sample_weights` 只能看到标签和行顺序。
- 最终验收只有一个保留批次时，统计力度有限；`team_adapt` 不做最终批次上的显著性检验，只做描述性比较。
