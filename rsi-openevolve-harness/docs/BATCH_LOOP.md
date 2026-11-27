# 批次循环与重训判断 team_loop

赛题的评测方式是：数据一批一批到，每批必须先输出"上线新模型"或"不上线"，才能拿到下一批。`team_loop` 就是这层外循环。每来一批数据，它做三件事：

1. **判断要不要重训。** 用当前上线的模型在最新一批数据上算指标，和它上线时的水平比较。
2. **要重训就跑一轮。** 先划分（team_split），再搜索（`scripts/run_evolve.py`），最后过验收门（`scripts/run_gate.py`）。
3. **输出决定。** 验收门采纳了新检查点就上线，否则保留当前模型。

整个模块只新增文件。harness、team_split、team_ace 都通过它们公开的函数和文档里的命令调用，没有改动。

```
新一批 ──► 重训判断 ──不重训──────────────────────────────────────────┐
              │重训                                                    ▼
              ▼                                                    输出决定
     划分 + 接配置 ─► OpenEvolve 搜索 ─► 验收门 ─► 采纳则上线，否则保留 ──┘
```

## 一批数据长什么样

```json
{"batch_id": "b03", "train_path": "history.jsonl", "validation_path": "newest.jsonl"}
```

- `train_path`：到目前为止所有可以用来训练的有标签数据。赛题说每次会把之前的数据再给一遍，对应的就是这个文件。
- `validation_path`：最新到的一批有标签数据。重训判断在它上面算指标；如果这一批要重训，它会再被拆成挑选集和验收集。

多批数据可以写成一个 `batches.json`：`{"batches": [批次, ...]}`，按到达顺序排列，相对路径以这个文件所在目录为准。官方平台真正的批次接口目前还没公布；公布后只需要把它给的数据转成上面这三个字段。

## 命令

在项目根目录运行。

```bash
# 建立循环目录：基础配置里的 parent_checkpoint 作为第一个上线模型
python -m team_loop init --base-config configs/run_A.json --out runs/loop/A

# 来了一批：判断、（需要时）训练、输出决定
python -m team_loop step --out runs/loop/A --batch-id b01 \
    --train history.jsonl --validation newest.jsonl \
    --oe-config configs/openevolve.split-v1.yaml --iterations 40

# 一次处理整个列表（目录没初始化时加 --base-config 自动初始化）
python -m team_loop run --out runs/loop/A --batches batches.json --base-config configs/run_A.json

# 只看判断，不训练
python -m team_loop check --out runs/loop/A --validation newest.jsonl

# 本地测试：把公开验证集按文件顺序切成 3 批
python -m team_loop simulate --base-config configs/run_A.json --chunks 3 --out runs/loop/A-sim/data

# 查看每一批的决定
python -m team_loop status --out runs/loop/A
```

`step` 和 `run` 的选项：

| 选项 | 含义 |
| --- | --- |
| `--retrain auto` | 按指标判断，默认值 |
| `--retrain always` | 每批都训 |
| `--retrain never` | 从不训。用来算赛题客观分里"训练和不训练差多少"的对照 |
| `--search evolve` | 用 OpenEvolve 搜索，默认值 |
| `--search continuation` | 不调大模型，验收门只在官方续训和当前模型之间选。适合没有模型服务时使用，也可作为低成本的兜底 |
| `--oe-config` `--iterations` `--api-base` `--model` `--workers` | 原样传给 `run_evolve.py`。密钥由 OpenEvolve 配置从环境变量读取，不走命令行 |
| `--timeout` | 搜索最多运行多少秒，超时后照常进入验收门 |
| `--primary-drop` `--confidence` `--min-rows` `--no-guards` | 重训判断的参数，见下一节 |

基础配置（例如 `configs/run_A.json`）只提供任务名、限制、种子、划分参数和 ACE 设置。它的 `train_path`、`validation_path` 和 `split_manifest` 会被每一批自己的数据和划分替换。

## 重训判断的规则

代码在 `team_loop/trigger.py`。用当前上线的模型在新一批数据上算指标，和"参照水平"比较，满足下面**任一**条件就重训：

- 主指标相对变差超过 `--primary-drop`，默认 5%。
- 某个辅助指标相对变差超过它在任务规格里的护栏阈值：A 的准确率 2%；B、C 的 MAE 5%；C 的 NASA 分 20%。用 `--no-guards` 可以只看主指标。

而且这个变差必须在统计上站得住：对新批次的行做自助法重采样（默认 500 次），指标确实变差的比例要达到 `--confidence`，默认 0.90。这样小批次的随机波动不会白白触发一次搜索。

另外有两条例外：

- 新批次少于 `--min-rows` 行（默认 50）时不重训，因为证据不足。
- 还没有参照水平时直接重训。

输入分布的偏移（新数据的特征均值相对训练标准化偏了多少个标准差）和标签分布的变化（类别占比的总变差距离，或目标均值变化了几个标准差）会写进报告，但**不单独触发重训**。原因是每批都有标签，指标能直接说明这些变化有没有伤到模型。

**参照水平怎么定：**

- **初始化时**：在基础配置的 `train_path` 上量，也就是模型训练用的数据。这是样本内的成绩，偏乐观，所以第一批通常会触发重训。想换一份数据，用 `init --reference-data`。
- **跑过一轮重训后**（不论验收门采纳与否）：参照水平更新为上线模型在这批数据上的成绩。如果新模型上线，它在这批上的成绩部分来自挑选集，略微偏乐观，所以下一次判断会稍微敏感一些。
- **跳过的批次**：参照水平不变。这样慢慢累积的漂移最终会触发重训，而不会被逐批"吸收"掉。

**NASA 分在这里按每行平均比较。** 它本来是对所有行求和，不同大小的批次不能直接比。C 在训练集上的 NASA 分被少数几个样本的指数惩罚放得极大，所以第一次重训之前，这一项基本不会触发。

## 输出

```
runs/loop/A/
  current.json          当前上线的模型：{"checkpoint", "policy", "since"}，外部系统只需读这个文件
  decisions.jsonl       每批一行完整决定
  state.json            当前模型、参照水平、历史
  deployed/<批次>/       上线过的检查点和对应策略（initial 是起点）
  batches/<批次>/
    check.json          重训判断报告
    stage.json split.json run.json   这一批的阶段配置、划分清单、接好清单的运行配置
    evolve.log gate.log  两个子进程的输出
    work-split/         harness 的工作目录（搜索记录、gate/decision.json、selected/）
    decision.json       这一批的决定
```

每批结束时屏幕上会打印一行：

```
DECISION batch=b02 retrain=yes deploy=yes checkpoint=.../deployed/b02/checkpoint.npz seconds=6.49
```

几个行为细节：

- 下一批的父检查点是当前上线的检查点。下一轮搜索的初始程序是它对应的策略，所以采纳过的进化策略会接着往下进化。
- 同一个 `batch_id` 再处理一次，直接返回上次的决定，不重复训练。
- 无人值守的设计：搜索失败或超时时，验收门照常运行，只在官方续训和当前模型之间选；验收门本身也失败时，保留当前模型，并把错误记在 `decision.json` 的 `error` 字段。每一批都一定会给出决定。

## 在公开数据上的模拟结果

把公开验证集按文件顺序切成 3 批，用 `--search continuation` 运行：

| 任务 | b01 | b02 | b03 | 每批用时 |
| --- | --- | --- | --- | --- |
| A | 重训，上线（官方续训） | 重训，上线（官方续训） | 不重训 | 0.3–2.6 s |
| B | 重训，验收门保留原模型 | 不重训 | 不重训 | 0.1–2.5 s |
| C | 不重训 | 不重训 | 重训（RMSE 变差 5.4%），上线 | 0.8–20 s |

A 用模拟大模型（`scripts/mock_llm_server.py`）、每批 6 次迭代跑 `--search evolve`：b01 上线官方续训，b02 由验收门判定 `adopt_candidate`，b03 不重训，三批共约 17 秒。这一次的候选恰好就是初始策略本身，因为模拟大模型的改动都没有更好，所以它说明的是流程能走通，不代表效果。

A 的几批之间类别构成差别很大（类别占比的总变差距离 0.56–0.77），Macro-F1 只在出现过的类别上平均，所以批次之间的波动主要来自类别构成。b03 的 Macro-F1 掉了 25%，但自助法判定"确实变差"的概率只有 0.86，所以没有触发重训。

## 已知限制

- **官方批次接口未定。** 现在按"历史数据 + 最新一批"两个文件设计，平台公布接口后需要写一个转换层。
- **最新一批只用来挑选和验收，不用来训练。** 训练只用 `train_path` 里的历史数据。下一批到来时，这一批会进入历史数据。
- **经验库不跨批积累。** team_ace 的分区包含数据摘要，每批都是新分区、空经验库。
- **使用 team_ace 的基础配置会在 `state/live/` 下建新分区。** 本地测试请用不带 team_ace 的基础配置，tests 里就是这样做的。
- **还没有接 SwanLab。** 接口可以挂在 `step.process` 的结尾：每批的 `decision.json` 已经包含需要记录的全部字段。
