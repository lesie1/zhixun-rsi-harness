# 智训引擎 / 工业模型自进化：续训策略 harness

> 2026-11-15：已合并 `team_ace`。A/B/C 配置已连接共享数据和分区经验库；真实模型调用保持关闭。使用方法见 [ACE 接入说明](docs/ACE_ADAPTER.txt)。
>
> 2026-11-23：新增 `team_sim`（用 12 根公开轴承模拟 10 轮数据流、比较更新策略）、`harness/policy_rul_cap.py`（可截断回归目标的训练策略，默认不截断时与 policy_initial.py 训练结果逐字节一致）和预置经验 `team_sim/seeds.json`。只新增文件，现有 A/B/C 运行不受影响。见 [模拟器说明](docs/SIMULATOR.md)。

这是一套已经跑通 CPU 测试链路的比赛实验框架（harness，即把生成、训练、评测和验收串起来的程序）。OpenEvolve 负责提出和搜索策略；被修改的是一个 NumPy 多层感知机（MLP）的**续训策略**，不是网络结构，也不是 Qwen 本身。当前可交接代码、反馈、ACE 接口和验收流程；尚未验证真实 Qwen 27B 的效果。

| 任务 | 预测目标 | 主指标 | 辅助护栏 |
| --- | --- | --- | --- |
| A | 气体六分类 | Macro-F1（各类 F1 的平均，越大越好） | accuracy 相对变差不超过 2% |
| B | 轴承剩余寿命，分钟 | RMSE（均方根误差，越小越好） | MAE（平均绝对误差）相对变差不超过 5% |
| C | 发动机剩余寿命，循环 | RMSE | MAE 5%、NASA 分数 20% |

NASA 分数对寿命高估的惩罚更重，越小越好。有可用报告分组时，还检查该分组下各组主指标相对变差是否超过 10%。这些是当前本地规则，不代表主办方已确认的线上评分。

## 先把环境和数据准备好

以下命令均在项目根目录执行，使用 Linux。环境建议放在项目旁的专用目录，避免被打包；使用能安装 NumPy 1.26.4 的 Python（本项目已在 Python 3.12 环境运行）。

~~~bash
python3 -m venv ../rsi-openevolve-env
source ../rsi-openevolve-env/bin/activate
python -m pip install -r requirements.txt
~~~

如果系统的 venv 缺少 pip：先用 `python3 -m venv --without-pip ../rsi-openevolve-env`，再下载 Python Packaging 官方的 get-pip.py 并用该环境的 Python 执行，最后安装上述 requirements。已有可用环境时直接激活即可。无需再安装 PyPI 版 OpenEvolve：运行脚本优先加载随项目携带的修改版。

将三个官方包解压，使下表目录直接包含 `checkpoint.npz` 和 `public/`。完整说明见 [data/README.md](data/README.md)。数据不随项目 ZIP 分发。

| 任务 | 解压后的包根目录 | 运行配置 |
| --- | --- | --- |
| A | `data/cpu-a-guide-20261110-v1/` | [run_A.json](configs/run_A.json) |
| B | `data/cpu-b-rul-20261110-v1/` | [run_B.json](configs/run_B.json) |
| C | `data/cpu-c-guide-20261110-v1/` | [run_C.json](configs/run_C.json) |

配置内的相对路径以**配置文件所在目录**为基准，因此写成 `../data/...` 和 `../runs/...`。已有只读数据时可在新运行配置中填绝对路径，无需复制数据。每个阶段固定训练文件、验证文件、起点检查点和工作目录；修改这些内容应新建阶段与工作目录，不能沿用旧缓存。

## 四步完成一个阶段

下面以 A 为例；B/C 换对应包名和运行配置。第一步检查实现等价，第二步检查本地评测，第三步搜索，第四步决定是否采用。挑选集用于搜索；验收集仅供最后验收，不能把验收反馈送回搜索。

### 1. 验证初始策略与官方续训等价

~~~bash
python scripts/check_equivalence.py --package data/cpu-a-guide-20261110-v1 --profile A
~~~

同一个起点分别由官方训练器和 runner 续训 5 轮，比较检查点文件 SHA-256（文件内容摘要）。终端应显示 `IDENTICAL`；退出码 0 表示一致。检查点放在临时目录并自动删除，**这一步没有持久输出文件**；需要留证时保存终端输出。

### 2. 单独评估策略

~~~bash
python scripts/evaluate_policy.py --run-config configs/run_A.json
# 查看完整 JSON；自写策略用 --policy 路径，强制重算用 --no-cache
python scripts/evaluate_policy.py --run-config configs/run_A.json --json
~~~

默认评估 [policy_initial.py](harness/policy_initial.py)。首先准备固定划分、起点指标和官方续训参考，再进行静态检查、子进程限时训练、续训规则检查、自行重算指标、评分、生成反馈。评分以本阶段起点为基准；所有候选都从同一个起点检查点训练，并非从搜索父策略的训练结果继续接力。

本例工作目录为 `runs/A/round-01/`。主要产出：

| 相对工作目录的路径 | 内容 |
| --- | --- |
| `splits/manifest.json` | 默认生成的划分清单；配置外部清单时读取外部路径 |
| `splits/select.jsonl`、`splits/gate.jsonl`、`splits/materialized.json` | 挑选/验收数据副本和物化标记 |
| `stage_context.json` | 阶段、规则、分组和参考指标摘要 |
| `reference/` | 初始策略产生的官方续训参考 |
| `candidates/<代码摘要前16位>/` | 成功候选的 checkpoint.npz、training.json、history.jsonl、runner_report.json |
| `reports/<完整代码摘要>.json` | 完整评测报告，包括中文反馈；失败报告也会保留 |
| `cache/`、`locks/` | 数据缓存和并行互斥锁 |

相同阶段按策略源码 SHA-256 缓存，锁用于避免相同代码并发训练。命令显示成功退出不等于策略有效：还要看报告 `status == "ok"` 和护栏。

### 3. 用驱动模型搜索策略

~~~bash
python scripts/run_evolve.py --run-config configs/run_A.json --iterations 40   --api-base http://127.0.0.1:8000/v1 --model qwen
~~~

先启动本地 Qwen 27B 的 OpenAI 兼容服务，将地址和 `--model` 换为服务实际值；需要鉴权时使用 `--api-key`。这里不负责部署模型服务。默认搜索采用 SEARCH/REPLACE（精确匹配原文后替换）并限制在 EVOLVE-BLOCK；默认 60 轮、2 个并行评估，可用 `--iterations`、`--workers` 覆盖。配置见 [openevolve.yaml](configs/openevolve.yaml)。

产出 `run_config.resolved.json`、`initial_policy.py`、`system_message.txt`、`evolve_summary.json`，及 `openevolve_output/` 下的日志、进化轨迹、检查点和 `best/best_program.py`。每轮记录写入 `records/iterations.jsonl`；默认 ACE 占位模块每 10 条记录更新 `playbook/bullet_stats.json`，不会自动生成经验条目。分数为 0 的候选默认不进种群（可作为下一轮父程序的候选集合）。

无 GPU 时，可另开终端运行：

~~~bash
python scripts/mock_llm_server.py --port 8765
~~~

搜索命令改用 `--api-base http://127.0.0.1:8765/v1 --model mock`。这个服务按脚本返回修改和故意构造的错误，只验证链路，不代表真实大模型能力。

### 4. 运行验收门并选出下一阶段起点

~~~bash
python scripts/run_gate.py --run-config configs/run_A.json
~~~

默认读取搜索得到的最佳策略，也可用 `--policy` 指定文件。配对 bootstrap（在同一批重采样样本上比较两个模型）默认重采样 2000 次，检查主指标胜出频率及护栏：独立验收阈值 0.90，无独立验收时 0.95。

结果写入 `gate/decision.json`：

| decision | 实际结果 |
| --- | --- |
| `adopt_candidate` | 采用候选，复制到 `selected/`，补充对应 policy.py |
| `adopt_reference` | 候选未获采用，采用官方续训，复制到 `selected/` |
| `keep_parent` | 保留原起点；不创建 selected/，selected_checkpoint 指向原 parent_checkpoint |

采用时 `selected/` 包含 checkpoint.npz、training.json、history.jsonl、runner_report.json 和 policy.py。重复执行验收会替换该阶段已有 selected/。候选必须通过对起点的验收；当官方续训也通过时，候选还需主指标不差于官方续训。代码不会要求 candidate_vs_reference 的 bootstrap `passes` 必须为真。

下一阶段复制一份运行配置，更新 `stage_id`、`work_dir`，以及 `parent_checkpoint`。若本阶段采用了模型，后者填前一阶段 `selected/checkpoint.npz`；统一稳妥的取法是读取 `gate/decision.json` 的 `selected_checkpoint`（也覆盖 keep_parent）。例如 A 的第二阶段可用 `round-02`、`../runs/A/round-02`、`../runs/A/round-01/selected/checkpoint.npz`。阶段数据变化时同时更新 train_path、validation_path 和外部 split_manifest。新的配置文件由运行者创建，不能覆盖上一阶段产物。

## 目录与交接入口

| 路径 | 作用 |
| --- | --- |
| [harness/](harness/) | 策略、冻结 runner、评测评分、详细反馈、划分、ACE 适配和验收门 |
| [scripts/](scripts/) | 四步命令及假 LLM 服务 |
| [configs/](configs/) | A/B/C 运行配置和 OpenEvolve 搜索配置 |
| [prompts/](prompts/) | 固定系统约定、任务说明和用户提示模板 |
| [baselines/](baselines/) | 官方 NumPy 模型、训练和预测实现 |
| [third_party/openevolve/](third_party/openevolve/) | OpenEvolve 0.4.1，交接版本 commit 3e72b5b，仅扩展 process_parallel.py |
| [tests/](tests/) | 27 个 harness 测试 |
| [docs/](docs/) | [钩子](docs/HOOKS.md)、[反馈](docs/FEEDBACK.md)、[ACE](docs/ACE_INTERFACE.md)、[划分](docs/SPLIT_INTERFACE.md)、[补丁](docs/OPENEVOLVE_PATCH.md) |
| [docs/samples/](docs/samples/) | 真实执行的反馈、提示、记录、验收决定及截短的划分清单 |
| [data/](data/) | 只提供放置说明，官方包另行准备 |
| `runs/` | 运行时生成，不入最终 ZIP |
| [requirements.txt](requirements.txt)、[.gitignore](.gitignore) | 依赖与忽略规则 |

只有 policy_initial.py 的 EVOLVE-BLOCK 内 CONFIG 和五个钩子参与进化。runner、evaluator、指标、数据、标准化、Adam 状态和检查点结构均属于固定部分。规则选择顺序是：运行配置的 `spec_file` JSON → 已知 A/B/C 表 → 按任务类型的默认值；不会自动解析任意自然语言说明文件。

评分公式见 scoring.py：主指标比值 − penalty_weight × 护栏超限扣分 + 极小平局分，最低为 0。主指标比值大于 1 表示改善；平局分系数 0.001。该实现不是严格的字典序排序，极小的主指标差距仍可能受平局分影响。

## 已验证的内容

1. 初始策略续训 5 轮与官方 `python -m baselines.train --from-baseline ... --epochs 5` 的检查点逐字节一致。A/B/C SHA-256 前 16 位分别为 `90150659753f6e58`、`a2926c0606cc2680`、`c3367b6f8c8f6468`。
2. 设置 HARNESS_DATA_A/B/C 指向包根目录时，27 个 harness 测试全部通过；未设置时跳过 8 个依赖数据的测试。指标函数在三个基线检查点上与官方 training.json 对应指标一致到 12 位小数。
3. 已留存的假 LLM 运行中，A/B/C 各完成 24 轮搜索及验收，无链路崩溃。假服务故意返回无效候选，因此记录中的 policy_rejected、train_failed、rejected_before_eval 是预期测试路径，不是“所有候选都成功”。A 采用官方续训：搜索最优未通过验收，展示了对选择偏差的拦截；B 因分组护栏违规保留起点；C 采用候选。
4. 官方 baselines/tests 有 2 个测试因缺少主办方未随包发布的 `scripts.package_baselines` 而报错；原始 B 包同样在这两个打包相关测试的导入处报错；原始 A/C 包各只有 9 个测试且全部通过。不能笼统说三个原始包都失败；这是 B 包对应测试依赖的上游打包模块缺失。与本项目 27 个 harness 测试是两套测试。
5. 本次收尾先后复测 harness，核对现有受保护文件的 SHA-256；文档和打包不改变代码、配置或提示模板行为。详细执行结果见 [VALIDATION.md](docs/VALIDATION.md)。

复现测试：

~~~bash
HARNESS_DATA_A=data/cpu-a-guide-20261110-v1 HARNESS_DATA_B=data/cpu-b-rul-20261110-v1 HARNESS_DATA_C=data/cpu-c-guide-20261110-v1 python -m unittest discover -s tests
~~~

## 已知限制和待办

- 尚未用真实 Qwen 27B 跑过；SEARCH/REPLACE 的精确匹配率、生成有效率和收益需要实测。
- ACE 目前只有接口与计数占位实现，需要同学接入经验总结/整理模块；SwanLab 尚未接入，可由完整 iterations.jsonl 写适配器；Dockerfile 尚未编写。
- 数据划分是临时实现。接入正式实现前按 SPLIT_INTERFACE 验收；当前 validate_manifest 只做部分结构和集合检查，不会自动核验清单摘要、来源摘要、分组隔离或时间顺序。
- B 验证集只有 3 根轴承，无法留出独立验收集。代码使用分组等权评分、提高验收阈值并警告，但选择偏差仍存在。
- 平台 Macro-F1 具体算法、每阶段提供哪些数据、是否返回采纳结果，均待与主办方确认。当前 macro_f1 对真值或预测中出现的类别求平均；macro_f1_all 是包含全部配置类别的另一个字段。
- 策略静态检查不是安全沙箱。当前已有子进程超时；真正隔离还需受限容器等部署措施。本项目没有完成容器配置，也没有通过子进程本身提供文件/网络/内存权限隔离。
- 代码阅读发现验收 reason 在“候选通过、官方续训不通过”分支可能仍写“且不差于官方续训”；以 decision 和 comparisons 数值为准。本次只记录，不改逻辑。
- 本地验收只说明当前数据和规则下的结果，不能替代平台的最终采纳结论。不要反复用验收集调策略。
