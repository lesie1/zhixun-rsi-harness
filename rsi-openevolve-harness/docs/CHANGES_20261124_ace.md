# 2026-11-24 第二次改动：ACE 返回空正文

起因：打上 11-24 补丁后，在 captain 上重跑了试跑（`rsi-sim-20261124/runs/B-live-01`，第 0 折，给法 1 和 2c，策略 v3）。

- 搜索：5 次全部交出了候选。
- ACE：10 次调用全部失败，报错都是 `empty or oversized response`。每次调用约 2 分钟，一条经验也没写出。

## 原因

pytrio 上的 Qwen3.8-27B 默认先思考再回答。思考内容放在单独的字段里返回，但计入输出 token 上限。ACE 的提示词很长，思考阶段就把 4096 token 的上限用完了，所以正文是空的。

用 4 个小请求确认过（2026-11-24，提示词：只回复 `{"ok": true}`）：

| 写法 | 结果 |
|---|---|
| 默认 | 先思考（单独返回 134 字），共输出 42 token |
| 请求里加 `chat_template_kwargs: {"enable_thinking": false}` | HTTP 400：`chat_template_kwargs is not supported` |
| 请求顶层加 `enable_thinking: false` | 不思考，6 个 token 直接给出 `{"ok": true}` |
| 提示词末尾加 `/no_think` | 无效，仍然思考 |

另外，失败的调用不记录 token 用量，所以汇总里的 ACE token 是 0。实际大约花了 10 × 4096 token。

## 修改

| 路径 | 内容 |
|---|---|
| `team_ace/backend.py` | 改动很小，只影响打开新开关的运行：<br>- 新增 `thinking_body`：只有配置里写了 `ace.enable_thinking`，请求才带上 `enable_thinking` 字段；不写时请求和原来完全一样。<br>- 回复达到输出上限被截断时，报 `reply cut off at the output-token cap (N tokens)`。<br>- 空回复的报错里写明结束原因。<br>- 失败的调用也记录 token 用量。 |
| `team_sim/search.py` | 新参数 `ace_thinking`，写进每轮 ACE 配置的 `enable_thinking` |
| `team_sim/__main__.py` | `run` 新增 `--ace-no-thinking`；新子命令 `ace-retry` |
| `team_sim/ace_retry.py` | 新文件，在某一轮 ACE 状态的副本上重跑排队任务，原运行不动。<br>- 每轮的数据文件被删除时（运行没加 `--keep-data`），用 `--package` 重新生成，并核对哈希。<br>- 密钥环境变量没设置时，会提示输入（输入时不显示）。 |
| `tests/test_team_sim.py` | 新增 7 个测试：<br>- 开关只在设置时发送<br>- 搜索配置带上开关<br>- 本地假服务器上的 live 请求：带开关、默认不带、截断时报错并记 token<br>- `ace-retry` 只改副本、live 模式必须显式批准<br>- 数据文件重建与哈希核对 |
| `docs/SIMULATOR.md` | 第 6 节补充 `--ace-no-thinking` 和 `ace-retry` |

## 验证（本机）

- `tests.test_team_sim` 加 `tests.test_ace_adapter`：35 个测试全部通过，没有跳过。
- 端到端：模拟器配本地假模型搜索，ACE 用 fixture 模式，加 `--ace-no-thinking`，第 0 折，给法 2c，2 轮。
  - 每轮 run.json 里都有 `enable_thinking: false`。
  - 跑完后数据目录已删除。
- `ace-retry`：
  - 不加 `--package` 时，提示数据已删除。
  - 加 `--package` 后，数据重建并通过哈希核对，副本里的排队任务处理完成。
  - 原运行的 `memory.sqlite3` 哈希前后一致。

## 补充：Curator 回复包在代码块里（同日）

在 captain 上用 `ace-retry` 重试 B-live-01 ②c 第 10 轮，结果如下。

- 关闭思考生效：Reflector 和 Curator 两次调用都成功，整个任务 43 秒，共 5576 token（Reflector 输入 1767、输出 791；Curator 输入 2577、输出 441）。
- 任务仍然失败，报 `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`。原因：适配层在上游 Curator 解析之后，又对它的回复做了一次严格的整段 `json.loads`；Qwen 把 JSON 包在代码块里（或在前面加了一句话），这一步就失败了。上游自己用 `extract_json_from_text` 解析，能接受这种写法。本机用假服务器返回代码块里的 JSON，复现出了同样的报错。
- 第 2 个任务报 `model-call budget exhausted`：那次运行每轮预算是 4 次，原运行已经用掉 2 次，第 1 个任务又用掉 2 次。

修改：

| 路径 | 内容 |
|---|---|
| `team_ace/backend.py` | Reflector 和 Curator 的回复改用上游的 `extract_json_from_text` 读取（整段 JSON、代码块，或第一个配对完整的 {…}），完全没有 JSON 时任务失败。之后的检查不变：只接受 ADD、条数要和上游解析的一致，写入前还要过严格的字段和证据检查。 |
| `team_sim/ace_retry.py`、`team_sim/__main__.py` | `ace-retry` 新增 `--ace-max-calls`，只改副本的调用预算 |
| `tests/test_team_sim.py` | 新增 2 个测试：回复包在代码块里、或前面带一句话时能被接受；回复里完全没有 JSON 时任务失败 |

验证（本机）：`tests.test_team_sim` 加 `tests.test_ace_adapter`，37 个测试全部通过。改动前，新增的代码块测试报出的错误和 captain 上相同。dino 只有 2 GB 内存，可用内存接近 250 MB 时，worker 会被内存保护先杀掉，个别测试会因此失败或跳过；单独重跑都能通过。

## 补充 2：经验的 content 写成了普通文字（同日）

打上 ace2 补丁后，再用 `ace-retry` 试了一次（②c 第 10 轮，2 个任务，4 次调用，共 11568 token）。4 次调用都成功了，但两个任务仍然报 `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`。

- 上一节"代码块"的判断不完整。ace2 补丁已经生效（`--ace-max-calls` 被接受了），读 Reflector 和 Curator 回复的地方都不会再抛这个错。剩下唯一对模型输出做严格 `json.loads` 的地方，是 `team_ace/worker.py` 写入经验那一步：它要求每条 ADD 的 content 是一个 JSON 字符串。
- 上游 Curator 提示词模板里的示例是 `"content": "[New calculation method...]"`，也就是普通文字；而我们对格式的要求只写在任务说明里。Qwen 照着模板写了普通文字。对一段中文普通文字做 `json.loads`，报错和 captain 上一字不差。
- fixture 的模拟回复永远是规范的 JSON 字符串，所以本机测试一直没测出这个问题。

修改：

| 路径 | 内容 |
|---|---|
| `team_ace/backend.py` | - 任务说明里给出完整的操作示例，content 写成 JSON 对象<br>- 新增 `memory_content`：content 是 JSON 对象，或者是包含 JSON 对象的字符串（整段、代码块、前面带一句话），都接受；普通文字拒绝，报错里带上前 200 字<br>- Reflector 或 Curator 的回复里没有 JSON 时，报错也带上前 200 字 |
| `team_ace/worker.py` | 已有代码，用 diff 补丁打（基准是 11-23 完整包里的版本，sha256 开头 `fcc40db8`）：<br>- 写入经验改用 `memory_content`<br>- content 里多出来的字段丢弃，不存<br>- 缺字段时报出缺的是哪些<br>- 证据 ID 允许写成唯一前缀（至少 8 个字符；模型常把 64 位的 ID 写短），对不上任何 ID、或者同时对上多个，都拒绝<br>- 文字过长时报出是哪个字段 |
| `tests/test_team_sim.py` | 新增 2 个测试：<br>- live 模式下 worker 完整走一遍：content 写成对象、ID 写短、带多余字段时写入成功；content 是普通文字时任务失败，报错里带原文<br>- 证据 ID 前缀规则 |

原有的"伪造经验必须拒绝"测试照样通过：编造的 ID、跨任务的经验仍然会被拒绝。

验证（本机，先关掉 Chrome 并禁用 Pylance，腾出内存）：A/B/C 全部 138 个测试通过，没有跳过（其中 tests.test_team_sim 29 个，tests.test_ace_adapter 10 个）。

## 补充 3：输入填错了，输出格式和模板冲突（同日，第四次修改）

打上 ace3 补丁后，再用 `ace-retry` 试了一次：4 次调用都成功，共 13159 token。报错第一次带出了 Qwen 写的原文，例如：

> 待验证｜B｜症状假设：模型可能因‘假设非事实’约束而拒绝输出或混淆反馈与现有条目；建议：无论反馈是否匹配现有条目，必须生成结构化JSON；……证据：reflection-01；……

从原文能看出两层问题，前三次修改都没碰到。

1. **输入填错了**。上游 Reflector 的模板是“诊断一个模型为什么答错”：题目、模型的推理过程、模型的答案、环境反馈。我们把给 Curator 的格式要求填成了题目，把 `Experiment hypotheses are unverified.` 填成了推理过程，把 `No supervised answer; …` 填成了答案。Qwen 看到的是“有个模型被要求给出结构化建议，却拒绝了”，就认真诊断起“模型因约束拒绝输出”。所以即使格式对了，写出来的经验也全在讲提示词，不是在讲轴承寿命模型怎么训练。
2. **输出格式和模板冲突**。上游 Curator 模板的示例里，content 是一句文字，前面还附着现有经验库（每条都是“待验证｜B｜症状假设：…；建议：…；范围：…；证据：…”）。Qwen 照着经验库的格式写，非常自然；我们“写成 JSON 对象”的要求放在最后，被盖过去了。另外，它照格式写了“证据：reflection-01”，这是编造的 ID：27B 模型几乎不可能原样抄出 64 位的 ID。

这次不再要求模型按我们的格式写，而是：输入按上游的本意填，输出按模型最自然的写法接。

| 路径 | 内容 |
|---|---|
| `team_ace/backend.py` | Reflector 的题目改为任务和指标的说明；推理过程填搜索模型的假设，答案填它改的代码，环境反馈填实测结果（附各数值的含义）。<br>Curator 的说明改为按经验库的格式写“症状假设/建议/范围”，并要求只写关于训练本任务模型的建议。<br>Curator 的回复我们自己读：只要求有 operations 列表，不再依赖上游对 reasoning、section 字段的检查；非 ADD 的操作忽略；最多取 5 条。Reflector 的回复不再要求是 JSON。<br>`memory_content` 能读三种写法：JSON 对象、带标签的文字、无标签的一句话。<br>每次调用的完整提示词和回复存到 `replies/`。 |
| `team_ace/worker.py` | 证据改为系统附上（该任务的实验记录），模型写的 ID 一律忽略（去掉 ace3 加的 ID 前缀规则）。<br>内容写明属于别的任务时，整批拒绝。<br>某条操作读不出建议时跳过；全部读不出时任务失败。<br>缺“症状”或“范围”时填默认说明；每段超过 250 字截断。 |
| `team_sim/ace_retry.py` | 输出里给出 `replies/` 的路径 |
| `tests/test_team_sim.py` | 用 Qwen 在 captain 上写出的两段原文回放：两段都能写成经验，证据是该任务自己的记录，不会是 reflection-01。<br>另外检查：Reflector 的提示词里有假设、代码改动和实测结果，不再有 No supervised answer；跨任务的内容被拒绝；没有建议的内容失败并带原文；完整回复存了下来。 |
| `tests/test_ace_adapter.py` | 已有测试改了一处：编造的证据 ID 现在不再导致整批失败，而是被忽略并换成该任务自己的记录；跨任务的内容照样整批拒绝，且一条都不写入。 |

验证（本机）：A/B/C 全部 138 个测试通过，没有跳过；本机模拟输出上的 `ace-retry`（真实 worker 子进程）处理成功，`replies/` 里有完整的提示词和回复。

还没验证的：Qwen 按新输入写出的经验，内容是否真的在讲训练。这只能在 captain 上用真实模型看，`ace-retry` 的输出会列出每条新经验的全文。

## 补充 4：ACE 跑通了；经验库进提示词的规则（同日，第五次修改）

ace4 补丁后，在 captain 上用 `ace-retry` 重试了 ②c 第 10 轮和第 4 轮，共 4 个任务，全部成功，写出 4 条经验，共约 3.06 万 token，每个任务 50 到 102 秒。经验讲的是训练本身：例如“加 epochs 后出现后期退化 → 改用 pick_epoch 而不是继续加轮数”，“WORSE_THAN_START → 降学习率、减轮数”。其中 1 条比较弱，基本在复述搜索模型自己的假设。每条只来自 1 次实验，都是待验证的候选。

在本机按 team_ace 的渲染逻辑模拟时，发现完整运行会有一个问题：经验库按 ID 排序、截到 1900 字。ACE 经验的 ID 是哈希值（`B-…`），排在预置经验（`R-00-BASE-…`）前面。加 2 条时还都能显示；加 4 条时，预置的“特征”和“指标”两条就被挤掉了。完整运行里，每轮的新经验会带到下一轮，几轮之后预置经验会被逐渐挤掉，而显示哪几条 ACE 经验只由哈希值决定。

| 路径 | 内容 |
|---|---|
| `team_ace/store.py` | 已有代码，用 diff 打：<br>- 新增 `render_settings`。配置里有 `ace.render` 时，先放全部预置经验，再放最多 N 条学到的经验（按 `updated` 从新到旧），总长上限 C。<br>- 没有 `ace.render` 时，输出和原来逐字相同。 |
| `team_ace/worker.py` | 写入经验时记下 `created` 和 `updated`。同一条建议被新证据再次提出时，`updated` 更新为当时。 |
| `team_ace/backend.py` | 请 Curator 每段写在 100 字以内（原来是 250 字；写入时仍按 250 字截断） |
| `team_sim/search.py`、`team_sim/__main__.py` | 每轮配置写入 `ace.render = {base_first: true, max_learned: --ace-max-shown（默认 3）, max_chars: 2400}`。2400 小于搜索提示词对经验库的 2500 字上限。 |
| `scripts/sim_status.py` | 新文件，只读，查看一次运行的搜索和 ACE。每轮列出：带入了几条经验、搜索提示词里实际出现了几条 ACE 经验（按记录里的经验库版本号查）、新经验全文。 |
| `tests/test_team_sim.py` | 新增 2 个测试：<br>- 预置经验在前、最新 3 条在后、长度不超过上限<br>- 没有 `ace.render` 时仍按原来的 ID 顺序和 1900 字<br>另在已有测试里检查：搜索配置带上了渲染规则；Curator 收到“100 字以内”；新经验记下了时间。 |

用 captain 上那 4 条真实经验模拟新规则：2c 显示全部 5 条预置经验和最新 3 条，共 1906 字；给法 1 显示全部 6 条预置经验和最新 3 条，共 2066 字。

验证（本机）：
- A/B/C 全部 140 个测试通过，没有跳过。
- 端到端：5 轮模拟，配假模型搜索和 fixture 模式 ACE，每轮都搜索。
  - 每轮带入的经验分别是 0、1、2、3、4 条；搜索提示词里实际出现的 ACE 经验是 0、1、2、3、3 条。
  - 第 5 轮的经验库是：5 条预置经验在前，后面是第 4、3、2 轮写的经验，共 1579 字。
- `scripts/sim_status.py` 在这次输出上能正常列出以上信息。
