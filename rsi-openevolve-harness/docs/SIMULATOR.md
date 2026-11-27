# 数据流模拟器 team_sim（第一层：只用公开数据）

2026-11-23 新增。目的：在 12 根公开轴承上模拟平台"10 轮数据流"，比较不同的更新策略，提前发现问题。只读官方 B 包，不调用大模型（除非显式打开搜索）。

## 1. 怎么跑

```bash
# 完整模拟：4 折 × 4 种给法 × 4 种策略，结果在 runs/sim/B/report.md
python -m team_sim run --package /path/to/cpu-baselines-20261110-v1/B --out runs/sim/B

# 只跑一部分
python -m team_sim run --package ... --out runs/sim/B-quick --folds 0 --protocols 1,2c --policies never,v3

# 加消融：只用上限 1.0 / 只用不截断
python -m team_sim run --package ... --policies never,always,v1,v3,v3_cap1,v3_nocap

# 重新汇总报告；查看预置经验进入提示词后的样子
python -m team_sim report --out runs/sim/B
python -m team_sim seeds --protocol 2b
```

在 1 核机器上，一折一种给法跑 4 种策略约 1 分钟，全量约 15–30 分钟；B 的内存峰值不到 200 MB。

## 2. 角色和"时期"

每个工况 4 根公开轴承，按折轮换角色，每根轴承都会当一次隐藏轴承：

| 角色 | 每工况 | 用途 |
|---|---|---|
| train | 2 根，整段寿命 | 用官方配方从头训出本折的伪基线（官方检查点见过 _1–_3，不能直接用） |
| stream | 1 根 | 逐段给出带标签的新数据（②），或作为固定验证设备（①） |
| hidden | 1 根 | 逐段测试计分，相当于平台的隐藏测试 |
| stream_val（仅 ②c） | 从 train 里拿 1 根 | 只做验证、永不训练；②c 的伪基线只用剩下 1 根训练 |

"时期"按相对寿命对齐：每根轴承的寿命切成 10 段，第 k 轮对应各自寿命的第 k 个 10%。

## 3. 四种给法（第 k 轮）

| 给法 | 训练集 | 验证集（验收门用） | 测试 |
|---|---|---|---|
| 1 先测后训 | 初始 + 隐藏第 1…k−2 段 | stream 整根 + 隐藏第 k−1 段（刚回传） | 隐藏第 k 段（下一时期） |
| 2a | 初始 + stream 第 1…k−1 段 + 第 k 段前 70% | stream 第 k 段后 30% | 隐藏第 k 段（同期，测完作废） |
| 2b | 初始 + stream 第 1…k−1 段 | stream 第 k 段 | 同上 |
| 2c | 初始 + stream 第 1…k 段 | stream_val 第 k 段（这台设备从不训练） | 同上 |

注意：2a、2b 里验证用的设备前几段已经进了训练集，验收门会偏爱"记住这几台设备"的候选，而这对没见过的隐藏设备没有帮助。冒烟测试里 v3 在 2b 第 0 折平均比基线差 17%，原因就在这里。2c 的验证设备从不训练，测的才是隐藏测试需要的"跨设备"能力。平台上要先确认验证数据和训练数据是不是同一批设备（第一次试运行看 id）。

## 4. 比较的更新策略

| 策略 | 规则 |
|---|---|
| never | 一直用静态基线 |
| always | 训练集有新标签就续训，直接上线，不验收 |
| v1 | 现有 team_loop 规则（team_loop/trigger.py 原样调用）：和模型自己的"参照水平"比，主指标变差 >5% 且概率 ≥0.9 才续训，过验收再上线，然后在这批数据上重测参照水平 |
| v3 | 本次确定的规则，见下 |
| v3_cap1 / v3_nocap | v3 的消融：每轮只训上限 1.0 / 只训不截断 |

v3（第三版 + 两处补充）：

1. 有新训练标签的每一轮，从当前模型做便宜续训，候选上限 = {1.0, 0.8, 1.2, 不截断} × 本轮训练标签均值（`harness/policy_rul_cap.py`，评分永远用真实标签）。
2. 静态基线始终在候选池里。
3. 尺子：在同一批最新带标签数据上的相对优势 =(基线 RMSE − 模型 RMSE)/基线 RMSE，可以跨轮比较。
4. 红灯：当前模型比基线差，概率 ≥0.8 → 立刻回退到 {当前、候选、基线} 中验证集上最好的；只有所有候选都打不过基线时才请求搜索。
5. 黄灯：优势低于前两轮平均的一半，且至少缩 5 个百分点，概率 ≥0.8 → 请求搜索；本轮先上线池中验证集上最好的。
6. 绿灯：候选更好的概率 ≥0.9 才替换；1.0 以外的上限要比 1.0 好 2% 以上且概率 ≥0.9。
7. 兜底：任何非基线模型若在最新数据上输给基线（概率 ≥0.8），改用基线。

所有概率都是同一轮数据上的配对行自助法（同一轮所有模型共用同一组重采样）。门槛是起点，用本模拟器校准。

## 5. 输出

```
<out>/f<折>/static/checkpoint.npz            伪基线（②c 另有 static-2c/）
<out>/f<折>/p<给法>/<策略>/rounds.jsonl      每轮：数据量、决定（红黄灯、候选分数、理由）、测试分数
<out>/f<折>/p<给法>/<策略>/summary.json      按客观分项汇总
<out>/summary.json, report.md               所有组合
```

summary 的字段对应评分项：`mean_advantage`（平均相对优势）、`overall_advantage`（全程按行）、`mean_unit_advantage`（按轴承等权）、`worst_advantage` / `rounds_below_static` / `mean_rounds_to_recover`（最差阶段与恢复）、`advantage_std`（稳定性）、`trainings` / `search_requests`（资源）。打开搜索时另有 `searches_run`（搜出新候选并训练的次数）、`search_adopted`（搜索候选上线次数）、`search_time_limited`（搜索到时限停下的次数），report.md 里对应"搜出新候选 / 搜索候选上线 / 搜索到时限"三列；旧的 summary 没有这些字段时显示"—"。

## 6. 搜索与 ACE（可选，默认关闭）

红灯/黄灯请求搜索时，默认只记录 `search_requested`。打开搜索：

```bash
export OPENEVOLVE_API_KEY=...        # 只放环境变量，不写进任何文件
python -m team_sim run --package ... --out runs/sim/B-search --policies v3 \
    --search evolve --api-base <OpenAI 兼容地址> --model <模型名> --iterations 6 \
    --ace team_ace --ace-mode disabled
```

- 搜索调用原有 `scripts/run_evolve.py`，起点策略是上限 1.0 的 policy，起点检查点是当前模型；搜到的最佳策略续训后和其他候选一起过验收门。最佳策略和起点策略一模一样时不算新候选，不重训。
- 时限 `--search-timeout`（默认 1800 秒）：到时限给搜索主进程发 SIGTERM，OpenEvolve 停止派发新迭代、保存已完成迭代里最好的策略并正常退出，所以慢的搜索也能交出候选；90 秒内没退出才强杀。搜索在独立进程组里运行，结束后组里残留的进程一律清掉（OpenEvolve 的工作进程在主进程被杀后会继续调用模型）。`search_result.json` 的 `stopped` 为 `time_limit` 表示到时限停下，`killed` 表示被强杀。
- ACE：`--ace team_ace` 为该轮建立 team_ace 分区（在 `state/sim/<输出目录名>-<路径哈希>/f<折>-p<给法>-<策略>/` 下，不同运行、不同策略互不共用），并写入预置经验；`--ace-mode disabled` 只把经验放进提示词、把记录入队，不调用模型；`fixture` 用模拟回复测接口；`live` 需要 `--ace-approve-live`、`--ace-max-calls`、`--ace-model`、`--ace-api-base`、`--ace-api-key-env`，会花额度。`--ace-worker-seconds N --ace-max-jobs M`：搜索后最多处理 M 个排队任务（每个任务一次 Reflector + 一次 Curator 调用），每次限时 N 秒。team_ace 的分区按数据和检查点划分，每轮都是新分区；同一次运行里上一轮 ACE 写出的候选经验会带入下一轮分区（标记 carried_from），否则 ACE 学到的东西永远到不了后面的提示词。调用预算 `--ace-max-calls` 按分区计，也就是每轮各算一次。
- ACE 单次调用：`--ace-request-timeout`（默认 300 秒）和 `--ace-max-tokens`（默认 4096）。team_ace 自己的默认值是 20 秒、1024 token，对 Qwen3.8-27B 这类先思考再回答的模型太小：2026-11-23 captain 试跑时每次 ACE 调用都在 20 秒超时，一条经验也没写出。`--ace-worker-seconds` 必须不小于 2 × 单次超时 + 30（一个任务是一次 Reflector 加一次 Curator），否则启动时报错。worker 在系统可用内存低于 250 MB 时会拒绝运行（`stop_reason: memory_limit`），任务留在队列里。
- 关闭思考 `--ace-no-thinking`：每次 ACE 调用带上 `enable_thinking: false`。pytrio 上的 Qwen3.8-27B 默认先思考再回答，思考内容单独返回，但占输出 token 上限。2026-11-24 第二次试跑时，10 次 ACE 调用都在约 2 分钟后返回空正文，原因是思考把 4096 token 用完了。同日用小请求测过三种写法：`enable_thinking: false` 有效（6 个 token 直接给出 JSON）；`chat_template_kwargs` 被拒（HTTP 400）；提示词末尾加 `/no_think` 无效。不加这个参数时，请求和原来完全一样。失败的调用现在也记录 token 用量，报错里会写明结束原因，例如 `reply cut off at the output-token cap`。
- 只重跑 ACE：`python -m team_sim ace-retry --search-dir <out>/f0/p2c/v3/round-10/search --ace-approve-live --ace-no-thinking [--ace-max-jobs 2] [--package <B 包>]`。
  - 原运行不动：命令把这一轮的 ACE 状态复制到 `state/sim/retry-<时间>-<哈希>/`，只在副本上处理排队任务。
  - 沿用该轮 run.json 里的模型、地址、密钥变量名和调用预算；预算会扣掉该轮已经用掉的次数。
  - 原运行没加 `--keep-data` 的话，每轮的数据文件已被删除。这时加 `--package`，命令会按折、给法、轮次重新生成数据文件，放进副本目录，并用 ACE 分区里记录的哈希核对。
  - 密钥环境变量没设置时，命令会提示输入（输入时不显示）。
  - `--include-failed` 把失败过一次的任务也放回队列。
  - `--ace-max-calls N` 只改副本的调用预算（原运行已用掉的次数照样计入）。B-live-01 每轮预算是 4 次，原运行已经用掉 2 次，所以不加这个参数时，副本里只够跑 1 个任务。
- ACE 怎么用上游的两个角色（2026-11-24 第四次修改后）：
  - Reflector 的模板是“诊断一个模型的推理过程和答案”。我们把搜索里生成策略的模型当成这个模型：它的假设填作推理过程，它改的代码填作答案，实测结果（状态、提升/持平/退步、RMSE 等数值、诊断标记）填作环境反馈。之前这里填的是格式要求和“没有答案”，Qwen 就去诊断“模型拒绝作答”，写出的经验全在讲提示词。
  - Curator 的模板要求 content 是一句文字，前面还附着现有经验库，所以我们直接请它按经验库的格式写：“症状假设：…；建议：…；范围：…”（每段 100 字以内），并说明只写关于训练本任务模型的建议。
  - 读回复：两个回复都用上游的提取函数读（整段 JSON、Markdown 代码块、前面带一句话都行）。Reflector 的回复只是给 Curator 的参考，不要求是 JSON；Curator 的回复里必须有 operations 列表，非 ADD 的操作忽略，最多取 5 条。
  - 读每条经验：JSON 对象（中英文键都行）、经验库那种带标签的文字（按“症状假设/建议/范围”等标签切分，部分内部的“；”保留）、没有标签的一句建议（当作“建议”）都接受。一条都读不出建议时任务失败，报错里带模型写的前 200 字。
  - 证据由系统附上：一条经验的证据就是产生它的那个任务的实验记录，模型写的 ID 一律忽略，所以编造的 ID 进不了经验库。内容写明属于别的任务（如“待验证｜A｜”）时整批拒绝。每段文字超过 250 字时截断。
  - 每次调用的完整提示词和回复都存在该分区的 `replies/` 目录下，`ace-retry` 的输出会给出路径。
- 经验库怎么进提示词（`--ace-max-shown`，默认 3）：每轮搜索的提示词里，先放该给法适用的全部预置经验（按 ID 顺序），再放最多 3 条 ACE 写出的经验，按写入时间从新到旧，总长不超过 2400 字（搜索提示词对经验库的上限是 2500 字）。以前是所有经验按 ID 排、截到 1900 字；ACE 经验的 ID 是哈希值，排在预置经验 `R-00-BASE-…` 前面，一轮轮带下来以后会把预置经验挤掉，留下哪几条也只由哈希值决定。这个规则只在 team_sim 里打开（配置 `ace.render`），A/B/C 正式流程的渲染不变。
- 查看一次运行的搜索和 ACE：`python scripts/sim_status.py <out>`（只读）。每次搜索列出模型调用和 token、完成迭代、最佳分；ACE 的任务、失败原因、token；从上一轮带入了几条经验、这一轮搜索提示词里实际出现了几条 ACE 经验（按搜索记录里的经验库版本号查）；以及 ACE 新写的每条经验全文。
- 预置经验在 `team_sim/seeds.json`（7 条，条件写成"回归、剩余寿命"等通用条件；FEATURES-03 和 NOCOND-07 只限 B，NOCAP-06 只限给法 1。2026-11-27 给 CAP-01 补了 C 的反例并收窄范围，给 EARLYSTOP-02 补了 C 的证据，新增 NOCOND-07「轴承任务不要按工况分支」），证据脚本在 `team_sim/evidence/`。

## 7. 已知限制

- 只支持 B（回归、按轴承划分）；C 需要另写单元划分。
- 平台每轮要求的"至少一次大模型调用"在模拟器里只记理由，不真的调用。
- 搜索打开后每轮可能很慢且花额度；建议先在本地用 `scripts/mock_llm_server.py` 跑通。
- 结果只来自 12 根公开轴承的轮换，折数少、噪声大，用来比较策略的相对好坏，不代表平台分数。
