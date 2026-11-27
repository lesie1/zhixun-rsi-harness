# 接入数据划分后的运行交接

本次已在 dino 接入 team_split。运行时使用 configs/run_A.split-v1.json、run_B.split-v1.json、run_C.split-v1.json；原配置和已有代码未修改。真实模型 API、密钥和首轮预算由运行同学填写，本次未调用真实模型。

## 1. 本次生成的文件

项目根目录：/home/dino/Documents/rsi-openevolve-harness。以下路径均相对项目根目录。

|任务|运行配置|固定划分清单|方法|挑选 / 验收行数|
|---|---|---|---|---|
|A|configs/run_A.split-v1.json|splits/A-v1.json|class_time|623 / 621|
|B|configs/run_B.split-v1.json|splits/B-v1.json|no_split|1679 / 1679（同一批）|
|C|configs/run_C.split-v1.json|splits/C-v1.json|group_stratified|11817 / 11232|

每个清单旁有同名 .report.json，保存划分参数和质量报告。A 在每类内部按文件行顺序前后划分，六类两边都有，验收侧甲苯仅 2 个。C 按 source_subset + unit 整台发动机划分，106 台分为挑选 55 台、验收 51 台。B 只有 3 根轴承，保留 no_split 警告；不能把 B 的结果当作独立留出集的泛化验证。

新配置的 split_manifest 已指向清单；stage_id 分别为 ace-live-A/B/C-split-v1，work_dir 分别为 runs/live/A-split-v1、B-split-v1、C-split-v1。ACE scope 已重新计算，playbook_path 指向新的空经验库，旧经验库保留。清单一旦开始使用就不要改；换划分要生成 v2、重新 wire。

## 2. 两处 API 接口都要填

OpenEvolve 用模型生成续训策略代码；ACE 的 Reflector（反思）和 Curator（整理）用模型把实验记录变成候选经验。两者均使用 OpenAI 兼容的 chat completions 接口，可以共用同一服务、模型和密钥，但配置独立，不会自动继承。Reflector 和 Curator 共用 ace 下的配置，无需再填两套。训练、划分、指标计算和验收门本身不调用模型 API。

|用途|填写位置|字段|
|---|---|---|
|OpenEvolve 地址|configs/openevolve.split-v1.yaml|llm.api_base，例如 http://模型服务器:端口/v1|
|OpenEvolve 模型|同上|llm.models[0].name，必须是服务实际提供的模型名|
|OpenEvolve 密钥|启动进程的环境变量|OPENEVOLVE_API_KEY；新 YAML 的 llm.api_key 已写为 ${OPENEVOLVE_API_KEY}|
|ACE 地址|当前任务 configs/run_*.split-v1.json|ace.api_base|
|ACE 模型|同上|ace.model|
|ACE 密钥变量名|同上|ace.api_key_env，例如 ACE_API_KEY；这里只填变量名|
|ACE 密钥值|启动 ACE 进程的环境变量|ACE_API_KEY|
|启用 ACE 在线处理|任务 JSON|ace.mode 改为 live，ace.live_calls_approved 改为 true，并填写 ace.max_total_calls|

新 YAML 中的地址和模型名是待填写项；60 次迭代、3000 输出 tokens 等是原配置数值，不代表本次已批准的首轮预算。新 YAML 的 evaluator.parallel_evaluations 设为 1，运行同学可按资源调整。原 configs/openevolve.yaml 未改。

ACE 配置当前仍为 disabled / false / 0；请只修改上述字段，保留 module、storage_root、scope、record_origin 和 playbook_path。直接把密钥填进 api_key_env 是错误的。api_base 填到 /v1，通常不要再拼 /chat/completions。若服务无需鉴权，也要给对应密钥环境变量设置服务允许的非空占位值。

代码位置：scripts/run_evolve.py 读取 OpenEvolve YAML，接受 --api-base、--model、--api-key 覆盖；third_party/openevolve/openevolve/llm/openai.py 发出搜索请求。team_ace/backend.py 的 Transport 发出 ACE 请求。

## 3. 服务器环境与首轮预算接口

|要决定的内容|填写位置|说明|
|---|---|---|
|Python 解释器|下方 shell 变量 PYTHON|所有启动命令都用同一个环境|
|数据和起点|任务 JSON 的 train_path、validation_path、parent_checkpoint|当前已指向 dino 的正式 A/B/C 包|
|搜索迭代次数|--iterations 或 YAML 的 max_iterations|命令行覆盖 YAML；由运行同学填写|
|搜索单次输出上限|YAML 的 llm.max_tokens|不包含输入 tokens；不是总 token 预算|
|搜索请求超时 / 重试|YAML 的 llm.timeout、llm.retries、llm.retry_delay|重试可能增加请求与消耗|
|训练并行数|--workers 或 YAML 的 evaluator.parallel_evaluations|本机约 2 GB 内存，新模板先设 1|
|每个候选的续训上限|任务 JSON 的 limits.max_epochs、train_timeout_s、cpu_threads|原值为 5 轮、20 秒、1 线程；按实际环境核定|
|ACE 总调用次数|任务 JSON 的 ace.max_total_calls|live 模式有效整数 1–20，按当前任务 / scope / mode 累计，不因重启重置；失败尝试也占次数|
|每批实验记录数|任务 JSON 的 ace.batch_size|当前为 1；通常一个成功处理批次有反思和整理两次 API 调用|
|一次 ACE 处理任务数 / 存活时长|team_ace serve 的 --max-jobs、--seconds|分别允许 1–3、1–120；不是无限常驻服务|

OpenEvolve 当前没有整个运行的硬性金额或总 token 上限；迭代数不能直接当作 API 次数或费用。ACE 的调用次数预算与 OpenEvolve 分开计算。需要总费用硬限制时，由运行同学在所用服务端设置额度。

当前解释器已验证可用：

```bash
cd /home/dino/Documents/rsi-openevolve-harness
PYTHON=/tmp/claude-1000/-home-dino-Documents/86632056-68e7-4a98-8935-c84aa3c13046/scratchpad/oeenv/bin/python
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

该解释器在临时目录；如果运行时已不存在，由运行同学在持久目录创建环境并按 requirements.txt 安装，再修改 PYTHON。ACE 还依赖 /home/dino/Documents/environment/vendor/ace 的上游源码，路径目前固定在 team_ace/backend.py 的 ACE_ROOT。若迁移到其他服务器，仅复制项目和改 PYTHON 不足以完成部署，还要准备该目录和数据路径。

ACE 现有固定限制：每次请求 20 秒、无 SDK 重试、输出最多 1024 tokens、输入最多 24000 字符；每个 worker 最多约 35 秒，另有内存限制。慢速的 27B 服务可能触发超时。它们目前不是 JSON 可调字段，本次未改代码；上线时先用小预算核实响应时长及 usage（token 用量）返回。

## 4. 运行顺序

先在新 YAML 和当前任务 JSON 填好接口、启用项、预算，再运行。以下以 A 为例；B/C 换任务配置。各终端都需要设置工作目录、PYTHON，以及各自的密钥环境变量。

先校验已接好的清单（不调用 API）：

```bash
"$PYTHON" -B -m team_split verify --config configs/run_A.split-v1.json --manifest splits/A-v1.json
```

终端一：让运行同学输入密钥和批准的搜索迭代次数，然后启动搜索：

```bash
read -r -s -p "OpenEvolve API key: " OPENEVOLVE_API_KEY
export OPENEVOLVE_API_KEY
printf "\n"
read -r -p "本轮搜索迭代次数: " ITERATIONS
"$PYTHON" -B scripts/run_evolve.py --run-config configs/run_A.split-v1.json --oe-config configs/openevolve.split-v1.yaml --iterations "$ITERATIONS" --workers 1
```

终端二：使用相同的任务配置，在搜索期间运行 ACE，使新经验能被后续迭代读取：

```bash
read -r -s -p "ACE API key: " ACE_API_KEY
export ACE_API_KEY
printf "\n"
"$PYTHON" -B -m team_ace serve --config configs/run_A.split-v1.json --max-jobs 3 --seconds 120
"$PYTHON" -B -m team_ace status --config configs/run_A.split-v1.json
```

这里 3 个任务 / 120 秒是单次 worker 管理命令的执行窗口；真正允许的 API 总次数由运行同学填写的 ace.max_total_calls 决定。serve 到时间或任务数上限会退出；搜索仍在进行、预算尚有余量时可再次执行。OpenEvolve 不会自动启动它。只在搜索结束后才处理 ACE，经验不会影响已经结束的搜索。

搜索结束后运行验收门：

```bash
"$PYTHON" -B scripts/run_gate.py --run-config configs/run_A.split-v1.json
```

查看 runs/live/A-split-v1/evolve_summary.json、records/iterations.jsonl、gate/decision.json；ACE 使用 JSON 中 playbook_path 指向的文件和同目录 memory.sqlite3。验收门可能保留起点，此时以 decision.json 的 selected_checkpoint 为准，不要假定 selected/checkpoint.npz 一定存在。

先分开运行 A/B/C，核实资源、SEARCH/REPLACE 成功率、ACE usage 和更新后经验是否进入后续提示，再扩大预算。

## 5. 接入核验与边界

本次通过 team_split make / wire / verify 接入，未修改划分策略或 harness 逻辑。完整测试、原文件摘要、已有经验库是否保持不变及新增空库状态见 checks/split-wiring-20261115.json；对应日志放在 checks/split-wiring-20261115/。

本次只完成划分接入和运行参数交接，未开始真实 Qwen 搜索。ACE 仍按数据 / 检查点 / 划分的 scope 隔离；新数据或新划分下自动迁移旧经验尚未实现，当前功能不能作为跨数据集经验迁移已验证的证据。
