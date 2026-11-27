# dino 部署说明

此包来自 captain 的 FINAL，经 37 项测试全部通过后生成。保留 FINAL 原文件；本包仅替换 A/B/C 配置、ACE 数据根目录与上游源码根目录的机器路径，并补上 python-dotenv、tiktoken 依赖声明。训练、评估、评分与提示核心文件保持原样。

- 项目：/home/dino/Documents/rsi-openevolve-harness
- ACE 源码：/home/dino/Documents/environment/vendor/ace
- 数据入口：/home/dino/Documents/cpu-baselines-20261110-v1/{A,B,C}（链接到已有官方数据，数据不随包分发）
- 正式模式仍为 disabled，模型配置未填写，未调用真实 Qwen。
- state/、runs/ 不在包中，部署后初始化新的正式空经验库。
- 原 checks/、ACE_MERGE_RESULT.txt 等是 captain 的历史证据，其中旧绝对路径不作为 dino 的运行配置。
- 新任务尚需扩展 team_ace.store.TASKS；当前只支持 A/B/C。
- skill 是带来源证据的候选文字经验，不是可执行技能包，也不是已验证有效的方法。
