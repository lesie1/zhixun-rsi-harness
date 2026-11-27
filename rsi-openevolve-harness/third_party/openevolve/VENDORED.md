# 随项目携带的 OpenEvolve

- 来源：[codelion/openevolve](https://github.com/codelion/openevolve)。
- 版本：0.4.1；交接固定提交：3e72b5b（短 commit）。版本号可在 openevolve/_version.py 核对。
- 许可证：Apache-2.0，原文保存在 [LICENSE](LICENSE)，原有版权与许可说明保留。
- 本项目修改的代码文件仅为 [openevolve/process_parallel.py](openevolve/process_parallel.py)：新增 OPENEVOLVE_PROMPT_EXTRAS 与 OPENEVOLVE_ITERATION_HOOK 两个可选扩展点。
- 修改内容见 [补丁](../../docs/openevolve_extension_points.patch)，调用契约和升级方法见 [OPENEVOLVE_PATCH.md](../../docs/OPENEVOLVE_PATCH.md)。process_parallel.py 内也有 Harness extension points 的修改说明。

这是项目随附的源码副本，不应被未打补丁的 PyPI 安装替代。scripts/run_evolve.py 会把此目录加入导入路径。本次收尾仅新增本文档，没有修改已有 OpenEvolve 代码。
