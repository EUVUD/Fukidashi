# IDEAS

> checkpoint 之外冒出来的想法先记在这里，当下不做。规则：改范围只在 checkpoint（10/25、12/13），见 [SCOPE.md](SCOPE.md) 第 1 条和 [OUTLINE.md](OUTLINE.md) 第 9 节；其他时候的想法留到下一个 checkpoint 再决定做不做。
> 每条写日期、想法、为什么、什么时候回头看。做了就划掉并注明在哪个 milestone 或 PR；决定不做就写原因。

| 日期 | 想法 | 为什么 | 什么时候回头看 | 状态 |
|---|---|---|---|---|
| 2026-10-24 | 给 `pipeline/baseline.py` 一个不依赖 `openai` 的异常类型：`LLMClient` 抛自己的 error，pipeline 不再 `import openai` | 现在 pipeline 为了捕获 provider 错误直接 import SDK，SDK 漏进了 pipeline 层 | M2，写 tool loop、要统一处理 provider 错误时 | open |
| 2026-10-24 | 把 `Page` / `Segment` 从 `datasets/openmantra.py` 挪到和数据集无关的位置 | pipeline 和 API 都依赖它；M7 的 end-to-end path（上传页面，不是 OpenMantra）需要一个独立的类型 | M7，做 detection + OCR 之前 | open |
| 2026-10-24 | `runs.status = 'running'` 的行在服务中途被杀后会一直停着；同一页连点 Translate 没有服务端保护 | 属于 OUTLINE 第 3.6 节 Reliability（page 是 retry unit） | M2，做 reliability 时一起处理 | open |
| 2026-10-24 | 补一个缺 `.env` 时返回 503 的测试 | 这条路径现在只有代码，没有测试，也没手动跑过 | 下一次动 `api/` 或 `LLMClient` 时顺手 | open |
| 2026-10-24 | CI：GitHub Actions 跑 `pytest`、`tsc`、`oxlint` | 现在测试和 lint 全靠手动跑 | M3 之后，或 M9 仓库公开之前 | open |
| 2026-10-24 | 查清 `.claude/launch.json` 在 Claude app 的 preview 里为什么起不来 | 进程在跑但一直不监听端口，`uv run` 和直接调 `.venv/bin/python` 都一样；怀疑和 macOS 对 `~/Documents` 的权限提示有关，没有确认 | 下次需要在 app 里预览前端时 | open |
