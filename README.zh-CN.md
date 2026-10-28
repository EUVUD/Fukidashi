# Fukidashi

[English](README.md) | 简体中文

多 agent 漫画翻译编辑器（日 → 简体中文）。在 [OpenMantra](https://github.com/mantra-inc/open-mantra-dataset) 数据集上，用标准的 MT evaluation framework 和单次调用的 baseline 做对比。

> **现状（2026-10-28）：** M0、M1 和 M2 已完成。vertical slice 已经跑通：导入 OpenMantra，对一页做一次模型调用完成翻译，在浏览器里逐句对照图上的 box 检查译文。M2 新增了用 chrF / BLEU 给各个 config 打分的 eval harness；第一版 Translator Agent（即 `translator` config），它跑在调用工具的 agent loop 上，底下是带计量的 client；还有可选的 tracing，trace 发到本机的 Arize Phoenix（[记录](docs/m2.md)）。M3 正在进行：两部 dev 作品的 gold glossary、Reviewer Agent、运行 `full_no_kb` config 的 Orchestrator、Knowledge Base 的存储，以及两个 agent 的 KB tools 已经有了；接下来是 Curator、写入 KB 和 `full` config。

## 它做什么

- **带上下文翻译一页。** 模型同时看到整页图片和每个气泡的日文原文、box，为每个 box 返回一句简体中文。
- **在原位检查。** 编辑器在页面上画出每个 box，并把原文、译文和参考译文并排列出。
- **第一版 Translator Agent（M2）。** `translator` config 给模型同样的一页，外加一个读取前几页的工具；答案不合格时修复一次，仍不行就回退到 `baseline`，并且会明确记录下来。
- **翻译、审校、修改（M3，进行中）。** 在 `full_no_kb` config 里，由 Orchestrator（普通代码）先运行 Translator，再运行 Reviewer Agent。Reviewer 在 independent context 里工作：它看得到页面、译文和 Translator 标为不确定的区域，看不到 Translator 的推理，审校意见用简体中文写。被退回的页面由 Translator 修改，最多两次。
- **计划中（M3）：** `full` config，让两个 agent 用上每本书的 Knowledge Base：Translator 查询并提议 Glossary 条目和 Character Profiles，Reviewer 只查询，Curator Agent 是唯一的写者。KB 的存储、写入规则和两个 agent 的 tools 都已经有了，但还没有 config 用上它们。

## 怎么评测

Fukidashi 按产品评测，不做研究型对照实验（细节见 [OUTLINE.md 第 4 节](OUTLINE.md#4-evaluation)）：

- **主对比：** full system vs `baseline`（每页一次调用）。**KB ablation：** full system 开 / 关 Knowledge Base。
- **Metric：** chrF / BLEU（sacreBLEU，附 signature）、COMET / xCOMET、GEMBA-MQM（LLM-as-judge，先在约 30 个人工标注的 segment 上检查一致性）、对照 gold glossary 的 term accuracy、每页成本和 latency。
- **严谨性：** 按作品划分（2 部 dev、3 部 test），冻结前写好 eval protocol，test split 只跑一次，按页 paired bootstrap 95% CI。

## 工程上的几点

- **只有一个带计量的 client。** 所有模型调用都经过 `LLMClient`：它限制每页的 token budget，并把每次调用记进费用账本。provider 没返回的 usage 记为 unknown，不记成 0。
- **参考译文不会泄漏进 prompt。** prompt 里只有日文和 box；和它们存在一起的参考译文不会进 prompt，有测试检查。
- **Run 只追加，不覆盖。** Run 先以 `running` 插入，再标成 `succeeded` 或 `failed`；失败的 Run 保留原始响应，页面继续显示上一次成功的译文。
- **Agent loop 不绕过计量。** 工具调用跑在 LangChain 的 `create_agent` 上，但它的模型是包在 `LLMClient` 外面的一层薄适配，所以每次 agent 调用都和其他调用一样计量，DeepSeek 的推理内容也按它 API 的要求传回。保护措施：步数上限、token budget、重复调用检测，被截断的回复不会被当成答案。
- **Trace 留在本机。** 可选的 tracing 通过 OpenTelemetry 把每一页的 trace 发到本机的 Arize Phoenix。每个 span 里的页面图片和 API key 都会被去掉；后端只是一个 exporter，以后可以替换。
- **Provider 可替换。** 任何支持图片输入、tool calling 和 JSON output 的 OpenAI-compatible 模型都可以通过 `.env` 接入；目前用 DeepSeek `deepseek-flash`（约 $0.0006 / 页）。

设计图（Domain Model、顺序图、Object Model）：[docs/design.md](docs/design.md)（英文）。

## 运行

需要 Python 3.12（用 [uv](https://docs.astral.sh/uv/) 管理）和 Node.js。

```bash
uv sync
git clone --depth 1 https://github.com/mantra-inc/open-mantra-dataset.git data/open-mantra-dataset
cp .env.example .env   # 填 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL

uv run python scripts/import_openmantra.py      # 214 页导入 runs/fukidashi.db，可重复运行
uv run uvicorn fukidashi.main:app --port 8000
npm install --prefix frontend                   # 第一次
npm run dev --prefix frontend                   # http://localhost:5173/tojime_no_siora/002
```

选一页点 Translate：后端对这一页跑一次 `baseline`（一次调用，不用 tools 和 KB），把 Run 和 zh-Hans 译文存进 SQLite。Translate 会真实调用 `.env` 里配置的模型。DB 路径可以用环境变量 `FUKIDASHI_DB` 改。

```bash
uv run python eval/run.py --config baseline --book tojime_no_siora --pages 2,3,4     # 真实调用：每页一次
uv run python eval/run.py --config translator --book tojime_no_siora --pages 2,3,4   # Translator Agent：每页几次调用
uv run python eval/run.py --config full_no_kb --book tojime_no_siora --pages 2,3,4   # Translator + Reviewer：调用再多几次
```

Eval harness（评测执行器）走和编辑器相同的代码路径翻译这几页（每页在 SQLite 里留下一个 Run），用 OpenCC（`t2s`）把 zh-Hant 参考译文转成 zh-Hans，然后打印 chrF 和 BLEU 及其 sacreBLEU 签名，以及每页成本和延迟；跑 `translator` 和 `full_no_kb` 时还会统计回退到 `baseline` 的页数，跑 `full_no_kb` 时还会统计 Reviewer 退回了几次、循环为什么停下。运行清单（Manifest）、逐 Segment 的输出和分数写入 `runs/eval/<id>/`。有页面失败或 harness 无法启动时，退出码非零。

可选的 tracing：每翻译一页就生成一条 trace（包含每次模型调用、每个 agent 步骤和每次工具调用），发到你本机运行的 [Arize Phoenix](https://github.com/Arize-ai/phoenix)。在另一个终端启动 Phoenix，设置 `FUKIDASHI_TRACING=phoenix`（在 shell 或 `.env` 里都可以），然后打开 http://localhost:6006：

```bash
PHOENIX_HOST=127.0.0.1 PHOENIX_TELEMETRY_ENABLED=false uvx --from arize-phoenix phoenix serve
FUKIDASHI_TRACING=phoenix uv run python eval/run.py --config translator --book tojime_no_siora --pages 3
```

Tracing 默认关闭，出错也不会让翻译失败。每个 span 里的页面图片和 API key 都会被去掉，endpoint 必须在本机。即使你的 shell 设置了 HTTP 代理，span 也会直接发到这个 endpoint。eval 输出会记录每一页的 trace ID。

```bash
uv run pytest                                       # 不联网，不发真实 LLM 请求
uv run python scripts/m0_first_call.py --dry-run    # 打印 M0 的请求内容，不发送
uv run python eval/glossary.py check                # 对照数据集检查 gold glossary；不调用模型
```

## Roadmap

| Milestone | 目标日期 | 状态 |
|---|---|---|
| M0 决定与第一次调用 | 10/18 | 完成（[记录](docs/m0.md)） |
| M1 Vertical slice：页面 → `baseline` → 浏览器 | 10/25 | 完成（[记录](docs/m1.md)） |
| M2 Agent layer 与 eval harness | 11/1 | 完成（[记录](docs/m2.md)） |
| M3 Knowledge Base 与 Reviewer | 11/8 | 进行中 |
| M4–M5 Dev 评测、冻结 eval protocol | 11/22 | 计划中 |
| M6–M8 Test 运行、编辑器产品化、分析 | 1/3 | 计划中 |
| M9 报告与发布 | 1/17 | 计划中 |

完整计划：[OUTLINE.md](OUTLINE.md)（设计、milestones、决定、风险）和 [SCOPE.md](SCOPE.md)（范围和 cut order），均为英文。

## 这个仓库是怎么开发的

大部分代码由 AI coding agent（Claude Code）按 architect → executor → reviewer 的流程编写（[docs/workflow.md](docs/workflow.md)）：每个 task 有书面的验收标准，executor session 在自己的分支上实现，一个全新的 reviewer session 对照验收标准检查 diff，每个 PR 由我审阅并合并。设计决定（范围、评测方案、模型、预算）由我做出，记录在 [OUTLINE.md 第 8 节](OUTLINE.md#8-decisions)。

## 数据与署名

OpenMantra 来自 Hinami et al., *Towards Fully Automated Manga Translation*（AAAI 2021），许可为 **CC BY-NC 4.0**。图片不在本仓库里，请按上面的命令另外 clone。文档里为了举例引用了其中几句短文本；`eval/gold_glossary/` 里的 gold glossary 含有取自其日文原文和中文参考译文的短术语（人名和反复出现的词）。数据笔记：[docs/openmantra.md](docs/openmantra.md)（英文）。

## 许可

本仓库的代码以 [MIT License](LICENSE) 发布。该许可不覆盖 OpenMantra：数据集本身不在本仓库里分发，文档中引用的原文和 `eval/gold_glossary/` 里的术语仍适用 CC BY-NC 4.0，使用数据集须遵守该许可。
