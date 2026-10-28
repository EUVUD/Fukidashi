# Fukidashi — Project Outline

> v0.15 · 2026-10-28
> **v0.2 (2026-10-22):** evaluation changed from a research-style controlled experiment to a standard evaluation framework plus a baseline comparison (plan B); see §4.
> **v0.3 (2026-10-22):** M0 decisions recorded in §8: name Fukidashi, ja → zh-Hans, an OpenAI-compatible provider, a $100 budget cap.
> **v0.4 (2026-10-24):** M1 added the term Run (§0).
> **v0.5 (2026-10-24):** §0 added Translation and Reference translation (used by `docs/design.md`).
> **v0.6 (2026-10-24):** §6 updated to the actual M1 layout: Python package `fukidashi`, plus `runs.py`, `store.py`, `pipeline/`.
> **v0.7 (2026-10-24):** §7 marks M1 done.
> **v0.8 (2026-10-26):** docs translated to English; the README also has a Simplified Chinese version (`README.zh-CN.md`).
> **v0.9 (2026-10-26):** §8 decides the OpenCC config (`t2s`); §6 moves the eval library into `backend/fukidashi/evaluation/`, leaving the entry point and eval data in `eval/`.
> **v0.10 (2026-10-27):** §8 records that the provider SDK's automatic retries are off; §3.5 and §3.6 note that a failed request is metered as unknown and is not resent, as implemented in M2.
> **v0.11 (2026-10-27):** §8 decides the agent loop: LangChain v1 `create_agent`, with `LLMClient` as its model through an adapter; §5 adds it to the tech stack.
> **v0.12 (2026-10-27):** after M2's tool-calling loop: §0 adds Agent loop, Step and Stop reason; §3.4 records the guardrails as built and the smoke findings; §9 records that `deepseek-flash` can call tools.
> **v0.13 (2026-10-27):** after the Translator Agent: §0 adds `translator`, Repair and Fallback; §3.2, §3.4 and §4.2 record the Translator Agent as built (`flag_uncertain` moves to M3); §8 decides the tracing tool (Arize Phoenix, local).
> **v0.14 (2026-10-28):** M2 done (§7, `docs/m2.md`); §3.5 records tracing as built.
> **v0.15 (2026-10-28):** §3.5: spans reach the local endpoint without any proxy, and a page that fails before its Run exists keeps its trace ID.
> Convention: code, docs, commit messages and reports use the canonical terms in §0.

---

## 0. Terminology

Use the English term in the left column everywhere, in code and in docs. The Chinese column is a glossary for Chinese readers.

| Canonical term | Chinese | Notes |
|---|---|---|
| Vertical slice | 竖切 | Get data → model → browser working end to end before adding any infrastructure |
| Translator Agent | 翻译 agent | The only agent that calls tools on its own |
| Reviewer Agent | 独立审校 agent | Independent context, read-only, sends a page back at most 2 times |
| Curator Agent | 知识管理 agent | The only agent that writes to the Knowledge Base |
| Orchestrator | 编排层 | Plain code, not an LLM; permissions are enforced here |
| Knowledge Base (KB) | 故事知识库 | Holds the Glossary and Character Profiles |
| Glossary / Term entry | 术语表 / 词条 | |
| Character Profile | 角色档案 | |
| Proposal Queue | 提议队列 | The Translator can only propose; the Curator merges |
| Human Review Queue | 人工待确认 | Names and conflicting entries go here |
| Single-writer | 单一写者 | Concurrency control for shared state |
| Provenance / History | 来源 / 修改历史 | Every KB entry carries its evidence pages |
| Self-Refine | 自我修改 | Not a config since v0.2; only mentioned in Known Limitations |
| Oracle boxes / Oracle path | 标注框 / oracle 路径 | Use OpenMantra's bounding boxes and source text directly, skipping detection and OCR |
| Text Detection | 文本检测 | |
| OCR | 文字识别 | manga-ocr |
| Inpainting | 擦字 | OpenCV; LaMa optional |
| Typesetting | 排版 / 嵌字 | Written in-house |
| Overflow rate | 嵌字溢出率 | Product metric |
| Token budget (per page) | 每页 token 预算 | |
| Step limit | 步数上限 | `max_steps` model calls per agent run, default 6 |
| Agent loop | agent 循环 | `run_agent` in `agents/`: LangChain's `create_agent` with our metered model and the §3.4 guardrails (§8) |
| Step | 步 | One model call in an agent run, with the tool calls it asked for; a failed call is a step too |
| Stop reason | 停止原因 | Why an agent run ended: `final`, `step_limit`, `incomplete`, `budget` or `provider_error` |
| Metering | 计量 | Written in-house; unknown usage is counted separately, never recorded as 0 |
| Trace | 执行轨迹 | Recorded by the tracing tool since v0.2, no longer custom JSONL |
| Tracing tool | tracing 工具 | Arize Phoenix, run locally (§8) |
| Manifest | 运行清单 | Data hash, model ID, config, seed, git commit |
| Evaluation framework | 评测框架 | The standard metrics and tools listed in §4.4 |
| Eval harness | 评测执行器 | `eval/run.py`: runs configs, scores them, writes the manifest |
| Eval protocol | 评测规程 | One-page `eval/PROTOCOL.md`, written before the freeze; replaces the earlier `PREREG.md` |
| Baseline (`baseline`) | 基线 | Formerly C0: one call per page, no tools, no KB |
| Translator (`translator`) | 翻译 agent 配置 | M2's smoke config: the Translator Agent alone, no KB, no Reviewer; not a reported config (§4.2) |
| Repair | 修复调用 | One extra call that sends the model its unparseable answer's error and asks for valid JSON (§3.4) |
| Fallback | 回退到 baseline | When an agent run ends without a usable answer, the page is translated by `baseline` instead; the Run records it and the eval counts it (§3.4) |
| Full system (`full`) | 完整系统 | Translator + Reviewer + Curator / KB |
| KB ablation (`full_no_kb`) | KB 消融 | Identical to `full` except the KB is off |
| Primary / Secondary metric | 主指标 / 次要指标 | Primary metrics are written into the eval protocol before the freeze |
| Segment | 评测单元 | One OpenMantra text region (one box): its source text and translation |
| zh-Hans / zh-Hant | 简体中文 / 繁体中文 | The system outputs zh-Hans; OpenMantra's reference translations are zh-Hant |
| OpenCC | 繁简转换工具 | Converts the zh-Hant references to zh-Hans before evaluation |
| Dev split / Test split | 验证集 / 测试集 | Split by book: 2 dev books, 3 test books |
| Gold glossary | 标准术语表 | Built from the reference translation before looking at any model output |
| Term accuracy / Term consistency | 术语准确率 / 一致性 | Term accuracy is the KB ablation's primary metric |
| sacreBLEU signature | sacreBLEU 签名 | Records the tokenizer, smoothing and other settings; included in the report |
| Neural metric | 神经网络评测指标 | COMET / xCOMET in this project |
| COMET / xCOMET | — | Unbabel's neural metrics; xCOMET also marks error spans |
| LLM-as-judge | 用 LLM 当评委 | |
| Judge model | 评委模型 | Different from the translation model, to reduce self-preference bias |
| MQM (Multidimensional Quality Metrics) | 多维质量指标 | A framework for annotating translation errors by error category and severity |
| GEMBA-MQM | — | MQM annotation done by an LLM-as-judge |
| Error category / Severity | 错误类别 / 严重度 | The two MQM dimensions; severity is minor / major / critical |
| MQM score / Error breakdown | MQM 分数 / 错误分布 | Replaces the earlier custom failure taxonomy |
| Human agreement check | 人工一致性检查 | About 30 human-annotated segments that show whether GEMBA-MQM can be trusted |
| Latency (p50 / p95) | 延迟 | End-to-end wall-clock time per page |
| Freeze (`exp-frozen`) | 冻结 | After this git tag, the eval code path does not change |
| Paired bootstrap | 配对 bootstrap | Paired by page; reports a 95% CI |
| Confidence Interval (CI) | 置信区间 | |
| Adapter | 适配器 | Detection, OCR and inpainting are swappable |
| Checkpoint | 检查点 | The only points where a change of direction is considered |
| Translation | 译文 | One Run's zh-Hans output for one Segment, stored in `translations`; a Segment collects one Translation per Run |
| Reference translation | 参考译文 | OpenMantra's human zh-Hant and en translations of each Segment; used for comparison and evaluation only, never in a prompt |
| Run | 一次运行 | One execution of one config on one page, stored in `runs` (config, model, usage, cost, status, timestamps); a page shows its latest succeeded Run. Unlike a Manifest, which records a whole batch of eval runs |

**Removed in v0.2:** Cost parity, Preregistration (now Eval protocol), Failure taxonomy (now MQM error breakdown), Blind evaluation, Reviewer false positive.

---

## 1. Overview

### 1.1 One-liner

A **web-based manga translation editor**. A Translator Agent looks up the Glossary and Character Profiles as it needs them, a Reviewer Agent checks the result in an independent context, and a Curator Agent maintains the Knowledge Base as the book is translated. It is evaluated on OpenMantra against a baseline with a standard evaluation framework (sacreBLEU, COMET / xCOMET, GEMBA-MQM, term accuracy).

### 1.2 Evaluation Goal

Fukidashi is a product, so it is evaluated with general, objective metrics rather than an experiment built around a research question.

- **Main comparison:** `full` vs `baseline`. How much better is the whole system than one call per page, and how much more does it cost in money and latency?
- **KB ablation:** `full` vs `full_no_kb` in the same eval harness, measuring what the KB contributes to term accuracy and term consistency.

**The result goes into the report whether it is positive or negative**, as long as the numbers and CIs are real.

### 1.3 Design Principles

1. **Vertical slice first:** a real model translation of one page in the browser by 10/25; any piece of infrastructure is added only in the week it is needed.
2. **Code orchestration, agent autonomy only where uncertain:** the overall flow (translate → review → finalize) is ordinary code; only "what to look up, and how often" is left to the agent.
3. **Least privilege:** the Reviewer is read-only and only the Curator writes to the KB. Permissions are enforced in the Orchestrator, not by prompt.
4. **Same code path:** the editor and the evaluation share one code path, and human edits flow back into the KB.
5. **Scope is written down:** `SCOPE.md` fixes the scope; when time runs out, cut from the bottom of its list.
6. **Carry over designs from PanelFlow, not code:** split by book, paired bootstrap, unknown usage never recorded as 0, a per-page token budget.
7. **Standard over custom:** metrics and tracing use common tools (sacreBLEU, COMET, GEMBA-MQM, Langfuse / Phoenix); write our own only where no common tool fits (term accuracy, metering).

---

## 2. Scope

> [SCOPE.md](SCOPE.md) is authoritative; this table is a summary.

| In scope | Out of scope |
|---|---|
| One direction: Japanese → Simplified Chinese (zh-Hans), decided in M0 | Other language pairs, Traditional Chinese output, user accounts, real-time multi-user editing |
| Oracle path (for evaluation) + end-to-end visual path (for the demo) | Training our own detection / OCR / inpainting models |
| 3 agents + code orchestration | A general agent framework or our own workflow DSL |
| SQLite KB + Proposal Queue | A vector database (RAG is not the focus) |
| Evaluation framework + `baseline` vs `full` + KB ablation, dev/test split | Evaluating the Reviewer on its own (independent review vs Self-Refine); KB gains on long books (discussed in the report only) |
| Tracing with Langfuse or Arize Phoenix | Custom trace storage and dashboards |
| Canvas editor: move, resize and add boxes; edit text, font size, horizontal / vertical text; simple undo | Full typesetting features, font management |
| SSE live trace + `Last-Event-ID` reconnect | WebSocket, a distributed task queue |
| A minimal MCP server (exposes `lookup_term` only) | Turning the whole system into an MCP service |

**Cut order (cut from the bottom when time runs out):** `SCOPE.md` §3 lists 12 product features, the first to cut at the bottom. The eval path and the vertical slice are in `SCOPE.md` §2 (Never cut) and are not part of the cut order.

---

## 3. Architecture

### 3.1 System Overview

```
┌──────────────────────────────┐        REST + SSE        ┌───────────────────────────────────────┐
│  Web Editor                  │ ◀──────────────────────▶ │  FastAPI Backend                      │
│  React + TS + react-konva    │   events: seq + type     │                                       │
│  - page canvas / boxes       │   Last-Event-ID replay   │  Orchestrator (plain code)            │
│  - translation list          │                          │   ├─ Translator Agent ── tools ──┐    │
│  - live agent trace          │                          │   ├─ Reviewer Agent (read-only)  │    │
│  - review comments / flags   │                          │   └─ Curator Agent (KB writer)   │    │
│  - proposal approve / reject │                          │                                  ▼    │
└──────────────────────────────┘                          │  Knowledge Base (SQLite)              │
                                                          │   Glossary · Character Profiles ·     │
                                                          │   Proposal Queue · Human Review Queue │
                                                          │                                       │
                                                          │  LLM Client (OpenAI-compatible)       │
                                                          │   metering · token budget · tracing   │
                                                          │                                       │
                                                          │  Vision Adapters                      │
                                                          │   detector · OCR · inpainter ·        │
                                                          │   typesetter · PNG export             │
                                                          └───────────────────────────────────────┘
                     eval/run.py ──▶ the same Orchestrator (oracle path) ──▶ manifest + scores + trace IDs
                     tracing ──▶ Arize Phoenix (local)
```

### 3.2 Agents, Tools, Permissions

| Agent | Context | Tools (proposed names, final in M2) | KB access |
|---|---|---|---|
| Translator Agent | Page image + source text + relevant KB snippets | `lookup_term`, `lookup_character`, `get_previous_context`, `flag_uncertain`, `propose_entry` | Propose only |
| Reviewer Agent | **Independent context:** sees the source, the translation and the KB, not the Translator's reasoning | `lookup_term`, `lookup_character` (read-only) | Read |
| Curator Agent | Proposal Queue + current KB | Merges or rejects proposals; sends names and conflicts to the Human Review Queue | **Sole agent writer** |
| Human (editor user) | Editor UI | Approves or rejects proposals; renaming a term updates the whole book | Write (through the Curator's path) |

**As built in M2:** the Translator Agent has one tool, `get_previous_context`: the Japanese text of up to 2 earlier pages and their latest `translator` translations, read before the run starts, never reference translations. `flag_uncertain` moves to M3, because a flag needs somewhere to be stored and someone to read it (the Reviewer in M3, the editor in M7). The KB tools arrive with the KB in M3.

**Review loop:** the Reviewer sends a page back at most twice. This is fixed; since v0.2 it is not tuned for cost parity.

### 3.3 Knowledge Base

- **Entities:**
  - `GlossaryEntry`: `id`, `source_term`, `target_term`, `category`, `status` (proposed / approved / rejected / pending_human), `provenance` (agent / human), `evidence_pages`, `history`
  - `CharacterProfile`: `id`, `name_src`, `name_tgt`, `aliases`, `notes` (speech style and so on), `evidence_pages`, `history`
  - `Proposal`: `proposal_id`, `kind`, `payload`, `proposer`, `page_id`, `status`
- **Write policy:** single-writer + Proposal Queue. Proposals carry IDs and the Curator deduplicates by ID, so a retry never writes twice.
- **Conflicts:** two proposals with different target terms for the same source term, or anything involving a name, go to the Human Review Queue instead of being merged automatically.
- **Retrieval:** relevant terms and characters are matched against the source text and truncated to a character budget; the agent looks up the rest with tools. No vector database, and the report says why: terms and names are identifiers, where exact match beats embeddings.

### 3.4 Agent Loop Guardrails

- Step limit; per-page token budget; duplicate tool call detection; tool errors are passed back to the agent as they are. **Built in M2** (`agents/loop.py`): the Step limit is `max_steps` model calls (default 6); a duplicate is the same tool with the same validated arguments; the model's mistakes (an unknown tool, arguments that fail the tool's pydantic model, a handler that raises) go back as the tool's result. Nothing in the loop retries: a provider error ends the run (§3.6).
- A reply that ended for any reason other than `stop` or `tool_calls` is never taken as the answer (stop reason `incomplete`): with thinking on, reasoning can use up `max_tokens` and leave an empty, truncated reply. Further checks (no progress, a cap on tool calls, near-duplicates, a wall-clock deadline) wait for traces that show a need (`IDEAS.md`).
- Structured output: validate against a JSON schema, retry once on a parse failure, then **fall back visibly** to `baseline` (one call per page). **Built in M2** (`agents/translator.py`): the answer must pass `parse_translations`; a bad one gets one Repair call (the history plus the parse error, JSON output, no tools); `step_limit`, `incomplete` or a failed Repair fall back to `baseline` with a budget of its own. The Run keeps config `translator`, its attempts say why, and the eval harness counts the pages that fell back. `provider_error` and `budget` never fall back (§3.6).
- The prompt keeps a stable prefix for prompt caching: `system → tool definitions → glossary → page`.
- **Two DeepSeek rules:** with tools, the `reasoning_content` of every turn must be passed back unchanged in later requests (done by `LLMClientChatModel` since M2; `deepseek-flash` sends `""` on its final turn, which goes back as it is); JSON output occasionally comes back empty, which the retry above covers. Thinking is switched in `extra_body`; effort is `reasoning_effort` (`low` / `high` / `max`). The M2 smoke runs (2026-10-27, text only) also showed that `response_format: json_object` is accepted together with tools.

### 3.5 Metering & Tracing

- **Metering is ours:** every LLM call records the model ID, input / output / cached tokens and cost. When the provider returns no usage, including a request that failed or timed out, the call is recorded as **unknown** and counted separately, never as 0. Cost per page comes from this data.
- **Tracing uses an off-the-shelf tool:** Arize Phoenix, run locally (§8), replacing custom trace JSONL. LangSmith is never used: `run_agent` refuses to start while its tracing variables are on. **Built in M2** (`tracing/`): off unless `FUKIDASHI_TRACING=phoenix`; each `translate_page` is one trace whose root span carries book, page, config, Run id, status and, for `translator`, the stop reason and fallback reason; OpenInference adds a span per provider call and the agent's node and tool spans; an exporter wrapper removes page images and the API key, and spans go straight to the local endpoint, never through a proxy; eval rows and the manifest record the trace ID, also for a page that fails before its Run exists. One trace per page; agent steps, tool calls and review rounds are spans. Eval results store the trace ID, so a score links straight to its trace.
- **Choosing:** Phoenix runs locally as one Python process, which is light on an 8 GB Mac. Self-hosting Langfuse needs more components, and Langfuse Cloud sends traces to a third party.
- The editor's live agent trace (SSE) is a **product feature** and separate from the tracing tool: the first is for users, the second for development and evaluation.

### 3.6 Reliability

- **The page is the retry unit;** task state is persisted in SQLite.
- LLM calls have no side effects and are safe to retry; the KB write, which does have side effects, happens once, at finalize, by the Curator.
- When a result is unknown (for example a timeout that may still have been billed), it is not resent automatically; the user decides. Since M2, `LLMClient` turns off the provider SDK's own retries (§8) and records a failed request as a call with unknown usage, and `baseline` fails the Run instead of resending.
- SSE events carry an increasing `seq`; after a disconnect the client reconnects with `Last-Event-ID` and the server replays what was missed.

### 3.7 Vision Pipeline

| Stage | Evaluation (oracle path) | Demo (end-to-end path) |
|---|---|---|
| Text Detection | OpenMantra bounding boxes | comic-text-detector (GPL-3.0, optional adapter only) |
| OCR | OpenMantra source text | manga-ocr (Apache-2.0, runs on CPU) |
| Translation | Orchestrator + agents | Same |
| Inpainting | — | OpenCV; LaMa optional |
| Typesetting | — | In-house; records the overflow rate |
| Export | — | PNG |

Every vision component is a swappable **adapter**.

---

## 4. Evaluation

> Since v0.2, Fukidashi is evaluated as a product: a standard evaluation framework and a baseline comparison.

### 4.1 Data

- **OpenMantra:** Japanese source text with professional English and Chinese translations and bounding boxes; **CC BY-NC 4.0**. Fields and statistics in `docs/openmantra.md`.
- **The references are zh-Hant:** before evaluation, OpenCC converts `text_zh` to zh-Hans, and the gold glossary is built from the converted references. The OpenCC config is `t2s` (§8).
- Split by book: **dev = 2 books** (for tuning prompts and tools), **test = 3 books** (run once, after the freeze).
- Metrics are computed per segment; the paired bootstrap resamples pages.
- The public repository holds no OpenMantra images, only eval scripts, the gold glossary and attribution.

### 4.2 Configurations

| Config | Contents |
|---|---|
| `baseline` | Formerly C0: one call per page, no tools, no KB |
| `full` | Full system: Translator Agent (tools) + Reviewer Agent (at most 2 rounds) + Curator Agent / KB |
| `full_no_kb` | KB ablation: identical to `full` except the KB is off |

**`translator` (M2):** the Translator Agent alone, without KB or Reviewer, run on the M2 smoke pages. It is not one of the reported configs; M3 builds `full` and `full_no_kb` on top of it.

**KB off means:** no KB snippets in the prompt; `lookup_term`, `lookup_character` and `propose_entry` disabled; the Curator does not run; the Reviewer does not see the KB either. The Translator, the Reviewer, `get_previous_context` and the token budget are the same as in `full`.

**Removed in v0.2:** C1, C2 (Self-Refine) and C3 as separate configs; `mem_window3`; cost-parity tuning; running each config twice.

### 4.3 Comparisons

| Comparison | Primary metric | Question |
|---|---|---|
| `full` vs `baseline` | COMET (xCOMET too, if it runs) | How much better is the whole system than one call per page, and what does it cost in money and latency? |
| `full` vs `full_no_kb` | Term accuracy | How much does the KB contribute? |

Every quality metric in both comparisons gets a paired bootstrap 95% CI. Primary metrics are written into the eval protocol before the freeze, so no metric is picked after seeing results.

### 4.4 Evaluation Framework

| Metric | Tool | Use | Notes |
|---|---|---|---|
| **chrF / BLEU** | sacreBLEU | Corpus-level; the report includes the **sacreBLEU signature** | With Chinese as the target, BLEU uses `tokenize=zh`; chrF needs no tokenization |
| **COMET / xCOMET** | Unbabel COMET | The main neural metric, reference-based; **required** | If it does not run on the Mac, use the Windows machine's GPU; check license and memory of the checkpoints (for example `Unbabel/wmt22-comet-da`, `Unbabel/XCOMET-XL`) in M4 |
| **GEMBA-MQM** | LLM-as-judge | Marks error spans with error category and severity (minor / major / critical) per segment, aggregated into an MQM score and error breakdown; **replaces the custom failure taxonomy** | The judge model differs from the translation model; run the human agreement check (§4.5) first |
| **Term accuracy / consistency** | Ours, against the gold glossary | Measures the KB | Primary metric of the KB ablation |
| **Cost per page** | Our metering | Cost of all of the system's own LLM calls; unknown usage counted separately | The evaluation's own cost (judge calls, COMET GPU time) is tracked separately, not as system cost |
| **Latency** | Our timer | End-to-end wall-clock time per page, reported as p50 / p95 | |
| Overflow rate | Typesetter | Product metric | Not part of the §4.3 comparisons |

### 4.5 GEMBA-MQM Human Agreement Check

- Sample about 30 segments from the dev split (covering both `baseline` and `full` outputs) and annotate error category and severity by hand, using the same MQM guideline. **The human annotation is done before looking at GEMBA-MQM's output.**
- Run GEMBA-MQM on the same segments and compare two things: the correlation of segment-level MQM scores (for example Kendall τ), and whether the major / critical errors found by hand were caught.
- The decision rule goes into the eval protocol before the freeze: with enough agreement, the MQM error breakdown is the report's main error analysis; otherwise it is reported for reference only, and the report says so.

### 4.6 Rigor & Statistics

- Every config uses the same translation model and the same inputs.
- Prompts and tools are tuned on the dev split only; each config runs **once** on the test split.
- Before the freeze, write the one-page `eval/PROTOCOL.md`: metrics (primary marked), splits, test runs once, exclusion rules, the GEMBA-MQM decision rule. Then tag `exp-frozen`.
- **Statistics:** paired bootstrap 95% CI by page, for the two §4.3 comparisons only.

### 4.7 Gold Glossary Protocol

- 20–40 entries per dev book (M3), and one glossary for each of the three test books (M5).
- Built from the reference translation, **before looking at any model output**.

### 4.8 Reproducibility

- Every `eval/run.py` run writes a manifest: data hash, model IDs (translation model and judge model), config, seed, git commit, sacreBLEU signature, COMET checkpoint.
- Raw output and trace ID are kept for every page.

### 4.9 Known Limitations (stated in the report)

- The oracle path has no OCR errors, so the results do not fully carry over to real OCR input.
- OpenMantra's books are short, so KB gains may not show; hence term accuracy, not consistency alone.
- **There is no equal-budget review comparison** (independent review vs Self-Refine). The Reviewer Agent's value rests on design rationale, not on numbers.
- `full` vs `baseline` measures the whole system; it cannot separate what the Translator's tools and the Reviewer each contribute. Only the KB is ablated.
- GEMBA-MQM is an LLM-as-judge whose agreement with humans is checked on about 30 segments only.
- Reference-based metrics rely on OpenMantra's single reference translation, while manga translation allows a lot of freedom, so scores run conservative.
- The references are converted from Traditional to Simplified Chinese with OpenCC. Characters map almost one to one, but Taiwan and mainland vocabulary differences do not convert cleanly, which lowers reference-based metrics slightly.

---

## 5. Tech Stack

| Layer | Choice |
|---|---|
| Frontend | React + TypeScript + react-konva |
| Backend | FastAPI + SQLite + SSE |
| Runtime | Python 3.12, dependencies managed with uv |
| LLM | OpenAI-compatible API through the `openai` Python SDK; provider and `LLM_MODEL` are configured in `.env` (template: `.env.example`); requires vision, tool calling and JSON output. Currently DeepSeek `deepseek-flash` |
| Agents | LangChain v1 `create_agent` runs the tool-calling loop. Its model is a small adapter over our `LLMClient`, so metering and DeepSeek's `reasoning_content` stay ours; LangChain stays inside `agents/` |
| Vision | manga-ocr, comic-text-detector (optional), OpenCV, LaMa (optional) |
| Evaluation | sacreBLEU (chrF / BLEU), COMET / xCOMET (Unbabel), GEMBA-MQM (LLM-as-judge), paired bootstrap (Python), OpenCC (reference conversion) |
| Observability | Arize Phoenix, run locally, fed through OpenTelemetry with OpenInference instrumentation, so the backend can change (tracing); metering is ours |
| Protocol demo | A minimal MCP server that exposes only `lookup_term` |

---

## 6. Repository Layout (proposal)

```
fukidashi/
├── README.md              # English README, including AI vs human authorship and how code is reviewed
├── README.zh-CN.md        # Simplified Chinese README, kept in sync with README.md
├── LICENSE                # MIT, code only; OpenMantra stays CC BY-NC 4.0
├── SCOPE.md
├── OUTLINE.md             # this file
├── IDEAS.md               # ideas that come up outside a checkpoint
├── .env.example           # LLM provider template (.env is not committed)
├── docs/                  # design diagrams, milestone records, data notes
├── scripts/               # one-off scripts, for example m0_first_call.py
├── backend/
│   ├── fukidashi/         # installable Python package (editable install via uv sync); scripts and eval import fukidashi
│   │   ├── api/           # REST + SSE endpoints (request parsing, error mapping, response shaping only)
│   │   ├── runs.py        # translates a page and records a Run; shared by the API and the eval harness (M1)
│   │   ├── store.py       # SQLite queries (M1)
│   │   ├── datasets/      # OpenMantra loader and importer
│   │   ├── pipeline/      # baseline (M1); from M2 the orchestrator takes over and baseline stays as a config
│   │   ├── orchestrator/  # translate → review → finalize
│   │   ├── agents/        # loop.py + model.py (the agent loop), translator.py (M2); reviewer.py, curator.py (M3)
│   │   ├── tools/         # lookup_term, lookup_character, ...
│   │   ├── kb/            # Glossary, Character Profiles, Proposal Queue
│   │   ├── llm/           # client, metering, token budget
│   │   ├── vision/        # detector / ocr / inpainter / typesetter adapters
│   │   ├── evaluation/    # eval library, importable by tests: OpenCC references, metrics (sacreBLEU, later COMET, GEMBA-MQM, term accuracy), manifest, bootstrap
│   │   └── tracing/       # OpenTelemetry + OpenInference setup; the backend (Phoenix) is one exporter
│   └── tests/             # focused on KB write permissions and orchestration
├── frontend/              # React + react-konva editor
├── eval/                  # entry point and eval data; the logic lives in backend/fukidashi/evaluation/
│   ├── run.py             # eval harness entry point (CLI)
│   ├── configs/           # baseline.yaml, full.yaml, full_no_kb.yaml
│   ├── gold_glossary/
│   ├── mqm_human/         # human MQM annotation of about 30 segments
│   ├── analysis/          # MQM error breakdown tables, plots
│   └── PROTOCOL.md        # one-page eval protocol, written before the freeze
├── mcp/                   # lookup_term MCP server
├── runs/                  # Run outputs, eval outputs (runs/eval/) and the usage.jsonl cost ledger (gitignored)
└── data/                  # OpenMantra (gitignored)
```

---

## 7. Milestones (about 38.5 h)

| Milestone | Week | Dates | Hours | Deliverables |
|---|---|---|---|---|
| **M0** Decisions & first call | 1 | 10/16–10/18 | 2.5 | 4 decisions; `SCOPE.md`; download and check OpenMantra (fields in `annotation.json`, book title spelling, page and text-region counts); first real model output (model ID and token usage recorded). **Done 2026-10-24**, see `docs/m0.md` |
| **M1** Vertical slice | 2 | 10/19–10/25 | 3.5 | M1-a: FastAPI + SQLite + OpenMantra import (oracle boxes + source text); M1-b: C0 → the frontend shows the page image, boxes and a translation list. **Checkpoint 10/25: a real translation of one page in the browser.** **Done 2026-10-24**, see `docs/m1.md` |
| **M2** Agent layer & eval harness | 3 | 10/26–11/1 | 3.5 | Tool-calling loop + guardrails; Translator Agent; metering; tracing tool (Phoenix, §8); eval harness: `eval/run.py` + manifest + sacreBLEU (chrF / BLEU, with signature); `baseline` scored on 3 pages, Translator Agent runs on the same 3 pages (smoke test, not a reported config). **Done 2026-10-28**, see `docs/m2.md`. `translator` scored chrF 10.67 on the 3 pages, inside `baseline`'s run-to-run spread (11.17–12.87), at about $0.0016 per page against $0.0009 |
| **M3** KB & Reviewer | 4 | 11/2–11/8 | 3 | KB + Curator + Proposal Queue + unit tests for KB write permissions; Reviewer Agent in the full system (at most 2 rounds, fixed); gold glossary for the dev split; `baseline` / `full` / `full_no_kb` each run on 1 page |
| **M4** Dev evaluation | 5 | 11/9–11/15 | 3 | Estimate tokens and cost on 3 pages first (judge calls included) against the budget cap; set up COMET / xCOMET (Windows GPU if the Mac cannot run it) and check checkpoint licenses and memory; run all 3 configs on the dev split. **11/15: first dev numbers** (chrF / BLEU, COMET, term accuracy, cost, latency) |
| **M5** Freeze | 6 | 11/16–11/22 | 3 | Term accuracy / consistency; GEMBA-MQM + human agreement check on about 30 segments; tune prompts and tools from the MQM error breakdown; gold glossary for the test split; one-page `eval/PROTOCOL.md`. **11/22 freeze, tag `exp-frozen`** |
| **M6** Test run | 9 | 12/7–12/13 | 1 | From `exp-frozen`, run the test split in the background, each of the 3 configs once, then score with COMET. **Checkpoint 12/13** |
| **M7** Productization | 10 | 12/14–12/20 | **12** | Editor features; SSE live trace + `Last-Event-ID` reconnect; review comments and uncertain flags highlighted; approve / reject proposals, renaming a term updates and re-typesets the whole book; end-to-end visual path; overflow rate; first demo recording. **Does not touch the eval code path.** v0.2: +3 h from the simpler evaluation |
| **M8** Analysis | 12 | 12/28–1/3 | 2.5 | Paired bootstrap 95% CIs for both comparisons; GEMBA-MQM on the test split with an MQM error breakdown (by category and severity); cost / latency summary; charts |
| **M9** Report & release | 13–14 | 1/4–1/17 | 4.5 | Report (method, data, results, cost, latency, MQM error breakdown, limitations); README; demo recording; blog posts in Chinese and English |

### 7.1 Hours: v0.1 → v0.2

| Milestone | v0.1 | v0.2 | Why |
|---|---|---|---|
| M2 | 3.5 | 3.5 | Custom trace JSONL replaced by a tracing tool; C1 is no longer a config, so the Translator Agent gets a smoke test only |
| M3 | 3.5 | 3 | Self-Refine (C2) removed |
| M4 | 3.5 | 3 | Cost-parity tuning removed and configs cut from C0–C3 plus 3 memory variants to 3 (−1.5); COMET / xCOMET moved here from M5 and made required (+1) |
| M5 | 3 | 3 | COMET moved to M4 (−1); GEMBA-MQM and the human agreement check added (+1.25); PREREG simplified to an eval protocol, reading traces by hand replaced by the MQM breakdown (−0.25) |
| M6 | 1.5 | 1 | Each of the 3 configs runs once on the test split (previously twice) |
| M8 | 4 | 2.5 | Custom failure taxonomy replaced by the GEMBA-MQM breakdown; only two bootstrap comparisons left; optional blind evaluation removed |
| **Subtotal (M2–M6, M8)** | **19** | **16** | **−3** |
| M7 | 9 | 12 | Receives the 3 hours saved |
| **Total** | **38.5** | **38.5** | Unchanged |

**Why the hours go to M7:** M7 had 12 features to build in 9 hours, and "the editor eats the schedule" is a medium risk (§9). With Fukidashi framed as a product, the editor's quality is what the demo shows.

---

## 8. Decisions

| Decision | When | Choice / options | Status |
|---|---|---|---|
| Project name | M0 | **Fukidashi** | Decided (2026-10-22) |
| Translation direction | M0 | **ja → zh-Hans.** OpenMantra's references are zh-Hant and are converted to Simplified Chinese with OpenCC before evaluation; the demo is in Simplified Chinese | Decided (2026-10-22) |
| Model | M0 | **DeepSeek `deepseek-flash`** (DeepSeek-V4.1-Flash-1017), OpenAI-compatible, base URL `https://api.deepseek.com`. Supports image input and JSON output; thinking is on by default (effort `high`). Per 1M tokens: $0.15 input, $0.60 output off-peak, double at peak; `.env` logs spend at peak prices | Decided (2026-10-24); tool calling to be tested in M2 |
| API budget cap | M0 | **$100** for the whole project, judge calls included. `runs/usage.jsonl` logs every call; M4 estimates on 3 pages first | Decided (2026-10-22) |
| OpenCC config | M2 | **`t2s`**: converts characters only and leaves the reference's wording as the translator wrote it. Taiwan and mainland vocabulary differences remain and lower reference-based metrics slightly (§4.9). `tw2sp` was rejected because it rewrites the human translation's wording | Decided (2026-10-26) |
| Provider SDK retries | M2 | **Off** (`max_retries=0`). By default the `openai` SDK silently resends a request up to twice on connection errors, timeouts, 429 and 5xx; earlier attempts that may have been billed would never reach the Meter, and cost would be a partial sum. Now a 429 or 5xx fails the page and the user reruns it. An explicit retry for 429, which is rejected and not billed, can be added if 429s are common in M4 (`IDEAS.md`) | Decided (2026-10-26) |
| Agent loop | M2 | **LangChain v1 `create_agent`** as the loop skeleton, customised through its model, tools and middleware (the Step limit is its `ModelCallLimitMiddleware`); start here and change it if the project needs to. Its model is a small custom chat model over `LLMClient`, not `ChatOpenAI` / `ChatDeepSeek`: those bypass the Meter and retry by default, and DeepSeek rejects tool-call turns that do not send `reasoning_content` back. LangChain stays inside `agents/`, behind a plain result type, so replacing it touches nothing else. The alternative was a hand-written loop | Decided (2026-10-27) |
| Tracing tool | M2 | **Arize Phoenix, run locally** (`phoenix serve` or Docker): one process, no account, no cost, and traces kept as long as we like, so the report's trace links (§3.5) still work in January. Hosted free tiers keep traces 14 days (LangSmith) or 30 days (Langfuse Cloud), and self-hosting Langfuse needs several services. Traces stay on this machine by default; that is a choice, not a licence requirement. **Kept replaceable:** the code uses only OpenTelemetry and OpenInference, and the backend is one exporter in `tracing/`, so moving to Langfuse or another OTLP backend later changes that exporter, not the instrumentation (`IDEAS.md`). The Phoenix server is not a project dependency | Decided (2026-10-27) |
| COMET checkpoint | M4 | Which COMET and xCOMET models, by license and memory | Open |
| Judge model (GEMBA-MQM) | M5 | A strong model different from the translation model | Open |

---

## 9. Risks & Pivot Rules

| Risk | Mitigation |
|---|---|
| Sliding into over-engineering again (PanelFlow is the cautionary tale) | Vertical slice first; `SCOPE.md` fixes the scope; the 10/25 checkpoint only asks whether the browser shows a real translation |
| The editor eats the schedule | Product features are concentrated in M7 (12 h since v0.2); when time runs out, cut from the bottom of `SCOPE.md` |
| Books are too short for KB gains to show | Use term accuracy, not consistency alone; report honestly; long books are discussed in the report only |
| Model cost exceeds the budget | $100 cap; `runs/usage.jsonl` logs tokens and cost for every call (cost needs prices in `.env`); estimate on 3 pages first, judge calls included; tune on a subset of dev pages if costs run high |
| GEMBA-MQM disagrees with humans | Per the eval protocol's decision rule, report it for reference only and say so |
| COMET / xCOMET does not run on the Mac | Use the Windows machine's GPU; if xCOMET does not fit in memory, report COMET only |
| `deepseek-flash`'s vision is new (2026-10-17 release) and its tool calling is untested | Read translations by hand in M0 and M1; test the tool loop in M2; if it falls short, switch providers in `.env` without code changes. **Tool calling answered (2026-10-27):** three smoke runs through the agent loop with thinking on all called the tool and used its result (#16) |

**Pivot rules (decided only at checkpoints):**
- **10/25:** no real translation in the browser → drop the frontend for now, keep only the CLI path, and move all editor work to M7. The project itself does not change.
- **12/13:** if only one project has first numbers, the whole winter break goes to it.
- Any other "should we change direction" idea goes into `IDEAS.md` until the next checkpoint.

---

## 10. Development Workflow

- Each task has a goal and acceptance criteria → AI implements it → **one task, one PR** → the author reviews the diff. Details in `docs/workflow.md`.
- The README states which code is AI-written, which is the author's, and how it is reviewed.
- Tests cover the places that can break: KB write permissions, proposal deduplication, metering (unknown usage), bootstrap.

---

## 11. Related Work

**Where the system design comes from:**
- Lippmann et al., *Context-Informed Machine Translation of Manga* (COLING 2025), arXiv 2411.02589: how the translation unit and context affect manga translation.
- **DelTA** (ICLR 2025), arXiv 2410.08143: multi-level memory for document-level translation consistency.
- **TransAgents**, arXiv 2405.11804: multi-agent translation of long literary works.

**Where the evaluation framework comes from (verify before citing):**
- Post, *A Call for Clarity in Reporting BLEU Scores* (WMT 2018): sacreBLEU and its signature.
- Rei et al., *COMET* (EMNLP 2020); Guerreiro et al., *xCOMET* (TACL 2024).
- Freitag et al., *Experts, Errors, and Context* (TACL 2021): human MT evaluation with MQM.
- Kocmi & Federmann, *GEMBA-MQM* (WMT 2023): MQM error spans annotated by an LLM.

This project does not pose a new research question. It evaluates a product with a standard evaluation framework; the system-design papers above explain where the design comes from.

---

## 12. Design FAQ

| Question | Answer |
|---|---|
| How do you show the system really got better, not just that the demo feels smoother? | Standard metrics (COMET / xCOMET, chrF, GEMBA-MQM, term accuracy) against a baseline, with paired bootstrap 95% CIs by page; the eval protocol is written before the freeze and the test split is run once |
| Why code orchestration instead of letting an agent decide every step? | The overall flow is fixed, and ordinary code is controllable, evaluable and lets the budget be allocated precisely. Autonomy is kept for the uncertain parts (Anthropic, "Building effective agents") |
| Why does the Reviewer get an independent context? Is there data? | Design rationale: self-checks tend to justify themselves, and the Reviewer cannot see the Translator's reasoning. There is no equal-budget comparison (Known Limitations); validating it would take an equal-budget Self-Refine baseline and a paired bootstrap |
| What happens when several agents write the same shared state? | Single-writer + Proposal Queue; proposals are deduplicated by ID; names and conflicts go to the Human Review Queue |
| Does the KB help? Why no vector database? | The KB ablation's term accuracy difference and its CI; terms and names are identifiers, where exact match fits better |
| Can an LLM-as-judge be trusted? | A human agreement check on about 30 annotated segments; the judge model differs from the translation model; with too little agreement it is reported for reference only |
| Do chrF / BLEU say whether a manga translation is good? | Not on their own, hence COMET / xCOMET and GEMBA-MQM; the single-reference limitation is in the report |
| Why not write our own tracing? | Langfuse and Phoenix are mature; metering stays in-house because the definition of unknown usage has to be ours |
| What if SSE drops or the server restarts? Could the KB be written twice? | `seq` + `Last-Event-ID`; the page is the retry unit; the KB is written once, at finalize |
| Why rewrite PanelFlow? | The order was wrong: infrastructure came before a single real model call. Lessons: vertical slice first, scope written down, tests only where things break |
| How much of the code is AI-written, and how is quality kept? | See §10 and `docs/workflow.md` |

---

## 13. Definition of Done (2027-02-15)

- [ ] Public repository; a working editor + demo recording
- [ ] `baseline` / `full` / `full_no_kb` results on the test split: chrF / BLEU (with signature), COMET / xCOMET, GEMBA-MQM score and error breakdown, term accuracy / consistency, cost per page, latency (p50 / p95)
- [ ] Paired bootstrap 95% CIs for both comparisons; GEMBA-MQM human agreement results; report (including Known Limitations)
- [ ] Blog posts in Chinese and English
- [ ] Someone new to the project can reproduce the main table from the README
