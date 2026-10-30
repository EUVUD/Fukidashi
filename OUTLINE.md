# Fukidashi — Project Outline

> v0.29 · 2026-10-30
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
> **v0.16 (2026-10-28):** M3 decisions in §8: the dev / test split (§4.1), who drafts the gold glossary (§4.7), a fresh KB for each eval run with readable `pending_human` entries (§3.3), and the Curator as code rules plus one model call per page (§3.3).
> **v0.17 (2026-10-28):** after the gold glossary (#25) and the Reviewer (#26): §0 adds Flag, Verdict / Issue and Unusable review; §3.2 and §4.7 record what is built; §8 decides that the Reviewer writes its comments in Simplified Chinese; §10: each executor works in its own git worktree.
> **v0.18 (2026-10-28):** after the Orchestrator and `full_no_kb` (#29) and the KB store (#28): §0 adds Send-back / Revision and KB snapshot; §3.2, §3.3 and §4.2 record what is built; §3.5 no longer claims review rounds are spans (only the root span carries the loop's outcome).
> **v0.19 (2026-10-28):** after the KB tools (#31): §3.2 and §3.4 record the agents' KB tools and where the snippets go; §3.3 marks the order of parallel proposals as settled.
> **v0.20 (2026-10-28):** after hardening the KB seams (#33): §3.3 records the conflict across kinds and the first call's payload as built, and updates what is still open.
> **v0.21 (2026-10-28):** M3 done (§7, `docs/m3.md`) after the Curator and `full` (#35): §3.2, §3.3, §3.5, §3.6 and §4.2 record what is built; §3.3 updates what is still open; §9 notes what the first `full` run showed about the KB.
> **v0.22 (2026-10-28):** M4 decisions in §8: the COMET checkpoint, the term accuracy rules (§4.7) and the budget for the dev runs. §7 moves term accuracy from M5 to M4, before the dev runs, and records M4's cost estimate. §4.8 records that the eval harness writes its outputs page by page (#37).
> **v0.23 (2026-10-28):** after KB reach (#40), the COMET scorer (#41), term accuracy (#42) and CI (#39): §0 adds KB reach; §3.3, §4.4, §4.7 and §4.8 record what is built; §8 and §9 record that XCOMET-XL does not fit the Windows machine's GPU, so the report gives COMET only (§1.1, §4.3, §5, §12 and §13 follow); §8 adds CI; §6 shows the scorers and the workflow.
> **v0.24 (2026-10-29):** M4 done (§7, `docs/m4.md`) after the dev runs and the pooled report (#44): §0 adds Pooled report and Common pages; §3.3 records the dev runs' KB reach and settles normalised matching (not needed); §4.8 records the COMET checkpoint's revision; §4.9 adds what term accuracy cannot see; §6 shows the report; §8 and §9 record the dev runs' outcomes.
> **v0.25 (2026-10-29):** M5 decisions in §8: the judge model (GPT-4o mini, with GPT-5.1 as its fallback) and the agreement check's decision rule (§4.5), the KB ablation's primary metric (term accuracy stays; term consistency is added as a secondary metric), a second `baseline` pass for the run-to-run spread, and the rules of term consistency (§4.7). §0 adds Rendering and Run-to-run spread; §4.4, §4.5, §4.9 and §7 follow.
> **v0.26 (2026-10-29):** §4.5 records the hash of the agreement sample's labels, which Claude annotated because the author does not read Japanese, and what these labels allow the check to show.
> **v0.27 (2026-10-30):** M5 done except the freeze tag, which is the author's (§7, `docs/m5.md`), after one character, one profile (#47), the judge with the GEMBA-MQM scorer and the agreement sample (#48), term consistency (#49), the test glossaries (#51), the agreement check with the dev MQM scores and the second `baseline` pass (#52), the tuning (#53) and the eval protocol (#54): §0 adds Agreement sample, Keep rule and Merged; §3.3, §4.2, §4.4 to §4.9, §6, §7, §8 and §9 record what is built and what the agreement check and the tuned dev runs showed (neither judge passed, so MQM is for reference only); §8 adds the tuning's keep rule and the bootstrap's resamples and seed.
> **v0.28 (2026-10-30):** the author placed `exp-frozen` at `0c1fe05`. M6 decisions in §8: the test runs start now, from the tag, with no request timeout; the judge scores the test runs in M6 instead of M8; the paired bootstrap's code is merged before the first test page, and MQM gets an interval too, as reference only. §7's M6 and M8 rows follow, with M6's cost estimate; §9 adds the model behind `deepseek-flash`.
> **v0.29 (2026-10-30):** the paired bootstrap is merged (#58). §8 decides, before any test page, that the label of a term consistency interval is decided per run, not on a config's pooled counts.
> Convention: code, docs, commit messages and reports use the canonical terms in §0.

---

## 0. Terminology

Use the English term in the left column everywhere, in code and in docs. The Chinese column is a glossary for Chinese readers.

| Canonical term | Chinese | Notes |
|---|---|---|
| Vertical slice | 竖切 | Get data → model → browser working end to end before adding any infrastructure |
| Translator Agent | 翻译 agent | The only agent that calls tools on its own |
| Reviewer Agent | 独立审校 agent | Independent context, read-only, sends a page back at most 2 times |
| Flag | 不确定标记 | A region the Translator marks with `flag_uncertain`: its id and a short reason. The Reviewer reads the flags |
| Verdict / Issue | 审校结论 / 审校意见 | The Reviewer's answer for one page: `approve` or `revise`, with at most one issue (a comment, optionally a suggestion) per region. Only `revise` sends the page back |
| Unusable review | 无效审校 | A review that ended without a verdict (still unparseable after one Repair, or `step_limit` / `incomplete`): the translation stands. Unlike a failed review (`provider_error`, `budget`), after which nothing more is sent |
| Send-back / Revision | 退回 / 修改 | A `revise` Verdict sends the page back; the Revision is another Translator run that also gets the current translation and the Issues. It never falls back to `baseline`: an unusable Revision leaves the previous translation |
| KB snapshot | KB 快照 | An immutable in-memory copy of one KB's readable entries and profiles, taken before an agent run. Agents read the KB only through it, so they hold no database connection |
| KB reach | KB 触达 | How often the KB reached a page in a `full` eval: the items its snippets showed, the lookups that found something or not, and the distinct items that reached the page. In an eval run the KB starts empty, so every item that reached a page came from an earlier page (§3.3) |
| Curator Agent | 知识管理 agent | The only agent that writes to the Knowledge Base |
| Merged | 合并 | The result of a character proposal that joins an existing profile instead of making a new one (Curator rule 6, §3.3) |
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
| Pooled report | 汇总报告 | `eval/report.py`: one finished eval run per config and book, scored by the scorers, pooled into one report per config, over the books and per book (#44) |
| Common pages | 共同页 | The pages that every config in a pooled report scored, per book; the report's quality numbers cover only these, and lists the pages left out |
| Eval protocol | 评测规程 | One-page `eval/PROTOCOL.md`, written before the freeze (#54); replaces the earlier `PREREG.md` |
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
| Term accuracy / Term consistency | 术语准确率 / 一致性 | Term accuracy is the KB ablation's primary metric, term consistency its secondary metric (§4.7) |
| Rendering | 译名 | The part of a Translation that renders one gold glossary term; for term consistency the judge model extracts it (§4.7) |
| sacreBLEU signature | sacreBLEU 签名 | Records the tokenizer, smoothing and other settings; included in the report |
| Neural metric | 神经网络评测指标 | COMET in this project |
| COMET / xCOMET | — | Unbabel's neural metrics; xCOMET also marks error spans. The report gives COMET only (§8) |
| LLM-as-judge | 用 LLM 当评委 | |
| Judge model | 评委模型 | Different from the translation model, to reduce self-preference bias; GPT-4o mini (§8) |
| MQM (Multidimensional Quality Metrics) | 多维质量指标 | A framework for annotating translation errors by error category and severity |
| GEMBA-MQM | — | MQM annotation done by an LLM-as-judge |
| Error category / Severity | 错误类别 / 严重度 | The two MQM dimensions; severity is minor / major / critical |
| MQM score / Error breakdown | MQM 分数 / 错误分布 | Replaces the earlier custom failure taxonomy |
| Human agreement check | 人工一致性检查 | About 30 segments annotated before any judge output, which show whether GEMBA-MQM can be trusted (§4.5). M5's annotator was an LLM, and neither judge passed |
| Agreement sample | 一致性检查样本 | The 30 segments drawn blind from the dev runs' `baseline` and `full` outputs for the check, with the annotator's labels (§4.5) |
| Latency (p50 / p95) | 延迟 | End-to-end wall-clock time per page |
| Freeze (`exp-frozen`) | 冻结 | After this git tag, the eval code path does not change |
| Paired bootstrap | 配对 bootstrap | Paired by page; reports a 95% CI |
| Confidence Interval (CI) | 置信区间 | |
| Keep rule | 保留规则 | Fixed before the tuned dev runs: the tuning stays unless `full`'s pooled COMET or primary term accuracy falls, against M4's `full`, by more than the largest per-book difference between the two `baseline` passes (§8) |
| Run-to-run spread | 运行间波动 | How far two eval runs of the same config differ on the same pages; the model is not deterministic. Measured with a second `baseline` pass (§8) |
| Adapter | 适配器 | Detection, OCR and inpainting are swappable |
| Checkpoint | 检查点 | The only points where a change of direction is considered |
| Translation | 译文 | One Run's zh-Hans output for one Segment, stored in `translations`; a Segment collects one Translation per Run |
| Reference translation | 参考译文 | OpenMantra's human zh-Hant and en translations of each Segment; used for comparison and evaluation only, never in a prompt |
| Run | 一次运行 | One execution of one config on one page, stored in `runs` (config, model, usage, cost, status, timestamps); a page shows its latest succeeded Run. Unlike a Manifest, which records a whole batch of eval runs |

**Removed in v0.2:** Cost parity, Preregistration (now Eval protocol), Failure taxonomy (now MQM error breakdown), Blind evaluation, Reviewer false positive.

---

## 1. Overview

### 1.1 One-liner

A **web-based manga translation editor**. A Translator Agent looks up the Glossary and Character Profiles as it needs them, a Reviewer Agent checks the result in an independent context, and a Curator Agent maintains the Knowledge Base as the book is translated. It is evaluated on OpenMantra against a baseline with a standard evaluation framework (sacreBLEU, COMET, GEMBA-MQM, term accuracy).

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

**As built in M3 so far (#26):** with its `flagging` switch on, the Translator also has `flag_uncertain`; the `translator` config keeps the switch off and sends exactly M2's requests. The Reviewer Agent exists (`agents/reviewer.py`) but is not part of any config until the Orchestrator arrives. In a fresh conversation it gets the page image, every region's Japanese text and translation, and the flags, and answers with a Verdict and per-region Issues. It has no tools yet (the KB's read-only lookups come with the KB) and a token budget of its own. One smoke run (`tojime_no_siora` page 3, $0.005 for both agents) flagged 1 of 12 regions and approved the page with a note on that region.

**As built in M3, the review loop (#29):**
- **The Orchestrator** (`orchestrator/flow.py`) runs `full_no_kb` as plain code:
  - the flagging Translator first (it may fall back to `baseline`, as in M2);
  - then the Reviewer on the current translation, with the flags of the run that produced it;
  - on `revise`, a Revision and another review, at most 2 Send-backs;
  - no review after the second Revision.
- **What ends the loop:**
  - an unusable review keeps the current translation;
  - an unusable Revision keeps the previous one;
  - a provider error or a spent budget in any stage fails the whole Run, with nothing more sent (§3.6).
- **Budgets.** Each Translator run and each review has its own token budget.
- **The Reviewer's comments** are in Simplified Chinese (§8).
- **First real run** (`tojime_no_siora` pages 2–4, $0.013): the Reviewer approved all three pages on the first review, each with a note, so no page was sent back. A separate check (#31) sent one Revision, with a made-up Issue, to the real provider: it was accepted and usable, and changed only the region the Issue named.

**As built in M3, the KB tools (#31):**
- **The switch.** `run_orchestrator` takes a KB snapshot. Without one it is `full_no_kb`, and its requests are unchanged. With one:
  - the Translator has `lookup_term`, `lookup_character` and `propose_entry`;
  - the Reviewer has the two lookups only;
  - each agent has a system prompt of its own for the KB.
- **One snapshot per page.** A page uses one snapshot and one proposal collector for all its stages, so its own proposals stay invisible to it until finalize (§3.6).
- **Proposals.** Proposals are validated by the same code that stores them. The page's proposals come back in call order: Translator run, step, position in the reply. A failed Run returns none.
- **Wired in #35:** the `full` config passes the snapshot (below).
- **Real check** (`tojime_no_siora` page 3, empty KB, $0.009): the Translator looked a term up, proposed one term (a family name, as a term) and flagged one region; the Reviewer approved without a lookup. A non-empty snippets part, a character proposal and a Reviewer lookup have run only against scripted providers.

**As built in M3, the Curator and `full` (#35):**
- **`full`** is the Orchestrator with the KB of the eval run: the snapshot is read once, before the page. After a successful page comes finalize (§3.6).
- **The Curator Agent** (`agents/curator.py`) makes one call per page, without tools and without the page image. It sees:
  - the page's KB snippets;
  - each region's Japanese text and stored translation;
  - the page's term proposals.
  It never sees a name proposal (names go to a person whatever it says), the other agents' conversations or a reference translation. It accepts or rejects each term with a reason in Simplified Chinese; a bad answer gets one repair call.
- **Without an answer** (unusable, provider error, spent budget), the page still succeeds and its terms stay undecided. The rules still apply names, confirmations and conflicts.
- **First real run** (`tojime_no_siora` pages 2–4, $0.033; §4.2): the Curator was called on 2 pages and needed no repair. It accepted 3 terms and rejected 1, a rendering the stored translation did not use. Both names went to the Human Review Queue. Page 3's Translator found the name page 2 had proposed, so the KB carried a name from one page to the next.

**Review loop:** the Reviewer sends a page back at most twice. This is fixed; since v0.2 it is not tuned for cost parity.

### 3.3 Knowledge Base

- **Entities:**
  - `GlossaryEntry`: `id`, `source_term`, `target_term`, `category`, `status` (proposed / approved / rejected / pending_human), `provenance` (agent / human), `evidence_pages`, `history`
  - `CharacterProfile`: `id`, `name_src`, `name_tgt`, `aliases`, `notes` (speech style and so on), `evidence_pages`, `history`
  - `Proposal`: `proposal_id`, `kind`, `payload`, `proposer`, `page_id`, `status`
- **Write policy:** single-writer + Proposal Queue. Proposals carry IDs and the Curator deduplicates by ID, so a retry never writes twice.
- **Curator (§8):** the rules are code: deduplication by ID, conflicts and names to the Human Review Queue, and one KB write per page, at finalize (§3.6). The model is called once per page, without tools, to accept or reject each remaining proposal with a reason; code applies its decisions and enforces the rules whatever the model answers.
- **Conflicts:** two proposals with different target terms for the same source term, or anything involving a name, go to the Human Review Queue instead of being merged automatically.
- **Readable states:** lookups and KB snippets return `approved` and `pending_human` entries, the latter marked as pending; `proposed` and `rejected` entries are not returned. So a name proposed on one page reaches later pages before a human has confirmed it (§8).
- **One KB per eval run:** each eval run starts every book with an empty KB, which fills page by page in page order; nothing carries over between eval runs. The editor keeps one KB per book (M7).
- **Retrieval:** relevant terms and characters are matched against the source text and truncated to a character budget; the agent looks up the rest with tools. No vector database, and the report says why: terms and names are identifiers, where exact match beats embeddings.
- **As built in M3, storage and rules (#28)** (`kb/`; the agents read it since #31, and `full` writes it at finalize since #35):
  - **Tables.** A `kbs` row is one KB (book, purpose `eval` or `editor`, label). The entries, profiles and proposals each belong to one KB.
  - **Profile status.** Character Profiles carry a `status` like entries do, because names go to the Human Review Queue. That queue is the set of rows with status `pending_human`, not a table of its own.
  - **Proposal IDs** come from the KB, the page, the kind and the normalised pair, so a retried page writes nothing twice. The same pair proposed on a later page becomes evidence.
  - **`curator.apply`** applies one page's proposals and the Curator's decisions in one transaction, by the rules above. It is the only code that writes KB entries, which a test checks.
  - **Agents** get only a KB snapshot and a proposal collector, neither of which holds a database connection (built in #31). Since #33 that holds for the whole import graph: loading the agents or the Orchestrator does not load the database module, and a test checks it. Least privilege therefore comes from what an agent has, not from its prompt.
  - **Settled in #31:** the order of parallel `propose_entry` calls within one reply. Proposals follow the calls' position, not their arrival.
  - **Settled in #33:**
    - **Payload per id.** For one proposal id, the details kept are those of the first call in that order.
    - **Conflict across kinds.** A term whose source is a readable character's name or alias goes to the Human Review Queue and writes nothing. So does a character whose name is a readable term's source. Within a page, the earlier proposal counts as existing for the later one.
  - **Settled in #35:** a conflict across kinds has its own result (`conflict_across_kinds`), so it is counted apart from a same-kind conflict; what is written is unchanged.
  - **Rule 6, one character, one profile (M5, #47).** A character proposal matches a readable profile when either name contains the other, or a name or alias of one equals a name or alias of the other (aliases only by equality). One match whose rendering agrees (one contains the other) takes the proposal in: its names become aliases of that profile, its page joins the evidence, the history gets a `merged` line, and the result is `merged`, the ninth result. A disagreeing rendering, or more than one matching profile, is a `conflict` and writes nothing. The rule comes after duplicate and confirmation and before a new profile is made, and the Curator's model cannot change it. `store.add_profile_aliases` writes the aliases. A test replays M4's `boureisougi` proposals and ends with one profile for each of four characters; in the tuned dev runs no proposal was merged.
  - **KB reach (#40).** A `full` Run records the items its snippets part showed (`kb_snippets` in its attempts: kind, key and status, in prompt order), even when the Orchestrator failed. The eval harness counts, per page and in total, the snippets shown, the lookups that found something or not (from the stored steps), and the distinct items that reached the page. A Run stored before #40 has no snippet record, so its snippets count as unknown, not as 0. On the M3 `full` run, 1 of 12 lookups found something (a name proposed on the page before); its snippets were not recorded. **On the dev runs (#44):** the snippets part was not empty on 49 of 79 pages, and 44 of 110 lookups found something. Of the 66 that found nothing, 38 asked for an item that entered the KB only on that page or later and 27 for one that never did; at most two were near misses (`docs/m4.md`).
  - **Still open for task design:**
    - a proposal left undecided (the Curator's call failed) cannot be decided by applying its page again;
    - a failure after the KB write (when the Run is finished) leaves the KB written under a failed Run. A retry of that page on the same KB would find its own proposals as duplicates, and the Curator's decisions would be lost. It matters once a page can be retried on a kept KB (the editor, M7);
    - a name proposed as a term when the KB holds nothing it collides with: the first real KB run proposed a family name that way. Only the Curator's judgment can catch it;
    - a character proposed under another character's name or alias became a second pending profile in M3 and M4 (**seen on the dev runs:** a given name and a full name became two profiles for three characters, and one KB held two pending renderings of one name, which spread to later pages, `docs/m4.md`). **Closed in M5 (#47)** for a character against a character. Still open (`IDEAS.md`): a name with an honorific whose bare form sits inside a full name; merges grow a profile's aliases without limit; a character's aliases are not checked against term sources;
    - **variant spellings.** Retrieval and the rules match exact strings. In the first `full` run the dataset spelled one family name two ways on consecutive pages, and the KB ended with two approved entries with two different renderings. No rule sees it, and the Curator judges one page at a time. The same exact match left the snippets part empty on all 3 pages (§9). **Settled on the dev runs (#44): no normalisation.** No lookup missed an item that its exact spelling would have found, apart from at most two near misses; the misses were items not yet, or never, in the KB.

### 3.4 Agent Loop Guardrails

- Step limit; per-page token budget; duplicate tool call detection; tool errors are passed back to the agent as they are. **Built in M2** (`agents/loop.py`): the Step limit is `max_steps` model calls (default 6); a duplicate is the same tool with the same validated arguments; the model's mistakes (an unknown tool, arguments that fail the tool's pydantic model, a handler that raises) go back as the tool's result. Nothing in the loop retries: a provider error ends the run (§3.6).
- A reply that ended for any reason other than `stop` or `tool_calls` is never taken as the answer (stop reason `incomplete`): with thinking on, reasoning can use up `max_tokens` and leave an empty, truncated reply. Further checks (no progress, a cap on tool calls, near-duplicates, a wall-clock deadline) wait for traces that show a need (`IDEAS.md`).
- Structured output: validate against a JSON schema, retry once on a parse failure, then **fall back visibly** to `baseline` (one call per page). **Built in M2** (`agents/translator.py`): the answer must pass `parse_translations`; a bad one gets one Repair call (the history plus the parse error, JSON output, no tools); `step_limit`, `incomplete` or a failed Repair fall back to `baseline` with a budget of its own. The Run keeps config `translator`, its attempts say why, and the eval harness counts the pages that fell back. `provider_error` and `budget` never fall back (§3.6).
- The prompt keeps a stable prefix for prompt caching: `system → tool definitions → glossary → page`. **Built in M3** (#31): with the KB on, the glossary is the user message's first text part (the page's snippets, at most about 1,000 characters), before the page image and the regions; the system prompt and the tool definitions do not change from page to page.
- **Two DeepSeek rules:** with tools, the `reasoning_content` of every turn must be passed back unchanged in later requests (done by `LLMClientChatModel` since M2; `deepseek-flash` sends `""` on its final turn, which goes back as it is); JSON output occasionally comes back empty, which the retry above covers. Thinking is switched in `extra_body`; effort is `reasoning_effort` (`low` / `high` / `max`). The M2 smoke runs (2026-10-27, text only) also showed that `response_format: json_object` is accepted together with tools.

### 3.5 Metering & Tracing

- **Metering is ours:** every LLM call records the model ID, input / output / cached tokens and cost. When the provider returns no usage, including a request that failed or timed out, the call is recorded as **unknown** and counted separately, never as 0. Cost per page comes from this data.
- **Tracing uses an off-the-shelf tool:** Arize Phoenix, run locally (§8), replacing custom trace JSONL. LangSmith is never used: `run_agent` refuses to start while its tracing variables are on. **Built in M2** (`tracing/`): off unless `FUKIDASHI_TRACING=phoenix`; each `translate_page` is one trace whose root span carries book, page, config, Run id, status and, for `translator`, the stop reason and fallback reason; OpenInference adds a span per provider call and the agent's node and tool spans; an exporter wrapper removes page images and the API key, and spans go straight to the local endpoint, never through a proxy; eval rows and the manifest record the trace ID, also for a page that fails before its Run exists. One trace per page; agent steps and tool calls are spans. For `full_no_kb` and `full` (M3) the root span also carries the number of Send-backs and why the loop stopped, and for `full` the Curator's status and how many proposals got each result; the stages (translate, review, revise) have no spans of their own, so a trace shows the agents' spans one after another (`IDEAS.md`). Eval results store the trace ID, so a score links straight to its trace.
- **Choosing:** Phoenix runs locally as one Python process, which is light on an 8 GB Mac. Self-hosting Langfuse needs more components, and Langfuse Cloud sends traces to a third party.
- The editor's live agent trace (SSE) is a **product feature** and separate from the tracing tool: the first is for users, the second for development and evaluation.

### 3.6 Reliability

- **The page is the retry unit;** task state is persisted in SQLite.
- LLM calls have no side effects and are safe to retry; the KB write, which does have side effects, happens once, at finalize, by the Curator. **Built in M3** (#35): finalize runs only after a successful page, in the order Curator call → `curator.apply` (one transaction) → the Run is finished. A failing write fails the Run and leaves the KB unchanged; a failure after the write is still open (§3.3).
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
- Split by book: **dev = 2 books** (for tuning prompts and tools): `tojime_no_siora`, `boureisougi`; **test = 3 books** (run once, after the freeze): `balloon_dream`, `tencho_isoro`, `rasetugari` (§8).
- Metrics are computed per segment; the paired bootstrap resamples pages.
- The public repository holds no OpenMantra images, only eval scripts, the gold glossary and attribution.

### 4.2 Configurations

| Config | Contents |
|---|---|
| `baseline` | Formerly C0: one call per page, no tools, no KB |
| `full` | Full system: Translator Agent (tools) + Reviewer Agent (at most 2 rounds) + Curator Agent / KB |
| `full_no_kb` | KB ablation: identical to `full` except the KB is off |

**`translator` (M2):** the Translator Agent alone, without KB or Reviewer, run on the M2 smoke pages. It is not one of the reported configs; M3 builds `full` and `full_no_kb` on top of it.

**`full_no_kb` as built (M3, #29):** the Orchestrator with the KB off (§3.2). On `tojime_no_siora` pages 2–4 it scored chrF 9.28 / BLEU 5.30, against `baseline`'s two runs (11.17 / 6.72 and 12.87 / 8.83) and `translator` (10.67 / 6.46). No page was sent back, so its stored translations are the flagging Translator's first answers. The gap to `translator` shows run-to-run noise on 27 segments, not an effect of the Reviewer; whole dev books come in M4.

**`full` as built (M3, #35):** the Orchestrator with the KB on, a fresh KB for each eval run, and the Curator at finalize (§3.2). On `tojime_no_siora` pages 2–4 it scored chrF 10.86 / BLEU 7.69 at $0.011 per page (22 calls; latency p50 20.6 s, p95 53.8 s). It is the first eval run in which the Reviewer sent a page back: page 4, twice, ending at the Send-back limit. As with `full_no_kb`, 27 segments say nothing about quality.

**Tuned in M5 (#53):** one round of prompt and tool-description text, read from the dev books' M4 outputs and kept by the keep rule (§8, `docs/m5.md`). The tuned dev pass: `full_no_kb` chrF 20.18, COMET 0.8192, primary term accuracy 67 of 117, consistency 72.2%; `full` chrF 20.75, COMET 0.8254, primary term accuracy 73 of 117, consistency 83.2%, at $0.0066 and $0.0105 per page. They are dev numbers on the books the prompts were tuned on, one run each.

**KB off means:** no KB snippets in the prompt; `lookup_term`, `lookup_character` and `propose_entry` disabled; the Curator does not run; the Reviewer does not see the KB either. The Translator, the Reviewer, `get_previous_context` and the token budget are the same as in `full`.

**Removed in v0.2:** C1, C2 (Self-Refine) and C3 as separate configs; `mem_window3`; cost-parity tuning; running each config twice.

### 4.3 Comparisons

| Comparison | Primary metric | Question |
|---|---|---|
| `full` vs `baseline` | COMET | How much better is the whole system than one call per page, and what does it cost in money and latency? |
| `full` vs `full_no_kb` | Term accuracy | How much does the KB contribute? |

Every quality metric in both comparisons gets a paired bootstrap 95% CI. Primary metrics are written into the eval protocol before the freeze, so no metric is picked after seeing results.

### 4.4 Evaluation Framework

| Metric | Tool | Use | Notes |
|---|---|---|---|
| **chrF / BLEU** | sacreBLEU | Corpus-level; the report includes the **sacreBLEU signature** | With Chinese as the target, BLEU uses `tokenize=zh`; chrF needs no tokenization |
| **COMET** | Unbabel COMET | The main neural metric, reference-based; **required** | `Unbabel/wmt22-comet-da` (Apache-2.0), run on the Mac's CPU. xCOMET (`Unbabel/XCOMET-XL`) does not fit the Windows machine's GPU, so it is not reported (§8). **As built (#41):** `eval/comet.py` scores a finished eval run's `segments.jsonl` (source: the Japanese text; hypothesis: the stored Translation; reference: the converted reference) in an environment of its own and imports nothing from the project, so PyTorch never enters its lock or CI. It writes `comet-wmt22-comet-da.json` next to the eval run's outputs. On the M3 `full` run (27 segments) it took about 5 s on the CPU after loading the checkpoint, used about 1.4 GB of memory, and two runs gave the same scores |
| **GEMBA-MQM** | LLM-as-judge | Marks error spans with error category and severity (minor / major / critical) per segment, aggregated into an MQM score and error breakdown; **replaces the custom failure taxonomy** | The judge model differs from the translation model; run the human agreement check (§4.5) first. **Judge (§8, decided 2026-10-29):** GPT-4o mini, a dated snapshot, through `LLMClient` with a configuration and a ledger of its own; its calls are the evaluation's cost, not the system's. **As built (M5, #48, #52):** `eval/mqm.py` scores a finished eval run with the GEMBA-MQM prompt template at a pinned GEMBA commit (CC BY-SA 4.0 material, kept in three files that the README lists), through the judge's own `LLMClient`, Meter and ledger (`runs/judge_usage.jsonl`), and writes `mqm-<model>.json` next to the run's outputs: a score for each segment (GEMBA's weights: critical 25, major 5, minor 1, the first five errors, capped at 25, so a segment scores from −25 to 0), its errors and the judge's raw answer. A segment with an empty Japanese text is skipped. `--reparse` reads stored answers again with the current parser, without a call. **Outcome (§4.5): neither judge passed the agreement check, so MQM is for reference only** |
| **Term accuracy / consistency** | Ours, against the gold glossary | Measures the KB | Term accuracy is the primary metric of the KB ablation and term consistency its secondary metric (§8); both sets of rules are in §4.7. Term accuracy was built in #42 (`eval/terms.py`); consistency in #49 (`eval/consistency.py`), and the report pools both |
| **Cost per page** | Our metering | Cost of all of the system's own LLM calls; unknown usage counted separately | The evaluation's own cost (judge calls, COMET compute) is tracked separately, not as system cost |
| **Latency** | Our timer | End-to-end wall-clock time per page, reported as p50 / p95 | |
| Overflow rate | Typesetter | Product metric | Not part of the §4.3 comparisons |

### 4.5 GEMBA-MQM Human Agreement Check

- Sample about 30 segments from the dev split (covering both `baseline` and `full` outputs) and annotate error category and severity by hand, using the same MQM guideline. **The human annotation is done before looking at GEMBA-MQM's output.**
- Run GEMBA-MQM on the same segments and compare two things: the correlation of segment-level MQM scores (for example Kendall τ), and whether the major / critical errors found by hand were caught.
- The decision rule goes into the eval protocol before the freeze: with enough agreement, the MQM error breakdown is the report's main error analysis; otherwise it is reported for reference only, and the report says so.
- **Blind, and first (M5).** The sample is drawn from the dev runs' `baseline` and `full` outputs and shown in random order without its config, with only the source segment and the translation, as the judge sees them. The decision rule's thresholds are written down, and the annotation file's hash is recorded in git, before the judge runs on the sample.
- **The decision rule (decided 2026-10-29, before any judge output on the sample).**
  - **Pass.** A judge passes when both hold:
    - Kendall's τ_b between the author's and the judge's segment-level MQM scores over the sample is at least 0.30;
    - on at least 60% of the items where the author marked a major or critical error, the judge marks at least one major or critical error too.

    If the author marked no major or critical error, τ_b decides alone.
  - **Then.** For a judge that passes, the MQM breakdown is the report's main error analysis. If GPT-4o mini fails, its fallback (§8) annotates the same sample once, under the same rule. If neither passes, MQM is reported for reference only.
- **Recorded (2026-10-29):** the labels, `runs/mqm-human/20261029T172846Z-sample/sheet.csv`, sha256 `c400077e12c677f31b510bff6cb76b846fdd3d9d053487ab5e216974c51d68c2`, recorded before any judge output on the sample. **The annotator is Claude, an LLM, and not a person**: the author does not read Japanese and decided on 2026-10-29 to have a fresh Claude session annotate the sample. The session was told to read only the annotation guide, the sample's README and the sheet, and its tool calls show it opened nothing else: no `key.json`, no judge output, no reference translation, no web search. Wherever this section says "the author", read "the annotator". The rule and its thresholds are unchanged. The check therefore measures the judge's agreement with another LLM, which is weaker evidence than agreement with a person, and every report that uses it says so.
- **What these labels allow (2026-10-29, before any judge output).** The annotator marked an error on 1 of the 30 items (one minor `accuracy/addition`) and none on the other 29, so there is no major or critical error and τ_b decides alone. Of the 435 pairs of items, only the 29 that include the marked item can be concordant or discordant. The best case is a judge that scores that item below every other and gives many items the same score: at least 113 of the 435 pairs must be tied (16 items with one score would give 120). A judge whose scores mostly differ from item to item cannot reach 0.30, whatever it does. The procedure is unchanged, and a fail is reported as weak evidence against such a judge, not as proof that its errors are wrong.
- **A later judge** (GPT-4o mini leaves the API on 2027-01-14, §8) reuses the human labels: the new judge annotates the same sample, and the same rule decides.
- **Outcome (2026-10-30, #52): neither judge passed.** `eval/mqm_agreement.py` compares the labels (checked against the recorded hash) with a judge's scores on the sample's 30 items. GPT-4o mini: τ_b −0.042 (8 concordant and 12 discordant pairs). GPT-5.1, the fallback, scored the same runs once: τ_b 0.283 (28 concordant pairs, none discordant, but so many ties that 0.293 was the most it could reach). With one marked error and no major or critical one, τ_b decided alone, and both are under 0.30. **The MQM breakdown is therefore reported for reference only, with GPT-4o mini's scores, and every report says so and says that a fail on these labels is weak evidence** (an LLM's labels, one marked error in 30 items). A person's labels, or a sample with more marked errors, would be needed to tell the judges apart (`IDEAS.md`). One parser change followed the first look: an answer made only of "No-error" lines, with no severity heading, is read as no error, as GEMBA reads it. The author chose it, it was set before GPT-5.1's agreement was computed, and the stored answers were read again without a call.

### 4.6 Rigor & Statistics

- Every config uses the same translation model and the same inputs.
- Prompts and tools are tuned on the dev split only; each config runs **once** on the test split.
- Before the freeze, write the one-page `eval/PROTOCOL.md`: metrics (primary marked), splits, test runs once, exclusion rules, the GEMBA-MQM decision rule. Then tag `exp-frozen`.
- **Written (#54):** `eval/PROTOCOL.md` fixes the two comparisons and their metrics, the test runs, the exclusions, the agreement rule and its outcome, the bootstrap (pages pooled over the test books, 10,000 resamples, seed 2027) and the pinned inputs. **The author placed the tag** on 2026-10-30, at `0c1fe05`.
- **Statistics:** paired bootstrap 95% CI by page, for the two §4.3 comparisons only.

### 4.7 Gold Glossary Protocol

- 20–40 entries per dev book (M3), and one glossary for each of the three test books (M5).
- Built from the reference translation, **before looking at any model output**.
- **Who builds it (§8):** an AI coding assistant (Claude, not the translation model) drafts the entries from the Japanese text and the converted reference translations only, without access to stored Runs, Translations or eval outputs; the author reviews every entry. The glossary file says so. **What that meant (2026-10-30):** the author does not read Japanese (§4.5), so "reviews" meant approving the drafts as drafted, not checking them word by word; the reference translations were the only source.
- **Not fully blind for `tojime_no_siora`:** its pages 2–4 were translated in M0–M2 and their output was read. It is a dev book; each test book's glossary is built before any page of that book is translated.
- **As built (#25):**
  - **Files.** One JSON file per book in `eval/gold_glossary/`. The header has `book_title`, `split`, `opencc_config`, `drafted_by` and `reviewed_by_author`. Each entry has an `id`, a `category` (`character` or `term`), `source` and `source_aliases` (Japanese), `target` and `target_variants` (zh-Hans), and an optional `note`. Since #42 each entry also has a required boolean `borderline`.
  - **Checks.** `eval/glossary.py check` validates the files against `annotation.json`, never the database:
    - every term occurs where it should;
    - each entry occurs in at least 2 segments;
    - every target appears in the converted reference of a segment that contains its source (since #42, except a variant that is the target in mainland forms, which `t2s` never produces: see Character forms below);
    - no term is longer than 20 characters, so no sentence of the dataset enters git.
  - **Tools for the author.** `candidates` and `sheet` write candidate lists and review sheets to `runs/glossary/`; they quote the dataset and are gitignored.
  - **Entries.** `tojime_no_siora` has 20 and `boureisougi` 17 (its names and coined terms ran out). Every entry occurs on 2 or more pages. The author approved all of them as drafted.
  - **The test books' glossaries (M5, #51).** One file for each test book, drafted from the Japanese text and the converted references only, before any page of those books was translated: `balloon_dream` 27 entries (13 not borderline), `tencho_isoro` 21 (7) and `rasetugari` 28 (14). "Not borderline" holds names, coined terms and story-specific nouns, not only names and coined terms. `tencho_isoro`'s primary score will rest on few entries. The author approved the drafts as drafted.
  - **Ordinary words.** 15 of the 37 entries are recurring ordinary words, marked as borderline in their notes and, since #42, in their `borderline` field. Term accuracy leaves them out of its primary score (below).
- **Term accuracy (§8, decided 2026-10-28; built in #42, before the dev runs):**
  - **Entries.** The primary score covers the entries that are not borderline (22 of the dev split's 37: names and coined terms). A second score covers all entries. Borderline is the entry's `borderline` field, not a note.
  - **Occurrence.** A segment counts for an entry when its Japanese text contains the entry's source or an alias *and* its converted reference contains the target or a variant. Where the reference translator did not use the term, the reference cannot say what is right, so the segment is left out. This also leaves out most compounds that contain the source by accident.
  - **Hit.** The system's translation of that segment contains the target or a variant.
  - **Aggregation.** The primary score pools all occurrences (hits ÷ occurrences). The mean over entries is reported beside it.
  - **Character forms.** A form that `t2s` keeps but a mainland reader would write differently (著 / 着) is listed as a variant in the glossary, not handled by normalisation code. In the dev glossaries one entry needed one (#42).
  - **Consistency** is not part of term accuracy; it has rules of its own (below).
  - **As built (#42).** `eval/terms.py` scores one finished eval run, that is one book, and writes `terms.json` next to its outputs: both scores, the mean over entries, and each entry's occurrences, hits and left-out segments, with the glossary's hash and the rules' version. It reads only the Japanese text, the Translation and the converted reference, never the database or the zh-Hant reference, so a change of rules can score old runs again for free. The M3 runs of `tojime_no_siora` pages 2–4 hold 10 primary occurrences: too few to compare configs.
- **Term consistency (§8, decided 2026-10-29; built in M5, before any consistency number is computed):** the KB ablation's secondary metric. Term accuracy asks whether the system uses the reference translator's rendering of a term; consistency asks whether the system keeps one rendering of it across a book, which is what the KB is for (§4.9).
  - **Entries.** The entries that are not borderline (names and coined terms). An ordinary word may change with its context, and one rendering for it is not a goal.
  - **Occurrence.** A segment whose Japanese text contains the entry's source or an alias. The reference is not consulted. The dev split's 22 non-borderline entries have 126 such occurrences (70 in `tojime_no_siora`, 56 in `boureisougi`), counted from the Japanese text alone, and every entry has at least 2.
  - **Rendering.** For each occurrence, the judge model extracts the part of the Translation that renders the term, without honorifics, or answers that the term was left out or that the match is not this term. Each occurrence is its own call, so no call sees another occurrence's rendering. Code checks that the extracted rendering occurs in the Translation. Renderings are compared after NFKC, with whitespace removed and the middle-dot separators unified.
  - **Score.** Per book and entry, the share of rendered occurrences that use the entry's most common rendering. The pooled score takes the entries with at least 2 rendered occurrences, sums the counts of each one's most common rendering, and divides by their rendered occurrences. Left-out terms and non-matches are reported beside it, not scored.
  - **The extractor's own check.** The same extraction runs on the converted reference. Where the reference contains the entry's target or a variant, the extracted rendering must be one of them; the share that is gives the extractor's accuracy without a human. Consistency is reported as a result only when that accuracy is at least 90%, and for reference otherwise. The reference translator's own consistency is reported beside the configs'.
  - **Why extraction.** The idea follows Lyu et al.'s (2021) lexical translation consistency, which finds the translations of a repeated source word by word alignment. Here the judge model replaces the aligner (§8). Lei et al. (2024) note that not every repeated word should be translated the same way, which is why borderline entries are left out.
  - **As built (M5, #49).** `eval/consistency.py` scores one finished eval run and writes `consistency-<model>.json` next to its outputs, with the glossary's hash and the rules' version. The judge answers each occurrence in its own call, once for the Translation and once for the reference (a retry shows the judge its own answer and why it failed). A rendering must occur in the text after normalisation, and in the extractor's check on the reference an invalid, omitted or not-this-term answer counts as not correct. Consistency covers the common pages only. **On M4's runs:** the Translations' pooled consistency was 68.8% (86 of 125 rendered occurrences) for `baseline`, 72.8% (91 of 125) for `full_no_kb` and 82.4% (103 of 125) for `full`; the references' own was 81.2%, 81.6% and 82.0% (the judge is not exactly repeatable at temperature 0); the extractor's accuracy on the references was 95.7%, 96.6% and 95.7%, so consistency is reportable. **The score is strict:** if renderings that contain one another counted as the same, it would be 80.8%, 79.2% and 91.2%. `full` still leads, and `baseline` and `full_no_kb` swap places. The rule stays strict (`IDEAS.md` has the extractor's over-long answers).

### 4.8 Reproducibility

- Every `eval/run.py` run writes a manifest: data hash, model IDs (translation model and judge model), config, seed, git commit, sacreBLEU signature, COMET checkpoint.
- Raw output and trace ID are kept for every page.
- **As built (#37):** the harness writes its outputs after every page. `finished_at` stays null until the last page is done, and pages not yet reached are `pending`, so an interrupted eval keeps the pages it finished. An interrupted eval is not continued; it is run again (`IDEAS.md`).
- **As built (#41, #42):** COMET and term accuracy write their own files next to an eval run's outputs instead of adding to the manifest. Each records what it scored: the hash of `segments.jsonl` and the model and package versions for COMET; the glossary's hash and the rules' version for term accuracy. Since #44 the COMET file also records the checkpoint's Hugging Face revision, and both scorers read `segments.jsonl` line by line on `\n` only.
- **As built (M5, #48, #49):** GEMBA-MQM and term consistency also write their own files next to an eval run's outputs (`mqm-<model>.json`, `consistency-<model>.json`; the judge's model is in the file name), each recording its inputs: the hash of `segments.jsonl`, the model, and the template's hash or the glossary's hash and the rules' version. The pooled report reads them (`--mqm-judge` picks the model's file when a run has more than one, `--mqm-agreement` adds the agreement outcome) and says in its MQM block that the scores are for reference only. The Meter writes no timestamps to the ledgers, so spend is measured by the ledgers' line ranges.

### 4.9 Known Limitations (stated in the report)

- The oracle path has no OCR errors, so the results do not fully carry over to real OCR input.
- OpenMantra's books are short, so KB gains may not show; hence term accuracy, not consistency alone.
- **There is no equal-budget review comparison** (independent review vs Self-Refine). The Reviewer Agent's value rests on design rationale, not on numbers.
- `full` vs `baseline` measures the whole system; it cannot separate what the Translator's tools and the Reviewer each contribute. Only the KB is ablated.
- GEMBA-MQM is an LLM-as-judge, checked on 30 segments against labels that an LLM made and that mark one error. Neither the small judge (GPT-4o mini, chosen for cost) nor the fallback (GPT-5.1) passed (§4.5), so its scores are reported for reference only; a fail on these labels is weak evidence about the judge.
- Term consistency is strict: a longer rendering that contains the usual one counts as a different rendering (§4.7).
- The prompts were tuned by reading the dev books, and the dev numbers are on those books, so they are not held-out estimates. Each config ran once, and the run-to-run spread was measured with one extra `baseline` pass (§8).
- Term accuracy compares the system's renderings with the reference translator's. Where both are valid (two transliterations of one name), no config can know the reference's choice, and the KB keeps the system's own choice consistent rather than moving it towards the reference. On the dev runs, three entries that no config hit even once held half of `tojime_no_siora`'s primary occurrences (`docs/m4.md`). Term consistency (M5) measures what the KB is for.
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
| Evaluation | sacreBLEU (chrF / BLEU), COMET (Unbabel, `wmt22-comet-da`), GEMBA-MQM (LLM-as-judge), paired bootstrap (Python), OpenCC (reference conversion) |
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
├── .github/workflows/     # CI: tests, lint and the frontend typecheck (#39)
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
│   │   ├── evaluation/    # eval library, importable by tests: OpenCC references, metrics (sacreBLEU, term accuracy), the judge-based scorers (GEMBA-MQM, the agreement check, term consistency), manifest, the pooled report; the bootstrap comes in M8
│   │   └── tracing/       # OpenTelemetry + OpenInference setup; the backend (Phoenix) is one exporter
│   └── tests/             # focused on KB write permissions and orchestration
├── frontend/              # React + react-konva editor
├── eval/                  # entry points and eval data; the logic lives in backend/fukidashi/evaluation/, except comet.py
│   ├── run.py             # eval harness entry point (CLI)
│   ├── terms.py           # term accuracy of a finished eval run (CLI)
│   ├── comet.py           # COMET of a finished eval run, in an environment of its own
│   ├── report.py          # pools finished, scored eval runs into one report (CLI; writes runs/report/)
│   ├── configs/           # baseline.yaml, full.yaml, full_no_kb.yaml
│   ├── gold_glossary/
│   ├── mqm.py             # GEMBA-MQM of a finished eval run, through the judge (CLI)
│   ├── mqm_sample.py      # draws the 30-item human sample blind, and checks the filled sheet (CLI)
│   ├── mqm_agreement.py   # the labels against a judge's scores (CLI; the sample's own files are in runs/mqm-human/)
│   ├── consistency.py     # term consistency of a finished eval run, through the judge (CLI)
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
| **M3** KB & Reviewer | 4 | 11/2–11/8 | 3 | KB + Curator + Proposal Queue + unit tests for KB write permissions; Reviewer Agent in the full system (at most 2 rounds, fixed); gold glossary for the dev split; `baseline` / `full` / `full_no_kb` each run on 1 page. **Done 2026-10-28**, see `docs/m3.md`. Each config ran on `tojime_no_siora` pages 2–4: `full` scored chrF 10.86 and `full_no_kb` 9.28, at about $0.011 and $0.004 per page; on 27 segments that is no comparison (two `baseline` runs of the same pages differed by 1.7 chrF). Seven tasks; about 6 hours of active sessions against 3 planned (`IDEAS.md`) |
| **M4** Dev evaluation | 5 | 11/9–11/15 | 3 | Estimate tokens and cost on 3 pages first (judge calls included) against the budget cap; set up COMET / xCOMET (Windows GPU if the Mac cannot run it) and check checkpoint licenses and memory; run all 3 configs on the dev split. **11/15: first dev numbers** (chrF / BLEU, COMET, term accuracy, cost, latency). **Cost estimate (2026-10-28):** M3's runs on 3 pages put one pass of the 3 configs over the dev split's 79 pages with text at about $1.3 ($0.07 `baseline`, $0.35 `full_no_kb`, $0.87 `full`); the judge's cost is estimated in M5, once the judge model is chosen. Before the dev runs: whole-book evals that keep the pages they finished (#37), counts of how often the KB reaches a page, a COMET scorer and term accuracy (§4.7, moved here from M5). **Built (2026-10-28):** whole-book evals (#37), KB reach (#40), the COMET scorer (#41), term accuracy (#42) and CI (#39); xCOMET does not fit the Windows GPU (§8). **Done 2026-10-29**, see `docs/m4.md`: the pooled report (#44) over one run of each config on both dev books (79 pages, 608 segments, no failed page, $1.42). COMET 0.8257 `baseline`, 0.8205 `full_no_kb`, 0.8235 `full`; chrF 20.05, 20.04, 20.21; primary term accuracy 52.1% (61/117) for all three; cost per page $0.0011, $0.0064, $0.0105; latency p50 4.5 s, 19.5 s, 31.0 s. One run each and no CIs, but no gain of `full` over `baseline` shows, at about 9.5 times the cost |
| **M5** Freeze | 6 | 11/16–11/22 | 3 | Term consistency; GEMBA-MQM + human agreement check on about 30 segments; tune prompts and tools from the MQM error breakdown; gold glossary for the test split; one-page `eval/PROTOCOL.md`. **11/22 freeze, tag `exp-frozen`**. **Decisions (2026-10-29, §8):** the judge is GPT-4o mini; term accuracy stays the KB ablation's primary metric, and term consistency is added as a secondary metric (§4.7); a second `baseline` pass measures the run-to-run spread. **Plan:** six tasks: the judge with the GEMBA-MQM scorer and the human sample; the term consistency scorer; the test split's gold glossaries; one character, one profile in the KB (`IDEAS.md`); the agreement check, the dev MQM breakdown and the second `baseline` pass; tuning and a dev pass of the tuned configs. The author annotates the human sample and reviews the test glossaries. **Cost estimate (2026-10-29):** about $2.5 for M5: the second `baseline` pass $0.09, a dev pass of the tuned `full_no_kb` and `full` about $1.3, judge calls on about 3,700 segments about $0.9, and the consistency extraction a few cents. **Done 2026-10-30** except the freeze tag, which is the author's, see `docs/m5.md`: six tasks and three docs PRs. Both judges failed the agreement check (τ_b −0.042 and 0.283; Claude annotated the sample, §4.5), so MQM is for reference only. Built: the judge and the GEMBA-MQM scorer (#48), term consistency (#49), the test glossaries of 27, 21 and 28 entries (#51), the agreement check, the dev MQM scores and the second `baseline` pass (#52), one character one profile (#47), one tuning round kept by its rule (#53) and `eval/PROTOCOL.md` (#54). Tuned dev runs (one each, on the books tuned on): `full` COMET 0.8254, chrF 20.75, primary term accuracy 73 of 117, consistency 83.2%; `full_no_kb` 0.8192, 20.18, 67 of 117, 72.2%; M4's `baseline` 0.8257, 20.05, 61 of 117, 68.8%. Spend $4.78 (translation $1.51, judge $3.26) against about $2.5 planned; about 27 hours from M4's close against 3 planned. **The author placed `exp-frozen` at `0c1fe05` on 2026-10-30** |
| **M6** Test run | 9 | 12/7–12/13 | 1 | From `exp-frozen`, run the test split in the background, each of the 3 configs once, then score with COMET. **Checkpoint 12/13**. **Plan (2026-10-30, §8):** M6 starts now, not on 12/7. First the paired bootstrap's code, written and tested on the dev runs and merged before any test page is translated (moved from M8). Then the test runs, from a worktree of `exp-frozen` (`0c1fe05`) with no request timeout: a rehearsal on dev pages that fires the launcher's stop path once on a real eval; each config once on each test book, book by book (`baseline`, `full_no_kb`, `full`), each run scored by term accuracy and COMET as it finishes; term consistency and GEMBA-MQM (GPT-4o mini) on all nine runs (moved from M8); the pooled report. No test page is run twice, and nothing is tried on a test book first. **Cost estimate (2026-10-30):** the test books have 130 pages with text (36, 40 and 54) and 984 segments. At the tuned dev runs' cost per page that is about $2.4 of translation ($0.14 `baseline`, $0.86 `full_no_kb`, $1.37 `full`) and about $0.7 of judge calls (about 3,000 MQM calls and 1,500 consistency calls), so about $3.1, which would bring the ledgers to about $9.4 of the $100 cap; about 5 hours elapsed, mostly waiting on the providers |
| **M7** Productization | 10 | 12/14–12/20 | **12** | Editor features; SSE live trace + `Last-Event-ID` reconnect; review comments and uncertain flags highlighted; approve / reject proposals, renaming a term updates and re-typesets the whole book; end-to-end visual path; overflow rate; first demo recording. **Does not touch the eval code path.** v0.2: +3 h from the simpler evaluation |
| **M8** Analysis | 12 | 12/28–1/3 | 2.5 | Paired bootstrap 95% CIs for both comparisons; GEMBA-MQM on the test split with an MQM error breakdown (by category and severity); cost / latency summary; charts. **Since v0.28 (§8):** the bootstrap's code and the test split's GEMBA-MQM scores come in M6; M8 computes the intervals once, as the protocol says, and keeps the MQM breakdown, the cost / latency summary and the charts |
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
| Provider SDK retries | M2 | **Off** (`max_retries=0`). By default the `openai` SDK silently resends a request up to twice on connection errors, timeouts, 429 and 5xx; earlier attempts that may have been billed would never reach the Meter, and cost would be a partial sum. Now a 429 or 5xx fails the page and the user reruns it. An explicit retry for 429, which is rejected and not billed, can be added if 429s are common in M4 (`IDEAS.md`). **Outcome (2026-10-29):** no 429 in the dev runs' 684 calls, so no retry for now | Decided (2026-10-26) |
| Agent loop | M2 | **LangChain v1 `create_agent`** as the loop skeleton, customised through its model, tools and middleware (the Step limit is its `ModelCallLimitMiddleware`); start here and change it if the project needs to. Its model is a small custom chat model over `LLMClient`, not `ChatOpenAI` / `ChatDeepSeek`: those bypass the Meter and retry by default, and DeepSeek rejects tool-call turns that do not send `reasoning_content` back. LangChain stays inside `agents/`, behind a plain result type, so replacing it touches nothing else. The alternative was a hand-written loop | Decided (2026-10-27) |
| Tracing tool | M2 | **Arize Phoenix, run locally** (`phoenix serve` or Docker): one process, no account, no cost, and traces kept as long as we like, so the report's trace links (§3.5) still work in January. Hosted free tiers keep traces 14 days (LangSmith) or 30 days (Langfuse Cloud), and self-hosting Langfuse needs several services. Traces stay on this machine by default; that is a choice, not a licence requirement. **Kept replaceable:** the code uses only OpenTelemetry and OpenInference, and the backend is one exporter in `tracing/`, so moving to Langfuse or another OTLP backend later changes that exporter, not the instrumentation (`IDEAS.md`). The Phoenix server is not a project dependency | Decided (2026-10-27) |
| Dev / test split | M3 | **Dev: `tojime_no_siora`, `boureisougi`** (82 pages, 608 segments). **Test: `balloon_dream`, `tencho_isoro`, `rasetugari`** (132 pages, 984 segments). `tojime_no_siora` cannot be a test book: M0–M2 translated its pages 2–4 and read the output. Of the other four books `boureisougi` is the smallest, so the test split keeps the most segments (narrower CIs) and the dev runs in M4 cost least | Decided (2026-10-28) |
| Who builds the gold glossary | M3 | **An AI coding assistant drafts it from the reference translations only, and the author reviews every entry** (§4.7). Drafting by hand would take the author about 2 hours for the two dev books. The drafting assistant (Claude) is not the translation model (DeepSeek), and it has no access to stored Runs, Translations or eval outputs | Decided (2026-10-28) |
| KB scope in evaluation | M3 | **Each eval run starts every book with an empty KB**, filled page by page in page order, so two eval runs of the same config start from the same state. **`pending_human` entries are readable**, marked as pending; rejected ones are not (§3.3). No human works the Human Review Queue during an eval, and every name goes there; if pending entries were hidden, no name would ever reach a later page and the KB ablation would measure little. The cost: a wrong pending entry can spread to later pages, which the Reviewer is there to catch | Decided (2026-10-28) |
| Curator form | M3 | **Rules in code, plus one model call per page without tools** (§3.3). Deduplication, routing conflicts and names to the Human Review Queue, and the single write at finalize are code, so they hold whatever the model answers (least privilege, §1.3). The model only judges the remaining proposals. Rejected: a tool-calling Curator, which would leave the rules to the prompt; a Curator with no model at all, which would leave two agents and change the scope (`SCOPE.md` §1) | Decided (2026-10-28) |
| Reviewer comment language | M3 | **Simplified Chinese.** The Translator reads the Issues when a page is sent back, and from M7 the editor shows them to a Chinese-reading user. The first smoke run returned English comments because the Reviewer's prompt is English; the Translator's flag reasons were already Chinese | Decided (2026-10-28) |
| COMET checkpoint | M4 | **COMET: `Unbabel/wmt22-comet-da`** (Apache-2.0, not gated; XLM-R large, about 560M parameters), run on this Mac's CPU (Apple M2, 8 GB). **xCOMET: `Unbabel/XCOMET-XL`** (CC BY-NC-SA 4.0, gated: the author accepts its terms on Hugging Face), about 3.5B parameters, so about 14 GB of weights in fp32, more than the Mac's memory; it runs on the Windows machine's GPU if it fits, otherwise the report gives COMET only (§9). Rejected: `XCOMET-XXL` (10.7B). COMET scores an eval run's outputs after the run, outside the translation path, so it can run on another machine and re-score old runs. **Outcome (2026-10-28): COMET only.** The Windows machine's GPU (RTX 4070 Ti) has 12 GB, less than XCOMET-XL's fp32 weights, and COMET's `predict` offers no lower precision. Casting the model to half precision by hand was rejected as a non-standard setup. Running it on that machine's CPU, whose 32 GB of RAM holds the weights, is in `IDEAS.md` | Decided (2026-10-28) |
| Term accuracy rules | M4 | **Built in M4, before the dev runs, not in M5.** It is the primary metric of the KB ablation, and its rules are better fixed before whole-book output has been read; M5 keeps consistency. The rules are in §4.7: the primary score covers the non-borderline entries, counts an occurrence only where the reference uses the term, and pools the occurrences; all entries and the mean over entries are reported beside it | Decided (2026-10-28) |
| Dev-run budget | M4 | **At most $3** for running the 3 configs once on both dev books (one pass is about $1.3, §7). Each config runs on its own, one book at a time, so latencies are not confounded. A run stops and is reported when more than about 10% of its pages fail or 429s repeat. **Outcome (2026-10-29):** $1.42 for the six eval runs (#44), with no failed page and no 429; the stop rule never fired | Decided (2026-10-28) |
| CI | M4 | **GitHub Actions** on pull requests and pushes to `main` (#39): `uv sync --locked` and `pytest`; `npm ci`, lint and the frontend build. Read-only permissions and no secrets, so CI can never reach a provider; the tests make no real calls anyway. A stale `uv.lock` fails it. uv is pinned to 0.11.27 and the actions to major tags (`IDEAS.md` has the follow-ups) | Decided (2026-10-28) |
| Judge model (GEMBA-MQM) | M5 | **GPT-4o mini** (OpenAI), pinned to a dated snapshot and chosen for cost: about $0.15 per 1M input tokens and $0.60 per 1M output tokens, so judging about 6,600 segments (M5's dev passes and M8's test split) costs about $1.6. It also extracts the renderings for term consistency (§4.7). It is not from DeepSeek, the translation model's provider. Like every model call it goes through `LLMClient`, with a configuration of its own (`JUDGE_*`), its own `Meter` and its own ledger: its spend is the evaluation's, not the system's, and still counts against the $100 cap. **Lifetime:** OpenAI shuts GPT-4o mini down on 2027-01-14 (announced 2026-10-14), so every judge call, M8's included, runs before then. A later change of judge reuses the human labels and repeats the agreement check (§4.5). **Risk:** a small model is the judge most likely to fail the agreement check. **Fallback, fixed before the check:** if GPT-4o mini fails, GPT-5.1 (a dated snapshot, about $1.25 input and $10 output per 1M tokens, so about $16.5 for the project; available until 2027-04-01) annotates the same sample once, under the same rule (§4.5). If it passes, it becomes the judge for every MQM score; if it fails too, the MQM breakdown is reported for reference only. No other judge is tried. Rejected: GPT-5.x at about $2–2.5 input and $12–15 output per 1M tokens (about $30); GPT-4o (about $26.5); GPT-5 and GPT-5 mini (shut down 2026-12-11, before M8); Claude, which drafted the gold glossaries; Gemini preview models, which may be retired before M8; Qwen, renamed and repriced often. The prices come from third-party listings and are checked against OpenAI's own pages when the judge is built. **Outcome (2026-10-30):** GPT-4o mini failed the agreement check (τ_b −0.042) and so did GPT-5.1 (τ_b 0.283), so MQM is for reference only (§4.5). The check ran both judges on whole eval runs, and GPT-5.1 alone cost $2.43, so M5's judge calls cost $3.26 against about $0.9 planned; the project's ledgers stand at $6.28 of the $100 cap. GPT-4o mini still extracts the renderings for term consistency, where its accuracy against the references was checked (§4.7) | Decided (2026-10-29) |
| KB ablation's primary metric | M5 | **Term accuracy stays the primary metric** (decided 2026-10-28, before any whole-book output), and **term consistency is added as a secondary metric** (§4.7). The dev runs showed why term accuracy barely moves: the KB keeps the system's own rendering consistent and cannot know the reference translator's (§4.9). Switching the primary metric after that null result would look like choosing a metric after seeing results, even though the test split is untouched. The eval protocol says that the consistency hypothesis (the KB raises term consistency) was formed after the dev runs. **Outcome (2026-10-30):** on the tuned dev runs `full` leads `full_no_kb` by 73 to 67 of 117 on primary term accuracy (61 to 61 before the tuning) and by 83.2% to 72.2% on consistency (82.4% to 72.8% before). The term accuracy gap is 6 occurrences, on the books the prompts were tuned on (`docs/m5.md`) | Decided (2026-10-29) |
| Run-to-run spread | M5 | **A second `baseline` pass on both dev books** (about $0.09, plus judge calls), scored by every metric. It measures how far two runs of one config differ on whole books; without it no dev difference can be read (`docs/m4.md`). One extra run gives one sample of the spread, not its distribution. No second `full` pass: M5's tuning replaces `full`, so a second run of the old one would measure a config that is about to change. **Outcome (2026-10-30):** the second pass differed from the first by chrF +0.55, BLEU +0.81, COMET +0.0007, primary term accuracy +1.7 pp, consistency −7.0 pp and MQM −0.24, pooled; per book by up to 0.0037 COMET and 3.7 pp of term accuracy. Only 191 of 608 segments were identical. M4's differences of `full` from `baseline` on COMET (−0.0022) and MQM (−0.38) are larger than the pooled spread; chrF, BLEU and term accuracy are within it (`docs/m5.md`) | Decided (2026-10-29) |
| Term consistency | M5 | **The share of a term's occurrences that use its most common rendering**, over the gold glossary's names and coined terms. The judge model extracts the renderings, and the extractor is checked against the converted references. The rules are in §4.7 and are fixed before any consistency number is computed. Rejected: a word aligner (Japanese and Chinese would first need word segmentation, and the aligner its own model environment); rendering lists harvested from the dev outputs and then reviewed (they would build in what was already seen). **Outcome (2026-10-30):** built (#49, §4.7); the extractor was accurate enough on the references (95.7% to 96.6%) for consistency to be reported on the dev runs | Decided (2026-10-29) |
| Tuning round and its keep rule | M5 | **One round of prompt and tool-description changes, tuned on the dev books only, and kept unless it hurts.** The evidence is the Executor's reading of the M4 `full` dev outputs against the Japanese and the page images, because the judge failed the agreement check; the MQM file only chose pages. The keep rule, fixed before the tuned runs: the tuning stays unless `full`'s pooled COMET or primary term accuracy falls, against M4's `full`, by more than the largest per-book difference between the two `baseline` passes (COMET 0.0037, term accuracy 3.7 pp). It is a regression guard: one run against one run cannot show that the tuning helped. Out of bounds: the plain `translator` prompt, `baseline`, any code logic and the KB rules; the KB-on and KB-off prompts change together, word for word, so the ablation stays fair. **Outcome (2026-10-30, #53):** kept (COMET +0.0019, primary term accuracy +10.3 pp). The term accuracy gain is all in `boureisougi`, and half of it also appears in `full_no_kb` | Decided (2026-10-29) |
| Bootstrap resamples and seed | M5 | **10,000 resamples, seed 2027,** over the pages of the three test books pooled, for the two §4.3 comparisons (`eval/PROTOCOL.md`). The numbers are arbitrary and fixed now so that nobody chooses them after seeing results | Decided (2026-10-30) |
| When the test runs start | M6 | **Now, not on 12/7.** The provider answers with the name `deepseek-flash` only (every manifest's `response_models` and every ledger line say just that), so a change of the model behind the name would not show anywhere; the prompts were tuned on DeepSeek-V4.1-Flash-1017 and the protocol pins it, so the sooner the runs happen, the likelier they meet the same model. GPT-4o mini leaves the API on 2027-01-14. The eval code path is frozen, so waiting changes nothing. The 12/13 checkpoint stays | Decided (2026-10-30) |
| Request timeout for the test runs | M6 | **None; the tag stays at `0c1fe05`.** A request that times out is recorded and not resent (§3.6), so a timeout cannot save a page, only waiting time, and it could turn a slow but valid call into a failed page. Without one, a hung call ends when the provider closes the connection (in M5 after about 17 and 10 minutes); its page fails and is listed, and the eval run goes on. Moving the tag after the protocol was written would weaken the freeze for minutes of waiting. **So nothing stops an eval run for silence alone:** a stopped eval run is not continued (`eval/PROTOCOL.md`), which would lose the rest of that book for that config. The launcher warns after 20 minutes without a finished page or a new ledger line; a person stops the run only after 45 minutes without either, and reports it as stopped. Rejected: a timeout in `LLMClient`, with the tag moved to the commit that adds it (`IDEAS.md` keeps the timeout for after the test runs) | Decided (2026-10-30) |
| Judge scoring of the test runs | M6 | **In M6, right after the translations,** not in M8: term consistency and GEMBA-MQM (GPT-4o mini) on each of the nine test eval runs, then the pooled report. About $0.7; it removes the risk of the 2027-01-14 shutdown. M8 keeps the bootstrap's computation, the MQM breakdown, the cost / latency summary and the charts | Decided (2026-10-30) |
| Bootstrap code before the test runs | M6 | **The paired bootstrap's code is written, tested on the dev runs and merged before the first test page is translated** (moved from M8), and computed on the test runs once, in M8, as the protocol says. The protocol fixes the resamples, the seed and the pairing, but not every detail of the implementation (each pair's common pages, how term accuracy and consistency are recomputed when a page is drawn twice); written before any test number exists, those details cannot follow a result. The code is new and reads files only, so nothing at the tag changes. **MQM gets an interval too, as reference only:** the protocol gives one to every quality metric of the two comparisons, and MQM is a secondary metric of `full` vs `baseline`, although the protocol's list of what is recomputed on each resample leaves it out. So the intervals are: for `full` vs `baseline`, COMET, chrF, BLEU, term accuracy (the primary score), term consistency and MQM; for `full` vs `full_no_kb`, term accuracy (the primary score) and term consistency | Decided (2026-10-30) |
| Label of a term consistency interval | M6 | **Decided per run, before any test page.** A run here is one config on one book. A comparison's consistency interval is a result only when every run of its two configs that has a page in the comparison passes the report's extractor rule (accuracy on the converted references of at least 90%), counted over the comparison's pages; otherwise the interval is for reference only. This takes `eval/PROTOCOL.md`'s "for a run" literally and is the stricter reading (a config that fails on its pooled counts has a run that fails); it is how #58 built it. The pooled report's own consistency line decides on a config's counts pooled over its books, so on the test runs one book with few checked occurrences can make the interval "for reference only" while the report's pooled line is reportable. M8's write-up states this rule for the intervals and shows each run's extractor accuracy (`label.extractor` in `bootstrap.json`) beside the pooled one. On the dev runs the two readings agree (92.1% to 100% per run) | Decided (2026-10-30) |

---

## 9. Risks & Pivot Rules

| Risk | Mitigation |
|---|---|
| Sliding into over-engineering again (PanelFlow is the cautionary tale) | Vertical slice first; `SCOPE.md` fixes the scope; the 10/25 checkpoint only asks whether the browser shows a real translation |
| The editor eats the schedule | Product features are concentrated in M7 (12 h since v0.2); when time runs out, cut from the bottom of `SCOPE.md` |
| Books are too short for KB gains to show | Use term accuracy, not consistency alone; report honestly; long books are discussed in the report only. **First sign (2026-10-28):** in the first `full` run (3 pages) the KB's snippets matched nothing on any page, and its one cross-page use came through a lookup. M4 measures how often the KB reaches a page on whole dev books before the ablation is read; the eval harness counts it since #40 (§3.3). **Dev runs (2026-10-29):** the KB reached 49 of 79 pages, but `full` and `full_no_kb` have the same primary term accuracy (61/117). Most misses are valid renderings that differ from the reference's, which the KB keeps consistent rather than corrects (§4.9, `docs/m4.md`). **Settled (2026-10-29):** term accuracy stays the primary metric, and term consistency is added as a secondary metric (§4.7, §8). **After the tuning (2026-10-30):** the KB reached 38 of 79 pages (49 before); `full` led `full_no_kb` on consistency (83.2% against 72.2%) and by 6 occurrences on term accuracy (`docs/m5.md`) |
| Model cost exceeds the budget | $100 cap; `runs/usage.jsonl` logs tokens and cost for every call (cost needs prices in `.env`); estimate on 3 pages first, judge calls included; tune on a subset of dev pages if costs run high. **M5 (2026-10-30):** $4.78 (translation $1.51, judge $3.26) against about $2.5 planned, because the agreement check ran both judges on whole eval runs; the project's ledgers stand at $6.28 of $100 |
| GEMBA-MQM disagrees with humans | Per the eval protocol's decision rule, report it for reference only and say so. **Happened (2026-10-30):** neither GPT-4o mini nor GPT-5.1 passed (§4.5), so MQM is for reference only |
| COMET / xCOMET does not run on the Mac | Use the Windows machine's GPU; if xCOMET does not fit in memory, report COMET only. **Outcome (2026-10-28):** COMET runs on the Mac's CPU (#41); XCOMET-XL does not fit the Windows GPU, so the report gives COMET only (§8) |
| The provider changes the model behind `deepseek-flash` between the tuning and the test runs | Responses name only `deepseek-flash`, so a change would not show in a manifest or the ledger. Run the test split soon after the freeze (§8, decided 2026-10-30), and state in the report that the snapshot could not be checked from the responses |
| `deepseek-flash`'s vision is new (2026-10-17 release) and its tool calling is untested | Read translations by hand in M0 and M1; test the tool loop in M2; if it falls short, switch providers in `.env` without code changes. **Tool calling answered (2026-10-27):** three smoke runs through the agent loop with thinking on all called the tool and used its result (#16) |

**Pivot rules (decided only at checkpoints):**
- **10/25:** no real translation in the browser → drop the frontend for now, keep only the CLI path, and move all editor work to M7. The project itself does not change.
- **12/13:** if only one project has first numbers, the whole winter break goes to it.
- Any other "should we change direction" idea goes into `IDEAS.md` until the next checkpoint.

---

## 10. Development Workflow

- Each task has a goal and acceptance criteria → AI implements it → **one task, one PR** → the author reviews the diff. Details in `docs/workflow.md`.
- Each executor works in a git worktree of its own, so tasks with no dependency between them can run side by side without sharing a branch (since M3).
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
| How do you show the system really got better, not just that the demo feels smoother? | Standard metrics (COMET, chrF, GEMBA-MQM, term accuracy) against a baseline, with paired bootstrap 95% CIs by page; the eval protocol is written before the freeze and the test split is run once |
| Why code orchestration instead of letting an agent decide every step? | The overall flow is fixed, and ordinary code is controllable, evaluable and lets the budget be allocated precisely. Autonomy is kept for the uncertain parts (Anthropic, "Building effective agents") |
| Why does the Reviewer get an independent context? Is there data? | Design rationale: self-checks tend to justify themselves, and the Reviewer cannot see the Translator's reasoning. There is no equal-budget comparison (Known Limitations); validating it would take an equal-budget Self-Refine baseline and a paired bootstrap |
| What happens when several agents write the same shared state? | Single-writer + Proposal Queue; proposals are deduplicated by ID; names and conflicts go to the Human Review Queue |
| Does the KB help? Why no vector database? | The KB ablation's term accuracy difference and its CI; terms and names are identifiers, where exact match fits better |
| Can an LLM-as-judge be trusted? | A human agreement check on about 30 annotated segments; the judge model differs from the translation model; with too little agreement it is reported for reference only |
| Do chrF / BLEU say whether a manga translation is good? | Not on their own, hence COMET and GEMBA-MQM; the single-reference limitation is in the report |
| Why not write our own tracing? | Langfuse and Phoenix are mature; metering stays in-house because the definition of unknown usage has to be ours |
| What if SSE drops or the server restarts? Could the KB be written twice? | `seq` + `Last-Event-ID`; the page is the retry unit; the KB is written once, at finalize |
| Why rewrite PanelFlow? | The order was wrong: infrastructure came before a single real model call. Lessons: vertical slice first, scope written down, tests only where things break |
| How much of the code is AI-written, and how is quality kept? | See §10 and `docs/workflow.md` |

---

## 13. Definition of Done (2027-02-15)

- [ ] Public repository; a working editor + demo recording
- [ ] `baseline` / `full` / `full_no_kb` results on the test split: chrF / BLEU (with signature), COMET, GEMBA-MQM score and error breakdown, term accuracy / consistency, cost per page, latency (p50 / p95)
- [ ] Paired bootstrap 95% CIs for both comparisons; GEMBA-MQM human agreement results; report (including Known Limitations)
- [ ] Blog posts in Chinese and English
- [ ] Someone new to the project can reproduce the main table from the README
