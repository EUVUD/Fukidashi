# Fukidashi（暂名）— Project Outline

> v0.2 · 2026-10-22 · 来源：Artifact「Agent 实习准备计划」v10 第 2 节、第 4 节逐周计划、5.1 / 5.7 / 7 节。
> **v0.2 改动：** 评测方案从 artifact v10 的「研究型对照实验」改为「标准 evaluation framework + baseline 对比」（方案 B），见第 4 节；评测相关的部分以本文件为准。
> **v0.3 改动（2026-10-22）：** 记录 M0 的决定（第 8 节）：项目名 Fukidashi，翻译方向 ja → zh-Hans，OpenAI-compatible provider，费用上限 $100。
> **v0.4 改动（2026-10-24）：** M1 新增术语 Run（第 0 节）。
> **v0.5 改动（2026-10-24）：** 第 0 节新增 Translation、Reference translation（`docs/design.md` 用到）。
> 写法约定：正文用中文，**专业术语一律用英文**，代码、文档、commit message 和报告统一使用第 0 节的 canonical term。

---

## 0. Terminology（术语对照）

代码和文档里只用左列的英文写法，不要混用中文或别的同义词。

| Canonical term | 中文含义 | 备注 |
|---|---|---|
| Vertical slice | 竖切 | 先打通「数据 → 模型 → 浏览器」整条链路，再加任何 infra |
| Translator Agent | 翻译 agent | 唯一会自主调用工具的 agent |
| Reviewer Agent | 独立审校 agent | independent context，read-only，最多打回 2 轮 |
| Curator Agent | 知识管理 agent | Knowledge Base 唯一的 agent writer |
| Orchestrator | 编排层 | 普通代码，不是 LLM；权限在这一层强制 |
| Knowledge Base (KB) | 故事知识库 | 包含 Glossary 和 Character Profiles |
| Glossary / Term entry | 术语表 / 词条 | |
| Character Profile | 角色档案 | |
| Proposal Queue | 提议队列 | Translator 只能 propose，Curator 负责 merge |
| Human Review Queue | 人工待确认 | 人名和冲突条目进这里 |
| Single-writer | 单一写者 | 共享状态的并发控制策略 |
| Provenance / History | 来源 / 修改历史 | 每个 KB entry 都带 evidence pages |
| Self-Refine | 自我修改 | v0.2 起不再作为 config，只在 Known Limitations 里提到 |
| Oracle boxes / Oracle path | 标注框 / oracle 路径 | 直接用 OpenMantra 的 bounding box 和原文，跳过 detection 和 OCR |
| Text Detection | 文本检测 | |
| OCR | 文字识别 | manga-ocr |
| Inpainting | 擦字 | OpenCV，LaMa 可选 |
| Typesetting | 排版 / 嵌字 | 自己写 |
| Overflow rate | 嵌字溢出率 | 产品侧指标 |
| Token budget (per page) | 每页 token 预算 | |
| Step limit | 步数上限 | |
| Metering | 计量 | 自己写；unknown usage 单独计数，不记为 0 |
| Trace | 执行轨迹 | v0.2 起由 tracing tool 记录，不再自写 JSONL |
| Tracing tool | tracing 工具 | Langfuse 或 Arize Phoenix，M2 选定（v0.2 新增） |
| Manifest | 运行清单 | data hash、model ID、config、seed、git commit |
| Evaluation framework | 评测框架 | 第 4.4 节列出的一组标准 metric 和工具（v0.2 新增） |
| Eval harness | 评测执行器 | `eval/run.py`：跑 config、打分、写 manifest（v0.2 新增） |
| Eval protocol | 评测规程 | 一页的 `eval/PROTOCOL.md`，冻结前写好；替代原来的 `PREREG.md`（v0.2 新增） |
| Baseline (`baseline`) | 基线 | 原 C0：整页 single call，不用 tools 和 KB（v0.2 新增） |
| Full system (`full`) | 完整系统 | Translator + Reviewer + Curator / KB（v0.2 新增） |
| KB ablation (`full_no_kb`) | KB 消融 | 和 `full` 完全相同，只关掉 KB（v0.2 新增） |
| Primary / Secondary metric | 主指标 / 次要指标 | primary metric 冻结前写进 eval protocol（v0.2 新增） |
| Segment | 评测单元 | OpenMantra 的一个文本区（一个 box）的原文和译文（v0.2 新增） |
| zh-Hans / zh-Hant | 简体中文 / 繁体中文 | 系统输出 zh-Hans；OpenMantra 的参考译文是 zh-Hant（M0 新增） |
| OpenCC | 繁简转换工具 | 评测前把 zh-Hant 参考译文转成 zh-Hans（M0 新增） |
| Dev split / Test split | 验证集 / 测试集 | 按作品划分：2 部 dev，3 部 test |
| Gold glossary | 标准术语表 | 从 reference translation 整理，在看模型输出之前做 |
| Term accuracy / Term consistency | 术语准确率 / 一致性 | KB ablation 的 primary metric |
| sacreBLEU signature | sacreBLEU 签名 | 记录 tokenizer、smoothing 等设置，报告里附上（v0.2 新增） |
| Neural metric | 神经网络评测指标 | 本项目指 COMET / xCOMET（v0.2 新增） |
| COMET / xCOMET | — | Unbabel 的 neural metric；xCOMET 还会标出 error span（v0.2 新增） |
| LLM-as-judge | 用 LLM 当评委 | （v0.2 新增） |
| Judge model | 评委模型 | 和 translation model 不同，减少 self-preference bias（v0.2 新增） |
| MQM (Multidimensional Quality Metrics) | 多维质量指标 | 一套按 error category 和 severity 标注翻译错误的框架（v0.2 新增） |
| GEMBA-MQM | — | 用 LLM-as-judge 做 MQM 标注的方法（v0.2 新增） |
| Error category / Severity | 错误类别 / 严重度 | MQM 的两个维度；severity 分 minor / major / critical（v0.2 新增） |
| MQM score / Error breakdown | MQM 分数 / 错误分布 | 替代原来自定义的 failure taxonomy（v0.2 新增） |
| Human agreement check | 人工一致性检查 | 用约 30 个人工标注的 segment 检查 GEMBA-MQM 是否可信（v0.2 新增） |
| Latency (p50 / p95) | 延迟 | 每页 end-to-end wall-clock（v0.2 新增） |
| Freeze (`exp-frozen`) | 冻结 | 打 git tag 之后不再改 eval code path |
| Paired bootstrap | 配对 bootstrap | 按页配对，报 95% CI |
| Confidence Interval (CI) | 置信区间 | |
| Adapter | 适配器 | detection、OCR、inpainting 可以替换 |
| Checkpoint | 检查点 | 只在 checkpoint 考虑转向 |
| Translation | 译文 | 一个 Run 给一个 Segment 的 zh-Hans 输出，存在 `translations` 表；同一个 Segment 会随多次 Run 有多条 Translation（v0.5 新增） |
| Reference translation | 参考译文 | OpenMantra 里每个 Segment 的 zh-Hant 和 en 人工译文，只用于对照和评测，不进 prompt（v0.5 新增） |
| Run | 一次运行 | 一个 config 在一页上的一次执行，存在 `runs` 表（config、model、usage、cost、status、时间）；页面显示最新一次 succeeded 的 run 的译文。和 Manifest 不同：Manifest 记录一整批 eval 运行（M1 新增） |

**v0.2 删除的术语：** Cost parity、Preregistration（改为 Eval protocol）、Failure taxonomy（改为 MQM error breakdown）、Blind evaluation、Reviewer false positive。

---

## 1. Overview

### 1.1 One-liner

一个 **Web-based manga translation editor**：Translator Agent 按需查 Glossary 和 Character Profiles，Reviewer Agent 在 independent context 里挑错，Curator Agent 一边翻一边维护 Knowledge Base。在 OpenMantra 上用标准的 evaluation framework（sacreBLEU、COMET / xCOMET、GEMBA-MQM、term accuracy）和 baseline 做对比。

### 1.2 Evaluation Goal

Fukidashi 是一个产品，用通用、客观的 evaluation framework 来评测，不再围绕研究问题设计实验。

- **主对比：** `full` vs `baseline`。整个 system 比整页 single call 在质量上好多少，cost 和 latency 又多付了多少。
- **KB ablation：** 在同一个 eval harness 里比较 `full` vs `full_no_kb`，衡量 KB 对 term accuracy 和 term consistency 的贡献。

**结论无论正负都可以写进报告**，前提是数字和 CI 是真的。

### 1.3 Design Principles

1. **Vertical slice first：** 10/25 前浏览器里必须有一页真实的模型译文；任何 infra 只在需要它的那一周才加。
2. **Code orchestration, agent autonomy only where uncertain：** 大流程（translate → review → finalize）由代码编排；只有「查什么、查几次」由 agent 自主决定。
3. **Least privilege：** Reviewer 只读；只有 Curator 能写 KB；权限在 Orchestrator 层强制，不靠 prompt 约束。
4. **Same code path：** 编辑器和评测共用一条代码路径，人工修改会回流进 KB。
5. **Scope is written down：** `SCOPE.md` 写死范围，超时按 `SCOPE.md` 从下往上砍。
6. **从 PanelFlow 只带走设计，不搬代码：** dev/test 按作品划分、paired bootstrap、unknown usage 不记 0、每页 token budget。
7. **Standard over custom：** metric 和 tracing 都用社区通用的工具（sacreBLEU、COMET、GEMBA-MQM、Langfuse / Phoenix）；只在通用工具覆盖不到的地方自己写（term accuracy、metering）。

---

## 2. Scope

> 范围以 [SCOPE.md](SCOPE.md) 为准，下表只是摘要。

| In scope | Out of scope |
|---|---|
| 日→简体中文（zh-Hans）单方向翻译（M0 决定） | 多语言方向、繁体输出、用户系统、多人实时协作 |
| Oracle path（评测用）+ end-to-end visual path（demo 用） | 自训 detection / OCR / inpainting 模型 |
| 3 个 agent + code orchestration | 通用 agent framework、自己的 workflow DSL |
| SQLite KB + Proposal Queue | Vector database（RAG 不是本项目重点） |
| Evaluation framework + `baseline` vs `full` + KB ablation，dev/test split | 单独评测 Reviewer（independent review vs Self-Refine）；长篇作品上的 KB 收益（只在报告里讨论） |
| Tracing 用 Langfuse 或 Arize Phoenix | 自写 trace 存储和 dashboard |
| Canvas editor：拖动、缩放、新建框，改字、字号、横 / 竖排，simple undo | 完整的排版软件功能、字体管理 |
| SSE live trace + `Last-Event-ID` 断线重连 | WebSocket、分布式任务队列 |
| 一个最小 MCP server（只暴露 `lookup_term`） | 把整个系统做成 MCP 服务 |

**Cut order（超时从下往上砍）：** 见 `SCOPE.md` 第 3 节，12 项 product feature，最先砍的在最下面。Eval path 和 vertical slice 列在 `SCOPE.md` 第 2 节 Never cut，不参与 cut order。

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
                     eval/run.py ──▶ 同一个 Orchestrator（oracle path）──▶ manifest + scores + trace IDs
                     tracing ──▶ Langfuse 或 Arize Phoenix
```

### 3.2 Agents, Tools, Permissions

| Agent | Context | Tools（名称为 proposal，M2 定稿） | KB 权限 |
|---|---|---|---|
| Translator Agent | 页面图片 + 原文 + 相关 KB 片段 | `lookup_term`、`lookup_character`、`get_previous_context`、`flag_uncertain`、`propose_entry` | propose only |
| Reviewer Agent | **independent context**：只看原文、译文和 KB，看不到 Translator 的推理 | `lookup_term`、`lookup_character`（read-only） | read |
| Curator Agent | Proposal Queue + 现有 KB | merge / reject proposal；人名和冲突转 Human Review Queue | **sole agent writer** |
| Human（编辑器用户） | 编辑器 UI | approve / reject proposal，改译名后全书同步 | write（通过 Curator 同一条路径） |

**Review loop：** Reviewer 最多打回 2 轮。这是固定值，v0.2 起不再为了 cost parity 调参。

### 3.3 Knowledge Base

- **Entities：**
  - `GlossaryEntry`：`id`、`source_term`、`target_term`、`category`、`status`（proposed / approved / rejected / pending_human）、`provenance`（agent / human）、`evidence_pages`、`history`
  - `CharacterProfile`：`id`、`name_src`、`name_tgt`、`aliases`、`notes`（说话风格等）、`evidence_pages`、`history`
  - `Proposal`：`proposal_id`、`kind`、`payload`、`proposer`、`page_id`、`status`
- **Write policy：** single-writer + Proposal Queue。proposal 带 ID，Curator 合并时按 ID 去重，重试不会重复写入。
- **Conflict handling：** 两个 proposal 对同一个 source term 给出不同译名，或者涉及人名 → 进 Human Review Queue，不自动合并。
- **Retrieval：** 按原文匹配相关 term 和 character，按 character budget 截断，其余由 agent 通过工具按需查询。不用 vector database，理由写进报告：term 和人名都是 identifier，exact match 比 embedding 更合适。

### 3.4 Agent Loop Guardrails

- Step limit；per-page token budget；duplicate tool call detection；tool error 原样回传给 agent。
- Structured output：先做 JSON schema 校验，解析失败重试一次，仍失败就 **visible fallback** 到 `baseline`（整页 single call）。
- Prompt layout 按 prompt caching 排好稳定前缀：`system → tool definitions → glossary → page`。
- **DeepSeek 的两条规则：** 带 tools 时，每一轮返回的 `reasoning_content` 必须在后续请求里原样传回（M2 实现 tool loop 时处理）；JSON output 偶尔返回空 content，按上面的规则重试一次。thinking 的开关在 `extra_body` 里，effort 用 `reasoning_effort`（`low` / `high` / `max`）。

### 3.5 Metering & Tracing

- **Metering 自己写：** 每次 LLM call 记录 model ID、input / output / cached tokens、cost；provider 不返回 usage 时记为 **unknown**，单独计数，不记为 0。Cost per page 以这份数据为准。
- **Tracing 用现成的 tracing tool：** Langfuse 或 Arize Phoenix（M2 选定），不再自写 trace JSONL。每页一个 trace，agent step、tool call、review round 各是一个 span；eval 结果里存 trace ID，可以从分数直接跳到对应的 trace。
- **选型提示：** Phoenix 可以在本地作为一个 Python 进程运行，对 8 GB 的 Mac 比较轻；Langfuse 自托管需要的组件更多，用 Langfuse Cloud 的话 trace 会传到第三方服务。
- 编辑器里的 live agent trace（SSE）是**产品功能**，和 tracing tool 分开：前者给用户看，后者给开发和评测用。

### 3.6 Reliability

- **Page 是 retry unit**；任务状态持久化到 SQLite。
- LLM call 没有 side effect，可以安全重试；有 side effect 的 KB write 只在 finalize 时由 Curator 做一次。
- 结果未知（例如超时但可能已经扣费）时不自动重发，交给用户决定。
- SSE event 带递增 `seq`；客户端断线后用 `Last-Event-ID` 重连，服务端补发缺失的 event。

### 3.7 Vision Pipeline

| Stage | Evaluation（oracle path） | Demo（end-to-end path） |
|---|---|---|
| Text Detection | OpenMantra bounding boxes | comic-text-detector（GPL-3.0，只做 optional adapter） |
| OCR | OpenMantra 原文 | manga-ocr（Apache-2.0，CPU 可跑） |
| Translation | Orchestrator + agents | 同左 |
| Inpainting | — | OpenCV；LaMa 可选 |
| Typesetting | — | 自己写，记录 overflow rate |
| Export | — | PNG |

所有 vision 组件都包成可替换的 **adapter**。

---

## 4. Evaluation

> v0.2：Fukidashi 按产品来评测，用一套标准的 evaluation framework 和一个 baseline 做对比。

### 4.1 Data

- **OpenMantra**：日文原文 + 专业英文、中文译文，带 bounding box；**CC BY-NC 4.0**。字段和统计见 `docs/openmantra.md`。
- **参考译文是 zh-Hant：** 评测前用 OpenCC 把 `text_zh` 转成 zh-Hans，gold glossary 也从转换后的参考整理。具体的 OpenCC 配置（`t2s` 或 `tw2sp`）在 M2 定。
- Split 按作品划分：**dev = 2 部**（调 prompt 和 tools），**test = 3 部**（冻结后只跑一次）。
- Metric 在 segment 上计算；paired bootstrap 按 page 重采样。
- 公开仓库不包含 OpenMantra 图片，只放 eval 脚本、gold glossary 和 attribution。

### 4.2 Configurations

| Config | 内容 |
|---|---|
| `baseline` | 原 C0：整页 single call，不用 tools，不用 KB |
| `full` | Full system：Translator Agent（tools）+ Reviewer Agent（最多 2 轮）+ Curator Agent / KB |
| `full_no_kb` | KB ablation：和 `full` 完全相同，只关掉 KB |

**KB off 的定义：** prompt 里不注入 KB 片段；关闭 `lookup_term`、`lookup_character` 和 `propose_entry`；Curator 不运行；Reviewer 也看不到 KB。Translator、Reviewer、`get_previous_context` 和 token budget 都和 `full` 一样。

**v0.2 删除：** C1、C2（Self-Refine）、C3 不再作为独立的 config；删除 `mem_window3`、cost parity 调参，以及每个 config 跑 2 次的做法。

### 4.3 Comparisons

| Comparison | Primary metric | 回答什么 |
|---|---|---|
| `full` vs `baseline` | COMET（xCOMET 跑得动就一起报） | 整个 system 比整页 single call 好多少；cost 和 latency 多付多少 |
| `full` vs `full_no_kb` | Term accuracy | KB 的贡献有多大 |

两组对比里的所有 quality metric 都报 paired bootstrap 95% CI。Primary metric 在冻结前写进 eval protocol，避免看到结果后再挑指标。

### 4.4 Evaluation Framework

| Metric | 工具 | 用法 | 备注 |
|---|---|---|---|
| **chrF / BLEU** | sacreBLEU | corpus-level；报告里附 **sacreBLEU signature** | 目标语为中文时，BLEU 用 `tokenize=zh`；chrF 不依赖分词 |
| **COMET / xCOMET** | Unbabel COMET | 主要的 neural metric，reference-based；**必做** | Mac 上跑不动就用 Windows 机器的 GPU；具体 checkpoint（例如 `Unbabel/wmt22-comet-da`、`Unbabel/XCOMET-XL`）在 M4 核对 license 和显存需求 |
| **GEMBA-MQM** | LLM-as-judge | 每个 segment 标出 error span、error category 和 severity（minor / major / critical），汇总成 MQM score 和 error breakdown；**替代自定义的 failure taxonomy** | judge model 和 translation model 不同；使用前先做 human agreement check（4.5） |
| **Term accuracy / consistency** | 自己写，对照 gold glossary | 衡量 KB | KB ablation 的 primary metric |
| **Cost per page** | 自己的 metering | system 自身所有 LLM call 的成本；unknown usage 单独计数 | 评测本身的成本（judge 调用、COMET 的 GPU 时间）单独记，不算进 system 的 cost |
| **Latency** | 自己计时 | 每页 end-to-end wall-clock，报 p50 / p95 | |
| Overflow rate | typesetter | 产品侧指标 | 不进 4.3 的对比 |

### 4.5 GEMBA-MQM Human Agreement Check

- 从 dev split 抽约 30 个 segment（`baseline` 和 `full` 的输出都要覆盖），按同一份 MQM guideline 人工标注 error category 和 severity。**人工标注在看 GEMBA-MQM 的结果之前完成。**
- 在同样的 segment 上跑 GEMBA-MQM，比较两件事：segment-level MQM score 的相关性（例如 Kendall τ），以及人工标出的 major / critical 错误有没有被抓到。
- 判定规则在冻结前写进 eval protocol：一致性够，MQM error breakdown 就作为报告的主要 error analysis；不够，就只作参考，并在报告里写明。

### 4.6 Rigor & Statistics

- 所有 config 用同一个 translation model、同样的输入。
- 只在 dev split 上调 prompt 和 tools；test split 上每个 config **只跑一次**。
- 冻结前写好一页 `eval/PROTOCOL.md`：metric（标明 primary）、split、test 只跑一次、exclusion rules、GEMBA-MQM 的判定规则。写完打 tag `exp-frozen`。
- **统计：** 按页 paired bootstrap 95% CI，只用于 4.3 的两组对比。这部分 Python 代码和 Triage Agent 项目共用。

### 4.7 Gold Glossary Protocol

- dev split 每部作品 20–40 条（M3），test split 三部作品各做一份（M5）。
- 从 reference translation 整理，**在看任何模型输出之前完成**。

### 4.8 Reproducibility

- `eval/run.py` 每次运行写一份 manifest：data hash、model ID（translation model 和 judge model）、config、seed、git commit、sacreBLEU signature、COMET checkpoint。
- 每页保留 raw output 和 trace ID。

### 4.9 Known Limitations（报告里写明）

- Oracle path 不包含 OCR 误差，结论不能完全推广到真实的 OCR 场景。
- OpenMantra 每部作品较短，KB 的收益可能测不出来，所以用 term accuracy，而不只看 consistency。
- **没有做同预算的 review 对比**（independent review vs Self-Refine）。Reviewer Agent 的价值，面试时只能讲设计理由，拿不出数字。
- `full` vs `baseline` 衡量的是整个 system 的效果，拆不出 Translator tools 和 Reviewer 各自贡献多少；只有 KB 做了 ablation。
- GEMBA-MQM 是 LLM-as-judge，只在约 30 个 segment 上检查过和人工判断的一致性。
- Reference-based metric 依赖 OpenMantra 唯一的一份参考译文，而漫画翻译的意译空间大，分数会偏保守。
- 参考译文由 OpenCC 从繁体转成简体。字形基本一一对应，但台湾和大陆的用词差异转不干净，reference-based metric 会因此略微偏低。

---

## 5. Tech Stack

| Layer | Choice |
|---|---|
| Frontend | React + TypeScript + react-konva |
| Backend | FastAPI + SQLite + SSE |
| Runtime | Python 3.12，用 uv 管理依赖 |
| LLM | OpenAI-compatible API，用 `openai` Python SDK 调用；provider 和 `LLM_MODEL` 在 `.env` 里配置（模板见 `.env.example`）；要求支持 vision、tool calling 和 JSON output。当前用 DeepSeek `deepseek-flash` |
| Vision | manga-ocr、comic-text-detector（optional）、OpenCV、LaMa（optional） |
| Evaluation | sacreBLEU（chrF / BLEU）、COMET / xCOMET（Unbabel）、GEMBA-MQM（LLM-as-judge）、paired bootstrap（Python）、OpenCC（参考译文繁转简） |
| Observability | Langfuse 或 Arize Phoenix（tracing）；metering 自己写 |
| Protocol demo | 最小 MCP server，只暴露 `lookup_term`（第 5 周 A 线） |

---

## 6. Repository Layout（proposal）

```
fukidashi/
├── README.md              # 包括 AI vs human authorship、怎么审代码
├── SCOPE.md
├── OUTLINE.md             # 本文件
├── IDEAS.md               # checkpoint 之外冒出来的想法先记在这里
├── .env.example           # LLM provider 配置模板（.env 不进仓库）
├── docs/                  # 数据笔记，例如 openmantra.md
├── scripts/               # 一次性脚本，例如 m0_first_call.py
├── backend/
│   ├── app/
│   │   ├── api/           # REST + SSE endpoints
│   │   ├── datasets/      # OpenMantra loader
│   │   ├── orchestrator/  # translate → review → finalize
│   │   ├── agents/        # translator.py, reviewer.py, curator.py
│   │   ├── tools/         # lookup_term, lookup_character, ...
│   │   ├── kb/            # Glossary, Character Profiles, Proposal Queue
│   │   ├── llm/           # client, metering, token budget
│   │   ├── vision/        # detector / ocr / inpainter / typesetter adapters
│   │   └── tracing/       # Langfuse / Phoenix integration
│   └── tests/             # 重点测 KB write permission 和 orchestration
├── frontend/              # React + react-konva editor
├── eval/
│   ├── run.py             # eval harness 入口
│   ├── configs/           # baseline.yaml, full.yaml, full_no_kb.yaml
│   ├── metrics/           # sacrebleu, comet, gemba_mqm, term_accuracy
│   ├── gold_glossary/
│   ├── mqm_human/         # 约 30 个 segment 的人工 MQM 标注
│   ├── analysis/          # bootstrap, MQM error breakdown, plots
│   └── PROTOCOL.md        # 一页 eval protocol，冻结前写好
├── mcp/                   # lookup_term MCP server
├── runs/                  # 运行输出和 usage.jsonl 费用账本（gitignored）
└── data/                  # OpenMantra（gitignored）
```

---

## 7. Milestones（共约 38.5 h）

| Milestone | Week | Dates | Hours | Deliverables |
|---|---|---|---|---|
| **M0** Decisions & first call | 1 | 10/16–10/18 | 2.5 | 4 个 decisions；`SCOPE.md`；下载并核对 OpenMantra（`annotation.json` 的字段、作品标题写法、页数、文本区数）；第一条真实 model output（记录 model ID 和 token usage）。**2026-10-24 完成**，见 `docs/m0.md` |
| **M1** Vertical slice | 2 | 10/19–10/25 | 3.5 | M1-a：FastAPI + SQLite + OpenMantra import（oracle boxes + 原文）；M1-b：C0 → 前端显示页面图片、框和译文列表。**Checkpoint 10/25：浏览器里有一页真实译文** |
| **M2** Agent layer & eval harness | 3 | 10/26–11/1 | 3.5 | Tool-calling loop + guardrails；Translator Agent；metering；接入 tracing tool（M2 选定 Langfuse 或 Phoenix）；eval harness：`eval/run.py` + manifest + sacreBLEU（chrF / BLEU，附 signature）；`baseline` 在 3 页上出分数，Translator Agent 在同样 3 页上跑通（smoke test，不是报告里的 config） |
| **M3** KB & Reviewer | 4 | 11/2–11/8 | 3 | KB + Curator + Proposal Queue + KB write permission 的 unit tests；把 Reviewer Agent 接进 full system（最多 2 轮，固定值）；dev split 的 gold glossary；`baseline` / `full` / `full_no_kb` 各跑通 1 页 |
| **M4** Dev evaluation | 5 | 11/9–11/15 | 3 | 先用 3 页估算 token 和费用（包括 judge 调用），对照 budget cap；装好 COMET / xCOMET（Mac 跑不动就用 Windows GPU），核对 checkpoint 的 license 和显存；在 dev split 上全量跑 3 个 config。**11/15：dev split 第一版数字**（chrF / BLEU、COMET、term accuracy、cost、latency） |
| **M5** Freeze | 6 | 11/16–11/22 | 3 | Term accuracy / consistency；GEMBA-MQM + 约 30 个 segment 的 human agreement check；按 MQM error breakdown 调 prompt 和 tools；test split 的 gold glossary；一页 `eval/PROTOCOL.md`。**11/22 freeze，打 tag `exp-frozen`** |
| **M6** Test run | 9 | 12/7–12/13 | 1 | 从 `exp-frozen` 后台跑 test split，3 个 config 各跑一次，跑完用 COMET 打分。**Checkpoint 12/13** |
| **M7** Productization | 10 | 12/14–12/20 | **12** | 编辑器协作功能；SSE live trace + `Last-Event-ID` reconnect；review comments 和 uncertain flags 高亮；proposal approve / reject，改译名后全书同步并重新 typesetting；end-to-end visual path；overflow rate；demo 录屏草稿。**不碰 eval code path**。v0.2：+3 h，来自评测简化省下的时间 |
| **M8** Analysis | 12 | 12/28–1/3 | 2.5 | 两组对比的 paired bootstrap 95% CI；在 test split 上跑 GEMBA-MQM，出 MQM error breakdown（按 category 和 severity）；cost / latency 汇总；图表 |
| **M9** Report & release | 13–14 | 1/4–1/17 | 4.5 | 报告（method、data、results、cost、latency、MQM error breakdown、limitations）；README；demo 录屏；中英 blog；公开仓库 |

### 7.1 工时变化（v0.1 → v0.2）

| Milestone | v0.1 | v0.2 | 原因 |
|---|---|---|---|
| M2 | 3.5 | 3.5 | 自写 trace JSONL 换成 tracing tool；C1 不再是 config，Translator Agent 只做 smoke test |
| M3 | 3.5 | 3 | 删除 Self-Refine（C2） |
| M4 | 3.5 | 3 | 删除 cost parity 调参，config 从 C0–C3 + 3 个 memory 变体减到 3 个（−1.5）；COMET / xCOMET 从 M5 提前到这里，并改为必做（+1） |
| M5 | 3 | 3 | COMET 移到 M4（−1）；新增 GEMBA-MQM 和 human agreement check（+1.25）；PREREG 简化为 eval protocol，人工读 trace 改为看 MQM breakdown（−0.25） |
| M6 | 1.5 | 1 | test split 上 3 个 config 各跑一次（原来是每个 config 跑 2 次） |
| M8 | 4 | 2.5 | 自定义 failure taxonomy 换成 GEMBA-MQM breakdown；只剩两组 bootstrap；删除 optional blind evaluation |
| **小计（M2–M6、M8）** | **19** | **16** | **−3** |
| M7 | 9 | 12 | 接收省下的 3 h |
| **Fukidashi 合计** | **38.5** | **38.5** | 不变 |

**为什么挪给 M7，而不是 Triage Agent：** M7 原来要在 9 h 里做完 12 项 feature，风险表里「编辑器吃掉时间」是中等风险；而 Fukidashi 现在按产品定位，编辑器的质量直接决定 demo 的效果。如果 Triage Agent 那边更缺时间，可以把第 12 周 M8 省下的 1.5 h 改给同一周的 T3，不用调日历。

**日历上怎么挪（每周总小时不变）：** 第 4、5、9、12 周省下的 P 时间（0.5 / 0.5 / 0.5 / 1.5 h）给当周的 L 线；第 10 周 P 从 9 h 加到 12 h，L 从 4 h 减到 1 h。整份实习计划的总小时（约 161 h）和各线的总小时都不变。

---

## 8. Pending Decisions

| Decision | When | Options / 建议 | Status |
|---|---|---|---|
| Project name | M0 | **Fukidashi** | Decided（2026-10-22） |
| Translation direction | M0 | **ja → zh-Hans**。OpenMantra 的参考译文是 zh-Hant，评测前用 OpenCC 转成简体；demo 是简体，国内面试官能直接看懂 | Decided（2026-10-22） |
| Model | M0 | **DeepSeek `deepseek-flash`**（DeepSeek-V4.1-Flash-1017），OpenAI-compatible，base URL `https://api.deepseek.com`。支持图片输入和 JSON output；thinking 默认开（effort `high`）。每百万 token：输入 $0.15、输出 $0.60（错峰），高峰时段翻倍；`.env` 按高峰价记账 | Decided（2026-10-24）；tool calling 待 M2 实测 |
| API budget cap | M0 | **$100**（整个项目，包括 judge 调用）。`runs/usage.jsonl` 记每次调用；M4 先用 3 页估算 | Decided（2026-10-22） |
| Tracing tool | M2 | Langfuse 或 Arize Phoenix；建议 Phoenix（本地运行，较轻） | TBD |
| COMET checkpoint | M4 | COMET 和 xCOMET 的具体模型，按 license 和显存选 | TBD |
| Judge model（GEMBA-MQM） | M5 | 一个和 translation model 不同的强模型 | TBD |

---

## 9. Risks & Pivot Rules

| Risk | Mitigation |
|---|---|
| 又滑向 over-engineering（PanelFlow 是前车之鉴） | Vertical slice first；`SCOPE.md` 写死范围；10/25 checkpoint 只看浏览器里有没有真实译文 |
| 编辑器吃掉时间 | Product feature 集中在 M7（v0.2 起 12 h）；超时按 `SCOPE.md` 从下往上砍 |
| 作品太短，KB 收益测不出 | 用 term accuracy，不只看 consistency；如实报告；长篇场景只在报告里讨论 |
| 模型费用超预算 | 总上限 $100；`runs/usage.jsonl` 记每次调用的 token 和费用（`.env` 里填了价格才算得出费用）；先用 3 页估算（包括 judge 调用）；超出预期时 dev 调参只跑部分页面 |
| GEMBA-MQM 和人工判断不一致 | 按 eval protocol 的判定规则，降为参考，并在报告里写明 |
| COMET / xCOMET 在 Mac 上跑不动 | 用 Windows 机器的 GPU；xCOMET 显存不够就只报 COMET |
| `deepseek-flash` 的图片能力很新（2026-10-17 的版本），tool calling 也没实测 | M0、M1 先人工看译文；M2 实测 tool loop；不行就在 `.env` 里换 provider，代码不用改 |

**Pivot rules（只在 checkpoint 做决定）：**
- **10/25：** 浏览器里没有真实译文 → 先放下前端，只走 CLI 实验路径，编辑器全部推到 M7。不再换项目。
- **12/13：** 如果只有一个项目有第一版数字，寒假的 P 线全部给它。
- 其他时候冒出「要不要换方向」的想法 → 先记进 `IDEAS.md`，留到下一个 checkpoint 再决定。

---

## 10. Development Workflow

- 每个 task 写 goal + acceptance criteria → AI 实现 → **一个 task 一个 PR** → 你审 diff。
- README 写清楚哪些代码是 AI 写的、哪些是你写的、你怎么审。
- 测试只覆盖会出事的地方：KB write permission、Proposal 去重、metering（unknown usage）、bootstrap。

---

## 11. Related Work

**系统设计的来源：**
- Lippmann et al., *Context-Informed Machine Translation of Manga*（COLING 2025）— arXiv 2411.02589：translation unit 和 context 对漫画翻译的影响。
- **DelTA**（ICLR 2025）— arXiv 2410.08143：用 multi-level memory 做 document-level translation consistency。
- **TransAgents** — arXiv 2405.11804：用 multi-agent 翻译长篇文学。

**Evaluation framework 的来源（引用前核对）：**
- Post, *A Call for Clarity in Reporting BLEU Scores*（WMT 2018）：sacreBLEU 和 signature。
- Rei et al., *COMET*（EMNLP 2020）；Guerreiro et al., *xCOMET*（TACL 2024）。
- Freitag et al., *Experts, Errors, and Context*（TACL 2021）：用 MQM 做 MT 的人工评测。
- Kocmi & Federmann, *GEMBA-MQM*（WMT 2023）：用 LLM 标注 MQM error span。

本项目不提出新的研究问题，而是用标准的 evaluation framework 评测一个产品；上面的系统设计文献只用来说明设计从哪里来。

---

## 12. Interview Hooks

| 预期问题 | 回答依据 |
|---|---|
| 怎么证明 system 真的变好了，而不是 demo 看起来更流畅？ | 标准 metric（COMET / xCOMET、chrF、GEMBA-MQM、term accuracy）+ baseline + 按页 paired bootstrap 95% CI；eval protocol 冻结前写好，test split 只跑一次 |
| 为什么用 code orchestration，而不是让 agent 决定所有步骤？ | 大流程是确定的，代码编排可控、可评测、能精确分配预算；对照 Triage Agent 的完全自主 ReAct（Anthropic "Building effective agents"） |
| 为什么 Reviewer 要用 independent context？有数据吗？ | 设计理由：self-check 容易自圆其说，Reviewer 看不到 Translator 的推理。坦白说没有做同预算对比（Known Limitations），并讲清楚如果要验证会怎么设计：同预算的 Self-Refine 对照 + paired bootstrap |
| 多个 agent 写同一份共享状态怎么办？ | single-writer + Proposal Queue；proposal ID 去重；人名和冲突进 Human Review Queue |
| KB 有没有用？为什么不用 vector database？ | KB ablation 的 term accuracy 差值和 CI；term 和人名是 identifier，exact match 更合适 |
| LLM-as-judge 可信吗？ | 在约 30 个人工标注的 segment 上做了 human agreement check；judge model 和 translation model 不同；一致性不够就只作参考 |
| chrF / BLEU 能说明漫画翻译好吗？ | 单独不能，所以加了 COMET / xCOMET 和 GEMBA-MQM；参考译文只有一份的局限写在报告里 |
| 为什么不自己写 trace？ | Langfuse / Phoenix 已经很成熟；只保留自己写的 metering，因为 unknown usage 的口径要自己控制 |
| SSE 断了、服务重启怎么办？会不会重复写 KB？ | `seq` + `Last-Event-ID`；page 是 retry unit；KB 只在 finalize 时写一次 |
| 为什么重写 PanelFlow？ | 30 秒版本：顺序错了；学到 vertical slice、scope 写死、测试只测会出事的地方 |
| 多少代码是 AI 写的？你怎么保证质量？ | Development Workflow（第 10 节） |

---

## 13. Definition of Done（2027-02-15）

- [ ] 公开仓库；可以运行的编辑器 + demo 录屏
- [ ] `baseline` / `full` / `full_no_kb` 在 test split 上的结果：chrF / BLEU（附 signature）、COMET / xCOMET、GEMBA-MQM score 和 error breakdown、term accuracy / consistency、cost per page、latency（p50 / p95）
- [ ] 两组对比的 paired bootstrap 95% CI；GEMBA-MQM human agreement 的结果；报告（含 Known Limitations）
- [ ] 中英 blog
- [ ] 不了解项目的人能按 README 复现主表
