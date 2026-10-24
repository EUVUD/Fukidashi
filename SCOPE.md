# SCOPE

> Fukidashi 的范围文件。v3 · 2026-10-22 · 由 [OUTLINE.md](OUTLINE.md) 第 2 节生成，范围以本文件为准。
> 术语沿用 OUTLINE.md 第 0 节的 Terminology 表。

**三条规则：**

1. 不在 In scope 里的不做。要加功能先改本文件，而且只在 checkpoint（10/25、12/13）改；其他时候冒出的想法先记进 `IDEAS.md`。
2. 超时就按第 3 节的表，**从最下面一行往上砍**。
3. 砍 product feature 时不动 eval code path。

---

## 1. In scope / Out of scope

| In scope | Out of scope |
|---|---|
| 日→简体中文（zh-Hans）单方向翻译（M0 决定） | 多语言方向、繁体输出、用户系统、多人实时协作 |
| Oracle path（评测用）+ end-to-end visual path（demo 用） | 自训 detection / OCR / inpainting 模型 |
| 3 个 agent + code orchestration | 通用 agent framework、自己的 workflow DSL |
| SQLite KB + Proposal Queue | Vector database（RAG 不是本项目重点） |
| Evaluation framework（sacreBLEU、COMET / xCOMET、GEMBA-MQM、term accuracy、cost、latency）+ `baseline` vs `full` + KB ablation（`full_no_kb`），dev/test split | 单独评测 Reviewer（independent review vs Self-Refine）、cost parity 调参；长篇作品上的 KB 收益（只在报告里讨论） |
| Tracing 用 Langfuse 或 Arize Phoenix | 自写 trace 存储和 dashboard |
| Canvas editor（第 3 节列出的 feature） | 完整的排版软件功能、字体管理 |
| SSE live trace + `Last-Event-ID` reconnect | WebSocket、分布式任务队列 |
| 一个最小 MCP server（只暴露 `lookup_term`） | 把整个系统做成 MCP 服务 |

---

## 2. Never cut

下面两组不在 cut order 里。如果连它们都保不住，就按 10/25 的 pivot rule 处理：放下前端，只走 CLI 实验路径。

- **Eval path（M2–M6、M8）：** OpenMantra import（oracle path）；Orchestrator + Translator / Reviewer / Curator Agent；`baseline` / `full` / `full_no_kb` 三个 config；metering + tracing tool（Langfuse 或 Phoenix）；eval harness（`eval/run.py`）+ manifest；sacreBLEU（chrF / BLEU，附 signature）+ COMET / xCOMET + GEMBA-MQM（含 human agreement check）+ term accuracy / consistency + cost / latency；两组对比的 paired bootstrap；`eval/PROTOCOL.md` + tag `exp-frozen`。
- **Vertical slice（M1）：** 浏览器里显示页面图片、oracle boxes 和 translation list，能对一页触发 C0 并看到真实译文。

---

## 3. Product Features — Cut Order

M7 的预算是 12 h（12/14–12/20；v2 起从评测简化省下的 3 h 挪到这里）。**#1 最后砍，最下面一行最先砍。**

**规则：** 每一项只依赖排在它上面的项，所以从下往上砍，保留下来的功能不会坏。

| # | Feature | Done when（acceptance criteria） | Depends on | 为什么排在这里 |
|---|---|---|---|---|
| 1 | **Edit translation** | 在 translation list 里改一个 box 的译文，保存到 SQLite，刷新后还在 | Vertical slice | 编辑器最基本的功能；没有它就不算 editor |
| 2 | **Live agent trace（SSE）** | 翻译一页时，前端实时显示每个 step（tool call、tool result、review round、finalize）；每个 event 带递增的 `seq` | — | 这是能直接演示多 agent 的地方，面试 demo 的核心 |
| 3 | **Canvas typesetting** | 译文渲染进对应的 box（canvas overlay），字号自动缩小到放得下；能统计 overflow rate | — | 没有它，demo 看不出是漫画；#6、#9、#11 都依赖它 |
| 4 | **Review comments & uncertain flags** | Reviewer 的意见和 `flag_uncertain` 标记在对应的 box 上高亮，点开能看到内容 | Reviewer Agent（M3） | 在产品里看得见 RQ1 的那个 Reviewer |
| 5 | **Proposal approve / reject** | KB 面板列出 Proposal Queue 和 Human Review Queue；approve 或 reject 之后 KB 更新，entry 的 `provenance` 记为 human | KB + Curator（M3） | human-in-the-loop，能回答「错误条目怎么防」 |
| 6 | **Rename → whole-book sync** | 改一个 term 的 `target_term` 后，全书用到它的 box 都更新，并重新 typesetting | #3、#5 | 产品的 headline feature，KB 在产品里最直观的体现 |
| 7 | **SSE reconnect** | 断网或刷新页面后，客户端用 `Last-Event-ID` 重连，服务端补发缺失的 event，不重复也不丢 | #2 | 便宜，面试必问；不过 reliability 的故事 Triage Agent 那边（lease、fencing、`kill -9`）也讲得到，所以排在 #6 后面 |
| 8 | **Detection + OCR（end-to-end path）** | 上传一张不在 OpenMantra 里的页面，自动得到 box 和原文：manga-ocr 必选，comic-text-detector 作为 optional adapter | Vision adapters | 证明系统不只能跑 oracle 数据；实验结论不依赖它 |
| 9 | **Inpainting + PNG export** | 用 OpenCV 擦掉原文，写入译文，导出 PNG | #3、#8 | 成品图好看，但对 agent 和实验的故事没有帮助 |
| 10 | **Box editing** | 能拖动、缩放和新建 box | — | 主要在 detection 漏框时才用；oracle path 用不上 |
| 11 | **Text style** | 能手动改字号，能切换横排 / 竖排 | #3 | 日→中以横排为主，价值最低的排版细节 |
| 12 | **Simple undo** | 撤销最近一次人工编辑（改字或 box 编辑）；不跨页面，不撤销 agent 的操作 | #1、#10 | 最先砍，没有它 demo 也能完整演示 |

**不是 feature，不参与 cut order：** demo 录屏（M7 草稿，M9 定稿）、README、报告。

---

## 4. 怎么砍

- M7 期间每完成一项就记下实际用时。预计会超出 12 h 时，从表的最下面一行开始砍，并在第 5 节记一行（日期、砍掉哪一项、原因）。
- 被砍的功能在 README 里写明 not implemented，**简历上不把设计稿写成已实现的功能**。
- 不能拿 Never cut 的时间去补 product feature。

---

## 5. Change Log

| Date | Change |
|---|---|
| 2026-10-22 | v1：由 OUTLINE.md 第 2 节生成，product feature 按 cut order 排序 |
| 2026-10-22 | v2：评测方案改为 evaluation framework + baseline 对比（方案 B）。第 1 节的评测行改为 `baseline` vs `full` + KB ablation，并新增 tracing 行；第 2 节 experiment path 改为 eval path；M7 预算从 9 h 改为 12 h。Cut order 不变 |
| 2026-10-22 | v3：M0 决定。项目名定为 Fukidashi；第 1 节翻译方向改为 ja → zh-Hans，繁体输出列入 Out of scope |
