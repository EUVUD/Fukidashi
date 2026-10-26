# SCOPE

> Fukidashi's scope file. v4 · 2026-10-26 · Derived from [OUTLINE.md](OUTLINE.md) §2; this file is authoritative for scope.
> Terms follow the Terminology table in OUTLINE.md §0.

**Three rules:**

1. What is not in scope does not get built. Adding a feature means changing this file first, and only at a checkpoint (10/25, 12/13); ideas that come up in between go into `IDEAS.md`.
2. When time runs out, cut from §3, **starting at the bottom row**.
3. Cutting a product feature never touches the eval code path.

---

## 1. In scope / Out of scope

| In scope | Out of scope |
|---|---|
| One direction: Japanese → Simplified Chinese (zh-Hans), decided in M0 | Other language pairs, Traditional Chinese output, user accounts, real-time multi-user editing |
| Oracle path (for evaluation) + end-to-end visual path (for the demo) | Training our own detection / OCR / inpainting models |
| 3 agents + code orchestration | A general agent framework or our own workflow DSL |
| SQLite KB + Proposal Queue | A vector database (RAG is not the focus) |
| Evaluation framework (sacreBLEU, COMET / xCOMET, GEMBA-MQM, term accuracy, cost, latency) + `baseline` vs `full` + KB ablation (`full_no_kb`), dev/test split | Evaluating the Reviewer on its own (independent review vs Self-Refine), cost-parity tuning; KB gains on long books (discussed in the report only) |
| Tracing with Langfuse or Arize Phoenix | Custom trace storage and dashboards |
| Canvas editor (the features in §3) | Full typesetting features, font management |
| SSE live trace + `Last-Event-ID` reconnect | WebSocket, a distributed task queue |
| A minimal MCP server (exposes `lookup_term` only) | Turning the whole system into an MCP service |

---

## 2. Never cut

These two groups are not part of the cut order. If even they are at risk, apply the 10/25 pivot rule: drop the frontend and keep only the CLI path.

- **Eval path (M2–M6, M8):** OpenMantra import (oracle path); Orchestrator + Translator / Reviewer / Curator Agent; the `baseline` / `full` / `full_no_kb` configs; metering + tracing tool (Langfuse or Phoenix); eval harness (`eval/run.py`) + manifest; sacreBLEU (chrF / BLEU, with signature) + COMET / xCOMET + GEMBA-MQM (with the human agreement check) + term accuracy / consistency + cost / latency; paired bootstrap for both comparisons; `eval/PROTOCOL.md` + tag `exp-frozen`.
- **Vertical slice (M1):** the browser shows the page image, the oracle boxes and the translation list, and can trigger C0 on a page and show the real translation.

---

## 3. Product Features — Cut Order

M7 has a 12-hour budget (12/14–12/20; since v2 it includes the 3 hours saved by the simpler evaluation). **#1 is cut last; the bottom row is cut first.**

**Rule:** each feature depends only on features above it, so cutting from the bottom never breaks what remains.

| # | Feature | Done when (acceptance criteria) | Depends on | Why it sits here |
|---|---|---|---|---|
| 1 | **Edit translation** | Editing one box's translation in the translation list saves it to SQLite, and it survives a reload | Vertical slice | The most basic editor feature; without it this is not an editor |
| 2 | **Live agent trace (SSE)** | While a page is translated, the frontend shows every step live (tool call, tool result, review round, finalize); every event carries an increasing `seq` | — | Where the multi-agent design is visible; the core of the demo |
| 3 | **Canvas typesetting** | Translations render inside their boxes (canvas overlay), with the font shrunk until it fits; the overflow rate can be measured | — | Without it the demo does not look like manga; #6, #9 and #11 depend on it |
| 4 | **Review comments & uncertain flags** | The Reviewer's comments and `flag_uncertain` marks highlight the matching box, and clicking shows the content | Reviewer Agent (M3) | Makes the Reviewer visible in the product |
| 5 | **Proposal approve / reject** | A KB panel lists the Proposal Queue and the Human Review Queue; approving or rejecting updates the KB and records `provenance` as human | KB + Curator (M3) | Human in the loop; answers "how do wrong entries get stopped" |
| 6 | **Rename → whole-book sync** | Changing a term's `target_term` updates every box in the book that uses it and re-typesets them | #3, #5 | The headline product feature; the most direct view of the KB in the product |
| 7 | **SSE reconnect** | After a network drop or a page reload, the client reconnects with `Last-Event-ID` and the server replays the missed events, with no duplicates and no gaps | #2 | Cheap and a common question, but the reliability story is also covered by the companion Triage Agent project (leases, fencing, `kill -9`), so it sits below #6 |
| 8 | **Detection + OCR (end-to-end path)** | Uploading a page that is not in OpenMantra yields boxes and source text automatically: manga-ocr required, comic-text-detector as an optional adapter | Vision adapters | Shows the system is not limited to oracle data; no evaluation result depends on it |
| 9 | **Inpainting + PNG export** | OpenCV erases the source text, the translation is written in, and the page exports as PNG | #3, #8 | Nice finished pages, but adds nothing to the agent or evaluation story |
| 10 | **Box editing** | Boxes can be moved, resized and added | — | Mostly needed when detection misses a box; the oracle path does not need it |
| 11 | **Text style** | Font size can be set by hand; text switches between horizontal and vertical | #3 | Japanese → Chinese is mostly horizontal; the least valuable typesetting detail |
| 12 | **Simple undo** | Undo the latest human edit (text or box); not across pages, and not agent actions | #1, #10 | Cut first; the demo is complete without it |

**Not features, so not in the cut order:** the demo recording (draft in M7, final in M9), the README, the report.

---

## 4. How to cut

- Record the actual time for each feature during M7. As soon as 12 hours look likely to be exceeded, cut from the bottom row and log it in §5 (date, feature, reason).
- A cut feature is listed as not implemented in the README, and **a design is never presented as a finished feature on a résumé**.
- Never take time from the Never-cut groups to finish a product feature.

---

## 5. Change Log

| Date | Change |
|---|---|
| 2026-10-22 | v1: derived from OUTLINE.md §2; product features ordered for cutting |
| 2026-10-22 | v2: evaluation changed to an evaluation framework + baseline comparison (plan B). §1's evaluation row became `baseline` vs `full` + KB ablation, and a tracing row was added; §2's experiment path became the eval path; the M7 budget went from 9 h to 12 h. Cut order unchanged |
| 2026-10-22 | v3: M0 decisions. Name fixed as Fukidashi; §1's direction became ja → zh-Hans, with Traditional Chinese output out of scope |
| 2026-10-26 | v4: translated to English; scope and cut order unchanged |
