# Fukidashi

English | [简体中文](README.zh-CN.md)

A multi-agent manga translation editor (Japanese → Simplified Chinese), evaluated on the [OpenMantra](https://github.com/mantra-inc/open-mantra-dataset) dataset against a single-call baseline with a standard MT evaluation framework.

> **Status (2026-10-29):** M0 to M4 are done. The vertical slice works end to end: import OpenMantra, translate a page with one model call, and check every line against its box in the browser. M2 added an eval harness that scores configs with chrF / BLEU, a first Translator Agent (the `translator` config) on a tool-calling loop over our metered client, and optional tracing in a local Arize Phoenix ([record](docs/m2.md)). M3 added the multi-agent system ([record](docs/m3.md)): a Reviewer Agent, an Orchestrator that sends pages back for revision, a per-book Knowledge Base with a Curator Agent as its only writer, the `full` and `full_no_kb` configs, and a gold glossary for the two dev books. M4 added whole-book evals, a COMET scorer, term accuracy against the gold glossary, a count of how often the KB reaches a page, a report that pools eval runs, and CI on GitHub Actions, then ran each config once on the two dev books ([record](docs/m4.md)). On the dev split the full system scored no higher than one call per page (COMET 0.8235 against 0.8257) at about 9.5 times the cost; that is one run of each config, without confidence intervals yet. Next is M5: the eval protocol, GEMBA-MQM and term consistency, then the freeze.

## What it does

- **Translate a page in context.** The model sees the whole page image together with the Japanese text and box of every speech bubble, and returns a Simplified Chinese line for each box.
- **Check it where it belongs.** The editor draws each box on the page and lists source, translation and reference side by side.
- **A first Translator Agent (M2).** The `translator` config gives the model the same page plus a tool that reads the earlier pages; a bad answer is repaired once, and otherwise the page falls back to `baseline`, visibly.
- **Translate, review, revise (M3).** In the `full_no_kb` config an Orchestrator, plain code, runs the Translator, then a Reviewer Agent. The Reviewer works in an independent context: it sees the page, the translation and the regions the Translator flagged as uncertain, never the Translator's reasoning, and comments in Simplified Chinese. A page it sends back is revised, at most twice.
- **A Knowledge Base that fills page by page (M3).** In the `full` config the agents share a per-book Knowledge Base. The Translator looks up Glossary entries and Character Profiles and proposes new ones; the Reviewer can only look them up. Once a page is done, a Curator Agent accepts or rejects each proposed term with a reason, and code rules write the KB once. They send every name, and every conflict with an existing entry, to a person instead. `full_no_kb` is the same system with the KB off, for the ablation.

## How it is evaluated

Fukidashi is evaluated as a product, not as a research experiment (details in [OUTLINE.md §4](OUTLINE.md#4-evaluation)):

- **Main comparison:** the full system vs `baseline` (one call per page). **KB ablation:** the full system with and without the Knowledge Base.
- **Metrics:** chrF / BLEU (sacreBLEU, with signature), COMET (`wmt22-comet-da`), GEMBA-MQM (LLM-as-judge, checked against about 30 human-annotated segments first), term accuracy against a gold glossary, cost per page and latency.
- **Rigor:** split by book (2 dev, 3 test), an eval protocol written before the freeze, the test split run once, paired bootstrap 95% CIs by page.

## Engineering notes

- **One metered client.** Every model call goes through `LLMClient`, which enforces a per-page token budget and logs each call to a cost ledger. Usage the provider does not report is recorded as unknown, never as 0.
- **No reference leakage.** Prompts contain only the Japanese text and boxes; the reference translations stored next to them never reach a prompt, and a test checks it.
- **Runs are recorded, not overwritten.** A Run is inserted as `running` and finished as `succeeded` or `failed`; a failed Run keeps the raw responses, and the page keeps showing the last good translation.
- **An agent loop that keeps the metering.** Tool calls run on LangChain's `create_agent`, but its model is a thin adapter over `LLMClient`, so every agent call is metered like any other and DeepSeek's reasoning is passed back as its API requires. Guardrails: a step limit, the token budget, duplicate-call detection, and a truncated reply is never taken as an answer.
- **One writer for the Knowledge Base.** The agents get a read-only snapshot and a way to propose, never a database connection, and a test checks that loading them does not load the database module. Only the Curator's rules write, once per page, after the page is final; names and conflicts wait for a person.
- **Traces that stay local.** Optional tracing sends one trace per page to an Arize Phoenix on this machine through OpenTelemetry. Page images and the API key are stripped from every span, and the backend is one exporter, so it can be swapped later.
- **Swappable provider.** Any OpenAI-compatible model with vision, tool calling and JSON output works through `.env`; the current choice is DeepSeek `deepseek-flash` (about $0.0006 per page).

Diagrams (domain model, sequence diagrams, object model): [docs/design.md](docs/design.md).

## Run it

Requires Python 3.12 with [uv](https://docs.astral.sh/uv/), and Node.js.

```bash
uv sync
git clone --depth 1 https://github.com/mantra-inc/open-mantra-dataset.git data/open-mantra-dataset
cp .env.example .env   # fill in LLM_BASE_URL, LLM_API_KEY, LLM_MODEL

uv run python scripts/import_openmantra.py      # imports 214 pages into runs/fukidashi.db; safe to re-run
uv run uvicorn fukidashi.main:app --port 8000
npm install --prefix frontend                   # first time only
npm run dev --prefix frontend                   # http://localhost:5173/tojime_no_siora/002
```

Pick a page and click Translate: the backend runs `baseline` on that page (one call, no tools, no KB) and stores the Run and its zh-Hans translations in SQLite. Translate makes a real call to the model configured in `.env`. `FUKIDASHI_DB` overrides the DB path.

```bash
uv run python eval/run.py --config baseline --book tojime_no_siora --pages 2,3,4     # real calls: one per page
uv run python eval/run.py --config translator --book tojime_no_siora --pages 2,3,4   # the Translator Agent: a few calls per page
uv run python eval/run.py --config full_no_kb --book tojime_no_siora --pages 2,3,4   # Translator + Reviewer: a few more
uv run python eval/run.py --config full --book tojime_no_siora --pages 2,3,4         # the same with a Knowledge Base and the Curator
```

The eval harness translates those pages through the same code path as the editor (each page leaves a Run in SQLite), converts the zh-Hant reference translations to zh-Hans with OpenCC (`t2s`), and prints chrF and BLEU with their sacreBLEU signatures, plus cost per page and latency; for the agent configs it also counts the pages that fell back to `baseline`, and for `full_no_kb` and `full` how often the Reviewer sent a page back and why the loop stopped. A `full` eval starts from a new, empty Knowledge Base, passes it through the pages in order, and reports what the Curator decided, how often the KB reached a page, and what the KB holds at the end. The Manifest, per-Segment outputs and scores go to `runs/eval/<id>/`, rewritten after every page, so an interrupted eval keeps the pages it finished. Leave out `--pages` to evaluate every page of a book in page order (a whole book makes many real calls). The exit code is non-zero if a page failed or the harness could not start.

Two scorers read a finished eval run and write their results next to its outputs, and a report pools scored eval runs, one per config and book. None of them calls a model provider:

```bash
uv run python eval/terms.py runs/eval/<id>   # term accuracy against the gold glossary
uv run eval/comet.py runs/eval/<id>          # COMET, in an environment of its own
uv run python eval/report.py runs/eval/<id> runs/eval/<id> ...   # one report per config, over the books and per book
```

`eval/comet.py` declares its own dependencies (PyTorch and `unbabel-comet`), so they stay out of the project's environment. Its first run downloads them and the `Unbabel/wmt22-comet-da` checkpoint (about 2.3 GB) from Hugging Face. A CPU is enough. The report compares the configs on the pages that every config scored, recomputes chrF and BLEU over the pooled segments, and refuses files that do not belong together; it writes `runs/report/<time>-report/report.json` and prints a summary.

Optional tracing: one trace per translated page (every model call, agent step and tool call) in an [Arize Phoenix](https://github.com/Arize-ai/phoenix) running on your machine. Start Phoenix in another terminal, set `FUKIDASHI_TRACING=phoenix` (in the shell or `.env`), and open http://localhost:6006:

```bash
PHOENIX_HOST=127.0.0.1 PHOENIX_TELEMETRY_ENABLED=false uvx --from arize-phoenix phoenix serve
FUKIDASHI_TRACING=phoenix uv run python eval/run.py --config translator --book tojime_no_siora --pages 3
```

Tracing is off by default and never fails a translation. Page images and the API key are removed from every span, and the endpoint must be on this machine. Spans go straight to it, even when your shell uses an HTTP proxy. The eval outputs record each page's trace ID.

```bash
uv run pytest                                       # no network, no real LLM calls; CI runs it on every pull request
uv run python scripts/m0_first_call.py --dry-run    # print the M0 request without sending it
uv run python eval/glossary.py check                # check the gold glossaries against the dataset; no model calls
```

## Roadmap

| Milestone | Target | Status |
|---|---|---|
| M0 Decisions & first call | 10/18 | Done ([record](docs/m0.md)) |
| M1 Vertical slice: page → `baseline` → browser | 10/25 | Done ([record](docs/m1.md)) |
| M2 Agent layer & eval harness | 11/1 | Done ([record](docs/m2.md)) |
| M3 Knowledge Base & Reviewer | 11/8 | Done ([record](docs/m3.md)) |
| M4 Dev evaluation | 11/15 | Done ([record](docs/m4.md)) |
| M5 Eval protocol & freeze | 11/22 | Planned |
| M6–M8 Test run, editor productization, analysis | 1/3 | Planned |
| M9 Report & release | 1/17 | Planned |

Full plan: [OUTLINE.md](OUTLINE.md) (design, milestones, decisions, risks) and [SCOPE.md](SCOPE.md) (scope and cut order).

## How this repository is built

Most of the code is written by AI coding agents (Claude Code) through an architect → executor → reviewer workflow ([docs/workflow.md](docs/workflow.md)): every task has written acceptance criteria, an executor session implements it on its own branch, a fresh reviewer session checks the diff against the criteria, and I review and merge each PR. The design decisions (scope, evaluation plan, model, budget) are mine and are recorded in [OUTLINE.md §8](OUTLINE.md#8-decisions).

## Data and attribution

OpenMantra is by Hinami et al., *Towards Fully Automated Manga Translation* (AAAI 2021), licensed **CC BY-NC 4.0**. The images are not part of this repository; clone the dataset separately as shown above. The docs quote a few short lines of its text for illustration, and the gold glossaries in `eval/gold_glossary/` hold short terms (names and recurring words) taken from its Japanese text and Chinese reference translations. Data notes: [docs/openmantra.md](docs/openmantra.md).

## License

The code in this repository is released under the [MIT License](LICENSE). The license does not cover OpenMantra: the dataset is not distributed here, and the lines of its text quoted in the docs and the terms in `eval/gold_glossary/` remain under CC BY-NC 4.0, so any use of the dataset must follow that license.
