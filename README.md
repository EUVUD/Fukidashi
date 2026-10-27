# Fukidashi

English | [简体中文](README.zh-CN.md)

A multi-agent manga translation editor (Japanese → Simplified Chinese), evaluated on the [OpenMantra](https://github.com/mantra-inc/open-mantra-dataset) dataset against a single-call baseline with a standard MT evaluation framework.

> **Status (2026-10-26):** M0 and M1 are done. The vertical slice works end to end: import OpenMantra, translate a page with one model call, and check every line against its box in the browser. The agents, the Knowledge Base, the Reviewer, tracing and the eval harness are planned for M2 onward and do not exist yet.

## What it does

- **Translate a page in context.** The model sees the whole page image together with the Japanese text and box of every speech bubble, and returns a Simplified Chinese line for each box.
- **Check it where it belongs.** The editor draws each box on the page and lists source, translation and reference side by side.
- **Planned (M2–M3):** a Translator Agent that looks up a Glossary and Character Profiles through tools, a Reviewer Agent that checks the result in an independent context, and a Curator Agent that is the only writer of the per-book Knowledge Base.

## How it is evaluated

Fukidashi is evaluated as a product, not as a research experiment (details in [OUTLINE.md §4](OUTLINE.md#4-evaluation)):

- **Main comparison:** the full system vs `baseline` (one call per page). **KB ablation:** the full system with and without the Knowledge Base.
- **Metrics:** chrF / BLEU (sacreBLEU, with signature), COMET / xCOMET, GEMBA-MQM (LLM-as-judge, checked against about 30 human-annotated segments first), term accuracy against a gold glossary, cost per page and latency.
- **Rigor:** split by book (2 dev, 3 test), an eval protocol written before the freeze, the test split run once, paired bootstrap 95% CIs by page.

## Engineering notes

- **One metered client.** Every model call goes through `LLMClient`, which enforces a per-page token budget and logs each call to a cost ledger. Usage the provider does not report is recorded as unknown, never as 0.
- **No reference leakage.** Prompts contain only the Japanese text and boxes; the reference translations stored next to them never reach a prompt, and a test checks it.
- **Runs are recorded, not overwritten.** A Run is inserted as `running` and finished as `succeeded` or `failed`; a failed Run keeps the raw responses, and the page keeps showing the last good translation.
- **Swappable provider.** Any OpenAI-compatible model with vision and JSON output works through `.env`; the current choice is DeepSeek `deepseek-flash` (about $0.0006 per page).

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
uv run pytest                                       # no network, no real LLM calls
uv run python scripts/m0_first_call.py --dry-run    # print the M0 request without sending it
```

## Roadmap

| Milestone | Target | Status |
|---|---|---|
| M0 Decisions & first call | 10/18 | Done ([record](docs/m0.md)) |
| M1 Vertical slice: page → `baseline` → browser | 10/25 | Done ([record](docs/m1.md)) |
| M2 Agent layer & eval harness | 11/1 | Planned |
| M3 Knowledge Base & Reviewer | 11/8 | Planned |
| M4–M5 Dev evaluation, protocol freeze | 11/22 | Planned |
| M6–M8 Test run, editor productization, analysis | 1/3 | Planned |
| M9 Report & release | 1/17 | Planned |

Full plan: [OUTLINE.md](OUTLINE.md) (design, milestones, decisions, risks) and [SCOPE.md](SCOPE.md) (scope and cut order).

## How this repository is built

Most of the code is written by AI coding agents (Claude Code) through an architect → executor → reviewer workflow ([docs/workflow.md](docs/workflow.md)): every task has written acceptance criteria, an executor session implements it on its own branch, a fresh reviewer session checks the diff against the criteria, and I review and merge each PR. The design decisions (scope, evaluation plan, model, budget) are mine and are recorded in [OUTLINE.md §8](OUTLINE.md#8-decisions).

## Data and attribution

OpenMantra is by Hinami et al., *Towards Fully Automated Manga Translation* (AAAI 2021), licensed **CC BY-NC 4.0**. The images are not part of this repository; clone the dataset separately as shown above. The docs quote a few short lines of its text for illustration. Data notes: [docs/openmantra.md](docs/openmantra.md).

## License

The code in this repository is released under the [MIT License](LICENSE). The license does not cover OpenMantra: the dataset is not distributed here, and the lines of its text quoted in the docs remain under CC BY-NC 4.0, so any use of the dataset must follow that license.
