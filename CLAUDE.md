# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fukidashi is a web editor that translates Japanese manga into Simplified Chinese (zh-Hans) and is evaluated on OpenMantra against a single-call `baseline`. The plan is written down; read it instead of guessing:

- `OUTLINE.md`: design, milestones (§7), decisions (§8), pivot rules (§9). Its §0 Terminology table is canonical.
- `SCOPE.md`: what is in and out of scope, and the cut order. `IDEAS.md`: where ideas outside a checkpoint go.
- `docs/design.md`: domain model, system sequence diagram, object interaction diagram, object model. `docs/m0.md`, `docs/m1.md`, `docs/m2.md`: milestone records.

M0, M1 and M2 are done. The vertical slice (page → `baseline` → browser) works. M2 added the eval harness v0 (`eval/run.py`: `baseline` and `translator`, chrF / BLEU, manifest), the tool-calling loop (`agents/loop.py`: `run_agent` on LangChain's `create_agent`), the Translator Agent (`agents/translator.py`: config `translator`, M2's smoke config, with one tool, `get_previous_context`) and tracing (`tracing/`: one trace per page in a local Arize Phoenix, off by default). The Knowledge Base and its tools, `flag_uncertain`, the Reviewer, COMET, GEMBA-MQM and term accuracy do not exist yet (M3 onward); do not describe them as implemented.

Docs are written in English. The README also exists in Simplified Chinese (`README.zh-CN.md`): change both together. Use the §0 terms verbatim and add any new term to that table. The repository is public: write docs for outside readers.

## Commands

```bash
uv sync                                        # installs backend/ as the editable package `fukidashi`
npm install --prefix frontend
git clone --depth 1 https://github.com/mantra-inc/open-mantra-dataset.git data/open-mantra-dataset
uv run python scripts/import_openmantra.py     # fills runs/fukidashi.db; safe to re-run

uv run uvicorn fukidashi.main:app --port 8000  # backend
npm run dev --prefix frontend                  # http://localhost:5173/tojime_no_siora/002, proxies /api to :8000

uv run pytest                                  # no network, no real LLM calls
uv run pytest backend/tests/test_api.py::test_empty_content_is_retried_once   # one test
uv run pytest -k run                           # by keyword
npm run lint --prefix frontend                 # oxlint
npm run build --prefix frontend                # tsc -b + vite build; this is the frontend typecheck
uv run python scripts/m0_first_call.py --dry-run   # build the request, send nothing
uv run python scripts/m2_tool_smoke.py --dry-run   # the agent loop's first request, with its tool; sends nothing
uv run python scripts/m2_tool_smoke.py --repair-check --dry-run   # also the Translator's repair request
PHOENIX_HOST=127.0.0.1 PHOENIX_TELEMETRY_ENABLED=false uvx --from arize-phoenix phoenix serve   # optional tracing UI on :6006;
                                               # then set FUKIDASHI_TRACING=phoenix (shell or .env)
uv run python eval/run.py --config baseline --book tojime_no_siora --pages 2,3,4   # real calls, one per page; writes runs/eval/<id>/
uv run python eval/run.py --config translator --book tojime_no_siora --pages 2,3,4  # the Translator Agent: a few calls per page
```

There is no Python linter and there are no frontend tests. `.env` (template: `.env.example`) holds the LLM provider, prices and the optional tracing switch: never read or print it. The API's `lifespan` and `eval/run.py` load it at start. `FUKIDASHI_DB` overrides the DB path (default `runs/fukidashi.db`).

## Architecture

One request shows the layering. In the browser, `PageView` POSTs `/api/pages/:book/:index/translate` (the Vite dev server proxies `/api`):

```
api/pages.py        parse the request, map errors to 404 / 422 / 503 / 502, shape the response; no SQL
 → runs.py          translate_page(config): validate_page → make_client() → store.start_run → baseline or translator
                    → store.finish_run
    → pipeline/baseline.py   one call per page: image + Japanese regions in, zh-Hans JSON out
       → llm/client.py       LLMClient, the only code that talks to the provider; failures leave llm/ as ProviderError
    → agents/translator.py   run_translator(): the agent with get_previous_context; repair once, else fall back to baseline
       → agents/loop.py      run_agent(): LangChain's create_agent with LLMClientChatModel (calls LLMClient.chat) as its
                             model and the OUTLINE §3.4 guardrails; returns a plain AgentResult
store.py            all SQL; schema.sql defines the tables
datasets/           OpenMantra loader and the importer
evaluation/         eval library: harness, OpenCC references, sacreBLEU metrics, manifest. Its CLI, eval/run.py,
                    calls runs.translate_page like the API does and writes runs/eval/<id>/
tracing/            setup(): OpenTelemetry + OpenInference instrumentors; page_span(): each page's root span;
                    backends.py: the backend (Phoenix) is one function; redact.py strips images and the key
```

Invariants that span several files:

- Every model call goes through `LLMClient`, which enforces the per-page `PageBudget` and records each call in a `Meter` (ledger `runs/usage.jsonl`). Usage the provider does not return is unknown, never 0: it shows up as `unknown_usage_calls` and NULL token counts on the Run. A request that raised (timeout, connection error, any HTTP error), or whose response `LLMClient` cannot read (not a chat completion, no readable first choice), is still a call: it is recorded once with `error` set, then raised as `ProviderError`. Its usage is unknown unless the response carried some, so a ledger line can have both `error` and usage; totals key on usage, never on `error`. The SDK's own retries are off (`max_retries=0`), so every request sent is one call the `Meter` sees. `cost_usd` is NULL when any call had no usage or no price (`Meter.summary`); it is never a partial sum, and a Run with no calls costs 0.0. The eval harness keeps the same rule for totals (`cost_summary`) and reports scored and failed Runs' cost separately.
- Only `llm/` imports `openai`, only `agents/` imports `langchain` / `langgraph`, and only `tracing/` imports `opentelemetry` / `openinference` (nothing imports `phoenix`). Code outside `llm/` sees a provider failure only as `ProviderError` (`description`, `status_code`, and `record`: the call as the `Meter` has it), whose text never contains the API key.
- The agent loop (`agents/loop.py`): `LLMClientChatModel` calls `LLMClient.chat` once per model call, so agent calls are metered like any other, and nothing in the loop retries. DeepSeek's `reasoning_content` goes back unchanged on every assistant message, `""` included. `run_agent` returns an `AgentResult` of plain values (text, stop reason, error, steps, messages as OpenAI-format dicts) with one step per model call, a failed call included. Stop reasons: `final`, `step_limit` (`max_steps`, default 6), `incomplete` (a finish reason other than `stop` or `tool_calls`), `budget`, `provider_error`. A model's mistakes (unknown tool, arguments that fail the tool's pydantic `args`, a handler that raises) go back to it as the tool's result; a repeated call with the same validated arguments is not run again. Tool handlers run in LangGraph worker threads, the calls of one reply in parallel, so they must be thread-safe. `langsmith` comes with `langchain-core`, so `run_agent` refuses to start (`TracingEnabled`) while `LANGSMITH_TRACING`, `LANGSMITH_TRACING_V2`, `LANGCHAIN_TRACING_V2` or `LANGCHAIN_TRACING` is on: prompts would leave the machine.
- The Translator Agent (`agents/translator.py`): the user message is `baseline`'s (page image and regions), only the system prompt differs, and the answer must pass `parse_translations`. A bad answer gets one repair call (the history plus the parse error, JSON output, no tools); `step_limit`, `incomplete` or a failed repair fall back to `run_baseline` with its own budget (`fallback_budget`; the agent has `agent_budget`), and the Run stays `translator` and records why. `provider_error` and `budget` never fall back (OUTLINE §3.6). `get_previous_context` returns up to 2 earlier pages' Japanese and their latest succeeded `translator` translations, read by `runs.previous_pages` before the run starts, so the handler touches no database; reference translations never enter it. The eval harness reports `fell_back` per page and `fallback_pages` in `scores.json`.
- Prompts contain only `text_ja` and the boxes. `segments` also stores the reference translations (`text_zh_hant`, `text_en`); they must never reach a prompt.
- A segment's `idx` (its position in `annotation.json`) is the region id in the prompt, the badge on the image and the row order. Import upserts with `ON CONFLICT DO UPDATE` so row ids stay stable; `INSERT OR REPLACE` would orphan `translations`.
- A Run is inserted as `running`, then finished. A failed Run stores `error` and `attempts_json` but no translations, and the page keeps showing the latest *succeeded* Run, whatever its config. A `baseline` Run's `attempts_json` is a list with one attempt per call, a failed request included (that attempt has `error` and its `record`); a `translator` Run's is a dict (`config`, `agent` with its steps, `repair`, `fallback`); a `RunCrashed` Run keeps `[]`. Empty or unparseable model output (DeepSeek sometimes returns empty content) is retried once, then the Run fails; output whose region ids are not exactly the page's ids (missing, unknown or duplicate) counts as unparseable. A `ProviderError` fails the Run at once, without a retry: the request may have been billed, and OUTLINE §3.6 leaves resending to the user. Once the Run row exists, an unexpected exception marks the Run failed before it propagates. Only a killed or interrupted process (Ctrl-C), or a database that fails during that recovery write, leaves a Run `running`.
- `runs.py` raises domain errors (`PageNotFound`, `NothingToTranslate`, `ImageMissing`; the client factory raises `LLMNotConfigured`), never `HTTPException`; the router does the mapping. These `TranslateError`s mean no Run was recorded. `RunCrashed` (carrying `run_id`) means a Run was recorded and marked failed; the API answers 500 for it. `validate_page` runs the same checks without a Run, and the eval harness calls it for every page before the first model call. `translate_page` takes a client factory so each Run gets its own `Meter` and its usage is its own. It picks the pipeline by `config` (`runs.CONFIGS`: `baseline`, `translator`); the API always runs `baseline`, and only the eval harness and callers of `translate_page` choose.
- Tracing (`tracing/`, OUTLINE §3.5, §8): off unless `FUKIDASHI_TRACING` names a backend (`phoenix` is the only one). Entry points (the API's `lifespan`, `eval/run.py`, the smoke script) call `tracing.setup()` once; library code never does. `runs.translate_page` runs inside `tracing.page_span`, so one page is one trace; `Outcome.trace_id` and `RunCrashed.trace_id` carry its ID, and eval rows and the manifest record it. Every exporter is wrapped in `redact.RedactingExporter`, which replaces data-URL images and the value of `LLM_API_KEY` in span and event attributes: `hide_input_images` alone is not enough, because the LangChain instrumentor writes whole conversations, images included, into `input.value`. The Phoenix endpoint must be on this machine, and whatever is specific to a backend lives in its one function in `tracing/backends.py`, so another backend is another function there. The exporter still honours `HTTP_PROXY`: with a proxy set, also set `NO_PROXY=localhost,127.0.0.1`. Tracing never fails a Run: spans leave through a batch processor and exporter errors are only logged. Before the first page, `eval/run.py` exits 2, with no Run, when a LangSmith variable is on or tracing refuses to start.
- Tests replace the FastAPI dependencies `get_db`, `get_dataset_root` and `get_client_factory` and use `FakeClient` and its variants (`UsageClient`, `CrashAfterCallClient`) from `backend/tests/conftest.py`. To test `LLMClient` itself, `mock_llm_client()` runs the real client and SDK on an `httpx.MockTransport` (responses built with `completion()`), so nothing leaves the process; the agent loop's tests (`test_agent_loop.py`) script the provider the same way. `FakeClient.chat` takes no `tools`. An autouse fixture in `conftest.py` clears the LangSmith tracing variables, `LLM_JSON_MODE`, `FUKIDASHI_TRACING` and `PHOENIX_*`, and calls `tracing.shutdown()` after each test, so a shell's variables cannot change the suite. Tracing tests plug an in-memory backend into `tracing.BACKENDS`, the same seam Phoenix uses.
- Eval references are `text_zh_hant` converted with OpenCC `t2s` (OUTLINE §8); hypotheses are the Run's stored Translations. Eval outputs live in `runs/eval/` (gitignored).
- `fukidashi.config.ROOT` comes from `__file__`, so the package must stay an editable install (`uv sync`).
- The frontend positions boxes in percent of the image's natural size. `frontend/src/api.ts` mirrors the pydantic models in `api/pages.py` by hand; change both together.

## Working agreements

- Outside the task workflow below, commit, push, open a PR and merge only when asked, each as its own step. PRs are merged with rebase and merge, so a merged branch looks unmerged to `git branch --merged`: check `git cherry -v origin/main <branch>` before `git branch -D`.
- Real LLM calls cost money (the whole project is capped at $100): manual verification is 1–2 pages.
- OpenMantra is CC BY-NC. Never commit `data/`, and keep screenshots that show manga pages under `runs/`.
- Scope changes only at the checkpoints (10/25, 12/13). Anything else goes into `IDEAS.md`.
- Starting the servers through the Claude app's `preview_start` hung in this setup (see `IDEAS.md`); start them from a shell.

## Task workflow (architect → executor → reviewer)

The user steers the project through three separate chat sessions. They share one thing: a task file. Human-oriented guide: `docs/workflow.md`.

| Session | Start with | Does | Never |
|---|---|---|---|
| Architect | `/architect` | Explains and steers the project with the user, edits plan docs (`OUTLINE.md`, `IDEAS.md`, `docs/`; `SCOPE.md` only at a checkpoint), writes tasks | Edits code, runs real LLM calls |
| Executor | `/execute-task NNN` | Implements one task on branch `task/NNN-slug`, verifies, commits, writes the Execution report | Edits the task's goal or criteria, widens scope, pushes, opens a PR, merges |
| Reviewer | `/review-task NNN` | Fresh context: checks the branch against the task, writes the Review and a verdict | Edits or commits repo files; trusts the Execution report without reproducing it |

Task files live in the main checkout's `tasks/` directory (gitignored), so every session sees them whether it runs in the main checkout or in a worktree. Resolve the directory with:

```bash
TASKS_DIR="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")/tasks"
```

The format is `docs/task-template.md`; the file is `$TASKS_DIR/NNN-slug.md`. The architect owns everything above `## Execution report`, the executor owns `## Execution report` and the `branch:` field, the reviewer owns `## Review`. Each role changes `status:` only for its own transitions:

```
proposed → in_progress → in_review → approved → done
```

- Executor: `proposed` or `changes_requested` → `in_progress` → `in_review`, or back to `proposed` when blocked (with a note).
- Reviewer: `in_review` → `approved` or `changes_requested`.
- Architect: `proposed`, `cancelled`, and `done` once the user has merged (record the PR number).

One task is one PR. The executor branches from a fresh `origin/main`, refuses to start in a working tree that has changes it did not make, stages explicit paths (never `git add -A`: another session may have left files), and may commit on its task branch, because being assigned the task is the authorization. Pushing, opening the PR and merging happen only when the user says so. Real LLM calls happen only if the task's Constraints allow them; the reviewer makes none unless they do too.
