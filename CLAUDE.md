# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fukidashi is a web editor that translates Japanese manga into Simplified Chinese (zh-Hans) and is evaluated on OpenMantra against a single-call `baseline`. The plan is written down; read it instead of guessing:

- `OUTLINE.md`: design, milestones (§7), decisions (§8), pivot rules (§9). Its §0 Terminology table is canonical.
- `SCOPE.md`: what is in and out of scope, and the cut order. `IDEAS.md`: where ideas outside a checkpoint go.
- `docs/design.md`: domain model, system sequence diagram, object interaction diagram, object model. `docs/m0.md`, `docs/m1.md`: milestone records.

M0 and M1 are done: the vertical slice (page → `baseline` → browser) works. M2 is in progress: the eval harness v0 exists (`eval/run.py`: `baseline` only, chrF / BLEU, manifest). Agents, tools, the Knowledge Base, Reviewer, tracing, COMET, GEMBA-MQM and term accuracy do not exist yet; do not describe them as implemented.

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
uv run python eval/run.py --config baseline --book tojime_no_siora --pages 2,3,4   # real calls, one per page; writes runs/eval/<id>/
```

There is no Python linter and there are no frontend tests. `.env` (template: `.env.example`) holds the LLM provider and prices: never read or print it. `FUKIDASHI_DB` overrides the DB path (default `runs/fukidashi.db`).

## Architecture

One request shows the layering. In the browser, `PageView` POSTs `/api/pages/:book/:index/translate` (the Vite dev server proxies `/api`):

```
api/pages.py        parse the request, map errors to 404 / 422 / 503 / 502, shape the response; no SQL
 → runs.py          translate_page(): validate_page → make_client() → store.start_run → baseline → store.finish_run
    → pipeline/baseline.py   one call per page: image + Japanese regions in, zh-Hans JSON out
       → llm/client.py       LLMClient, the only code that talks to the provider
store.py            all SQL; schema.sql defines the tables
datasets/           OpenMantra loader and the importer
evaluation/         eval library: harness, OpenCC references, sacreBLEU metrics, manifest. Its CLI, eval/run.py,
                    calls runs.translate_page like the API does and writes runs/eval/<id>/
```

Invariants that span several files:

- Every model call goes through `LLMClient`, which enforces the per-page `PageBudget` and records each call in a `Meter` (ledger `runs/usage.jsonl`). Usage the provider does not return is unknown, never 0: it shows up as `unknown_usage_calls` and NULL token counts on the Run. `cost_usd` is NULL when any call had no usage or no price (`Meter.summary`); it is never a partial sum. The eval harness keeps the same rule for totals (`cost_summary`) and reports scored and failed Runs' cost separately.
- Prompts contain only `text_ja` and the boxes. `segments` also stores the reference translations (`text_zh_hant`, `text_en`); they must never reach a prompt.
- A segment's `idx` (its position in `annotation.json`) is the region id in the prompt, the badge on the image and the row order. Import upserts with `ON CONFLICT DO UPDATE` so row ids stay stable; `INSERT OR REPLACE` would orphan `translations`.
- A Run is inserted as `running`, then finished. A failed Run stores `error` and `attempts_json` but no translations, and the page keeps showing the latest *succeeded* Run. Empty or unparseable model output (DeepSeek sometimes returns empty content) is retried once, then the Run fails; output whose region ids are not exactly the page's ids (missing, unknown or duplicate) counts as unparseable. Once the Run row exists, an unexpected exception marks the Run failed before it propagates. Only a killed or interrupted process (Ctrl-C), or a database that fails during that recovery write, leaves a Run `running`.
- `runs.py` raises domain errors (`PageNotFound`, `NothingToTranslate`, `ImageMissing`; the client factory raises `LLMNotConfigured`), never `HTTPException`; the router does the mapping. These `TranslateError`s mean no Run was recorded. `RunCrashed` (carrying `run_id`) means a Run was recorded and marked failed; the API answers 500 for it. `validate_page` runs the same checks without a Run, and the eval harness calls it for every page before the first model call. `translate_page` takes a client factory so each Run gets its own `Meter` and its usage is its own. It currently hard-wires `baseline`; M2 makes it pick by `config`.
- Tests replace the FastAPI dependencies `get_db`, `get_dataset_root` and `get_client_factory` and use `FakeClient` and its variants (`UsageClient`, `CrashAfterCallClient`) from `backend/tests/conftest.py`.
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
