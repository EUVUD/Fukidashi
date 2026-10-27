# IDEAS

> Ideas that come up outside a checkpoint go here and are not built yet. Scope changes only at checkpoints (10/25, 12/13); see [SCOPE.md](SCOPE.md) rule 1 and [OUTLINE.md](OUTLINE.md) §9. Anything that comes up in between waits for the next checkpoint.
> Each row has a date, the idea, why, and when to revisit it. When an idea is done, strike it through and note the milestone or PR; when it is dropped, give the reason.

| Date | Idea | Why | Revisit | Status |
|---|---|---|---|---|
| 2026-10-24 | Give `pipeline/baseline.py` an exception type that does not depend on `openai`: `LLMClient` raises its own errors and the pipeline stops importing `openai` | The pipeline imports the SDK directly to catch provider errors, so the SDK leaks into the pipeline layer | M2, when the tool loop needs uniform handling of provider errors | open |
| 2026-10-24 | Move `Page` / `Segment` out of `datasets/openmantra.py` into a dataset-independent place | The pipeline and the API both depend on them; M7's end-to-end path (an uploaded page, not OpenMantra) needs a type of its own | M7, before detection + OCR | open |
| 2026-10-24 | Rows with `runs.status = 'running'` stay that way if the server is killed mid-request; clicking Translate twice on one page has no server-side guard | Belongs to OUTLINE §3.6 Reliability (the page is the retry unit) | M2, together with reliability | open |
| 2026-10-24 | Add a test for the 503 returned when `.env` is missing | This path exists only in code; it has no test and has never been run by hand | Next time `api/` or `LLMClient` changes | open |
| 2026-10-24 | CI: GitHub Actions running `pytest`, `tsc` and `oxlint` | Tests and lint are run by hand | After M3, or before the repository goes public in M9 | open |
| 2026-10-24 | Find out why `.claude/launch.json` does not start in the Claude app's preview | The processes run but never listen on their ports, with both `uv run` and `.venv/bin/python`; possibly macOS's permission prompt for `~/Documents`, unconfirmed | Next time the frontend needs a preview in the app | open |
