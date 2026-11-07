"""The live trace's contract (OUTLINE §0, Live trace event; §8, "Where the live trace's events come from").

A Run reports what it does as it does it, to an optional sink: code inside the Run calls `emit(kind, data)`, and
`runs.translate_page` turns each call into a `RunEvent` of the Run (its id, a `seq` that rises by 1 from 1) for its
caller's `on_event`. Without a sink every `emit` is `None` and the code takes exactly the path it takes without a live
trace. A sink only observes: it changes no request, no stored Run and no cost.

This module imports only the standard library, so the agents and the Orchestrator can import it without loading a
database. Every `data` is a JSON object (`json.dumps(data, allow_nan=False)` accepts it). An event carries no page
image, no user message, no reasoning text (only its length) and no Reference translation.

The kinds, who emits them and what their `data` holds:

- `run_start`: `runs.py`, right after the Run row exists. `run_id`, `config`, `model`.
- `stage_start`: the Orchestrator; `runs.py` for finalize. `stage` (`translate`, `review`, `revise` or `finalize`) and
  `index`, the stage's position in the attempts' `stages` (null for `finalize`).
- `model_call`: `run_agent`, once per model call that returned or failed. `step` (1-based: the call's index in the
  agent's `steps`); `record` (the call as the Meter has it, or null), `finish_reason`, `text`, `reasoning_chars` (null:
  the reply had no `reasoning_content`) and `error` (null unless the call failed), as that step has them; and
  `tool_calls`, a list of `id`, `name` and `arguments` as the model asked for them. A Step limit or a spent budget sends
  no request, so it has no `model_call`; the stage's `stage_end` carries the error.
- `tool_result`: `run_agent`, once per tool message. `tool_call_id`, `name`, `result`, `error` (one of the two is null),
  `ran` (the tool's handler was called) and `duplicate` (the call was refused as a repeat), as the steps read them.
- `repair`: the Translator, the Reviewer and the Curator, after their repair call. `record` (the call as the Meter has
  it, or null), `error` (null unless the call failed) and `parse_error`, what was wrong with the answer that made them
  repair it. A repair that was not sent, because the budget was spent, has no event.
- `fallback`: the Translator, before it runs `baseline`. `reason`, the `fallback.reason` of its attempts.
- `stage_end`: the Orchestrator; `runs.py` for finalize. `stage`, `index`, `status` (`ok`, `unusable` or `failed`; for
  finalize the Curator's: `ok`, `unusable`, `failed` or `skipped`) and `error`. A `translate` or `revise` stage also
  has `flags`, a `review` stage `verdict` and `issues`, and finalize `results`: how many proposals got each of
  `kb.curator.RESULTS`. They are the stored stage's.
- `run_end`: `runs.py`, after the Run row is finished. `status` (`succeeded` or `failed`), `error`, `calls`,
  `unknown_usage_calls`, `input_tokens`, `output_tokens` and `cost_usd`, all as the stored row has them; for
  `full_no_kb` and `full` also `send_backs`, `final_stop` and `final_reviewed`.

A `baseline` Run emits `run_start` and `run_end` only (its one call is not an agent step). A `translator` Run has no
stages. A Run that crashes after its row exists ends with `run_end` and status `failed`; when marking it failed fails
too, there is no `run_end`. A page refused before its Run exists emits nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

RUN_START = "run_start"
STAGE_START = "stage_start"
MODEL_CALL = "model_call"
TOOL_RESULT = "tool_result"
REPAIR = "repair"
FALLBACK = "fallback"
STAGE_END = "stage_end"
RUN_END = "run_end"
KINDS = (RUN_START, STAGE_START, MODEL_CALL, TOOL_RESULT, REPAIR, FALLBACK, STAGE_END, RUN_END)

# The stage of the Run's end, after the Orchestrator; its start and end events are `runs.py`'s.
FINALIZE = "finalize"

# What code inside a Run calls: `emit(kind, data)`. It is called in the thread that called `translate_page`, and it must
# not raise (the one `translate_page` passes down never does).
Emit = Callable[[str, dict], None]


@dataclass(frozen=True)
class RunEvent:
    """One live trace event of one Run, as `translate_page`'s caller receives it. `seq` is 1 for a Run's first event and
    rises by 1 with each; `kind` is one of `KINDS`."""

    run_id: int
    seq: int
    kind: str
    data: dict
