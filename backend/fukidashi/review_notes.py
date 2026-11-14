"""The review of one page as the editor shows it: the Translator's Flags and the Reviewer's Issues (OUTLINE §0, Flag and
Verdict / Issue; §3.2). A note is a Flag or an Issue.

The notes are read from the stored Run, never produced again: nothing is called and nothing is stored for them. This module
is pure. It reads the rows it is given, and fetches another Run only through the `get_run` it is handed, so it holds no
database connection and imports neither `sqlite3` nor `store` (nor LangChain: the words of the stored records are
constants below, which a test pins to the modules that write them).

Which notes: those of the Run whose stages produced the displayed translation, the review's Run.
  - It is the page's displayed Run (its latest succeeded Run) when that is a `full` or `full_no_kb` Run. When the displayed
    Run is a `human` Run (a person's edit, OUTLINE §8), it is the first Run that is not `human` on the way down the
    `base_run_id`s, and every region the `human` Runs on the way edited is an edited region.
  - A `baseline` or `translator` Run has no review, and neither has a page with no succeeded Run or a chain that ends before
    it reaches a Run. All of these are None, never an error. So is a Run whose record cannot be read: the page still loads.

What is current: a note is current when it is about the stored translation and that text is still the model's.
  - A Flag is current when the stage that made it is the one that produced the stored translation (`final.stage`) and its
    region is not edited.
  - An Issue is current when the review that gave it came right after that stage and gave a verdict (status `ok`), and its
    region is not edited. After a `send_back_limit` no review follows `final.stage`, so no Issue is current. After a
    `revision_unusable` the review that sent the page back is the one right after it, so its `revise` verdict is current.
The notes that are not current are kept, with the pass or the review that made them, because they say what was doubted and
what was sent back for. A note on an edited region is not current whichever stage made it: it was about the text before the
edit.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Protocol

# The words of the stored records. They are the Orchestrator's (`orchestrator/flow.py`: its stage kinds, the status `ok`,
# the config names) and the store's (`store.HUMAN`); `test_review_notes.py` pins each of them to its source.
HUMAN = "human"
ORCHESTRATOR_CONFIGS = ("full_no_kb", "full")
TRANSLATE, REVIEW, REVISE = "translate", "review", "revise"
OK = "ok"


class RunRow(Protocol):
    """A row of `runs`: `id`, `config` and `attempts_json` are read. `store.get_run` gives a `sqlite3.Row`, a test a dict."""

    def __getitem__(self, key: str, /) -> Any: ...


def build_review(run: RunRow | None, get_run: Callable[[int], RunRow | None]) -> dict | None:
    """The review of the page whose displayed Run is `run` (None: it has no succeeded Run), or None when it has none.

    `get_run(run_id)` returns the Run with that id, or None; it is how the `base_run_id`s of `human` Runs are followed. The
    result is the JSON of the API's `PageReview`:
      {"run_id", "config",       # the review's Run: the Orchestrator Run, not the `human` Run that may sit on it
       "send_backs", "final_stop",  # from that Run's record
       "edited": [region, ...],     # sorted: every region a `human` Run on the way edited
       "flags":  [{"region", "reason", "pass", "current"}, ...],
       "issues": [{"region", "comment", "suggestion", "verdict", "review", "current"}, ...]}
    A Flag's `pass` is 0 for the first pass and n for the nth revision; an Issue's `review` is 1 for the first review, 2 for
    the second, and its `verdict` is that review's. Both lists are in the order of the stages, then of the stage's own list.
    """
    edited: set[int] = set()
    seen: set[int] = set()
    while run is not None and run["config"] == HUMAN:
        if run["id"] in seen:  # a record that leads back to a Run already read: none does, but this must not go round
            return None
        seen.add(run["id"])
        record = _record(run)
        edited |= _edited_regions(record)
        base = record.get("base_run_id")
        run = get_run(base) if _is_int(base) else None
    if run is None or run["config"] not in ORCHESTRATOR_CONFIGS:
        return None
    return _review_of(run, _record(run), edited)


def _review_of(run: RunRow, record: dict, edited: set[int]) -> dict | None:
    stages, final, send_backs = record.get("stages"), record.get("final"), record.get("send_backs")
    if not isinstance(stages, list) or not isinstance(final, dict) or not _is_int(send_backs):
        return None
    produced_by, stop = final.get("stage"), final.get("stop")
    if not _is_int(produced_by) or not 0 <= produced_by < len(stages) or not isinstance(stop, str):
        return None  # a Run that stored no translation, or a record of another shape

    flags: list[dict] = []
    issues: list[dict] = []
    revisions = reviews = 0  # the stages of each kind met so far: the pass of a Flag and the number of a review
    for at, stage in enumerate(stages):
        if not isinstance(stage, dict):
            continue
        kind = stage.get("kind")
        if kind in (TRANSLATE, REVISE):
            if kind == REVISE:
                revisions += 1  # the first pass is pass 0, the nth revision pass n
            about_stored = at == produced_by
            for flag in _entries(stage, "flags"):
                flags.append({"region": flag["id"], "reason": _text(flag.get("reason")), "pass": revisions,
                              "current": about_stored and flag["id"] not in edited})
        elif kind == REVIEW:
            reviews += 1
            about_stored = at == produced_by + 1 and stage.get("status") == OK
            verdict = stage.get("verdict")
            for issue in _entries(stage, "issues"):
                suggestion = issue.get("suggestion")
                issues.append({"region": issue["id"], "comment": _text(issue.get("comment")),
                               "suggestion": suggestion if isinstance(suggestion, str) else None,
                               "verdict": verdict if isinstance(verdict, str) else None, "review": reviews,
                               "current": about_stored and issue["id"] not in edited})
    return {"run_id": run["id"], "config": run["config"], "send_backs": send_backs, "final_stop": stop,
            "edited": sorted(edited), "flags": flags, "issues": issues}


def _record(run: RunRow) -> dict:
    """The Run's attempts record, or an empty one when it has none or cannot be read."""
    try:
        record = json.loads(run["attempts_json"])
    except (TypeError, ValueError):  # no record (NULL), or text that is not JSON
        return {}
    return record if isinstance(record, dict) else {}  # a `baseline` Run's record is a list: it has no review


def _edited_regions(record: dict) -> set[int]:
    edits = record.get("edits")
    if not isinstance(edits, list):
        return set()
    return {edit["idx"] for edit in edits if isinstance(edit, dict) and _is_int(edit.get("idx"))}


def _entries(stage: dict, field: str) -> list[dict]:
    """The Flags or Issues a stage stored: the entries that name a region. Anything else in the list is left out."""
    entries = stage.get(field)
    return [e for e in entries if isinstance(e, dict) and _is_int(e.get("id"))] if isinstance(entries, list) else []


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""
