"""Helpers shared by the rename tests (task 042): a small book of several pages, the Runs its pages display, an Editor KB with
an entry and a profile, and the snapshot of every table a rename could touch. A module of its own: conftest.py is not touched.

The book is `book_a` (what `kb_support` and the fixtures assume), with four pages. The entry is 結界 → 结界 and the profile is
ユキ → 小雪 (also called ゆき). Each box is (Japanese, displayed translation):

    page 1 (a `full` Run)     0 結界を張る！ | 布下结界！            the entry changes it
                              1 ユキ、結界の外へ | 小雪，到结界外面去  both the entry and the profile change it
                              2 バリアが消えた | 结界消失了          holds the entry's rendering but not its source: never touched
    page 2 (a `full` Run)     0 結界は結界だ | 结界就是结界          two occurrences
                              1 ゆきが笑う | 小雪笑了                the profile, through its alias
    page 3 (a `baseline` Run) 0 結界の中は安全 | 结界里很安全        the entry changes it
                              1 ユキの声 | 那是小雪的声音            the profile changes it
                              2 結界？ | 什么？                      the source, but no rendering: only listed
    page 4 (a failed Run)     0 結界とユキ | (none)                  no succeeded Run: only listed, with no `before`

So renaming the entry changes 4 boxes on pages 1 to 3 and lists 2, and renaming the profile changes 3 boxes on pages 1 to 3 and
lists 1.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kb_support import accept, name, term

from fukidashi import store
from fukidashi.kb.curator import apply
from fukidashi.llm.metering import Meter

BOOK = "book_a"
ENTRY_SOURCE, ENTRY_RENDERING = "結界", "结界"
NAME_SOURCE, NAME_RENDERING, NAME_ALIAS = "ユキ", "小雪", "ゆき"

PAGES: dict[int, list[tuple[str, str | None]]] = {
    1: [("結界を張る！", "布下结界！"), ("ユキ、結界の外へ", "小雪，到结界外面去"), ("バリアが消えた", "结界消失了")],
    2: [("結界は結界だ", "结界就是结界"), ("ゆきが笑う", "小雪笑了")],
    3: [("結界の中は安全", "结界里很安全"), ("ユキの声", "那是小雪的声音"), ("結界？", "什么？")],
    4: [("結界とユキ", None)],
}
# The Run each page displays (page 4 has a failed one only).
DISPLAYED_BY = {1: "full", 2: "full", 3: "baseline"}

# The tables a rename may write, and the ones it must not: a snapshot of all of them compares "nothing was written".
TABLES = ("runs", "translations", "run_events", "kbs", "glossary_entries", "character_profiles", "proposals")


def write_world(root: Path) -> Path:
    """The dataset of `PAGES` (an `annotation.json`; the rename tests need no image)."""
    root.mkdir(parents=True, exist_ok=True)
    pages = [{"page_index": index, "image_paths": {"ja": f"images/{BOOK}/ja/{index:03d}.jpg"}, "frame": [],
              "text": [{"x": 10 * i, "y": 20, "w": 30, "h": 40, "text_ja": ja, "text_en": "REF-EN", "text_zh": "REF-ZH"}
                       for i, (ja, _) in enumerate(boxes)]} for index, boxes in PAGES.items()]
    (root / "annotation.json").write_text(json.dumps([{"book_title": BOOK, "pages": pages}]), encoding="utf-8")
    return root


def finished_run(conn, page_index: int, config: str, texts: dict[int, str] | None, attempts=None) -> int:
    """A Run of `config` on the page that finishes with `texts` (segment idx -> zh-Hans); None makes it a failed Run."""
    page_id = store.find_page(conn, BOOK, page_index)["id"]
    seg_ids = {s["idx"]: s["id"] for s in store.page_segments(conn, page_id)}
    run_id = store.start_run(conn, page_id=page_id, config=config, params={}, model="some-model")
    store.finish_run(
        conn, run_id, error=None if texts is not None else "boom", response_models=[], usage=Meter().summary(),
        attempts={} if attempts is None else attempts,
        translations=None if texts is None else {seg_ids[i]: t for i, t in texts.items()},
    )
    return run_id


def displayed(conn, page_index: int) -> list[str | None]:
    """What the page displays: the translations of its latest succeeded Run, by box (None where there is none)."""
    page_id = store.find_page(conn, BOOK, page_index)["id"]
    run = store.latest_succeeded_run(conn, page_id)
    return [r["text_zh_hans"] for r in store.segments_with_translations(conn, page_id, run["id"] if run else None)]


def displayed_run(conn, page_index: int):
    return store.latest_succeeded_run(conn, store.find_page(conn, BOOK, page_index)["id"])


def snapshot(conn) -> dict[str, list[tuple]]:
    """Every row of every table a rename could write, timestamps included: equal snapshots mean nothing was written."""
    return {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")] for t in TABLES}


def attempts_of(conn, run_id: int) -> dict:
    return json.loads(store.get_run(conn, run_id)["attempts_json"])


def translations_of(conn, run_id: int) -> dict[int, str]:
    """The Run's Translations, by the box's `idx`."""
    return {r["idx"]: r["text_zh_hans"] for r in conn.execute(
        "SELECT s.idx, t.text_zh_hans FROM translations t JOIN segments s ON s.id = t.segment_id WHERE t.run_id = ?"
        " ORDER BY s.idx", (run_id,))}


def full_record(flag_regions=(), issue_regions=()) -> dict:
    """An Orchestrator Run's attempts (the shape `review_notes` reads): a first pass that flagged `flag_regions`, then one
    review that approved with notes on `issue_regions`."""
    return {"config": "full", "send_backs": 0,
            "stages": [{"kind": "translate", "status": "ok", "flags": [{"id": r, "reason": "unsure"} for r in flag_regions]},
                       {"kind": "review", "status": "ok", "verdict": "approve",
                        "issues": [{"id": r, "comment": "fine", "suggestion": None} for r in issue_regions]}],
            "final": {"stage": 0, "kind": "translate", "reviewed": True, "stop": "approved"}}


def build_world(conn) -> SimpleNamespace:
    """The book's Runs and its Editor KB, which holds the entry (`approved`, made by the agent on page 1) and the profile
    (`pending_human`: a new name waits for a person). Returns the ids a test needs. `conn` is committed and idle."""
    runs = {}
    for page_index, config in DISPLAYED_BY.items():
        texts = {i: zh for i, (_, zh) in enumerate(PAGES[page_index])}
        attempts = full_record(flag_regions=(0, 1), issue_regions=(1, 2)) if config == "full" else None
        runs[page_index] = finished_run(conn, page_index, config, texts, attempts)
    runs[4] = finished_run(conn, 4, "full", None)  # a failed Run: page 4 displays nothing
    kb = store.editor_kb(conn, BOOK)
    entry, profile = term(kb, 1, ENTRY_SOURCE, ENTRY_RENDERING), name(kb, 1, NAME_SOURCE, NAME_RENDERING, aliases=[NAME_ALIAS])
    apply(conn, kb, 1, runs[1], [entry, profile], {entry.proposal_id: accept()})
    return SimpleNamespace(
        kb=kb, runs=runs, entry=store.find_entry(conn, kb, ENTRY_SOURCE)["id"],
        profile=store.find_profile(conn, kb, NAME_SOURCE)["id"], new_name=profile.proposal_id,
    )
