"""A person's rename of a KB entry's or profile's rendering, and the book's boxes that follow (task 042; OUTLINE §8, "Rename →
whole-book sync"): `preview_*_rename` and `rename_*` of `kb/curator.py`, one transaction each, over the book of
`rename_support`. The rule that picks the boxes is tested apart (`test_kb_rename_boxes.py`), the routes in
`test_kb_rename_api.py`."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pytest
from kb_support import dump, name, term
from rename_support import (
    BOOK, ENTRY_SOURCE, NAME_SOURCE, attempts_of, build_world, displayed, displayed_run, snapshot, translations_of,
    write_world,
)

from fukidashi import runs, store
from fukidashi.datasets.importer import import_openmantra
from fukidashi.db import connect
from fukidashi.kb import KBNotFound
from fukidashi.kb.curator import (
    ENTRY, HUMAN_OUTCOMES, PROFILE, RESULTS, DecisionRefused, ItemNotFound, PageRun, RenamePlan, apply, decide_entry,
    decide_profile, preview_entry_rename, preview_profile_rename, rename_entry, rename_profile,
)
from fukidashi.kb.entities import make_proposal
from fukidashi.kb.rename import BoxChange, InvalidRendering
from fukidashi.review_notes import build_review


@pytest.fixture
def dataset(tmp_path):
    """The book of `rename_support`, in place of conftest's: no page of it needs an image."""
    return write_world(tmp_path / "world")


@pytest.fixture
def world(conn, dataset):
    import_openmantra(conn, dataset)
    return build_world(conn)


# --- what the tests need to read and to set up ---

ENTRY_CHANGES = [  # (page, idx, before, after) of renaming 结界 to 屏障
    (1, 0, "布下结界！", "布下屏障！"), (1, 1, "小雪，到结界外面去", "小雪，到屏障外面去"),
    (2, 0, "结界就是结界", "屏障就是屏障"), (3, 0, "结界里很安全", "屏障里很安全"),
]
ENTRY_LISTED = [(3, 2, "什么？", None), (4, 0, None, None)]
PROFILE_CHANGES = [  # renaming 小雪 to 雪乃: box 1 of page 2 is matched through the alias ゆき
    (1, 1, "小雪，到结界外面去", "雪乃，到结界外面去"), (2, 1, "小雪笑了", "雪乃笑了"), (3, 1, "那是小雪的声音", "那是雪乃的声音"),
]
PROFILE_LISTED = [(4, 0, None, None)]


def boxes(items: tuple[BoxChange, ...]) -> list[tuple]:
    return [(b.page_index, b.idx, b.before, b.after) for b in items]


def entry_row(conn, entry_id):
    return store.get_entry(conn, store.find_editor_kb(conn, BOOK)["id"], entry_id)


def profile_row(conn, profile_id):
    return store.get_profile(conn, store.find_editor_kb(conn, BOOK)["id"], profile_id)


def proposal_row(conn, proposal_or_id):
    pid = getattr(proposal_or_id, "proposal_id", proposal_or_id)
    return conn.execute("SELECT * FROM proposals WHERE proposal_id = ?", (pid,)).fetchone()


def history(row) -> list[dict]:
    return json.loads(row["history"])


def without_time(line: dict) -> dict:
    datetime.fromisoformat(line["at"])  # a time must be there
    return {k: v for k, v in line.items() if k != "at"}


def pages_of(row) -> list[int]:
    return json.loads(row["evidence_pages"])


def run_ids(conn) -> list[int]:
    return [r["id"] for r in conn.execute("SELECT id FROM runs ORDER BY id")]


def page_run(conn, page_index) -> int:
    """The id of the Run the page displays (the page has one)."""
    return displayed_run(conn, page_index)["id"]


def stage_running_run(conn, page_index) -> int:
    page_id = store.find_page(conn, BOOK, page_index)["id"]
    return store.start_run(conn, page_id=page_id, config="full", params={}, model="m")


def add_conflicts(conn, world):
    """Proposals on the entry's source that the rename to 屏障 meets: two term conflicts that propose 屏障 (pages 2 and 3), one
    that proposes 封锁, a character conflict that proposes 屏障 too (rule 6's), and an undecided term for 屏障."""
    kb, runs_ = world.kb, world.runs
    agree_2, agree_3, other = term(kb, 2, ENTRY_SOURCE, "屏障"), term(kb, 3, ENTRY_SOURCE, "屏障"), term(kb, 4, ENTRY_SOURCE, "封锁")
    character = name(kb, 3, "ユキちゃん", "屏障")  # rule 6: it matches the profile ユキ, and 屏障 does not agree with 小雪
    apply(conn, kb, 2, runs_[2], [agree_2])
    apply(conn, kb, 3, runs_[3], [agree_3, character])
    apply(conn, kb, 4, runs_[4], [other])
    undecided = make_proposal(kb, 1, "term", {"source_term": ENTRY_SOURCE, "target_term": "屏障"}, "translator")
    assert store.insert_proposal(conn, proposal_id=undecided.proposal_id, kb_id=kb, kind="term", payload=undecided.payload,
                                 proposer="translator", page_index=1, run_id=runs_[1])
    conn.commit()
    assert [proposal_row(conn, p)["status"] for p in (agree_2, agree_3, character, other, undecided)] == [
        "pending_human", "pending_human", "pending_human", "pending_human", "proposed"]
    return agree_2, agree_3, other, character, undecided


def refused(conn, call, match):
    """The rename is refused with `DecisionRefused`, and nothing is written, not even by a transaction left open."""
    before = snapshot(conn)
    with pytest.raises(DecisionRefused, match=match):
        call()
    assert snapshot(conn) == before
    assert not conn.in_transaction


# --- the preview ---


def test_the_preview_of_an_entry_says_which_boxes_change_and_which_are_only_listed(conn, world):
    plan = preview_entry_rename(conn, world.kb, world.entry, "屏障")

    assert (plan.kind, plan.id, plan.before, plan.after) == (ENTRY, world.entry, "结界", "屏障")
    assert boxes(plan.changes) == ENTRY_CHANGES
    assert boxes(plan.listed) == ENTRY_LISTED
    assert plan.settles == ()
    assert plan.changes[0].text_ja == "結界を張る！" and plan.listed[1].text_ja == "結界とユキ"
    assert (1, 2) not in {(b.page_index, b.idx) for b in (*plan.changes, *plan.listed)}  # the decoy: 结界, but no 結界


def test_the_preview_of_a_profile_matches_its_aliases_too(conn, world):
    plan = preview_profile_rename(conn, world.kb, world.profile, "雪乃")

    assert (plan.kind, plan.id, plan.before, plan.after) == (PROFILE, world.profile, "小雪", "雪乃")
    assert boxes(plan.changes) == PROFILE_CHANGES  # page 2's box says ゆき, an alias
    assert boxes(plan.listed) == PROFILE_LISTED
    assert plan.settles == (world.new_name,)  # the new-name proposal waits for a person, and the rename settles it


@pytest.mark.parametrize("preview, row, target", [
    (preview_entry_rename, "entry", "屏障"), (preview_profile_rename, "profile", "雪乃"),
], ids=["entry", "profile"])
def test_a_preview_writes_nothing(conn, world, preview, row, target):
    add_conflicts(conn, world)
    before = snapshot(conn)

    preview(conn, world.kb, getattr(world, row), target)

    assert snapshot(conn) == before  # no row of the KB, no Run, no Translation, no event
    assert not conn.in_transaction  # and the view it read in is over
    assert displayed(conn, 1) == ["布下结界！", "小雪，到结界外面去", "结界消失了"]


@pytest.mark.parametrize("preview, row, target", [
    (preview_entry_rename, "entry", "屏障"), (preview_profile_rename, "profile", "雪乃"),
], ids=["entry", "profile"])
def test_a_preview_takes_no_write_lock_so_it_never_waits_for_a_writer(conn, world, db_path, preview, row, target):
    """Another connection is in the middle of a write (a `full` Run's finalize, say): the preview reads one view of the
    database beside it, as a reader does in WAL mode. A preview that took the write lock would wait for the writer and fail."""
    writer = connect(db_path)
    writer.execute("BEGIN IMMEDIATE")
    try:
        plan = preview(conn, world.kb, getattr(world, row), target)
    finally:
        writer.rollback()
        writer.close()

    assert plan.changes


def test_the_preview_normalises_the_new_rendering_as_the_kb_does(conn, world):
    plan = preview_entry_rename(conn, world.kb, world.entry, "  屏障　")

    assert plan.after == "屏障" and boxes(plan.changes) == ENTRY_CHANGES


def test_the_write_makes_the_plan_the_preview_made(conn, world):
    add_conflicts(conn, world)
    previewed = preview_entry_rename(conn, world.kb, world.entry, "屏障")

    result = rename_entry(conn, world.kb, world.entry, "屏障")

    assert result.plan == previewed
    assert isinstance(result.plan, RenamePlan) and isinstance(result.runs[0], PageRun)
    again = preview_profile_rename(conn, world.kb, world.profile, "雪乃")
    assert rename_profile(conn, world.kb, world.profile, "雪乃").plan == again


# --- an entry renamed ---


def test_an_entry_renamed_holds_the_new_rendering_approved_and_made_by_a_person(conn, world):
    rename_entry(conn, world.kb, world.entry, "屏障")

    entry = entry_row(conn, world.entry)
    assert (entry["source_term"], entry["target_term"], entry["status"], entry["provenance"]) == (
        ENTRY_SOURCE, "屏障", "approved", "human")
    assert pages_of(entry) == [1]  # the evidence is what it was
    created, renamed = history(entry)
    assert created["event"] == "created" and "by" not in created  # the agent's line is kept as it was
    assert without_time(renamed) == {"event": "renamed", "by": "human", "from": "approved", "before": "结界", "after": "屏障"}
    assert entry["updated_at"] >= entry["created_at"]


def test_an_entry_renamed_writes_one_human_edit_for_each_page_it_changes(conn, world):
    bases = {i: page_run(conn, i) for i in (1, 2, 3)}
    assert displayed_run(conn, 4) is None  # page 4's only Run failed

    result = rename_entry(conn, world.kb, world.entry, "屏障")

    assert [(r.page_index, isinstance(r.run_id, int)) for r in result.runs] == [(1, True), (2, True), (3, True)]
    assert [r.run_id for r in result.runs] == sorted(r.run_id for r in result.runs)  # written in page order
    expected = {
        1: {0: "布下屏障！", 1: "小雪，到屏障外面去", 2: "结界消失了"},  # the decoy (2) is the base's, as it was
        2: {0: "屏障就是屏障", 1: "小雪笑了"},
        3: {0: "屏障里很安全", 1: "那是小雪的声音", 2: "什么？"},  # 2 is only listed
    }
    for made in result.runs:
        run = store.get_run(conn, made.run_id)
        assert (run["config"], run["model"], run["status"], run["config_json"]) == ("human", "human", "succeeded", "{}")
        assert (run["calls"], run["unknown_usage_calls"], run["cost_usd"]) == (0, 0, 0.0)  # nothing was called: $0, not unknown
        assert (run["input_tokens"], run["output_tokens"], run["response_model"], run["error"]) == (None, None, None, None)
        assert run["started_at"] and run["finished_at"] == run["started_at"]
        assert translations_of(conn, made.run_id) == expected[made.page_index]
        assert page_run(conn, made.page_index) == made.run_id  # so the page shows it
        assert translations_of(conn, bases[made.page_index]) != expected[made.page_index]  # the base keeps its own
    assert displayed_run(conn, 4) is None and displayed(conn, 4) == [None]  # no succeeded Run, so no Human edit there
    assert len(run_ids(conn)) == 4 + 3  # the book's four Runs (page 4's failed one included) and the three Human edits


def test_a_human_edit_names_its_base_lists_every_box_it_changed_and_says_which_rename_made_it(conn, world):
    bases = {i: page_run(conn, i) for i in (1, 2, 3)}

    result = rename_entry(conn, world.kb, world.entry, "屏障")

    by_page = {r.page_index: attempts_of(conn, r.run_id) for r in result.runs}
    rename = {"kind": "entry", "id": world.entry, "kb_id": world.kb, "before": "结界", "after": "屏障"}
    assert by_page == {
        1: {"config": "human", "base_run_id": bases[1], "rename": rename, "edits": [
            {"idx": 0, "before": "布下结界！", "after": "布下屏障！"},
            {"idx": 1, "before": "小雪，到结界外面去", "after": "小雪，到屏障外面去"}]},
        2: {"config": "human", "base_run_id": bases[2], "rename": rename, "edits": [
            {"idx": 0, "before": "结界就是结界", "after": "屏障就是屏障"}]},
        3: {"config": "human", "base_run_id": bases[3], "rename": rename, "edits": [
            {"idx": 0, "before": "结界里很安全", "after": "屏障里很安全"}]},
    }
    assert list(by_page[1]) == ["config", "base_run_id", "edits", "rename"]


def test_the_review_of_a_page_reads_a_renames_human_edit_as_edits_on_those_regions(conn, world):
    """Page 1 displays a `full` Run with Flags on regions 0 and 1 and Issues on 1 and 2. The rename's Human edit over it
    changed regions 0 and 1, so `build_review` finds the `full` Run behind it and reports exactly those as edited."""
    full = page_run(conn, 1)

    rename_entry(conn, world.kb, world.entry, "屏障")

    shown = displayed_run(conn, 1)
    assert shown["config"] == "human"
    review = build_review(shown, lambda run_id: store.get_run(conn, run_id))
    assert (review["run_id"], review["config"], review["edited"]) == (full, "full", [0, 1])
    assert [(f["region"], f["current"]) for f in review["flags"]] == [(0, False), (1, False)]  # about the text before the edit
    assert [(i["region"], i["current"]) for i in review["issues"]] == [(1, False), (2, True)]  # region 2 was not edited
    # a page whose displayed Run is `baseline` has no review, edited or not
    assert build_review(displayed_run(conn, 3), lambda run_id: store.get_run(conn, run_id)) is None


def test_a_rename_is_based_on_what_a_person_edited_before(conn, world):
    first = runs.edit_translation(conn, BOOK, 1, 0, "展开了一道结界的防御")  # the old rendering is still in it

    result = rename_entry(conn, world.kb, world.entry, "屏障")

    page_1 = next(r for r in result.runs if r.page_index == 1)
    assert attempts_of(conn, page_1.run_id)["base_run_id"] == first  # the page's latest succeeded Run, of any config
    assert attempts_of(conn, page_1.run_id)["edits"][0] == {"idx": 0, "before": "展开了一道结界的防御", "after": "展开了一道屏障的防御"}
    assert displayed(conn, 1) == ["展开了一道屏障的防御", "小雪，到屏障外面去", "结界消失了"]


def test_a_box_a_person_rewrote_without_the_old_rendering_is_listed_and_not_changed(conn, world):
    runs.edit_translation(conn, BOOK, 1, 0, "张开防护罩！")

    plan = preview_entry_rename(conn, world.kb, world.entry, "屏障")

    assert (1, 0, "张开防护罩！", None) in boxes(plan.listed)
    assert 0 not in {b.idx for b in plan.changes if b.page_index == 1}


def test_a_rename_that_changes_no_box_still_renames_the_row(conn, world):
    other = store.insert_entry(conn, kb_id=world.kb, source_term="魔法", target_term="魔法", category="", status="approved",
                               provenance="agent", page_index=1, event={"event": "created"})
    conn.commit()
    before_runs = run_ids(conn)

    result = rename_entry(conn, world.kb, other, "咒语")

    assert (result.plan.changes, result.plan.listed, result.runs) == ((), (), ())
    assert entry_row(conn, other)["target_term"] == "咒语" and history(entry_row(conn, other))[-1]["event"] == "renamed"
    assert run_ids(conn) == before_runs


def test_only_the_editor_kbs_row_is_written_and_an_eval_kbs_rows_of_the_same_book_are_not(conn, world):
    eval_kb = store.create_kb(conn, book_title=BOOK, purpose="eval", label="run-1")
    apply(conn, eval_kb, 1, world.runs[1], [term(eval_kb, 1, ENTRY_SOURCE, "结界")], {})
    before = dump(conn)
    eval_rows = {t: [r for r in rows if r[1] == eval_kb] for t, rows in before.items() if t != "kbs"}

    rename_entry(conn, world.kb, world.entry, "屏障")

    after = dump(conn)
    assert {t: [r for r in rows if r[1] == eval_kb] for t, rows in after.items() if t != "kbs"} == eval_rows
    assert after["kbs"] == before["kbs"]


# --- a profile renamed ---


def test_a_profile_renamed_is_approved_and_settles_the_new_name_proposal_it_was_made_from(conn, world):
    assert profile_row(conn, world.profile)["status"] == "pending_human"  # a new name, waiting for a person

    result = rename_profile(conn, world.kb, world.profile, "雪乃")

    profile = profile_row(conn, world.profile)
    assert (profile["name_src"], profile["name_tgt"], profile["status"]) == (NAME_SOURCE, "雪乃", "approved")
    assert json.loads(profile["aliases"]) == ["ゆき"] and pages_of(profile) == [1] and profile["notes"] == ""
    created, renamed = history(profile)
    assert created["event"] == "created" and created["proposal_id"] == world.new_name
    assert without_time(renamed) == {"event": "renamed", "by": "human", "from": "pending_human", "before": "小雪", "after": "雪乃"}
    queued = proposal_row(conn, world.new_name)
    assert queued["status"] == "approved"
    assert queued["reason"] == f"approved by a person, as profile {world.profile} (earlier: a new name waits for a human)"
    assert result.plan.settles == (world.new_name,)


def test_a_profile_renamed_writes_one_human_edit_for_each_page_it_changes(conn, world):
    bases = {i: page_run(conn, i) for i in (1, 2, 3)}

    result = rename_profile(conn, world.kb, world.profile, "雪乃")

    assert [r.page_index for r in result.runs] == [1, 2, 3]
    assert [displayed(conn, i) for i in (1, 2, 3, 4)] == [
        ["布下结界！", "雪乃，到结界外面去", "结界消失了"], ["结界就是结界", "雪乃笑了"],
        ["结界里很安全", "那是雪乃的声音", "什么？"], [None]]
    rename = {"kind": "profile", "id": world.profile, "kb_id": world.kb, "before": "小雪", "after": "雪乃"}
    assert attempts_of(conn, result.runs[1].run_id) == {
        "config": "human", "base_run_id": bases[2], "edits": [{"idx": 1, "before": "小雪笑了", "after": "雪乃笑了"}],
        "rename": rename}


def test_a_profile_that_a_person_approved_already_is_renamed_from_approved_and_settles_nothing(conn, world):
    decide_profile(conn, world.kb, world.profile, approve=True)
    assert proposal_row(conn, world.new_name)["status"] == "approved"
    proposal_before = tuple(proposal_row(conn, world.new_name))

    result = rename_profile(conn, world.kb, world.profile, "雪乃")

    assert result.plan.settles == ()
    assert history(profile_row(conn, world.profile))[-1]["from"] == "approved"
    assert tuple(proposal_row(conn, world.new_name)) == proposal_before


def test_an_entry_and_then_a_profile_renamed_on_one_box_both_show_there(conn, world):
    entry = rename_entry(conn, world.kb, world.entry, "屏障")
    profile = rename_profile(conn, world.kb, world.profile, "雪乃")

    assert displayed(conn, 1) == ["布下屏障！", "雪乃，到屏障外面去", "结界消失了"]  # box 1 holds both renderings
    second = next(r for r in profile.runs if r.page_index == 1)
    first = next(r for r in entry.runs if r.page_index == 1)
    assert attempts_of(conn, second.run_id)["base_run_id"] == first.run_id
    assert attempts_of(conn, second.run_id)["edits"] == [{"idx": 1, "before": "小雪，到屏障外面去", "after": "雪乃，到屏障外面去"}]
    review = build_review(displayed_run(conn, 1), lambda run_id: store.get_run(conn, run_id))
    assert review["edited"] == [0, 1]  # the chain of Human Runs is followed down to the `full` Run


# --- a row that waits for a person is approved by its rename, under an approval's checks ---


def test_a_pending_entry_renamed_becomes_approved(conn, world):
    pending = store.insert_entry(conn, kb_id=world.kb, source_term="魔法", target_term="魔法", category="", status="pending_human",
                                 provenance="agent", page_index=1, event={"event": "created"})
    conn.commit()

    rename_entry(conn, world.kb, pending, "咒语")

    entry = entry_row(conn, pending)
    assert (entry["target_term"], entry["status"], entry["provenance"]) == ("咒语", "approved", "human")
    assert history(entry)[-1]["from"] == "pending_human"


@pytest.mark.parametrize("call", [
    lambda conn, world, pending: preview_entry_rename(conn, world.kb, pending, "咒语"),
    lambda conn, world, pending: rename_entry(conn, world.kb, pending, "咒语"),
], ids=["preview", "write"])
def test_a_pending_entry_is_not_renamed_when_approving_it_would_leave_two_readable_rows(conn, world, call):
    rival = store.insert_entry(conn, kb_id=world.kb, source_term="魔法", target_term="魔术", category="", status="approved",
                               provenance="agent", page_index=1, event={"event": "created"})
    pending = store.insert_entry(conn, kb_id=world.kb, source_term="魔法", target_term="魔法", category="", status="pending_human",
                                 provenance="agent", page_index=1, event={"event": "created"})
    conn.commit()

    refused(conn, lambda: call(conn, world, pending), rf"entry {pending} cannot be approved: entry {rival} \(魔术\)")
    assert entry_row(conn, pending)["status"] == "pending_human"


def test_a_pending_entry_is_not_renamed_when_its_source_is_a_readable_profiles_name(conn, world):
    pending = store.insert_entry(conn, kb_id=world.kb, source_term=NAME_SOURCE, target_term="雪", category="", status="pending_human",
                                 provenance="agent", page_index=1, event={"event": "created"})
    conn.commit()

    refused(conn, lambda: rename_entry(conn, world.kb, pending, "白雪"), r"is the name of profile .*, a character")


def test_an_approved_entry_is_renamed_whatever_else_holds_its_source(conn, world):
    """The collision checks are an approval's: an entry that is approved already is not approved again by its rename."""
    store.insert_entry(conn, kb_id=world.kb, source_term=ENTRY_SOURCE, target_term="结界术", category="", status="pending_human",
                       provenance="agent", page_index=1, event={"event": "created"})
    conn.commit()

    rename_entry(conn, world.kb, world.entry, "屏障")

    assert entry_row(conn, world.entry)["target_term"] == "屏障"


def test_a_profile_is_not_renamed_when_approving_it_would_leave_two_readable_profiles(conn, world):
    rival = store.insert_profile(conn, kb_id=world.kb, name_src=NAME_SOURCE, name_tgt="雪", aliases=[], notes="", status="approved",
                                 page_index=1, event={"event": "created"})
    conn.commit()
    # `world.profile` is pending_human, and approved by a rename it would be the second readable profile for ユキ
    refused(conn, lambda: rename_profile(conn, world.kb, world.profile, "雪乃"),
            rf"profile {world.profile} cannot be approved: profile {rival} \(雪\)")
    refused(conn, lambda: preview_profile_rename(conn, world.kb, world.profile, "雪乃"), "two readable profiles")


def test_a_pending_profile_is_not_renamed_when_its_name_is_a_readable_entrys_source(conn, world):
    profile = store.insert_profile(conn, kb_id=world.kb, name_src=ENTRY_SOURCE, name_tgt="守护者", aliases=[], notes="",
                                   status="pending_human", page_index=1, event={"event": "created"})
    conn.commit()

    refused(conn, lambda: rename_profile(conn, world.kb, profile, "护卫"), rf"profile {profile} cannot be approved: .* a term")


# --- the term conflicts that the new rendering settles ---


def test_the_term_conflicts_that_now_agree_confirm_the_entry_and_the_rest_stay_as_they_are(conn, world):
    agree_2, agree_3, other, character, undecided = add_conflicts(conn, world)
    kept = {p.proposal_id: tuple(proposal_row(conn, p)) for p in (other, character, undecided)}
    previewed = preview_entry_rename(conn, world.kb, world.entry, "屏障")
    assert previewed.settles == (agree_2.proposal_id, agree_3.proposal_id)  # in the order they were queued

    result = rename_entry(conn, world.kb, world.entry, "屏障")

    assert result.plan.settles == previewed.settles
    for settled in (agree_2, agree_3):
        row = proposal_row(conn, settled)
        assert (row["status"], row["reason"]) == ("approved", f"approved by a person; confirms entry {world.entry}")
        assert row["decided_at"]
    entry = entry_row(conn, world.entry)
    assert pages_of(entry) == [1, 2, 3]  # the pages of the two proposals joined the evidence
    lines = [without_time(line) for line in history(entry)[1:]]
    assert lines == [
        {"event": "renamed", "by": "human", "from": "approved", "before": "结界", "after": "屏障"},
        {"event": "confirmed", "page": 2, "run_id": world.runs[2], "proposal_id": agree_2.proposal_id, "by": "human"},
        {"event": "confirmed", "page": 3, "run_id": world.runs[3], "proposal_id": agree_3.proposal_id, "by": "human"},
    ]
    for p in (other, character, undecided):  # another rendering, a character conflict, and an undecided term: as they were
        assert tuple(proposal_row(conn, p)) == kept[p.proposal_id]
    assert {proposal_row(conn, p)["status"] for p in (other, character)} == {"pending_human"}
    assert proposal_row(conn, undecided)["status"] == "proposed"


def test_no_conflict_is_settled_when_none_proposes_the_new_rendering(conn, world):
    add_conflicts(conn, world)

    result = rename_entry(conn, world.kb, world.entry, "结界术")

    assert result.plan.settles == ()
    assert [r["status"] for r in store.kb_proposals(conn, world.kb, "pending_human")] == ["pending_human"] * 5
    assert pages_of(entry_row(conn, world.entry)) == [1]


def test_a_conflict_the_entry_took_the_rendering_of_by_a_rename_is_what_take_this_rendering_does(conn, world):
    """The KB panel's "take this rendering": rename the entry to the conflict's rendering. The conflict is then a
    confirmation, and the boxes follow."""
    (conflict, *_) = add_conflicts(conn, world)
    waiting = conflict.payload["target_term"]

    result = rename_entry(conn, world.kb, world.entry, waiting)

    assert proposal_row(conn, conflict)["status"] == "approved" and result.plan.settles[0] == conflict.proposal_id
    assert entry_row(conn, world.entry)["target_term"] == waiting
    assert displayed(conn, 2) == ["屏障就是屏障", "小雪笑了"]


def test_a_profile_rename_settles_no_character_conflict(conn, world):
    """Character conflicts are rule 6's, whatever their rendering."""
    kb = world.kb
    conflict = name(kb, 2, "ユキちゃん", "雪乃")  # matches the profile by containment, and 雪乃 does not agree with 小雪
    apply(conn, kb, 2, world.runs[2], [conflict])
    assert proposal_row(conn, conflict)["status"] == "pending_human"
    kept = tuple(proposal_row(conn, conflict))

    result = rename_profile(conn, kb, world.profile, "雪乃")

    assert result.plan.settles == (world.new_name,)
    assert tuple(proposal_row(conn, conflict)) == kept


# --- the refusals: nothing is written, and the preview refuses what the write refuses ---

PREVIEWS_AND_WRITES = [
    pytest.param(preview_entry_rename, "entry", "屏障", id="entry-preview"),
    pytest.param(rename_entry, "entry", "屏障", id="entry-write"),
    pytest.param(preview_profile_rename, "profile", "雪乃", id="profile-preview"),
    pytest.param(rename_profile, "profile", "雪乃", id="profile-write"),
]


@pytest.mark.parametrize("call, row, target", PREVIEWS_AND_WRITES)
@pytest.mark.parametrize("status", ["rejected", "proposed"])
def test_a_row_that_is_not_readable_is_not_renamed(conn, world, call, row, target, status):
    table = {"entry": "glossary_entries", "profile": "character_profiles"}[row]
    conn.execute(f"UPDATE {table} SET status = ? WHERE id = ?", (status, getattr(world, row)))
    conn.commit()

    refused(conn, lambda: call(conn, world.kb, getattr(world, row), target),
            rf"{row} {getattr(world, row)} is {status}, so it cannot be renamed: approve it first")


@pytest.mark.parametrize("call, row, target", PREVIEWS_AND_WRITES)
def test_a_row_a_person_rejected_is_renamed_only_after_it_is_approved_again(conn, world, call, row, target):
    if row == "entry":
        decide_entry(conn, world.kb, world.entry, approve=False)
    else:
        decide_profile(conn, world.kb, world.profile, approve=False)
    refused(conn, lambda: call(conn, world.kb, getattr(world, row), target), "approve it first")

    (decide_entry if row == "entry" else decide_profile)(conn, world.kb, getattr(world, row), approve=True)

    call(conn, world.kb, getattr(world, row), target)  # now it goes through


@pytest.mark.parametrize("call, row, same", [
    (preview_entry_rename, "entry", "结界"), (rename_entry, "entry", "结界"), (rename_entry, "entry", "  结界　"),
    (preview_profile_rename, "profile", "小雪"), (rename_profile, "profile", "小雪"), (rename_profile, "profile", " 小雪 "),
], ids=["entry-preview", "entry-write", "entry-write-trimmed", "profile-preview", "profile-write", "profile-write-trimmed"])
def test_a_rendering_equal_to_the_old_one_after_normalising_is_refused(conn, world, call, row, same):
    refused(conn, lambda: call(conn, world.kb, getattr(world, row), same), rf"{row} {getattr(world, row)} already has the rendering")


@pytest.mark.parametrize("call, row, target", PREVIEWS_AND_WRITES)
def test_a_book_with_a_running_run_is_not_renamed_in_until_it_ends(conn, world, call, row, target):
    running = stage_running_run(conn, 3)

    refused(conn, lambda: call(conn, world.kb, getattr(world, row), target),
            rf"a Run is still running on page 3 \(Run {running}\) of {BOOK}.*rename after it ends")

    store.finish_run(conn, running, error="stopped", response_models=[], usage={
        "calls": 0, "unknown_usage_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
        attempts={}, translations=None)
    call(conn, world.kb, getattr(world, row), target)  # a Run that has ended no longer stops it


def test_every_running_page_is_named(conn, world):
    first, second = stage_running_run(conn, 4), stage_running_run(conn, 1)

    refused(conn, lambda: rename_entry(conn, world.kb, world.entry, "屏障"),
            rf"page 1 \(Run {second}\), page 4 \(Run {first}\) of {BOOK}")


def test_a_running_run_of_another_book_stops_nothing(conn, world):
    conn.execute("INSERT INTO books (book_title) VALUES ('book_b')")
    conn.execute("INSERT INTO pages (book_title, page_index, image_path) VALUES ('book_b', 1, 'x.jpg')")
    page_id = conn.execute("SELECT id FROM pages WHERE book_title = 'book_b'").fetchone()[0]
    conn.commit()
    store.start_run(conn, page_id=page_id, config="full", params={}, model="m")

    rename_entry(conn, world.kb, world.entry, "屏障")

    assert entry_row(conn, world.entry)["target_term"] == "屏障"


def test_running_pages_lists_each_page_with_its_latest_running_run_in_page_order(conn, world):
    assert store.running_pages(conn, BOOK) == []
    store.finish_run(conn, stage_running_run(conn, 2), error="x", response_models=[], usage={
        "calls": 0, "unknown_usage_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}, attempts={}, translations=None)
    older, newer, other = stage_running_run(conn, 3), stage_running_run(conn, 3), stage_running_run(conn, 1)

    assert store.running_pages(conn, BOOK) == [(1, other), (3, newer)] and older < newer  # a page once, its latest Run
    assert store.running_pages(conn, "no_such_book") == []


@pytest.mark.parametrize("call, row", [
    (preview_entry_rename, "entry"), (rename_entry, "entry"), (preview_profile_rename, "profile"), (rename_profile, "profile")])
@pytest.mark.parametrize("target, why", [
    ("", "at least 1"), ("   ", "at least 1"), ("雪" * 61, "at most 60"), ("a\nb", "one line"), ("a\tb", "one line"),
], ids=["empty", "spaces", "too-long", "newline", "tab"])
def test_a_rendering_the_kb_would_not_store_is_refused_and_nothing_is_written(conn, world, call, row, target, why):
    before = snapshot(conn)

    with pytest.raises(InvalidRendering, match=why):
        call(conn, world.kb, getattr(world, row), target)

    assert snapshot(conn) == before and not conn.in_transaction


@pytest.mark.parametrize("call, row, target", PREVIEWS_AND_WRITES)
def test_an_id_the_editor_kb_does_not_have_is_not_found_and_nothing_is_written(conn, world, call, row, target):
    conn.execute("INSERT INTO books (book_title) VALUES ('book_b')")
    conn.commit()
    other_kb = store.editor_kb(conn, "book_b")  # another book's Editor KB has rows of its own
    entry = store.insert_entry(conn, kb_id=other_kb, source_term="魔法", target_term="魔法", category="", status="approved",
                               provenance="agent", page_index=1, event={"event": "created"})
    profile = store.insert_profile(conn, kb_id=other_kb, name_src="ハル", name_tgt="小春", aliases=[], notes="", status="approved",
                                   page_index=1, event={"event": "created"})
    eval_kb = store.create_kb(conn, book_title=BOOK, purpose="eval", label="run-1")
    eval_entry = store.insert_entry(conn, kb_id=eval_kb, source_term="魔法", target_term="魔法", category="", status="approved",
                                    provenance="agent", page_index=1, event={"event": "created"})
    eval_profile = store.insert_profile(conn, kb_id=eval_kb, name_src="ハル", name_tgt="小春", aliases=[], notes="",
                                        status="approved", page_index=1, event={"event": "created"})
    conn.commit()
    before = snapshot(conn)
    foreign = {"entry": (entry, eval_entry, 424242), "profile": (profile, eval_profile, 424242)}[row]

    for row_id in foreign:  # another book's, an eval KB's, and one that does not exist
        with pytest.raises(ItemNotFound, match=rf"no {row} {row_id} in the KB"):
            call(conn, world.kb, row_id, target)

    assert snapshot(conn) == before and not conn.in_transaction


@pytest.mark.parametrize("call, row, target", PREVIEWS_AND_WRITES)
def test_a_kb_that_is_not_an_editor_kb_is_not_found_and_an_eval_kbs_rows_are_never_renamed(conn, world, call, row, target):
    eval_kb = store.create_kb(conn, book_title=BOOK, purpose="eval", label="run-1")
    entry = store.insert_entry(conn, kb_id=eval_kb, source_term=ENTRY_SOURCE, target_term="结界", category="", status="approved",
                               provenance="agent", page_index=1, event={"event": "created"})
    profile = store.insert_profile(conn, kb_id=eval_kb, name_src=NAME_SOURCE, name_tgt="小雪", aliases=[], notes="",
                                   status="approved", page_index=1, event={"event": "created"})
    conn.commit()
    before = snapshot(conn)

    for kb_id in (eval_kb, 99999):
        with pytest.raises(KBNotFound, match="no Editor KB has id"):
            call(conn, kb_id, {"entry": entry, "profile": profile}[row], target)

    assert snapshot(conn) == before and not conn.in_transaction


# --- all or nothing, in one transaction ---

FAULTS = [
    ("entry", "insert_human_run", 3),  # while the third page's Human edit is written: the KB row and two edits came first
    ("profile", "insert_human_run", 3),
    ("entry", "insert_human_run", 1),
    ("entry", "set_entry_rendering", 1),
    ("profile", "set_profile_rendering", 1),
    ("entry", "add_entry_evidence", 1),  # settling a conflict
    ("entry", "set_proposal_status", 1),
    ("profile", "set_proposal_status", 1),  # settling the new-name proposal
]


@pytest.mark.parametrize("row, function, nth", FAULTS, ids=[f"{r}:{f}:{n}" for r, f, n in FAULTS])
def test_when_a_write_fails_partway_nothing_of_the_rename_is_stored(conn, world, monkeypatch, row, function, nth):
    add_conflicts(conn, world)
    rename = {"entry": lambda: rename_entry(conn, world.kb, world.entry, "屏障"),
              "profile": lambda: rename_profile(conn, world.kb, world.profile, "雪乃")}[row]
    before = snapshot(conn)
    real, calls = getattr(store, function), []

    def failing(*args, **kwargs):
        calls.append(function)
        if len(calls) == nth:
            raise sqlite3.OperationalError("disk I/O error")
        return real(*args, **kwargs)

    monkeypatch.setattr(store, function, failing)

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        rename()

    assert len(calls) == nth  # it did get that far
    assert snapshot(conn) == before  # the KB row and its history, the proposals, every Run and every Translation
    assert displayed(conn, 1) == ["布下结界！", "小雪，到结界外面去", "结界消失了"]
    assert not conn.in_transaction
    monkeypatch.setattr(store, function, real)  # once the fault is gone the rename goes through, whole
    rename()
    assert snapshot(conn) != before


def test_the_third_pages_failure_is_after_two_human_edits_were_written_in_the_same_transaction(conn, world, monkeypatch):
    """The failure of FAULTS' first case is not a first write: two Human edits and the KB row are in the transaction when the
    third edit raises, and the connection sees them until it is rolled back."""
    seen = []
    real = store.insert_human_run

    def failing(c, **kwargs):
        seen.append((c.in_transaction, c.execute("SELECT COUNT(*) FROM runs WHERE config = 'human'").fetchone()[0],
                     entry_row(conn, world.entry)["target_term"]))
        if len(seen) == 3:
            raise sqlite3.OperationalError("disk I/O error")
        return real(c, **kwargs)

    monkeypatch.setattr(store, "insert_human_run", failing)

    with pytest.raises(sqlite3.OperationalError):
        rename_entry(conn, world.kb, world.entry, "屏障")

    assert seen == [(True, 0, "屏障"), (True, 1, "屏障"), (True, 2, "屏障")]  # the row was written, the edits came one by one
    assert entry_row(conn, world.entry)["target_term"] == "结界"  # and all of it is gone


@pytest.mark.parametrize("call", [
    lambda conn, world: rename_entry(conn, world.kb, world.entry, "屏障"),
    lambda conn, world: rename_profile(conn, world.kb, world.profile, "雪乃"),
    lambda conn, world: preview_entry_rename(conn, world.kb, world.entry, "屏障"),
    lambda conn, world: preview_profile_rename(conn, world.kb, world.profile, "雪乃"),
], ids=["entry", "profile", "entry-preview", "profile-preview"])
def test_a_rename_does_not_commit_what_the_caller_has_pending(conn, world, call):
    conn.execute("INSERT INTO books (book_title) VALUES ('pending_book')")  # an open transaction of the caller's

    with pytest.raises(sqlite3.OperationalError, match="within a transaction"):
        call(conn, world)

    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM books WHERE book_title = 'pending_book'").fetchone()[0] == 0
    assert entry_row(conn, world.entry)["target_term"] == "结界"


# --- what is not changed ---


def test_the_nine_results_and_the_nine_outcomes_are_as_they_were(conn, world):
    """A rename is neither something `apply` did to a proposal nor one of a person's decisions on one: it adds no word to
    either list, so what the eval harness counts and what the decision route answers do not change."""
    assert len(RESULTS) == len(set(RESULTS)) == 9 and len(HUMAN_OUTCOMES) == len(set(HUMAN_OUTCOMES)) == 9
    assert not [w for w in (*RESULTS, *HUMAN_OUTCOMES) if "renam" in w]


def test_a_human_edit_of_one_box_is_as_it_was_after_a_rename(conn, world):
    rename_entry(conn, world.kb, world.entry, "屏障")

    base = page_run(conn, 3)  # the rename's Human edit
    edited = runs.edit_translation(conn, BOOK, 3, 2, "怎么了？")

    assert attempts_of(conn, edited) == {  # no `rename` key: this edit is not a rename's
        "config": "human", "base_run_id": base, "edits": [{"idx": 2, "before": "什么？", "after": "怎么了？"}]}
    assert displayed(conn, 3) == ["屏障里很安全", "那是小雪的声音", "怎么了？"]
