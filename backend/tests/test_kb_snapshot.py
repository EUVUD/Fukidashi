"""A KB snapshot: what it shows, how it looks things up, that it stays as it was, and its snippets."""

from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Mapping

import pytest
from kb_support import accept, name, new_kb, new_run, term

from fukidashi import store
from fukidashi.datasets.importer import import_openmantra
from fukidashi.kb import KBNotFound, KBSnapshot, load_snapshot
from fukidashi.kb.curator import apply


@pytest.fixture
def imported(conn, dataset):
    import_openmantra(conn, dataset)
    return conn


@pytest.fixture
def kb(imported):
    return new_kb(imported)


def add_entry(conn, kb_id, source, target, status="approved", **extra):
    return store.insert_entry(conn, kb_id=kb_id, source_term=source, target_term=target,
                              category=extra.get("category", ""), status=status, provenance="agent",
                              page_index=extra.get("page", 1), event={"event": "created"})


def add_profile(conn, kb_id, source, target, status="approved", aliases=(), notes=""):
    return store.insert_profile(conn, kb_id=kb_id, name_src=source, name_tgt=target, aliases=aliases, notes=notes,
                                status=status, page_index=1, event={"event": "created"})


def lines(snapshot, text, budget=10_000):
    out = snapshot.snippets(text, budget)
    return out.split("\n") if out else []


def reachable(value, seen=None):
    """Every object that can be reached from `value` through attributes, items and elements."""
    seen = set() if seen is None else seen
    if id(value) in seen:
        return
    seen.add(id(value))
    yield value
    if isinstance(value, Mapping):
        children = [*value.keys(), *value.values()]
    elif isinstance(value, (list, tuple, set, frozenset)):
        children = list(value)
    elif hasattr(value, "__dict__") or hasattr(value, "__slots__"):
        children = [getattr(value, a) for a in {*getattr(value, "__dict__", {}), *getattr(value, "__slots__", ())}]
    else:
        children = []
    for child in children:
        yield from reachable(child, seen)


# --- what a snapshot shows ---


def test_a_snapshot_shows_approved_and_pending_entries_and_profiles_marked_with_their_status(imported, kb):
    for source, status in [("甲", "approved"), ("乙", "proposed"), ("丙", "rejected"), ("丁", "pending_human")]:
        add_entry(imported, kb, source, "x" + source, status)
    for source, status in [("あ", "approved"), ("い", "proposed"), ("う", "rejected"), ("え", "pending_human")]:
        add_profile(imported, kb, source, "x" + source, status, aliases=["別名" + source])

    snapshot = load_snapshot(imported, kb)

    assert [(e.source_term, e.status) for e in snapshot.entries] == [("甲", "approved"), ("丁", "pending_human")]
    assert [(p.name_src, p.status) for p in snapshot.profiles] == [("あ", "approved"), ("え", "pending_human")]
    for hidden in ("乙", "丙"):
        assert snapshot.term(hidden) is None
    for hidden in ("い", "う", "別名い", "別名う"):
        assert snapshot.character(hidden) is None
    text = "甲乙丙丁あいうえ"
    assert [l.split(" → ")[0] for l in lines(snapshot, text)] == ["term: 甲", "character: あ", "term: 丁", "character: え"]


def test_a_snapshot_carries_the_entry_as_it_was_stored(imported, kb):
    entry_id = add_entry(imported, kb, "封印", "封印", category="magic")
    store.add_entry_evidence(imported, entry_id, 3, {"event": "confirmed", "page": 3})
    add_profile(imported, kb, "ユキ", "小雪", aliases=["ユッキー"], notes="说话轻声细语")

    snapshot = load_snapshot(imported, kb)

    (entry,) = snapshot.entries
    assert (entry.id, entry.source_term, entry.target_term, entry.category) == (entry_id, "封印", "封印", "magic")
    assert (entry.status, entry.provenance, entry.evidence_pages) == ("approved", "agent", (1, 3))
    assert [h["event"] for h in entry.history] == ["created", "confirmed"] and entry.history[1]["page"] == 3
    (profile,) = snapshot.profiles
    assert (profile.name_tgt, profile.aliases, profile.notes) == ("小雪", ("ユッキー",), "说话轻声细语")
    assert (snapshot.kb_id, snapshot.book_title) == (kb, "book_a")


def test_an_unknown_kb_has_no_snapshot(imported):
    with pytest.raises(KBNotFound):
        load_snapshot(imported, 99)


# --- lookups ---


def test_a_term_is_found_by_its_exact_source_only(imported, kb):
    add_entry(imported, kb, "封印", "封印")
    snapshot = load_snapshot(imported, kb)

    assert snapshot.term("封印").target_term == "封印"
    assert snapshot.term("  封印 ").target_term == "封印"  # trimmed
    assert snapshot.term("封") is None and snapshot.term("封印術") is None and snapshot.term("") is None


def test_lookups_use_the_normalised_form(imported, kb):
    add_entry(imported, kb, "ABC", "甲")
    add_profile(imported, kb, "ユキ", "小雪", aliases=["ﾕｯｷｰ"])  # half-width katakana, as a page may spell it

    snapshot = load_snapshot(imported, kb)

    assert snapshot.term("ＡＢＣ").target_term == "甲"  # full-width letters
    assert snapshot.character("ﾕｷ").name_tgt == "小雪"


def test_a_character_is_found_by_name_or_alias(imported, kb):
    add_profile(imported, kb, "ユキ", "小雪", aliases=["ゆき", "雪ちゃん"])
    add_profile(imported, kb, "ハル", "小春", status="pending_human")
    snapshot = load_snapshot(imported, kb)

    assert snapshot.character("ユキ").name_tgt == "小雪"
    assert snapshot.character("雪ちゃん").name_tgt == "小雪" and snapshot.character("ゆき").name_tgt == "小雪"
    assert snapshot.character("ハル").status == "pending_human"
    assert snapshot.character("ユ") is None and snapshot.character("雪") is None


def test_a_name_beats_an_alias_and_an_approved_entry_beats_a_pending_one(imported, kb):
    add_profile(imported, kb, "雪", "雪B", status="approved", aliases=["ユキ"])  # "ユキ" is its alias...
    add_profile(imported, kb, "ユキ", "小雪", status="pending_human")  # ...and this one's name
    add_entry(imported, kb, "封印", "pending 封印", status="pending_human")
    add_entry(imported, kb, "封印", "approved 封印", status="approved")

    snapshot = load_snapshot(imported, kb)

    assert snapshot.character("ユキ").name_tgt == "小雪"
    assert snapshot.term("封印").target_term == "approved 封印"


# --- a snapshot is a copy ---


def test_a_snapshot_does_not_change_when_the_kb_does(imported, kb):
    add_entry(imported, kb, "封印", "封印")
    snapshot = load_snapshot(imported, kb)
    before = (snapshot.entries, snapshot.profiles, snapshot.snippets("封印と魔法とユキ", 1000))

    run_id = new_run(imported)
    proposals = [term(kb, 2, "封印", "封印"), term(kb, 2, "魔法", "魔法"), name(kb, 2, "ユキ", "小雪")]
    apply(imported, kb, 2, run_id, proposals, {proposals[1].proposal_id: accept()})
    imported.execute("UPDATE glossary_entries SET target_term = 'changed'")  # even a direct edit

    assert (snapshot.entries, snapshot.profiles, snapshot.snippets("封印と魔法とユキ", 1000)) == before
    assert snapshot.term("魔法") is None and snapshot.character("ユキ") is None
    assert snapshot.entries[0].evidence_pages == (1,)
    fresh = load_snapshot(imported, kb)  # a new snapshot sees the new state
    assert fresh.term("魔法") is not None and fresh.character("ユキ") is not None


def test_a_snapshot_cannot_be_changed(imported, kb):
    add_entry(imported, kb, "封印", "封印")
    add_profile(imported, kb, "ユキ", "小雪", aliases=["ゆき"])
    snapshot = load_snapshot(imported, kb)

    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.entries = ()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.entries[0].target_term = "x"
    with pytest.raises(dataclasses.FrozenInstanceError):
        snapshot.profiles[0].name_tgt = "x"
    with pytest.raises(TypeError):
        snapshot.entries[0].history[0]["event"] = "x"
    with pytest.raises(TypeError):
        snapshot._terms["x"] = snapshot.entries[0]
    assert isinstance(snapshot.entries, tuple) and isinstance(snapshot.profiles[0].aliases, tuple)


def test_a_snapshot_holds_no_database_connection_anywhere(imported, kb):
    add_entry(imported, kb, "封印", "封印")
    add_profile(imported, kb, "ユキ", "小雪", aliases=["ゆき"])
    snapshot = load_snapshot(imported, kb)

    objects = list(reachable(snapshot))

    assert len(objects) > 10  # the walk really went through the entries, the profiles and their histories
    assert not [o for o in objects if isinstance(o, (sqlite3.Connection, sqlite3.Cursor, sqlite3.Row))]
    assert isinstance(snapshot, KBSnapshot)


def test_a_snapshot_survives_the_database_being_closed(db_path, dataset):
    from fukidashi.db import connect, init_db

    conn = connect(db_path)
    init_db(conn)
    import_openmantra(conn, dataset)
    kb_id = new_kb(conn)
    add_entry(conn, kb_id, "封印", "封印")
    snapshot = load_snapshot(conn, kb_id)
    conn.close()

    assert snapshot.term("封印").target_term == "封印" and snapshot.snippets("封印", 100)


# --- snippets ---


def kb_with_a_bit_of_everything(imported, kb):
    add_entry(imported, kb, "魔法", "魔法", category="magic")  # id 1
    add_entry(imported, kb, "封印", "封印", status="pending_human")  # id 2
    add_entry(imported, kb, "王国", "王国")  # id 3
    add_entry(imported, kb, "不在", "不在")  # id 4, never in the text
    add_entry(imported, kb, "隠し", "隐藏", status="rejected")  # in the text, but rejected
    add_profile(imported, kb, "ユキ", "小雪", aliases=["雪ちゃん"], notes="说话轻声细语")
    add_profile(imported, kb, "ハル", "小春", status="pending_human")
    add_profile(imported, kb, "ナシ", "无")
    return load_snapshot(imported, kb)


TEXT = "雪ちゃん、封印の魔法は王国のものよ。隠しても無駄。ハルはどこ？"


def test_snippets_return_exactly_the_entries_and_profiles_found_in_the_text(imported, kb):
    snapshot = kb_with_a_bit_of_everything(imported, kb)

    assert lines(snapshot, TEXT) == [
        "character: ユキ → 小雪 (also 雪ちゃん); 说话轻声细语",  # approved, and first in the text: via its alias
        "term: 魔法 → 魔法 (magic)",
        "term: 王国 → 王国",
        "term: 封印 → 封印 [pending]",  # pending comes after every approved one, though it occurs before 魔法
        "character: ハル → 小春 [pending]",
    ]
    assert lines(snapshot, "何もない") == [] and snapshot.snippets("", 100) == ""


def test_snippets_put_approved_before_pending_then_order_by_first_occurrence(imported, kb):
    add_entry(imported, kb, "三", "3", status="pending_human")
    add_entry(imported, kb, "二", "2")
    add_entry(imported, kb, "一", "1")
    add_profile(imported, kb, "四", "4", status="pending_human")
    add_profile(imported, kb, "五", "5")
    snapshot = load_snapshot(imported, kb)

    assert [l.split(" → ")[0] for l in lines(snapshot, "五四三二一")] == [
        "character: 五", "term: 二", "term: 一", "character: 四", "term: 三"]
    assert [l.split(" → ")[0] for l in lines(snapshot, "一二三四五")] == [
        "term: 一", "term: 二", "character: 五", "term: 三", "character: 四"]


def test_snippets_match_the_normalised_text(imported, kb):
    add_entry(imported, kb, "ABC", "甲")
    snapshot = load_snapshot(imported, kb)

    assert lines(snapshot, "これはＡＢＣだ") == ["term: ABC → 甲"]


def test_a_term_found_twice_or_by_two_names_is_listed_once(imported, kb):
    add_entry(imported, kb, "封印", "封印")
    add_profile(imported, kb, "ユキ", "小雪", aliases=["ゆき", "雪"])
    snapshot = load_snapshot(imported, kb)

    assert len(lines(snapshot, "封印、封印。ユキ、ゆき、雪")) == 2


def test_snippets_never_exceed_the_budget_and_never_cut_a_line(imported, kb):
    snapshot = kb_with_a_bit_of_everything(imported, kb)
    full = snapshot.snippets(TEXT, 10_000)
    all_lines = full.split("\n")
    assert len(all_lines) == 5

    for budget in range(0, len(full) + 3):
        out = snapshot.snippets(TEXT, budget)
        assert len(out) <= budget
        kept = out.split("\n") if out else []
        assert kept == all_lines[:len(kept)]  # whole lines, in the same order
        # and no line is left out that would have fitted next
        if len(kept) < len(all_lines):
            assert len("\n".join(all_lines[:len(kept) + 1])) > budget
    assert snapshot.snippets(TEXT, len(full)) == full and snapshot.snippets(TEXT, -5) == ""


def test_snippets_are_the_same_for_the_same_input(imported, kb):
    snapshot = kb_with_a_bit_of_everything(imported, kb)
    other = load_snapshot(imported, kb)

    runs = {snapshot.snippets(TEXT, 80) for _ in range(5)} | {other.snippets(TEXT, 80)}

    assert len(runs) == 1


def test_snippets_hold_only_what_the_snapshot_holds(imported, kb):
    snapshot = kb_with_a_bit_of_everything(imported, kb)

    out = snapshot.snippets(TEXT, 10_000)
    assert "隠し" not in out and "隐藏" not in out  # rejected
    assert out.count("[pending]") == 2
