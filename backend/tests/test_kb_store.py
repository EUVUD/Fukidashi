"""The KB tables: created by init_db without touching old data, and one KB per row, so two KBs never meet."""

from __future__ import annotations

import sqlite3

import pytest
from kb_support import KB_TABLES, accept, name, new_kb, new_run, term

from fukidashi import store
from fukidashi.datasets.importer import import_openmantra
from fukidashi.db import connect, init_db
from fukidashi.kb import ProposalCollector, load_snapshot
from fukidashi.kb.curator import ACCEPTED, apply

# schema.sql as it was before the KB tables, to build a database from the old schema.
OLD_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (book_title TEXT PRIMARY KEY);
CREATE TABLE IF NOT EXISTS pages (
    id INTEGER PRIMARY KEY, book_title TEXT NOT NULL REFERENCES books(book_title), page_index INTEGER NOT NULL,
    image_path TEXT NOT NULL, UNIQUE (book_title, page_index)
);
CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY, page_id INTEGER NOT NULL REFERENCES pages(id), idx INTEGER NOT NULL,
    x INTEGER NOT NULL, y INTEGER NOT NULL, w INTEGER NOT NULL, h INTEGER NOT NULL, text_ja TEXT NOT NULL,
    text_zh_hant TEXT NOT NULL, text_en TEXT NOT NULL, UNIQUE (page_id, idx)
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY, page_id INTEGER NOT NULL REFERENCES pages(id), config TEXT NOT NULL,
    config_json TEXT NOT NULL, model TEXT NOT NULL, response_model TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')), error TEXT,
    calls INTEGER NOT NULL DEFAULT 0, unknown_usage_calls INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER,
    output_tokens INTEGER, cost_usd REAL, attempts_json TEXT, started_at TEXT NOT NULL, finished_at TEXT
);
CREATE INDEX IF NOT EXISTS runs_page ON runs(page_id, id);
CREATE TABLE IF NOT EXISTS translations (
    run_id INTEGER NOT NULL REFERENCES runs(id), segment_id INTEGER NOT NULL REFERENCES segments(id),
    text_zh_hans TEXT NOT NULL, PRIMARY KEY (run_id, segment_id)
);
"""
OLD_TABLES = ("books", "pages", "segments", "runs", "translations")


@pytest.fixture
def imported(conn, dataset):
    import_openmantra(conn, dataset)
    return conn


def table_info(conn, table):
    return [tuple(r) for r in conn.execute(f"PRAGMA table_info({table})")]


def test_init_db_creates_the_four_kb_tables_and_is_harmless_to_run_twice(conn):
    def names():
        return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'index')")}

    assert set(KB_TABLES) <= names()
    before = names()

    init_db(conn)

    assert names() == before


def test_init_db_on_a_database_from_before_the_kb_tables_adds_them_and_changes_no_row(tmp_path, dataset):
    path = tmp_path / "old.db"
    old = connect(path)
    old.executescript(OLD_SCHEMA)
    import_openmantra(old, dataset)  # works on the old tables alone
    page = store.find_page(old, "book_a", 1)
    run_id = store.start_run(old, page_id=page["id"], config="baseline", params={"a": 1}, model="m")
    store.finish_run(old, run_id, error=None, response_models=["m-1"], usage={
        "calls": 1, "unknown_usage_calls": 0, "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.01,
    }, attempts=[{"ok": True}], translations={s["id"]: "译文" for s in store.page_segments(old, page["id"])})
    old_info = {t: table_info(old, t) for t in OLD_TABLES}
    rows = {t: [tuple(r) for r in old.execute(f"SELECT * FROM {t} ORDER BY rowid")] for t in OLD_TABLES}
    assert all(rows[t] for t in OLD_TABLES)  # every old table has rows to lose
    old.close()

    conn = connect(path)
    init_db(conn)

    assert {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")] for t in OLD_TABLES} == rows
    assert {t: table_info(conn, t) for t in OLD_TABLES} == old_info  # schema.sql does not redefine them
    assert all(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0 for t in KB_TABLES)
    assert new_kb(conn)  # and the new tables work
    conn.close()


def test_a_kb_is_created_and_found_by_its_book_purpose_and_label(imported):
    kb_id = new_kb(imported, "run-7")

    kb = store.find_kb(imported, book_title="book_a", purpose="eval", label="run-7")
    assert (kb["id"], kb["book_title"], kb["purpose"], kb["label"]) == (kb_id, "book_a", "eval", "run-7")
    assert store.get_kb(imported, kb_id)["id"] == kb_id
    assert store.find_kb(imported, book_title="book_a", purpose="editor", label="run-7") is None
    assert store.find_kb(imported, book_title="book_a", purpose="eval", label="run-8") is None
    assert store.get_kb(imported, kb_id + 1) is None


@pytest.mark.parametrize("book, purpose, label", [
    ("no_such_book", "eval", "x"), ("book_a", "production", "x"), ("book_a", "eval", "run-1"),
], ids=["unknown book", "unknown purpose", "label taken"])
def test_a_kb_that_does_not_fit_is_refused(imported, book, purpose, label):
    new_kb(imported, "run-1")

    with pytest.raises(sqlite3.IntegrityError):
        store.create_kb(imported, book_title=book, purpose=purpose, label=label)


def test_the_database_refuses_values_outside_the_status_kind_and_provenance_lists(imported):
    kb_id, run_id = new_kb(imported), new_run(imported)
    entry = dict(kb_id=kb_id, source_term="a", target_term="b", category="", status="approved", provenance="agent",
                 page_index=1, event={})

    for bad in ({"status": "maybe"}, {"provenance": "model"}):
        with pytest.raises(sqlite3.IntegrityError):
            store.insert_entry(imported, **{**entry, **bad})
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_profile(imported, kb_id=kb_id, name_src="a", name_tgt="b", aliases=[], notes="", status="maybe",
                             page_index=1, event={})
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_proposal(imported, proposal_id="p", kb_id=kb_id, kind="phrase", payload={}, proposer="t",
                              page_index=1, run_id=run_id)


def test_two_kbs_of_one_book_never_see_each_others_entries_profiles_or_proposals(imported):
    kb_a, kb_b = new_kb(imported, "run-1"), new_kb(imported, "run-2")
    run_id = new_run(imported)
    proposals_a = [term(kb_a, 1, "封印", "封印"), name(kb_a, 1, "ユキ", "小雪")]
    proposals_b = [term(kb_b, 1, "封印", "封印"), name(kb_b, 1, "ユキ", "小雪")]
    assert {p.proposal_id for p in proposals_a}.isdisjoint(p.proposal_id for p in proposals_b)  # same content, two ids

    apply(imported, kb_a, 1, run_id, proposals_a, {proposals_a[0].proposal_id: accept()})

    snap_a, snap_b = load_snapshot(imported, kb_a), load_snapshot(imported, kb_b)
    assert [e.source_term for e in snap_a.entries] == ["封印"] and [p.name_src for p in snap_a.profiles] == ["ユキ"]
    assert snap_b.entries == () and snap_b.profiles == ()
    assert store.kb_proposals(imported, kb_b) == [] and len(store.kb_proposals(imported, kb_a)) == 2
    assert store.find_entry(imported, kb_b, "封印") is None and store.find_profile(imported, kb_b, "ユキ") is None

    # B is empty, so the same proposals are new there: no confirmation, no conflict against A's entry
    results = apply(imported, kb_b, 1, run_id, [term(kb_b, 1, "封印", "封缄"), proposals_b[1]],
                    {term(kb_b, 1, "封印", "封缄").proposal_id: accept()})
    assert [r.result for r in results] == [ACCEPTED, "pending_human"]
    assert load_snapshot(imported, kb_a).term("封印").target_term == "封印"  # A is as it was
    assert load_snapshot(imported, kb_b).term("封印").target_term == "封缄"


def test_a_collector_gives_the_same_proposal_another_id_in_another_kb(imported):
    kb_a, kb_b = new_kb(imported, "run-1"), new_kb(imported, "run-2")
    payload = {"source_term": "封印", "target_term": "封印"}

    (a, _), (b, _) = ProposalCollector(kb_a, 1).add("term", payload), ProposalCollector(kb_b, 1).add("term", payload)

    assert a.proposal_id != b.proposal_id and a.payload == b.payload
