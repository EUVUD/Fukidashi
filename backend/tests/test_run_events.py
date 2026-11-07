"""The `run_events` table and the store functions around it (task 035): what is stored, what is read back after a given
`seq`, that an event is committed at once and never inside another write's transaction, and that an existing database
gets the table with its Runs and Translations untouched."""

from __future__ import annotations

import json
import sqlite3

import pytest
from conftest import FakeClient

from fukidashi import runs, store
from fukidashi.datasets.importer import import_openmantra
from fukidashi.db import connect, init_db
from fukidashi.live_trace import RunEvent

ZH = json.dumps({"translations": [{"id": 0, "zh": "你好"}, {"id": 1, "zh": "再见"}]}, ensure_ascii=False)


@pytest.fixture
def imported(conn, dataset):
    import_openmantra(conn, dataset)
    return conn


def new_run(conn) -> int:
    page = store.find_page(conn, "book_a", 1)
    return store.start_run(conn, page_id=page["id"], config="baseline", params={}, model="m")


def events_of(run_id: int, *seqs: int) -> list[RunEvent]:
    return [RunEvent(run_id, seq, "model_call", {"step": seq, "text": f"こんにちは {seq}"}) for seq in seqs]


def test_the_table_has_the_columns_and_the_key_the_contract_names(conn):
    columns = conn.execute("PRAGMA table_info(run_events)").fetchall()

    assert [c["name"] for c in columns] == ["run_id", "seq", "kind", "data_json", "created_at"]
    assert all(c["notnull"] == 1 for c in columns)
    assert [c["name"] for c in sorted((c for c in columns if c["pk"]), key=lambda c: c["pk"])] == ["run_id", "seq"]
    assert [tuple(f)[2:4] for f in conn.execute("PRAGMA foreign_key_list(run_events)")] == [("runs", "run_id")]


def test_events_come_back_in_seq_order_with_their_data_whatever_the_order_they_were_stored_in(imported):
    run_id, other = new_run(imported), new_run(imported)
    stored = events_of(run_id, 1, 2, 3)

    for event in (stored[2], *events_of(other, 1), stored[0], stored[1]):
        store.add_run_event(imported, event)

    assert store.run_events(imported, run_id) == stored
    assert store.run_events(imported, other) == events_of(other, 1)  # another Run's events are not mixed in
    assert store.run_events(imported, run_id + 100) == []


def test_after_seq_returns_exactly_the_events_after_it_in_order(imported):
    run_id = new_run(imported)
    stored = events_of(run_id, *range(1, 8))
    for event in stored:
        store.add_run_event(imported, event)

    for after in range(0, 9):
        assert store.run_events(imported, run_id, after_seq=after) == [e for e in stored if e.seq > after]
    assert store.run_events(imported, run_id, 0) == store.run_events(imported, run_id) == stored


def test_an_event_is_committed_at_once_so_another_connection_reads_it(imported, db_path):
    run_id, other = new_run(imported), connect(db_path)

    store.add_run_event(imported, events_of(run_id, 1)[0])

    assert store.run_events(other, run_id) == events_of(run_id, 1)  # nothing else was committed in between
    assert imported.in_transaction is False
    other.close()


def test_an_event_is_refused_inside_a_transaction_and_nothing_is_stored(imported):
    run_id = new_run(imported)

    with store.transaction(imported):
        with pytest.raises(RuntimeError, match="transaction"):
            store.add_run_event(imported, events_of(run_id, 1)[0])
        assert imported.in_transaction is True  # the refusal did not end the caller's transaction

    assert store.run_events(imported, run_id) == []


def test_an_event_is_refused_after_a_write_that_has_not_been_committed_and_that_write_stays_open(imported):
    run_id = new_run(imported)
    imported.execute("UPDATE runs SET error = 'half done' WHERE id = ?", (run_id,))  # opens a transaction

    with pytest.raises(RuntimeError):
        store.add_run_event(imported, events_of(run_id, 1)[0])

    assert imported.in_transaction is True  # not committed behind the writer's back
    imported.rollback()
    assert store.get_run(imported, run_id)["error"] is None and store.run_events(imported, run_id) == []


def test_a_seq_stored_twice_an_unknown_run_and_data_that_is_not_json_are_errors_that_store_nothing(imported):
    run_id = new_run(imported)
    store.add_run_event(imported, events_of(run_id, 1)[0])

    with pytest.raises(sqlite3.IntegrityError):
        store.add_run_event(imported, RunEvent(run_id, 1, "run_end", {}))  # the same (run, seq)
    with pytest.raises(sqlite3.IntegrityError):
        store.add_run_event(imported, RunEvent(run_id + 100, 1, "run_start", {}))  # no such Run
    with pytest.raises(ValueError):
        store.add_run_event(imported, RunEvent(run_id, 2, "run_end", {"cost_usd": float("nan")}))

    assert store.run_events(imported, run_id) == events_of(run_id, 1)
    assert imported.in_transaction is False


def test_init_db_adds_the_table_to_a_database_that_has_runs_and_leaves_them_as_they_were(db_path, dataset):
    conn = connect(db_path)
    init_db(conn)
    import_openmantra(conn, dataset)
    client = FakeClient([ZH])
    assert runs.translate_page(conn, "book_a", 1, make_client=lambda: client, dataset_root=dataset).result.ok
    conn.execute("DROP TABLE run_events")  # the database as it was before this task
    conn.commit()

    def counts():
        return tuple(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("runs", "translations", "segments"))

    before = counts()
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'run_events'").fetchone() is None
    init_db(conn)
    init_db(conn)  # and again: it is the start of every API process and eval

    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'run_events'").fetchone() is not None
    assert counts() == before == (1, 2, 2)
    assert store.run_events(conn, 1) == []
    conn.close()
