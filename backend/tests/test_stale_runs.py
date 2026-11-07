"""At start the API marks every Run left `running` as failed (task 036): a Run is only `running` between `start_run` and
`finish_run`, so one found at start belongs to a process that stopped."""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from fukidashi import config as app_config
from fukidashi import main, store
from fukidashi.datasets.importer import import_openmantra
from fukidashi.db import connect, init_db

USAGE = {"calls": 1, "unknown_usage_calls": 0, "input_tokens": 100, "output_tokens": 20, "cost_usd": 0.0001}


def database(tmp_path, dataset):
    """A database holding every kind of Run: two `running` (on two pages), a succeeded one with Translations, a failed one,
    and a `running` one's events."""
    conn = connect(tmp_path / "api.db")
    init_db(conn)
    import_openmantra(conn, dataset)
    page_1, page_2 = (store.find_page(conn, "book_a", i)["id"] for i in (1, 2))
    segment_ids = [s["id"] for s in store.page_segments(conn, page_1)]
    succeeded = store.start_run(conn, page_id=page_1, config="baseline", params={}, model="m")
    store.finish_run(conn, succeeded, error=None, response_models=["m-1"], usage=USAGE, attempts=[],
                     translations={segment_ids[0]: "你好", segment_ids[1]: "再见"})
    failed = store.start_run(conn, page_id=page_1, config="baseline", params={}, model="m")
    store.finish_run(conn, failed, error="boom", response_models=[], usage=USAGE, attempts=[], translations=None)
    stale = [store.start_run(conn, page_id=page_1, config="full", params={}, model="m"),
             store.start_run(conn, page_id=page_2, config="baseline", params={}, model="m")]
    return conn, succeeded, failed, stale


def snapshot(conn, run_id):
    return tuple(store.get_run(conn, run_id))


def test_the_api_marks_every_running_run_failed_when_it_starts_and_logs_how_many(tmp_path, dataset, monkeypatch, caplog):
    monkeypatch.setattr(main, "load_dotenv", lambda path: None)
    monkeypatch.setattr(app_config, "DB_PATH", tmp_path / "api.db")
    conn, succeeded, failed, stale = database(tmp_path, dataset)
    before = {run_id: snapshot(conn, run_id) for run_id in (succeeded, failed)}
    translations_before = conn.execute("SELECT * FROM translations ORDER BY run_id, segment_id").fetchall()

    with caplog.at_level(logging.INFO, logger="fukidashi.main"):
        with TestClient(main.app):  # runs the app's lifespan
            marked = [store.get_run(conn, run_id) for run_id in stale]

    for row in marked:
        assert row["status"] == "failed" and row["error"] == main.STALE_RUN_ERROR and row["finished_at"] is not None
        assert conn.execute("SELECT COUNT(*) FROM translations WHERE run_id = ?", (row["id"],)).fetchone()[0] == 0
    assert {run_id: snapshot(conn, run_id) for run_id in (succeeded, failed)} == before  # the others are left alone
    assert conn.execute("SELECT * FROM translations ORDER BY run_id, segment_id").fetchall() == translations_before
    assert conn.execute("SELECT COUNT(*) FROM runs WHERE status = 'running'").fetchone()[0] == 0
    (record,) = [r for r in caplog.records if r.name == "fukidashi.main"]
    assert record.levelno == logging.WARNING and "marked 2 Run(s)" in record.getMessage()
    assert store.running_run_id(conn, store.find_page(conn, "book_a", 1)["id"]) is None  # the page can be translated again


def test_a_start_with_no_stale_run_changes_nothing_and_says_so(tmp_path, dataset, monkeypatch, caplog):
    monkeypatch.setattr(main, "load_dotenv", lambda path: None)
    monkeypatch.setattr(app_config, "DB_PATH", tmp_path / "api.db")
    conn, succeeded, failed, stale = database(tmp_path, dataset)
    conn.execute("UPDATE runs SET status = 'failed', error = 'earlier' WHERE status = 'running'")
    conn.commit()
    before = [snapshot(conn, r["id"]) for r in conn.execute("SELECT id FROM runs ORDER BY id")]

    with caplog.at_level(logging.INFO, logger="fukidashi.main"):
        with TestClient(main.app):
            pass

    assert [snapshot(conn, r["id"]) for r in conn.execute("SELECT id FROM runs ORDER BY id")] == before
    (record,) = [r for r in caplog.records if r.name == "fukidashi.main"]
    assert record.levelno == logging.INFO and "no Run was left running" in record.getMessage()


def test_the_store_function_returns_the_count_and_keeps_a_runs_events(tmp_path, dataset):
    conn, succeeded, failed, stale = database(tmp_path, dataset)
    from fukidashi.live_trace import RunEvent
    store.add_run_event(conn, RunEvent(stale[0], 1, "run_start", {"run_id": stale[0]}))

    assert store.fail_stale_runs(conn, "stopped") == 2
    assert store.fail_stale_runs(conn, "stopped") == 0  # nothing left to mark
    assert [e.kind for e in store.run_events(conn, stale[0])] == ["run_start"]  # its stream can still be replayed
    assert store.get_run(conn, stale[0])["error"] == "stopped"
