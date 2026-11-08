"""Translate from the editor runs the page in the background (task 036): `POST .../translate` answers 202 once the Run
exists, one Run per page at a time, the background Run's thread and connection, and the book's Editor KB.

The API runs on the `api` fixture (conftest.py): `FakeClient` for `baseline`, and for `full` the real LLMClient on a scripted
provider. A Run is held at the provider with a `Hold`, so that the test sees it while it is still `running`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from conftest import CrashAfterCallClient, FakeClient, GatedClient, mock_llm_client, post_translate
from live_trace_support import assert_well_formed
from test_agent_loop import Provider, answer
from test_reviewer import APPROVE

from fukidashi import background as background_module
from fukidashi import config, store
from fukidashi.api import deps
from fukidashi.api.deps import get_client_factory
from fukidashi.main import app

GOOD = json.dumps({"translations": [{"id": 0, "zh": "你好"}, {"id": 1, "zh": "再见"}]}, ensure_ascii=False)
PAGE = "/api/pages/book_a/1/translate"


def run_rows(conn) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM runs ORDER BY id").fetchall()


def kinds(conn, run_id) -> list[str]:
    return [e.kind for e in store.run_events(conn, run_id)]


def clone_page(conn, book_title: str, page_index: int) -> None:
    """Page `page_index` of `book_title` (the book too, when it is new), with book_a/1's image and regions."""
    conn.execute("INSERT OR IGNORE INTO books (book_title) VALUES (?)", (book_title,))
    page_id = conn.execute("INSERT INTO pages (book_title, page_index, image_path) "
                           "SELECT ?, ?, image_path FROM pages WHERE book_title = 'book_a' AND page_index = 1",
                           (book_title, page_index)).lastrowid
    conn.execute("INSERT INTO segments (page_id, idx, x, y, w, h, text_ja, text_zh_hant, text_en) "
                 "SELECT ?, idx, x, y, w, h, text_ja, text_zh_hant, text_en FROM segments WHERE page_id = "
                 "(SELECT id FROM pages WHERE book_title = 'book_a' AND page_index = 1)", (page_id,))
    conn.commit()


def held(hold, *replies):
    """A scripted provider whose first reply waits at `hold` until the test lets it go."""
    def first(_):
        hold.wait()
        return replies[0]

    return Provider(first, *replies[1:])


# --- AC1: Translate in the background ---------------------------------------------------------------------------------

def test_translate_answers_202_with_the_runs_id_while_the_run_is_still_running(api, background, clients, conn, new_hold):
    client, _ = api
    hold = new_hold()
    clients.append(GatedClient([GOOD], hold=hold))

    resp = client.post(PAGE)

    assert resp.status_code == 202
    run_id = resp.json()["run_id"]
    assert hold.entered.wait(10)  # the Run is at the provider: the response did not wait for the model
    row = store.get_run(conn, run_id)
    assert (row["status"], row["config"], row["finished_at"], row["calls"]) == ("running", "baseline", None, 0)
    assert client.get("/api/pages/book_a/1").json()["run"] is None  # nothing to show until the Run ends

    hold.release.set()
    assert background.wait_idle(10)
    row = store.get_run(conn, run_id)
    assert (row["status"], row["calls"]) == ("succeeded", 1)
    assert [s["text_zh_hans"] for s in client.get("/api/pages/book_a/1").json()["segments"]] == ["你好", "再见"]


def test_without_a_body_it_runs_baseline_and_with_full_it_runs_full(api, background, clients, conn):
    client, use_replies = api
    use_replies(GOOD)
    assert post_translate(client, background).status_code == 202
    use_replies(GOOD)
    assert post_translate(client, background, config="baseline").status_code == 202
    clients.append(mock_llm_client(Provider(answer(GOOD), answer(APPROVE))))
    resp = post_translate(client, background, config="full")

    assert resp.status_code == 202
    assert [r["config"] for r in run_rows(conn)] == ["baseline", "baseline", "full"]
    assert json.loads(run_rows(conn)[2]["config_json"])["max_send_backs"] == 2  # the Orchestrator's parameters
    assert [r["status"] for r in run_rows(conn)] == ["succeeded"] * 3


@pytest.mark.parametrize("body", [{"config": "translator"}, {"config": "full_no_kb"}, {"config": "FULL"}, {"config": ""},
                                  {"config": None}, {"config": 5}, {"config": "full", "extra": 1}, {"confg": "full"},
                                  ["full"], "full"])
def test_any_other_body_is_a_422_and_starts_nothing(api, background, conn, body):
    client, use_replies = api
    fake = use_replies(GOOD)

    resp = client.post(PAGE, json=body)

    assert resp.status_code == 422
    assert fake.requests == [] and run_rows(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM kbs").fetchone()[0] == 0


def test_a_body_that_is_not_json_is_a_422(api, background, conn):
    client, use_replies = api
    use_replies(GOOD)

    resp = client.post(PAGE, content=b"{nope", headers={"content-type": "application/json"})

    assert resp.status_code == 422 and run_rows(conn) == []


@pytest.mark.parametrize("config_name", ["baseline", "full"])
@pytest.mark.parametrize("book, page, status", [("book_a", 99, 404), ("no_book", 1, 404), ("book_a", 2, 422)])
def test_a_page_refused_before_its_run_exists_keeps_todays_status_and_records_nothing(
        api, background, conn, config_name, book, page, status):
    client, use_replies = api
    fake = use_replies(GOOD)

    resp = post_translate(client, background, book, page, config=config_name)

    assert resp.status_code == status
    assert fake.requests == [] and run_rows(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM kbs").fetchone()[0] == 0  # a refused page makes no Editor KB either


@pytest.mark.parametrize("config_name", ["baseline", "full"])
def test_a_missing_image_is_a_404_and_records_nothing(api, background, conn, dataset, config_name):
    client, use_replies = api
    use_replies(GOOD)
    (dataset / "images" / "book_a" / "ja" / "000.jpg").unlink()

    resp = post_translate(client, background, config=config_name)

    assert resp.status_code == 404 and "image file missing" in resp.json()["detail"]
    assert run_rows(conn) == [] and conn.execute("SELECT COUNT(*) FROM kbs").fetchone()[0] == 0


@pytest.mark.parametrize("config_name", ["baseline", "full"])
def test_missing_llm_settings_are_a_503_and_record_no_run(api, background, conn, tmp_path, monkeypatch, config_name):
    client, _ = api
    del app.dependency_overrides[get_client_factory]  # the real factory, as the app runs it
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "ROOT", tmp_path)  # so the real .env is never read

    resp = post_translate(client, background, config=config_name)

    assert resp.status_code == 503
    assert resp.json()["detail"] == "Set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in .env (see .env.example)"
    assert run_rows(conn) == []


# --- AC2: one Run per page ---------------------------------------------------------------------------------------------

def test_a_post_while_the_pages_run_is_running_is_a_409_and_records_no_run(api, background, clients, conn, new_hold):
    client, use_replies = api
    hold = new_hold()
    clients.append(GatedClient([GOOD], hold=hold))
    first = client.post(PAGE)
    assert first.status_code == 202 and hold.entered.wait(10)

    for body in (None, {"config": "baseline"}, {"config": "full"}):
        second = client.post(PAGE, json=body)
        assert second.status_code == 409 and "already being translated" in second.json()["detail"]
    assert [r["id"] for r in run_rows(conn)] == [first.json()["run_id"]]  # still the one Run

    hold.release.set()
    assert background.wait_idle(10)
    use_replies(GOOD)
    assert client.post(PAGE).status_code == 202  # the page is free again once its Run has ended
    assert background.wait_idle(10)
    assert [r["status"] for r in run_rows(conn)] == ["succeeded", "succeeded"]


def test_a_post_on_another_page_while_one_is_running_is_not_refused(api, background, clients, conn, new_hold):
    client, _ = api
    clone_page(conn, "book_a", 3)
    clone_page(conn, "book_b", 1)
    holds = [new_hold(), new_hold(), new_hold()]
    pages = ["/api/pages/book_a/1/translate", "/api/pages/book_a/3/translate", "/api/pages/book_b/1/translate"]
    ids = []
    for hold, page in zip(holds, pages):
        clients.append(GatedClient([GOOD], hold=hold))
        resp = client.post(page)
        assert resp.status_code == 202
        assert hold.entered.wait(10)  # each Run is at the provider before the next is started
        ids.append(resp.json()["run_id"])

    assert len(set(ids)) == 3
    assert [r["status"] for r in run_rows(conn)] == ["running"] * 3
    for hold in holds:
        hold.release.set()
    assert background.wait_idle(10)
    assert [r["status"] for r in run_rows(conn)] == ["succeeded"] * 3


def test_two_posts_at_once_give_one_202_one_409_and_one_run(api, background, clients, conn, new_hold):
    client, _ = api
    for _round in range(15):
        hold = new_hold()
        clients.append(GatedClient([GOOD], hold=hold))  # the Run cannot end before both POSTs have been answered
        barrier = threading.Barrier(2)

        def post():
            barrier.wait(10)
            return client.post(PAGE).status_code

        with ThreadPoolExecutor(2) as pool:
            statuses = sorted(f.result(30) for f in [pool.submit(post), pool.submit(post)])

        assert statuses == [202, 409]
        assert hold.entered.wait(10)
        hold.release.set()
        assert background.wait_idle(10)
    assert [r["status"] for r in run_rows(conn)] == ["succeeded"] * 15  # one Run in each of the 15 rounds


def test_a_running_row_in_the_database_blocks_the_page_even_when_this_process_has_no_run(api, background, conn):
    client, use_replies = api
    use_replies(GOOD)
    page_id = store.find_page(conn, "book_a", 1)["id"]
    stale = store.start_run(conn, page_id=page_id, config="baseline", params={}, model="m")  # left by another process

    resp = client.post(PAGE)

    assert resp.status_code == 409 and [r["id"] for r in run_rows(conn)] == [stale]
    assert client.get("/api/pages/book_a/1").json()["running_run_id"] == stale
    conn.execute("UPDATE runs SET status = 'failed' WHERE id = ?", (stale,))
    conn.commit()
    assert post_translate(client, background).status_code == 202


# --- AC3: the background Run -------------------------------------------------------------------------------------------

class Connections:
    """The connections a background Run opens (`background.connect`), to see afterwards that they are closed."""

    def __init__(self, monkeypatch):
        self.opened: list[sqlite3.Connection] = []
        real = background_module.connect

        def spy(path):
            self.opened.append(real(path))
            return self.opened[-1]

        monkeypatch.setattr(background_module, "connect", spy)

    def all_closed(self) -> bool:
        for c in self.opened:
            try:
                c.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                continue
            return False
        return bool(self.opened)


def test_a_succeeded_run_has_its_events_run_start_first_and_run_end_last(api, background, conn):
    client, use_replies = api
    use_replies(GOOD)

    run_id = post_translate(client, background).json()["run_id"]

    events = store.run_events(conn, run_id)
    assert_well_formed(events, run_id)
    assert [e.kind for e in events] == ["run_start", "run_end"]  # baseline's whole story
    assert events[0].data == {"run_id": run_id, "config": "baseline", "model": "fake-model"}
    row = store.get_run(conn, run_id)
    assert events[-1].data["status"] == row["status"] == "succeeded" and events[-1].data["calls"] == row["calls"] == 1


def test_a_full_run_reports_its_stages_and_ends_with_the_loops_outcome(api, background, clients, conn):
    client, _ = api
    clients.append(mock_llm_client(Provider(answer(GOOD), answer(APPROVE))))

    run_id = post_translate(client, background, config="full").json()["run_id"]

    events = store.run_events(conn, run_id)
    assert_well_formed(events, run_id)
    assert [f"{e.kind}:{e.data['stage']}" if e.kind.startswith("stage_") else e.kind for e in events] == [
        "run_start", "stage_start:translate", "model_call", "stage_end:translate", "stage_start:review", "model_call",
        "stage_end:review", "stage_start:finalize", "stage_end:finalize", "run_end"]  # no term proposals: finalize is skipped
    assert events[-1].data["status"] == "succeeded" and events[-1].data["final_stop"] == "approved"
    assert store.get_run(conn, run_id)["config"] == "full"


def test_a_provider_error_is_a_failed_run_whose_run_end_carries_the_error(api, background, clients, conn):
    client, _ = api
    clients.append(mock_llm_client(lambda request: httpx.Response(500, json={"error": {"message": "upstream exploded"}})))

    run_id = post_translate(client, background).json()["run_id"]

    row = store.get_run(conn, run_id)
    assert row["status"] == "failed" and "upstream exploded" in row["error"]
    events = store.run_events(conn, run_id)
    assert_well_formed(events, run_id)
    assert events[-1].data["status"] == "failed" and events[-1].data["error"] == row["error"]


def test_a_crash_after_the_row_exists_is_a_failed_run(api, background, clients, conn):
    client, _ = api
    clients.append(CrashAfterCallClient([GOOD]))

    resp = post_translate(client, background)

    assert resp.status_code == 202  # the crash is the Run's, reported by its events and the page
    row = store.get_run(conn, resp.json()["run_id"])
    assert (row["status"], row["error"]) == ("failed", "RuntimeError: crashed after the call")
    events = store.run_events(conn, row["id"])
    assert [e.kind for e in events] == ["run_start", "run_end"] and events[-1].data["error"] == row["error"]
    assert conn.execute("SELECT COUNT(*) FROM translations").fetchone()[0] == 0


@pytest.mark.parametrize("how", ["succeeded", "provider_error", "crash"])
def test_the_threads_connection_is_closed_and_no_run_is_left_running(api, background, clients, conn, monkeypatch, how):
    client, use_replies = api
    connections = Connections(monkeypatch)
    if how == "succeeded":
        use_replies(GOOD)
    elif how == "provider_error":
        clients.append(mock_llm_client(lambda request: httpx.Response(500, json={"error": {"message": "no"}})))
    else:
        clients.append(CrashAfterCallClient([GOOD]))

    assert post_translate(client, background).status_code == 202

    assert len(connections.opened) == 1 and connections.all_closed()
    assert conn.execute("SELECT COUNT(*) FROM runs WHERE status = 'running'").fetchone()[0] == 0
    assert len(run_rows(conn)) == 1


@pytest.mark.parametrize("how", ["page_refused", "client_not_built"])
def test_a_run_refused_before_it_exists_closes_the_threads_connection_too(api, background, conn, monkeypatch, how):
    client, use_replies = api
    connections = Connections(monkeypatch)
    use_replies(GOOD)
    if how == "page_refused":
        assert post_translate(client, background, page=2).status_code == 422
    else:  # an unexpected failure before the Run exists reaches the caller as a server error
        def broken():
            raise RuntimeError("no client")

        app.dependency_overrides[get_client_factory] = lambda: broken
        with pytest.raises(RuntimeError, match="no client"):
            client.post(PAGE)
        assert background.wait_idle(10)

    assert len(connections.opened) == 1 and connections.all_closed()
    assert run_rows(conn) == []


def test_the_route_learns_the_runs_id_even_when_storing_an_event_fails(api, background, conn, monkeypatch, caplog):
    client, use_replies = api
    use_replies(GOOD)

    def broken(conn, event):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "add_run_event", broken)
    with caplog.at_level(logging.WARNING):
        resp = post_translate(client, background)

    assert resp.status_code == 202
    row = store.get_run(conn, resp.json()["run_id"])
    assert row["status"] == "succeeded"  # the Run is not the sink's to fail
    assert store.run_events(conn, row["id"]) == []
    assert sum("live trace sink raised" in r.getMessage() for r in caplog.records) == 1


def test_the_thread_has_its_own_connection_so_the_requests_connection_is_never_shared(api, background, conn, monkeypatch):
    client, use_replies = api
    use_replies(GOOD)
    seen = Connections(monkeypatch)

    post_translate(client, background)

    assert len(seen.opened) == 1  # one connection for the one Run, opened from the database path


def test_the_page_is_free_before_run_end_is_stored(api, background, conn, monkeypatch):
    """The registry lets the page go before `run_end` is stored, so a client that has seen `run_end` can Translate at once.
    Held at the moment the Run's thread stores its `run_end`: the Run's row is finished, the event is not yet stored, and a
    POST on the page is already accepted."""
    client, use_replies = api
    use_replies(GOOD)
    at_run_end, go_on = threading.Event(), threading.Event()
    real = store.add_run_event

    def holding(c, event):
        if event.kind == "run_end" and not at_run_end.is_set():
            at_run_end.set()
            assert go_on.wait(10)
        return real(c, event)

    monkeypatch.setattr(store, "add_run_event", holding)
    first = client.post(PAGE).json()["run_id"]
    assert at_run_end.wait(10)
    assert store.get_run(conn, first)["status"] == "succeeded" and "run_end" not in kinds(conn, first)

    use_replies(GOOD)
    second = client.post(PAGE)  # accepted: the first Run's thread has not ended, but it no longer holds the page

    assert second.status_code == 202 and second.json()["run_id"] != first
    go_on.set()
    assert background.wait_idle(10)
    assert kinds(conn, first)[-1] == "run_end" == kinds(conn, second.json()["run_id"])[-1]
    assert [r["status"] for r in run_rows(conn)] == ["succeeded", "succeeded"]


def test_a_run_whose_events_cannot_be_stored_frees_its_page_all_the_same(api, background, conn, monkeypatch):
    """No `run_end` reaches the registry's sink (the first failed store stops the sink), so the thread's end must free the
    page: the registry may not wait for an event that never comes."""
    client, use_replies = api
    use_replies(GOOD)

    def broken(c, event):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "add_run_event", broken)
    assert post_translate(client, background).status_code == 202
    assert background._active == {}

    use_replies(GOOD)
    assert post_translate(client, background).status_code == 202  # not refused as busy
    assert [r["status"] for r in run_rows(conn)] == ["succeeded", "succeeded"]


def test_an_ended_runs_late_release_does_not_free_the_page_of_the_run_that_took_it(api, background, conn, monkeypatch, new_hold):
    """A's `run_end` frees the page; B takes it; A's thread then ends and releases again. The token says the key is B's, so
    the page stays held and a third POST is refused, even while B has no Run row yet (held in building its client)."""
    client, use_replies = api
    use_replies(GOOD)
    at_run_end, go_on = threading.Event(), threading.Event()
    real = store.add_run_event

    def holding(c, event):
        if event.kind == "run_end" and not at_run_end.is_set():
            at_run_end.set()
            assert go_on.wait(10)
        return real(c, event)

    monkeypatch.setattr(store, "add_run_event", holding)
    assert client.post(PAGE).status_code == 202 and at_run_end.wait(10)  # A: its row is finished, its thread is held
    thread_a = background._threads[0]

    building = new_hold()

    def slow_factory():
        building.wait()
        return FakeClient([GOOD])

    app.dependency_overrides[get_client_factory] = lambda: slow_factory
    holder: dict = {}
    poster = threading.Thread(target=lambda: holder.update(resp=client.post(PAGE)), daemon=True)
    poster.start()  # B: takes the page, then waits for its client, before any Run row exists
    assert building.entered.wait(10)

    go_on.set()  # A's thread goes on and ends, releasing its (old) token
    thread_a.join(10)
    assert not thread_a.is_alive()

    third = client.post(PAGE)
    assert third.status_code == 409  # B still holds the page

    building.release.set()
    poster.join(10)
    assert holder["resp"].status_code == 202
    assert background.wait_idle(10)
    assert [r["status"] for r in run_rows(conn)] == ["succeeded", "succeeded"]


def test_a_thread_that_cannot_start_leaves_the_page_free(api, background, conn, monkeypatch):
    client, use_replies = api
    use_replies(GOOD)
    real_start = threading.Thread.start

    def cannot(self):
        if self.name.startswith("run "):  # a Run's thread only: the test client's own threads start as usual
            raise RuntimeError("can't start new thread")
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", cannot)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        client.post(PAGE)
    monkeypatch.setattr(threading.Thread, "start", real_start)

    assert background._active == {} and run_rows(conn) == []
    assert post_translate(client, background).status_code == 202


def test_a_run_started_while_another_start_is_pruning_stays_listed_for_wait_idle(api, background, conn, monkeypatch):
    """`wait_idle` is what keeps a Run's thread from outliving a test's database. A thread is listed only once it is alive,
    so another `start` that prunes the ended threads at that moment cannot drop it. Held inside the first `Thread.start`,
    a second start (another page) prunes: both threads must be listed afterwards."""
    client, _ = api
    clone_page(conn, "book_a", 3)
    inside_start, go_on = threading.Event(), threading.Event()
    real_start = threading.Thread.start
    first = []

    def start(self):
        if self.name.startswith("run ") and not first:
            first.append(self)
            inside_start.set()
            assert go_on.wait(10)
        return real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    app.dependency_overrides[get_client_factory] = lambda: lambda: FakeClient([GOOD])  # one client for each Run
    poster = threading.Thread(target=lambda: client.post(PAGE), daemon=True)
    poster.start()  # run 1's start() is held before its thread is alive
    assert inside_start.wait(10)

    assert client.post("/api/pages/book_a/3/translate").status_code == 202  # this start prunes the list meanwhile
    go_on.set()
    poster.join(10)

    assert first[0] in background._threads  # (the second thread may have ended and been pruned by now: only the first matters)
    assert background.wait_idle(10)


@pytest.mark.parametrize("page, status", [(99, 404), (2, 422)])
def test_a_refused_page_is_free_again_before_the_caller_is_answered(api, background, monkeypatch, page, status):
    """The thread lets the page go and then reports the refusal, so a retry sent at once is not refused as busy."""
    client, use_replies = api
    use_replies(GOOD)
    real = background_module.BackgroundRuns._release

    def slow_release(self, key, token):
        time.sleep(0.2)  # were the caller told first, it would be answered while the key was still held
        real(self, key, token)

    monkeypatch.setattr(background_module.BackgroundRuns, "_release", slow_release)

    assert client.post(f"/api/pages/book_a/{page}/translate").status_code == status

    assert background._active == {}  # released already: no waiting for the thread to end


# --- AC4: the Editor KB ------------------------------------------------------------------------------------------------

def editor_kbs(conn) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM kbs ORDER BY id").fetchall()


def full_run(client, background, clients, book="book_a", page=1) -> int:
    clients.append(mock_llm_client(Provider(answer(GOOD), answer(APPROVE))))
    resp = post_translate(client, background, book, page, config="full")
    assert resp.status_code == 202
    return resp.json()["run_id"]


def kb_of(conn, run_id) -> int:
    return json.loads(store.get_run(conn, run_id)["attempts_json"])["finalize"]["kb_id"]


def test_the_first_full_run_of_a_book_makes_its_editor_kb_and_uses_it(api, background, clients, conn):
    client, _ = api
    assert editor_kbs(conn) == []

    run_id = full_run(client, background, clients)

    (kb,) = editor_kbs(conn)
    assert (kb["book_title"], kb["purpose"], kb["label"]) == ("book_a", "editor", store.EDITOR_KB_LABEL)
    assert kb_of(conn, run_id) == kb["id"]


def test_later_full_runs_of_the_book_share_the_kb_and_another_book_gets_its_own(api, background, clients, conn):
    client, _ = api
    clone_page(conn, "book_a", 3)
    clone_page(conn, "book_b", 1)

    first = full_run(client, background, clients, "book_a", 1)
    again = full_run(client, background, clients, "book_a", 1)
    other_page = full_run(client, background, clients, "book_a", 3)
    other_book = full_run(client, background, clients, "book_b", 1)

    kbs = editor_kbs(conn)
    assert [(k["book_title"], k["purpose"]) for k in kbs] == [("book_a", "editor"), ("book_b", "editor")]
    assert kb_of(conn, first) == kb_of(conn, again) == kb_of(conn, other_page) == kbs[0]["id"]
    assert kb_of(conn, other_book) == kbs[1]["id"] != kbs[0]["id"]


def test_two_first_requests_for_one_book_at_once_give_one_kb(api, background, conn):
    client, _ = api
    for n in range(8):
        clone_page(conn, f"book_c{n}", 1)
        clone_page(conn, f"book_c{n}", 2)
    barrier = threading.Barrier(2)
    app.dependency_overrides[get_client_factory] = lambda: lambda: mock_llm_client(Provider(answer(GOOD), answer(APPROVE)))

    def start(book, page):
        barrier.wait(10)
        return client.post(f"/api/pages/{book}/{page}/translate", json={"config": "full"}).status_code

    for n in range(8):
        barrier.reset()
        with ThreadPoolExecutor(2) as pool:  # two different pages of one book, so neither is refused as busy
            statuses = [f.result(30) for f in [pool.submit(start, f"book_c{n}", 1), pool.submit(start, f"book_c{n}", 2)]]
        assert statuses == [202, 202]
        assert background.wait_idle(30)

    assert [k["book_title"] for k in editor_kbs(conn)] == [f"book_c{n}" for n in range(8)]  # one KB per book


def test_the_stores_lookup_or_create_is_idempotent_and_refuses_an_unknown_book(conn, dataset):
    from fukidashi.datasets.importer import import_openmantra
    import_openmantra(conn, dataset)

    first = store.editor_kb(conn, "book_a")

    assert store.editor_kb(conn, "book_a") == first
    assert conn.execute("SELECT COUNT(*) FROM kbs").fetchone()[0] == 1
    with pytest.raises(sqlite3.IntegrityError):
        store.editor_kb(conn, "no_book")
    assert conn.execute("SELECT COUNT(*) FROM kbs").fetchone()[0] == 1


def test_an_eval_kb_of_the_same_book_is_not_the_editor_kb(conn, dataset):
    from fukidashi.datasets.importer import import_openmantra
    import_openmantra(conn, dataset)
    eval_kb = store.create_kb(conn, book_title="book_a", purpose="eval", label="20261107T000000Z-eval")

    editor = store.editor_kb(conn, "book_a")

    assert editor != eval_kb and conn.execute("SELECT COUNT(*) FROM kbs").fetchone()[0] == 2


def test_the_editor_does_not_run_other_configs(background, conn, dataset, db_path):
    with pytest.raises(ValueError, match="the editor runs"):
        background.start(conn, "book_a", 1, config="translator", make_client=lambda: None, dataset_root=dataset,
                         db_path=db_path)


def test_the_api_dependency_is_one_registry_for_the_process():
    assert deps.get_background_runs() is deps.get_background_runs()
