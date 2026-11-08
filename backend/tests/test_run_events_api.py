"""The live trace over SSE (task 036): `GET /api/runs/{id}/events`, replay with `Last-Event-ID`, and `running_run_id`.

A Run that has ended is read with the `TestClient`. A Run that is still running is read with `Stream` (sse_support.py), which
hands the test each event as the app sends it; a Run is held at the provider with a `Hold` while the test looks.
"""

from __future__ import annotations

import json
import sqlite3
import time

import httpx
import pytest
from conftest import mock_llm_client, post_translate
from fastapi import HTTPException
from fastapi import routing as fastapi_routing
from sse_support import Stream, parse_sse
from test_agent_loop import Provider, answer
from test_background_runs import PAGE, clone_page, held
from test_reviewer import APPROVE

from fukidashi import runs, store
from fukidashi.api import runs as runs_api
from fukidashi.live_trace import RunEvent
from fukidashi.main import app

GOOD = json.dumps({"translations": [{"id": 0, "zh": "你好"}, {"id": 1, "zh": "再见"}]}, ensure_ascii=False)
FULL_KINDS = ["run_start", "stage_start", "model_call", "stage_end", "stage_start", "model_call", "stage_end",
              "stage_start", "stage_end", "run_end"]  # a `full` Run that approves at once and proposes no term


@pytest.fixture(autouse=True)
def quick_polls(monkeypatch):
    monkeypatch.setattr(runs_api, "POLL_INTERVAL_S", 0.01)


@pytest.fixture
def finished(api, background, clients) -> int:
    """The id of a finished `full` Run with its ten events (stored through the API's own background path)."""
    client, _ = api
    clients.append(mock_llm_client(Provider(answer(GOOD), answer(APPROVE))))
    return post_translate(client, background, config="full").json()["run_id"]


def start_held_full(client, clients, new_hold):
    """Start a `full` Run that stops at the provider's first reply: the Run's id, its Hold and the events it has by then."""
    hold = new_hold()
    clients.append(mock_llm_client(held(hold, answer(GOOD), answer(APPROVE))))
    resp = client.post(PAGE, json={"config": "full"})
    assert resp.status_code == 202 and hold.entered.wait(10)
    return resp.json()["run_id"], hold


def path(run_id: int) -> str:
    return f"/api/runs/{run_id}/events"


# --- AC5: the stream --------------------------------------------------------------------------------------------------

def test_a_finished_runs_events_come_in_seq_order_with_id_event_and_data(api, conn, finished):
    client, _ = api

    resp = client.get(path(finished))

    assert resp.status_code == 200 and resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    stored = store.run_events(conn, finished)
    assert [e.kind for e in stored] == FULL_KINDS
    assert parse_sse(resp.text) == [{"id": str(e.seq), "event": e.kind, "data": e.data} for e in stored]
    assert resp.text.startswith(f"event: run_start\ndata: {json.dumps(stored[0].data, ensure_ascii=False)}\nid: 1\n\n")
    assert "你好" in resp.text  # the text is sent as it is, not as \uXXXX escapes


def test_the_stream_sends_events_stored_after_it_opened_and_ends_with_the_run(api, clients, conn, new_hold, background):
    client, _ = api
    run_id, hold = start_held_full(client, clients, new_hold)

    with Stream(app, path(run_id)) as stream:
        assert stream.status == 200 and stream.headers["content-type"].startswith("text/event-stream")
        first = [stream.next_event(), stream.next_event()]
        assert [e["event"] for e in first] == ["run_start", "stage_start"]  # what the Run had stored when it opened
        with pytest.raises(TimeoutError):  # the stream is open and the Run is still running: nothing more yet
            stream.next_event(timeout=0.3)
        hold.release.set()  # the Run goes on; its next events are stored after the stream opened
        rest = stream.read_all(timeout=10)

    events = first + rest
    assert [e["event"] for e in events] == FULL_KINDS and events[-1]["data"]["status"] == "succeeded"
    assert [e["id"] for e in events] == [str(n) for n in range(1, 11)]
    assert [e["data"] for e in events] == [e.data for e in store.run_events(conn, run_id)]


def test_a_stream_ends_once_the_run_is_over_even_when_its_run_end_is_missing(api, conn):
    client, _ = api
    page_id = store.find_page(conn, "book_a", 1)["id"]
    run_id = store.start_run(conn, page_id=page_id, config="baseline", params={}, model="m")
    for seq, kind in enumerate(["run_start", "stage_start", "model_call"], start=1):
        store.add_run_event(conn, RunEvent(run_id, seq, kind, {"n": seq}))

    with Stream(app, path(run_id)) as stream:
        assert [stream.next_event()["id"] for _ in range(3)] == ["1", "2", "3"]
        with pytest.raises(TimeoutError):  # still running: the stream stays open
            stream.next_event(timeout=0.3)
        store.finish_run(conn, run_id, error="the process died", response_models=[], attempts=[], translations=None,
                         usage={"calls": 0, "unknown_usage_calls": 0, "input_tokens": 0, "output_tokens": 0,
                                "cost_usd": None})
        assert stream.next_event(timeout=10) is None  # over: no run_end was ever stored, and it does not wait for one

    assert [e.kind for e in store.run_events(conn, run_id)] == ["run_start", "stage_start", "model_call"]
    again = client.get(path(run_id))  # opened after the end: the same three, then it ends
    assert [e["id"] for e in parse_sse(again.text)] == ["1", "2", "3"]


def test_a_run_end_stored_just_after_the_row_was_finished_is_still_sent(api, conn, monkeypatch):
    """`translate_page` finishes the Run's row and then stores `run_end`, so a stream can read 'not running' with the last
    event still to come. It looks once more before it ends."""
    client, _ = api
    page_id = store.find_page(conn, "book_a", 1)["id"]
    run_id = store.start_run(conn, page_id=page_id, config="baseline", params={}, model="m")
    store.add_run_event(conn, RunEvent(run_id, 1, "run_start", {}))
    real_read = runs_api._read
    reads = 0

    def racing_read(c, rid, after):
        nonlocal reads
        reads += 1
        result = real_read(c, rid, after)
        if reads == 2:  # the stream's first read saw the Run running; the Run's row is finished now
            conn.execute("UPDATE runs SET status = 'failed' WHERE id = ?", (rid,))
            conn.commit()
        elif reads == 3:  # its second read saw the finished row and no `run_end` yet; the event is stored right after
            store.add_run_event(conn, RunEvent(rid, 2, "run_end", {"status": "failed"}))
        return result

    monkeypatch.setattr(runs_api, "_read", racing_read)
    with Stream(app, path(run_id)) as stream:
        events = stream.read_all(timeout=10)

    assert [e["event"] for e in events] == ["run_start", "run_end"]


def test_a_reconnect_is_not_answered_204_when_the_run_ends_between_the_status_and_the_events_being_read(
        api, conn, monkeypatch):
    """`_read` reads the status first. A Run that finishes (and stores the rest of its events) between the two reads must
    not look 'not running and nothing after': that would be a 204, and the browser then stops for good and never gets the
    events it missed. The Run ends right after the second store call of the request: the route's 404 check is the first, so
    that is `_read`'s first read, whichever it is (swap the two and the events come back empty, the status 'failed')."""
    client, _ = api
    page_id = store.find_page(conn, "book_a", 1)["id"]
    run_id = store.start_run(conn, page_id=page_id, config="baseline", params={}, model="m")
    for seq, kind in ((1, "run_start"), (2, "stage_start")):
        store.add_run_event(conn, RunEvent(run_id, seq, kind, {}))
    real_get_run, real_run_events = store.get_run, store.run_events
    calls = 0

    def after_second_call(real):
        def wrapped(c, *args, **kwargs):
            nonlocal calls
            result = real(c, *args, **kwargs)
            calls += 1
            if calls == 2:  # `_read`'s first read has happened; now the Run finishes and stores the rest
                conn.execute("UPDATE runs SET status = 'succeeded' WHERE id = ?", (run_id,))
                conn.commit()
                store.add_run_event(conn, RunEvent(run_id, 3, "stage_end", {}))
                store.add_run_event(conn, RunEvent(run_id, 4, "run_end", {"status": "succeeded"}))
            return result

        return wrapped

    monkeypatch.setattr(store, "get_run", after_second_call(real_get_run))
    monkeypatch.setattr(store, "run_events", after_second_call(real_run_events))
    with Stream(app, path(run_id), last_event_id=2) as stream:
        assert stream.status == 200  # not a 204
        events = stream.read_all(timeout=10)

    assert [e["id"] for e in events] == ["3", "4"] and events[-1]["event"] == "run_end"


def test_an_unknown_run_is_a_404(api):
    client, _ = api

    resp = client.get(path(999))

    assert resp.status_code == 404 and "no run 999" in resp.json()["detail"]


@pytest.mark.parametrize("run_id", [0, -1, 2**63 - 1, 2**63, 10**30])
def test_a_run_id_sqlite_cannot_hold_is_an_unknown_run_not_a_server_error(api, run_id):
    client, _ = api

    assert client.get(path(run_id)).status_code == 404


def test_an_idle_stream_sends_a_comment_now_and_then_to_keep_the_connection_alive(api, clients, new_hold, monkeypatch):
    monkeypatch.setattr(fastapi_routing, "_PING_INTERVAL", 0.05)  # FastAPI's own keepalive for an SSE path operation
    client, _ = api
    run_id, hold = start_held_full(client, clients, new_hold)

    with Stream(app, path(run_id)) as stream:
        assert [stream.next_event()["event"], stream.next_event()["event"]] == ["run_start", "stage_start"]
        with pytest.raises(TimeoutError):
            stream.next_event(timeout=0.5)
        assert stream.comments and set(stream.comments) == {": ping"}
        hold.release.set()
        assert stream.read_all(timeout=10)[-1]["event"] == "run_end"


def test_a_stream_reads_through_a_connection_of_its_own_which_closes_when_the_client_goes(
        api, clients, new_hold, monkeypatch):
    client, _ = api
    opened: list[sqlite3.Connection] = []
    real = runs_api.connect

    def spy(db_path):
        opened.append(real(db_path))
        return opened[-1]

    monkeypatch.setattr(runs_api, "connect", spy)
    run_id, hold = start_held_full(client, clients, new_hold)

    def is_closed(c):
        try:
            c.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            return True
        return False

    with Stream(app, path(run_id)) as stream:
        stream.next_event()
        assert len(opened) == 1 and not is_closed(opened[0])  # alive as long as the stream
    deadline = time.monotonic() + 10
    while not is_closed(opened[0]):  # the client went away: the stream's generator is closed, and its connection with it
        assert time.monotonic() < deadline
        time.sleep(0.01)
    hold.release.set()


def test_the_finished_streams_connection_is_closed_too(api, finished, monkeypatch):
    client, _ = api
    opened: list[sqlite3.Connection] = []
    real = runs_api.connect
    monkeypatch.setattr(runs_api, "connect", lambda p: opened.append(real(p)) or opened[-1])

    client.get(path(finished))

    assert len(opened) == 1
    with pytest.raises(sqlite3.ProgrammingError):
        opened[0].execute("SELECT 1")


# --- AC6: replay with Last-Event-ID -----------------------------------------------------------------------------------

def ids(text: str) -> list[int]:
    return [int(e["id"]) for e in parse_sse(text)]


@pytest.mark.parametrize("after", range(0, 10))
def test_a_finished_run_replays_from_the_event_after_last_event_id(api, finished, after):
    client, _ = api

    resp = client.get(path(finished), headers={"Last-Event-ID": str(after)})

    assert resp.status_code == 200
    assert ids(resp.text) == list(range(after + 1, 11))  # no duplicate, no gap


@pytest.mark.parametrize("after", ["10", "11", "999", str(10**30), "0010"])
def test_a_finished_run_with_nothing_after_last_event_id_is_a_204(api, finished, after):
    client, _ = api

    resp = client.get(path(finished), headers={"Last-Event-ID": after})

    assert resp.status_code == 204 and resp.content == b""


def test_a_reconnect_to_a_running_run_gets_exactly_what_it_missed(api, clients, conn, new_hold):
    client, _ = api
    run_id, hold = start_held_full(client, clients, new_hold)

    with Stream(app, path(run_id)) as before:  # the first connection sees two events and drops
        seen = [before.next_event(), before.next_event()]
    assert [e["id"] for e in seen] == ["1", "2"]

    with Stream(app, path(run_id), last_event_id=seen[-1]["id"]) as after:  # the browser reconnects with the last id
        assert after.status == 200  # the Run is running and has nothing new yet: a stream, not a 204
        with pytest.raises(TimeoutError):
            after.next_event(timeout=0.3)
        hold.release.set()
        missed = after.read_all(timeout=10)

    events = seen + missed
    assert [e["id"] for e in missed] == [str(n) for n in range(3, 11)]
    assert [e["id"] for e in events] == [str(n) for n in range(1, 11)]
    assert [e["data"] for e in events] == [e.data for e in store.run_events(conn, run_id)]


def test_a_reconnect_after_the_last_event_of_a_finished_run_is_a_204_so_the_browser_stops(api, finished):
    client, _ = api
    with Stream(app, path(finished), last_event_id=10) as stream:
        assert stream.status == 204 and stream.read_all() == []


def test_a_run_with_no_events_is_a_204(api, background, conn):
    """An old Run (stored before events existed) and a human Run have none."""
    client, use_replies = api
    use_replies(GOOD)
    post_translate(client, background)
    conn.execute("DELETE FROM run_events")  # as a Run from before run_events was
    conn.commit()
    human = runs.edit_translation(conn, "book_a", 1, 1, "再会")

    for run_id in (1, human):
        assert store.get_run(conn, run_id)["status"] == "succeeded"
        assert client.get(path(run_id)).status_code == 204
        assert client.get(path(run_id), headers={"Last-Event-ID": "0"}).status_code == 204


def test_a_running_run_with_no_events_yet_is_a_stream_not_a_204(api, conn):
    client, _ = api
    page_id = store.find_page(conn, "book_a", 1)["id"]
    run_id = store.start_run(conn, page_id=page_id, config="baseline", params={}, model="m")

    with Stream(app, path(run_id)) as stream:
        assert stream.status == 200
        store.add_run_event(conn, RunEvent(run_id, 1, "run_start", {"run_id": run_id}))
        assert stream.next_event(timeout=10)["event"] == "run_start"


@pytest.mark.parametrize("value", ["abc", "-1", "1.5", "", "0x3", "1e3", "3,4", "+3"])
def test_a_last_event_id_that_is_not_a_non_negative_integer_is_a_422(api, finished, value):
    client, _ = api

    resp = client.get(path(finished), headers={"Last-Event-ID": value})

    assert resp.status_code == 422 and "Last-Event-ID" in resp.json()["detail"]


def test_a_bad_last_event_id_is_a_422_for_an_unknown_run_too(api):
    client, _ = api

    assert client.get(path(999), headers={"Last-Event-ID": "x"}).status_code == 422


def test_the_header_parser_takes_ascii_digits_only_and_clamps_what_sqlite_cannot_hold():
    with pytest.raises(HTTPException) as refused:
        runs_api.parse_last_event_id("٣")  # Arabic-Indic digits are digits to str.isdigit, not to this
    assert refused.value.status_code == 422
    assert runs_api.parse_last_event_id(None) == 0
    assert runs_api.parse_last_event_id("0") == 0 and runs_api.parse_last_event_id("0042") == 42
    assert runs_api.parse_last_event_id(str(10**30)) == 2**63 - 1
    assert runs_api.parse_last_event_id("9" * 5000) == 2**63 - 1  # more digits than int() will convert
    assert runs_api.parse_last_event_id("0" * 5000 + "7") == 7  # leading zeros do not count
    assert runs_api.parse_last_event_id("0" * 5000) == 0


def test_a_last_event_id_of_thousands_of_digits_is_nothing_after_the_last_event_not_a_server_error(api, finished):
    client, _ = api

    assert client.get(path(finished), headers={"Last-Event-ID": "9" * 5000}).status_code == 204


# --- AC7: running_run_id ----------------------------------------------------------------------------------------------

def test_running_run_id_is_the_pages_running_run_while_it_runs_and_null_before_and_after(
        api, clients, conn, new_hold, background):
    client, _ = api
    clone_page(conn, "book_a", 3)
    assert client.get("/api/pages/book_a/1").json()["running_run_id"] is None

    run_id, hold = start_held_full(client, clients, new_hold)

    detail = client.get("/api/pages/book_a/1").json()
    assert detail["running_run_id"] == run_id and detail["run"] is None
    assert client.get("/api/pages/book_a/3").json()["running_run_id"] is None  # another page's Run is not this page's
    hold.release.set()
    assert background.wait_idle(10)
    after = client.get("/api/pages/book_a/1").json()
    assert after["running_run_id"] is None and after["run"]["id"] == run_id


def test_a_failed_run_is_not_running(api, background, conn, clients):
    client, _ = api
    clients.append(mock_llm_client(lambda request: httpx.Response(500, json={"error": {"message": "x"}})))
    run_id = post_translate(client, background).json()["run_id"]

    assert store.get_run(conn, run_id)["status"] == "failed"
    assert client.get("/api/pages/book_a/1").json()["running_run_id"] is None


def test_a_running_row_left_by_another_process_is_the_pages_running_run_and_the_latest_wins(api, conn):
    client, _ = api
    page_id = store.find_page(conn, "book_a", 1)["id"]
    first = store.start_run(conn, page_id=page_id, config="baseline", params={}, model="m")
    second = store.start_run(conn, page_id=page_id, config="full", params={}, model="m")

    assert first < second
    assert client.get("/api/pages/book_a/1").json()["running_run_id"] == second
    assert store.running_run_id(conn, page_id) == second
    assert store.running_run_id(conn, store.find_page(conn, "book_a", 2)["id"]) is None


def test_the_edit_api_answers_the_page_with_its_running_run_id_too(api, background, conn):
    client, use_replies = api
    use_replies(GOOD)
    post_translate(client, background)

    resp = client.put("/api/pages/book_a/1/segments/1/translation", json={"text_zh_hans": "再会"})

    assert resp.status_code == 200 and resp.json()["running_run_id"] is None
