"""The live trace over SSE: `GET /api/runs/{run_id}/events` (OUTLINE §8, "Running pages from the editor").

The events are the rows of `run_events`, which the Run's thread stores as it goes. A stream sends the ones after the
client's `Last-Event-ID` (each as `id = seq`, `event = kind`, `data = data`), then polls the table for new ones while the
Run is `running`; the browser's EventSource sends the last `id` it saw when it reconnects, so a reconnect or a reload
replays exactly what was missed.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent

from .. import store
from ..db import connect
from ..live_trace import RUN_END, RunEvent
from .deps import get_db, get_db_path

router = APIRouter(prefix="/api")

# How often a stream looks for new events while the Run is running. A module constant so that a test can shorten it.
POLL_INTERVAL_S = 0.25

# SQLite's integers are 64-bit: an id beyond it has no event after it.
_MAX_SEQ = 2**63 - 1


def parse_last_event_id(value: str | None) -> int:
    """The `seq` a client saw last: 0 when the header is missing, else a non-negative integer (a 422 for anything else)."""
    if value is None:
        return 0
    if re.fullmatch(r"[0-9]+", value) is None:
        raise HTTPException(status_code=422, detail="Last-Event-ID must be a non-negative integer (the seq of the last event seen)")
    digits = value.lstrip("0") or "0"
    return _MAX_SEQ if len(digits) > len(str(_MAX_SEQ)) else min(int(digits), _MAX_SEQ)  # int() refuses over 4300 digits


def _read(conn: sqlite3.Connection, run_id: int, after_seq: int) -> tuple[bool, list[RunEvent]]:
    """Whether the Run is `running`, and its events after `after_seq`. The status is read first: when the Run was no longer
    running, every event stored before its row was finished is among the events read next. Only `run_end` can still be
    missing, because `translate_page` stores it after it has finished the row."""
    run = store.get_run(conn, run_id)
    running = run is not None and run["status"] == "running"
    return running, store.run_events(conn, run_id, after_seq)


async def follow_run_events(db_path: Path, run_id: int, after_seq: int) -> AsyncIterator[RunEvent]:
    """The Run's events after `after_seq`, then new ones as they are stored. Ends once the Run is no longer `running` and
    every stored event is sent; it does not wait for a `run_end`, which may be missing: when the last event sent is not
    one, it looks once more, one poll later, because `run_end` is stored just after the row is finished, and then ends.
    Reads through a connection of its own that lives as long as the stream. The reads are short (a primary-key range in a
    WAL database, which a writer does not block) and run in the event loop's thread."""
    conn = connect(db_path)
    try:
        sent = after_seq
        last_kind = None
        looked_again = False
        while True:
            running, events = _read(conn, run_id, sent)
            for event in events:
                yield event
                sent, last_kind = event.seq, event.kind
            if not running:
                if last_kind == RUN_END or looked_again:
                    return
                looked_again = True
            await asyncio.sleep(POLL_INTERVAL_S)
    finally:
        conn.close()


def requested_events(
    run_id: int, last_event_id: str | None = Header(default=None), conn: sqlite3.Connection = Depends(get_db)
) -> int:
    """What the stream is asked for, or why it is not one: the `seq` to start after. 422 for a `Last-Event-ID` that is not
    a non-negative integer, 404 for an unknown Run, and 204 (no body) for a Run that is not running and has no event
    after it, which is what makes the browser's EventSource stop reconnecting. Decided here, before the response is a
    stream: a status cannot change once it has begun. The 204 is raised, because a path operation that yields cannot
    answer anything but a stream; `HTTPException` is what FastAPI answers with an empty 204."""
    after_seq = parse_last_event_id(last_event_id)
    if run_id > _MAX_SEQ or store.get_run(conn, run_id) is None:  # an id SQLite cannot hold is an unknown Run, not a 500
        raise HTTPException(status_code=404, detail=f"no run {run_id}")
    running, events = _read(conn, run_id, after_seq)
    if not running and not events:
        raise HTTPException(status_code=204)
    return after_seq


@router.get("/runs/{run_id}/events", response_class=EventSourceResponse)
async def get_run_events(
    run_id: int, after_seq: int = Depends(requested_events), db_path: Path = Depends(get_db_path)
) -> AsyncIterator[ServerSentEvent]:
    """The Run's live trace as server-sent events, each with `id = seq`, `event = kind` and `data = data`. A comment line
    now and then keeps an idle connection open (FastAPI's)."""
    async for event in follow_run_events(db_path, run_id, after_seq):
        yield ServerSentEvent(raw_data=json.dumps(event.data, ensure_ascii=False), event=event.kind, id=str(event.seq))
