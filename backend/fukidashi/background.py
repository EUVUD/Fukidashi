"""Runs started from the editor, in the background (OUTLINE §8, "Running pages from the editor"). Knows nothing about HTTP.

`BackgroundRuns.start` begins `runs.translate_page` in a thread of its own and returns the Run's id as soon as the Run's
`run_start` event arrives, so a caller waits for the Run to exist, never for the model. Each Run has its own thread, its
own database connection and the sink `runs.store_events(that connection)`: the Run's events are in `run_events` while it
runs, for whoever reads them (the API's stream), and the connection is closed when the Run ends, whatever happened.

One Run per page at a time: a page is busy while a Run started here has not ended, or while the database has a `running`
Run of it. The check and the start are one step under a lock, so of two starts at once one gets the Run and the other a
`RunInProgress`.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

from . import runs, store
from .db import connect
from .live_trace import RUN_END, RUN_START, RunEvent
from .llm.client import LLMClient
from .pipeline import baseline

logger = logging.getLogger(__name__)

# The configs the editor can run (OUTLINE §8). `translator` and `full_no_kb` are the eval's; the editor has these two.
EDITOR_CONFIGS = (baseline.CONFIG_NAME, runs.FULL)

# How long `start` waits for the Run's first event. Before it only local work happens (validate, read, build the client,
# insert the row), never a model call, so this is a safety net, not a budget.
START_TIMEOUT_S = 60.0


class RunInProgress(RuntimeError):
    """The page already has a Run that has not ended; nothing was started and nothing was recorded."""


class BackgroundRuns:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: dict[tuple[str, int], object] = {}  # page -> the token of the Run in flight
        self._threads: list[threading.Thread] = []

    def start(
        self,
        conn: sqlite3.Connection,
        book_title: str,
        page_index: int,
        *,
        config: str,
        make_client: Callable[[], LLMClient],
        dataset_root: Path,
        db_path: Path,
    ) -> int:
        """Start `config` (`baseline` or `full`) on the page in a background thread and return the Run's id once its
        `run_start` event has arrived.

        `conn` is the caller's, used only to ask whether the page has a `running` Run; the thread opens its own from
        `db_path`. Raises `RunInProgress` (nothing started) when the page is busy. A page `translate_page` refuses before
        its Run exists raises what it raises (`PageNotFound`, `NothingToTranslate`, `ImageMissing`, `LLMNotConfigured`),
        in the caller's thread, with no Run recorded. A Run that fails or crashes later is not raised: it is reported by
        its `run_end` event and by the page. The caller waits for the Run's start, never for the model."""
        if config not in EDITOR_CONFIGS:
            raise ValueError(f"the editor runs {EDITOR_CONFIGS}, not {config!r}")
        key = (book_title, page_index)
        token = object()
        started: Future[int] = Future()
        thread = threading.Thread(
            target=self._run, name=f"run {book_title}/{page_index}", daemon=True,  # a stopped server loses its Runs
            args=(key, token, started, config, make_client, dataset_root, db_path),
        )
        with self._lock:
            if key in self._active:
                raise RunInProgress(f"{book_title}/{page_index} is already being translated")
            page_row = store.find_page(conn, book_title, page_index)
            if page_row is not None and store.running_run_id(conn, page_row["id"]) is not None:
                raise RunInProgress(f"{book_title}/{page_index} is already being translated")
            self._active[key] = token
            self._threads = [t for t in self._threads if t.is_alive()]
            self._threads.append(thread)
        try:
            thread.start()
        except BaseException:
            self._release(key, token)
            raise
        try:
            return started.result(timeout=START_TIMEOUT_S)
        except FutureTimeout as e:
            raise RuntimeError(f"the Run of {book_title}/{page_index} did not start within {START_TIMEOUT_S:.0f} s") from e

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Wait until every thread started here has ended; False if some is still alive after `timeout` seconds. For
        tests and for a shutdown that wants to see its Runs end."""
        with self._lock:
            threads = list(self._threads)
        for thread in threads:
            thread.join(timeout)
        return not any(t.is_alive() for t in threads)

    def _release(self, key: tuple[str, int], token: object) -> None:
        """The page is free again. Only the Run that took it releases it: a late release by an ended Run must not free
        the page for a newer one."""
        with self._lock:
            if self._active.get(key) is token:
                del self._active[key]

    def _run(
        self, key: tuple[str, int], token: object, started: Future[int], config: str,
        make_client: Callable[[], LLMClient], dataset_root: Path, db_path: Path,
    ) -> None:
        book_title, page_index = key
        conn = None
        try:
            conn = connect(db_path)  # this Run's own: `add_run_event` raises inside another write's transaction
            store_event = runs.store_events(conn)

            def sink(event: RunEvent) -> None:
                if event.kind == RUN_START and not started.done():
                    started.set_result(event.run_id)  # before storing: the caller learns the id even if storing raises
                if event.kind == RUN_END:
                    # The row is finished by now. Freed before the event is stored, so whoever sees `run_end` can start
                    # the page again at once.
                    self._release(key, token)
                store_event(event)

            kb_id = None
            if config == runs.FULL:
                runs.validate_page(conn, book_title, page_index, dataset_root)  # a refused page makes no KB
                kb_id = store.editor_kb(conn, book_title)
            runs.translate_page(
                conn, book_title, page_index, make_client=make_client, dataset_root=dataset_root, config=config,
                kb_id=kb_id, on_event=sink,
            )
        except BaseException as e:
            if not started.done():
                started.set_exception(e)  # refused before its Run existed: the caller maps it
            else:
                logger.warning("the Run of %s/%d ended with %s: %s", book_title, page_index, type(e).__name__, e)
        finally:
            try:
                if conn is not None:
                    conn.close()
            finally:
                self._release(key, token)
                if not started.done():
                    started.set_exception(RuntimeError(f"the Run of {book_title}/{page_index} ended without starting"))
