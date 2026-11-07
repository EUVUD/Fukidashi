"""FastAPI dependencies. Tests replace these through app.dependency_overrides."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends

from .. import config
from ..background import BackgroundRuns
from ..db import connect
from ..llm.client import LLMClient, LLMConfig
from ..llm.metering import Meter

# The SDK's request timeout for a Run started from the editor (OUTLINE §8, "Running pages from the editor"): a request the
# provider never answers fails the Run after this long. The eval harness and the scripts set none.
API_REQUEST_TIMEOUT_S = 180.0


def get_db() -> Iterator[sqlite3.Connection]:
    conn = connect(config.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def get_db_path() -> Path:
    """Where the database is. A background Run (and a stream of its events) opens a connection of its own from this."""
    return config.DB_PATH


def get_dataset_root() -> Path:
    return config.DATASET_ROOT


def api_request_timeout() -> float:
    return API_REQUEST_TIMEOUT_S


def get_client_factory(timeout: Annotated[float | None, Depends(api_request_timeout)] = None) -> Callable[[], LLMClient]:
    """A factory, not a client: each translate request gets its own Meter, so its usage is its own.

    As a dependency of the API it gets `API_REQUEST_TIMEOUT_S`. Called directly, as `eval/run.py` does, `timeout` is its
    plain default, None: the eval's clients have no timeout of their own, and its requests are what they were."""

    def make() -> LLMClient:
        load_dotenv(config.ROOT / ".env")
        return LLMClient(LLMConfig.from_env(), Meter(ledger_path=config.LEDGER_PATH), timeout=timeout)  # may raise LLMNotConfigured

    return make


# One registry for the process: it knows which pages have a Run in flight. Tests replace the dependency with a fresh one.
_BACKGROUND_RUNS = BackgroundRuns()


def get_background_runs() -> BackgroundRuns:
    return _BACKGROUND_RUNS
