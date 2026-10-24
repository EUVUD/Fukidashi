"""FastAPI dependencies. Tests replace these through app.dependency_overrides."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException

from .. import config
from ..db import connect
from ..llm.client import LLMClient, LLMConfig
from ..llm.metering import Meter


def get_db() -> Iterator[sqlite3.Connection]:
    conn = connect(config.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def get_dataset_root() -> Path:
    return config.DATASET_ROOT


def get_client_factory() -> Callable[[], LLMClient]:
    """A factory, not a client: each translate request gets its own Meter, so its usage is its own."""

    def make() -> LLMClient:
        load_dotenv(config.ROOT / ".env")
        try:
            llm_config = LLMConfig.from_env()
        except RuntimeError as e:  # names the missing variables, never their values
            raise HTTPException(status_code=503, detail=str(e)) from e
        return LLMClient(llm_config, Meter(ledger_path=config.LEDGER_PATH))

    return make
