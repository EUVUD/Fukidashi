"""FastAPI entry point.

  uv run uvicorn fukidashi.main:app --port 8000
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from . import config, store, tracing
from .api.pages import router as pages_router
from .api.runs import router as runs_router
from .db import connect, init_db

logger = logging.getLogger(__name__)

# What a Run left `running` by a stopped server is marked with when the API starts (OUTLINE §8, "Running pages from the editor").
STALE_RUN_ERROR = "The server stopped before this Run finished; it was marked failed when the server started."


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_dotenv(config.ROOT / ".env")
    tracing.setup()  # off unless FUKIDASHI_TRACING is set; raises, and so refuses to start, when it cannot be honoured
    conn = connect(config.DB_PATH)
    init_db(conn)  # an empty database is fine: the page list is just empty until the import script runs
    # No Run of this process exists yet, so a `running` one belongs to a process that stopped (or to an eval that is still
    # going: do not start the API while an eval runs).
    stale = store.fail_stale_runs(conn, STALE_RUN_ERROR)
    conn.close()
    if stale:
        logger.warning("marked %d Run(s) left running by a stopped server as failed", stale)
    else:
        logger.info("no Run was left running by a stopped server")
    yield
    tracing.shutdown()


app = FastAPI(title="Fukidashi", lifespan=lifespan)
app.include_router(pages_router)
app.include_router(runs_router)
