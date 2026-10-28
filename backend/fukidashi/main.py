"""FastAPI entry point.

  uv run uvicorn fukidashi.main:app --port 8000
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from . import config, tracing
from .api.pages import router as pages_router
from .db import connect, init_db


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_dotenv(config.ROOT / ".env")
    tracing.setup()  # off unless FUKIDASHI_TRACING is set; raises, and so refuses to start, when it cannot be honoured
    conn = connect(config.DB_PATH)
    init_db(conn)  # an empty database is fine: the page list is just empty until the import script runs
    conn.close()
    yield
    tracing.shutdown()


app = FastAPI(title="Fukidashi", lifespan=lifespan)
app.include_router(pages_router)
