"""FastAPI entry point.

  uv run uvicorn app.main:app --app-dir backend --port 8000
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import config
from .api.pages import router as pages_router
from .db import connect, init_db


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    conn = connect(config.DB_PATH)
    init_db(conn)  # an empty database is fine: the page list is just empty until the import script runs
    conn.close()
    yield


app = FastAPI(title="Fukidashi", lifespan=lifespan)
app.include_router(pages_router)
