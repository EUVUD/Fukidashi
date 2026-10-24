"""REST endpoints for pages: list, detail (segments + latest translation), image, translate.

This layer parses requests, maps errors to status codes and shapes responses. The work itself lives in
fukidashi.runs (translate and record a Run) and fukidashi.store (SQL).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .. import runs, store
from ..llm.client import LLMClient, LLMNotConfigured
from .deps import get_client_factory, get_dataset_root, get_db

router = APIRouter(prefix="/api")


class PageSummary(BaseModel):
    book_title: str
    page_index: int
    segment_count: int
    translated: bool  # has a succeeded run


class SegmentOut(BaseModel):
    index: int
    x: int
    y: int
    w: int
    h: int
    text_ja: str
    text_zh_hant: str  # reference
    text_en: str  # reference
    text_zh_hans: str | None  # from the latest succeeded run


class RunOut(BaseModel):
    id: int
    config: str
    model: str
    response_model: str | None
    status: str
    calls: int
    unknown_usage_calls: int
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    started_at: str
    finished_at: str | None


class PageDetail(BaseModel):
    book_title: str
    page_index: int
    image_url: str
    segments: list[SegmentOut]
    run: RunOut | None  # latest succeeded run; None until the page has been translated


def _page_or_404(conn: sqlite3.Connection, book_title: str, page_index: int) -> sqlite3.Row:
    row = store.find_page(conn, book_title, page_index)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no page {book_title}/{page_index}")
    return row


def _page_detail(conn: sqlite3.Connection, page_row: sqlite3.Row) -> PageDetail:
    run = store.latest_succeeded_run(conn, page_row["id"])
    segments = store.segments_with_translations(conn, page_row["id"], run["id"] if run else None)
    book, index = page_row["book_title"], page_row["page_index"]
    return PageDetail(
        book_title=book,
        page_index=index,
        image_url=f"/api/pages/{book}/{index}/image",
        segments=[
            SegmentOut(index=s["idx"], x=s["x"], y=s["y"], w=s["w"], h=s["h"], text_ja=s["text_ja"],
                       text_zh_hant=s["text_zh_hant"], text_en=s["text_en"], text_zh_hans=s["text_zh_hans"])
            for s in segments
        ],
        run=RunOut(**{k: run[k] for k in RunOut.model_fields}) if run else None,
    )


@router.get("/pages", response_model=list[PageSummary])
def list_pages(conn: sqlite3.Connection = Depends(get_db)) -> list[PageSummary]:
    return [PageSummary(**dict(r) | {"translated": bool(r["translated"])}) for r in store.list_page_summaries(conn)]


@router.get("/pages/{book_title}/{page_index}", response_model=PageDetail)
def get_page_detail(book_title: str, page_index: int, conn: sqlite3.Connection = Depends(get_db)) -> PageDetail:
    return _page_detail(conn, _page_or_404(conn, book_title, page_index))


@router.get("/pages/{book_title}/{page_index}/image")
def get_page_image(
    book_title: str,
    page_index: int,
    conn: sqlite3.Connection = Depends(get_db),
    dataset_root: Path = Depends(get_dataset_root),
) -> FileResponse:
    path = dataset_root / _page_or_404(conn, book_title, page_index)["image_path"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"image file missing: {path}")
    return FileResponse(path)


@router.post("/pages/{book_title}/{page_index}/translate", response_model=PageDetail)
def translate_page(
    book_title: str,
    page_index: int,
    conn: sqlite3.Connection = Depends(get_db),
    dataset_root: Path = Depends(get_dataset_root),
    make_client: Callable[[], LLMClient] = Depends(get_client_factory),
) -> PageDetail:
    try:
        outcome = runs.translate_page(
            conn, book_title, page_index, make_client=make_client, dataset_root=dataset_root
        )
    except runs.PageNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except runs.ImageMissing as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except runs.NothingToTranslate as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except LLMNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    if not outcome.result.ok:
        raise HTTPException(status_code=502, detail=outcome.result.error)
    return _page_detail(conn, _page_or_404(conn, book_title, page_index))
