"""REST endpoints for pages: list, detail (segments + latest translation), image, translate."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..datasets.openmantra import Page, Segment
from ..llm.client import LLMClient
from ..llm.metering import PageBudget
from ..pipeline import baseline
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _get_page_row(conn: sqlite3.Connection, book_title: str, page_index: int) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM pages WHERE book_title = ? AND page_index = ?", (book_title, page_index)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no page {book_title}/{page_index}")
    return row


def _page_detail(conn: sqlite3.Connection, page_row: sqlite3.Row) -> PageDetail:
    run = conn.execute(
        "SELECT * FROM runs WHERE page_id = ? AND status = 'succeeded' ORDER BY id DESC LIMIT 1",
        (page_row["id"],),
    ).fetchone()
    segments = conn.execute(
        """SELECT s.*, t.text_zh_hans FROM segments s
           LEFT JOIN translations t ON t.segment_id = s.id AND t.run_id = ?
           WHERE s.page_id = ? ORDER BY s.idx""",
        (run["id"] if run else None, page_row["id"]),
    ).fetchall()
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
    rows = conn.execute(
        """SELECT p.book_title, p.page_index,
                  (SELECT COUNT(*) FROM segments s WHERE s.page_id = p.id) AS segment_count,
                  EXISTS (SELECT 1 FROM runs r WHERE r.page_id = p.id AND r.status = 'succeeded') AS translated
           FROM pages p ORDER BY p.id"""
    ).fetchall()
    return [PageSummary(**dict(r) | {"translated": bool(r["translated"])}) for r in rows]


@router.get("/pages/{book_title}/{page_index}", response_model=PageDetail)
def get_page_detail(book_title: str, page_index: int, conn: sqlite3.Connection = Depends(get_db)) -> PageDetail:
    return _page_detail(conn, _get_page_row(conn, book_title, page_index))


@router.get("/pages/{book_title}/{page_index}/image")
def get_page_image(
    book_title: str,
    page_index: int,
    conn: sqlite3.Connection = Depends(get_db),
    dataset_root: Path = Depends(get_dataset_root),
) -> FileResponse:
    path = dataset_root / _get_page_row(conn, book_title, page_index)["image_path"]
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
    page_row = _get_page_row(conn, book_title, page_index)
    seg_rows = conn.execute("SELECT * FROM segments WHERE page_id = ? ORDER BY idx", (page_row["id"],)).fetchall()
    if not seg_rows:
        raise HTTPException(status_code=422, detail="this page has no text regions to translate")
    image_path = dataset_root / page_row["image_path"]
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail=f"image file missing: {image_path}")
    segment_ids = {s["idx"]: s["id"] for s in seg_rows}
    page = Page(
        book_title, page_index, image_path,
        [Segment(s["idx"], s["x"], s["y"], s["w"], s["h"], s["text_ja"], s["text_en"], s["text_zh_hant"])
         for s in seg_rows],
    )

    client = make_client()  # fails with 503 before any run row exists when .env is incomplete
    params = {"page_budget": baseline.DEFAULT_PAGE_BUDGET, "max_tokens": baseline.DEFAULT_MAX_TOKENS}
    with conn:
        run_id = conn.execute(
            """INSERT INTO runs (page_id, config, config_json, model, status, started_at)
               VALUES (?, ?, ?, ?, 'running', ?)""",
            (page_row["id"], baseline.CONFIG_NAME, json.dumps(params), client.config.model, _now()),
        ).lastrowid  # committed here, so the LLM call below holds no write lock

    budget = PageBudget(page.page_id, params["page_budget"])
    try:
        result = baseline.run_baseline(page, client, budget, max_tokens=params["max_tokens"])
    except Exception as e:  # unexpected: don't leave the run 'running'
        _finish_run(conn, run_id, client, None, f"{type(e).__name__}: {e}", [])
        raise
    by_segment = None if result.translations is None else {segment_ids[i]: t for i, t in result.translations.items()}
    _finish_run(conn, run_id, client, by_segment, result.error, result.attempts)

    if not result.ok:
        raise HTTPException(status_code=502, detail=result.error)
    return _page_detail(conn, page_row)


def _finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    client: LLMClient,
    translations: dict[int, str] | None,  # segment row id -> zh-Hans
    error: str | None,
    attempts: list[dict],
) -> None:
    usage = client.meter.summary()
    # tokens are summed over calls with known usage; with none known they stay NULL rather than 0
    known = usage["calls"] - usage["unknown_usage_calls"]
    response_models = sorted({r.response_model for r in client.meter.records if r.response_model})
    with conn:
        conn.execute(
            """UPDATE runs SET status = ?, error = ?, response_model = ?, calls = ?, unknown_usage_calls = ?,
                      input_tokens = ?, output_tokens = ?, cost_usd = ?, attempts_json = ?, finished_at = ?
               WHERE id = ?""",
            ("succeeded" if translations is not None else "failed", error,
             ", ".join(response_models) or None, usage["calls"], usage["unknown_usage_calls"],
             usage["input_tokens"] if known else None, usage["output_tokens"] if known else None,
             usage["cost_usd"], json.dumps(attempts, ensure_ascii=False), _now(), run_id),
        )
        conn.executemany(
            "INSERT INTO translations (run_id, segment_id, text_zh_hans) VALUES (?, ?, ?)",
            [(run_id, seg_id, text) for seg_id, text in (translations or {}).items()],
        )
