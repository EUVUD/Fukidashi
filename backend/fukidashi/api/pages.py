"""REST endpoints for pages: list, detail (segments + latest translation + review), image, translate, edit a translation.

This layer parses requests, maps errors to status codes and shapes responses. The work itself lives in
fukidashi.runs (translate or edit, and record a Run), fukidashi.background (a translate runs in the background),
fukidashi.review_notes (the review of a page, read from its Run's record) and fukidashi.store (SQL).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .. import runs, store
from ..background import BackgroundRuns, RunInProgress
from ..llm.client import LLMClient, LLMNotConfigured
from ..review_notes import build_review
from .deps import get_background_runs, get_client_factory, get_dataset_root, get_db, get_db_path

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


class ReviewFlag(BaseModel):
    """A Translator's Flag (OUTLINE §0) of the review's Run: a region it was unsure of, and why."""

    region: int
    reason: str  # Simplified Chinese
    pass_: int = Field(alias="pass")  # 0: the first pass; n: the nth revision (`pass` is a Python keyword, so an alias)
    current: bool  # made by the stage that produced the displayed translation, and the region is not edited


class ReviewIssue(BaseModel):
    """A Reviewer's Issue (OUTLINE §0) of the review's Run: its remark on one region."""

    region: int
    comment: str  # Simplified Chinese
    suggestion: str | None
    verdict: str | None  # the Verdict of the review that gave it: "approve" or "revise" (None: that review gave none)
    review: int  # 1 for the first review, 2 for the second
    current: bool  # from the review that judged the displayed translation, and the region is not edited


class PageReview(BaseModel):
    """What the Orchestrator's stages say about the displayed translation (`fukidashi.review_notes`)."""

    run_id: int  # the review's Run: the `full` or `full_no_kb` Run whose stages produced the translation
    config: str
    send_backs: int
    final_stop: str  # "approved", "review_unusable", "revision_unusable" or "send_back_limit"
    edited: list[int]  # the regions a person edited since that Run, sorted
    flags: list[ReviewFlag]
    issues: list[ReviewIssue]


class PageDetail(BaseModel):
    book_title: str
    page_index: int
    image_url: str
    segments: list[SegmentOut]
    run: RunOut | None  # latest succeeded run; None until the page has been translated
    running_run_id: int | None  # the id of the page's `running` Run, or None: a reload finds the Run to watch
    review: PageReview | None  # None when the displayed translation has no review: `baseline`, `translator`, no Run


class TranslateRequest(BaseModel):
    """The optional body of Translate. The editor runs `baseline` or `full` (OUTLINE §8)."""

    model_config = ConfigDict(extra="forbid")  # a misspelt key is a 422, not a quiet `baseline`
    config: Literal["baseline", "full"] = "baseline"


class RunStarted(BaseModel):
    run_id: int


class TranslationEdit(BaseModel):
    text_zh_hans: str  # stored exactly as sent; empty or only whitespace is refused


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
        running_run_id=store.running_run_id(conn, page_row["id"]),
        review=build_review(run, lambda run_id: store.get_run(conn, run_id)),
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


@router.post("/pages/{book_title}/{page_index}/translate", response_model=RunStarted, status_code=202)
def translate_page(
    book_title: str,
    page_index: int,
    body: TranslateRequest | None = None,
    conn: sqlite3.Connection = Depends(get_db),
    dataset_root: Path = Depends(get_dataset_root),
    make_client: Callable[[], LLMClient] = Depends(get_client_factory),
    background: BackgroundRuns = Depends(get_background_runs),
    db_path: Path = Depends(get_db_path),
) -> RunStarted:
    """Start the page in the background and answer once its Run exists. The Run's end, and a failure, are reported by its
    events (`GET /api/runs/{run_id}/events`) and by the page, not by this response."""
    try:
        run_id = background.start(
            conn, book_title, page_index, config=(body or TranslateRequest()).config, make_client=make_client,
            dataset_root=dataset_root, db_path=db_path,
        )
    except RunInProgress as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except runs.PageNotFound as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except runs.ImageMissing as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except runs.NothingToTranslate as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except LLMNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return RunStarted(run_id=run_id)


@router.put("/pages/{book_title}/{page_index}/segments/{idx}/translation", response_model=PageDetail)
def edit_translation(
    book_title: str,
    page_index: int,
    idx: int,
    edit: TranslationEdit,
    conn: sqlite3.Connection = Depends(get_db),
) -> PageDetail:
    try:
        runs.edit_translation(conn, book_title, page_index, idx, edit.text_zh_hans)
    except (runs.PageNotFound, runs.SegmentNotFound) as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except runs.NothingToEdit as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except runs.EmptyTranslation as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _page_detail(conn, _page_or_404(conn, book_title, page_index))
