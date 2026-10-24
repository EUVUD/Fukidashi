"""Translate one page and record the Run. Knows nothing about HTTP.

The API calls this today; the eval harness calls the same function from M2 (OUTLINE 1.3, Same code path).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import store
from .datasets.openmantra import Page, Segment
from .llm.client import LLMClient
from .llm.metering import PageBudget
from .pipeline import baseline
from .pipeline.baseline import BaselineResult


class TranslateError(Exception):
    """The page cannot be translated; no Run was recorded."""


class PageNotFound(TranslateError):
    pass


class NothingToTranslate(TranslateError):
    pass


class ImageMissing(TranslateError):
    pass


@dataclass(frozen=True)
class Outcome:
    run_id: int
    result: BaselineResult  # ok, error and attempts of the baseline call


def translate_page(
    conn: sqlite3.Connection,
    book_title: str,
    page_index: int,
    *,
    make_client: Callable[[], LLMClient],
    dataset_root: Path,
    page_budget: int = baseline.DEFAULT_PAGE_BUDGET,
    max_tokens: int = baseline.DEFAULT_MAX_TOKENS,
) -> Outcome:
    """Run `baseline` on one page and store the Run (and its Translations when it succeeds).

    `make_client` is a factory so that every Run gets its own client and Meter, and its usage is its own.
    It is called after the page is validated and before the Run row exists, so a client that cannot be
    built (missing .env) leaves no Run behind. A failed model call or unparseable output is not an
    exception: it is a failed Run, reported in `Outcome.result`.
    """
    page_row = store.find_page(conn, book_title, page_index)
    if page_row is None:
        raise PageNotFound(f"no page {book_title}/{page_index}")
    seg_rows = store.page_segments(conn, page_row["id"])
    if not seg_rows:
        raise NothingToTranslate("this page has no text regions to translate")
    image_path = dataset_root / page_row["image_path"]
    if not image_path.is_file():
        raise ImageMissing(f"image file missing: {image_path}")
    segment_ids = {s["idx"]: s["id"] for s in seg_rows}
    page = Page(
        book_title, page_index, image_path,
        [Segment(s["idx"], s["x"], s["y"], s["w"], s["h"], s["text_ja"], s["text_en"], s["text_zh_hant"])
         for s in seg_rows],
    )

    client = make_client()
    run_id = store.start_run(
        conn, page_id=page_row["id"], config=baseline.CONFIG_NAME, model=client.config.model,
        params={"page_budget": page_budget, "max_tokens": max_tokens},
    )

    def finish(translations: dict[int, str] | None, error: str | None, attempts: list[dict]) -> None:
        store.finish_run(
            conn, run_id, error=error, attempts=attempts, translations=translations,
            usage=client.meter.summary(),
            response_models=sorted({r.response_model for r in client.meter.records if r.response_model}),
        )

    try:
        result = baseline.run_baseline(
            page, client, PageBudget(page.page_id, page_budget), max_tokens=max_tokens
        )
    except Exception as e:  # unexpected: don't leave the run 'running'
        finish(None, f"{type(e).__name__}: {e}", [])
        raise
    by_segment = None if result.translations is None else {segment_ids[i]: t for i, t in result.translations.items()}
    finish(by_segment, result.error, result.attempts)
    return Outcome(run_id, result)
