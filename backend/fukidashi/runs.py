"""Translate one page and record the Run. Knows nothing about HTTP.

The API calls this with `baseline`; the eval harness calls the same function with any config (OUTLINE 1.3, Same code
path).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import store
from .agents import translator
from .agents.loop import DEFAULT_MAX_STEPS
from .agents.translator import TranslatorResult
from .datasets.openmantra import Page, Segment
from .llm.client import LLMClient
from .llm.metering import PageBudget
from .pipeline import baseline
from .pipeline.baseline import BaselineResult

CONFIGS = (baseline.CONFIG_NAME, translator.CONFIG_NAME)


class TranslateError(Exception):
    """The page cannot be translated; no Run was recorded."""


class PageNotFound(TranslateError):
    pass


class NothingToTranslate(TranslateError):
    pass


class ImageMissing(TranslateError):
    pass


class RunCrashed(RuntimeError):
    """An unexpected exception after the Run row existed. The Run is marked failed; `run_id` says which.

    Not a `TranslateError`: those mean no Run was recorded. The message is the error stored on the Run, and
    the original exception is the `__cause__`. When the write that marks the Run failed raised too, the Run
    stays `running`, nothing was stored, and the message says so.
    """

    def __init__(self, run_id: int, error: str):
        super().__init__(error)
        self.run_id = run_id
        self.error = error


@dataclass(frozen=True)
class Outcome:
    run_id: int
    result: BaselineResult | TranslatorResult  # ok, error and attempts of the config's run


@dataclass(frozen=True)
class ValidPage:
    row: sqlite3.Row
    segment_rows: list[sqlite3.Row]
    image_path: Path


def validate_page(conn: sqlite3.Connection, book_title: str, page_index: int, dataset_root: Path) -> ValidPage:
    """Check that the page can be translated, without a model call or a Run. Raises a `TranslateError`.

    `translate_page` starts with this; the eval harness calls it for every page before its first model call.
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
    return ValidPage(page_row, seg_rows, image_path)


def run_params(
    config: str,
    *,
    page_budget: int,
    max_tokens: int,
    agent_budget: int = translator.DEFAULT_PAGE_BUDGET,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict:
    """The parameters a Run records. `page_budget` is the token budget of the `baseline` call: the page's own for
    `baseline`, the fallback's for `translator`, which also has the agent's budget and its Step limit."""
    if config == translator.CONFIG_NAME:
        return {"agent_budget": agent_budget, "fallback_budget": page_budget, "max_steps": max_steps,
                "max_tokens": max_tokens}
    return {"page_budget": page_budget, "max_tokens": max_tokens}


def previous_pages(
    conn: sqlite3.Connection, book_title: str, page_index: int, config: str
) -> list[translator.PreviousPage]:
    """The pages before this one (up to `translator.MAX_PAGES_BACK`), oldest first: each region's Japanese text and
    its zh-Hans from the page's latest succeeded Run of `config`, if there is one.

    Read before a `translator` Run starts: the tool handlers run in worker threads and only return this. The
    reference translations are not selected.
    """
    pages = []
    for page_row in store.earlier_pages(conn, book_title, page_index, translator.MAX_PAGES_BACK):
        run = store.latest_succeeded_run(conn, page_row["id"], config)
        rows = store.segments_with_translations(conn, page_row["id"], run["id"] if run else None)
        pages.append(translator.PreviousPage(
            page_row["page_index"],
            tuple(translator.PreviousRegion(r["idx"], r["text_ja"], r["text_zh_hans"]) for r in rows),
            translated=run is not None,
        ))
    return pages


def translate_page(
    conn: sqlite3.Connection,
    book_title: str,
    page_index: int,
    *,
    make_client: Callable[[], LLMClient],
    dataset_root: Path,
    config: str = baseline.CONFIG_NAME,
    page_budget: int = baseline.DEFAULT_PAGE_BUDGET,
    max_tokens: int = baseline.DEFAULT_MAX_TOKENS,
    agent_budget: int = translator.DEFAULT_PAGE_BUDGET,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> Outcome:
    """Run `config` (`baseline` or `translator`) on one page and store the Run (and its Translations when it succeeds).

    `agent_budget` and `max_steps` are the `translator`'s; `page_budget` is `baseline`'s, which is the fallback's
    budget for the `translator`. A `translator` Run that fell back to `baseline` is still a `translator` Run: its
    `attempts_json` says so. An unknown `config` is a `ValueError`, raised before anything else happens.

    `make_client` is a factory so that every Run gets its own client and Meter, and its usage is its own.
    It is called after the page is validated and before the Run row exists, so a client that cannot be
    built (missing .env) leaves no Run behind. A failed model call or unparseable output is not an
    exception: it is a failed Run, reported in `Outcome.result`. Once the Run row exists, any other exception
    marks that Run failed and is raised as `RunCrashed`. The one case that leaves a Run `running` is a
    database that fails the write marking it failed too: `RunCrashed` is still raised, with the `run_id`
    and a message that says the Run was not marked.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {CONFIGS}")
    valid = validate_page(conn, book_title, page_index, dataset_root)
    page_row = valid.row
    segment_ids = {s["idx"]: s["id"] for s in valid.segment_rows}
    page = Page(
        book_title, page_index, valid.image_path,
        [Segment(s["idx"], s["x"], s["y"], s["w"], s["h"], s["text_ja"], s["text_en"], s["text_zh_hant"])
         for s in valid.segment_rows],
    )

    previous = previous_pages(conn, book_title, page_index, config) if config == translator.CONFIG_NAME else []
    client = make_client()
    run_id = store.start_run(
        conn, page_id=page_row["id"], config=config, model=client.config.model,
        params=run_params(config, page_budget=page_budget, max_tokens=max_tokens, agent_budget=agent_budget,
                          max_steps=max_steps),
    )

    def finish(translations: dict[int, str] | None, error: str | None, attempts: list[dict] | dict) -> None:
        store.finish_run(
            conn, run_id, error=error, attempts=attempts, translations=translations,
            usage=client.meter.summary(),
            response_models=sorted({r.response_model for r in client.meter.records if r.response_model}),
        )

    try:
        if config == translator.CONFIG_NAME:
            result = translator.run_translator(
                page, client, previous=previous, agent_budget=agent_budget, fallback_budget=page_budget,
                max_tokens=max_tokens, max_steps=max_steps,
            )
        else:
            result = baseline.run_baseline(
                page, client, PageBudget(page.page_id, page_budget), max_tokens=max_tokens
            )
        by_segment = (None if result.translations is None
                      else {segment_ids[i]: t for i, t in result.translations.items()})
        finish(by_segment, result.error, result.attempts)
    except Exception as e:  # unexpected: don't leave the run 'running'
        error = f"{type(e).__name__}: {e}"
        try:
            finish(None, error, [])
        except Exception as recovery:  # the database is failing: the Run stays 'running', the caller still gets its id
            raise RunCrashed(
                run_id, f"{error} (marking the Run failed also failed: {type(recovery).__name__}: {recovery})"
            ) from e
        raise RunCrashed(run_id, error) from e
    return Outcome(run_id, result)
