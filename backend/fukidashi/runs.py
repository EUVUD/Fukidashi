"""Translate one page and record the Run. Knows nothing about HTTP.

The API calls this with `baseline`; the eval harness calls the same function with any config (OUTLINE 1.3, Same code
path).
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from . import orchestrator, store, tracing
from .agents import curator as curator_agent
from .agents import reviewer, translator
from .agents.loop import DEFAULT_MAX_STEPS
from .agents.translator import TranslatorResult
from .datasets.openmantra import Page, Segment
from .kb import curator as curator_rules
from .kb.entities import KEY_FIELDS
from .kb.loader import load_snapshot
from .kb.snapshot import KBSnapshot
from .llm.client import LLMClient
from .llm.metering import PageBudget
from .orchestrator import OrchestratorResult
from .pipeline import baseline
from .pipeline.baseline import BaselineResult

FULL = orchestrator.KB_CONFIG_NAME  # `full_no_kb` with the Knowledge Base and the Curator (OUTLINE §4.2)
CONFIGS = (baseline.CONFIG_NAME, translator.CONFIG_NAME, orchestrator.CONFIG_NAME, FULL)
# the configs whose Translator has earlier pages, each read from the Runs of its own config
AGENT_CONFIGS = (translator.CONFIG_NAME, orchestrator.CONFIG_NAME, FULL)


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
    stays `running`, nothing was stored, and the message says so. `trace_id` is the page's trace, None when
    tracing is off.
    """

    def __init__(self, run_id: int, error: str, trace_id: str | None = None):
        super().__init__(error)
        self.run_id = run_id
        self.error = error
        self.trace_id = trace_id


@dataclass(frozen=True)
class Outcome:
    run_id: int
    result: BaselineResult | TranslatorResult | OrchestratorResult  # ok, error and attempts of the config's run
    trace_id: str | None = None  # the page's trace; None when tracing is off


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
    review_budget: int = reviewer.DEFAULT_REVIEW_BUDGET,
    curator_budget: int = curator_agent.DEFAULT_CURATOR_BUDGET,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> dict:
    """The parameters a Run records. `page_budget` is the token budget of the `baseline` call: the page's own for
    `baseline`, the fallback's for `translator`, `full_no_kb` and `full`, which also have the agent's budget and its
    Step limit. `full_no_kb` has the Reviewer's budget too, and the number of times a page can be sent back; `full` also
    has the Curator's budget."""
    if config == translator.CONFIG_NAME:
        return {"agent_budget": agent_budget, "fallback_budget": page_budget, "max_steps": max_steps,
                "max_tokens": max_tokens}
    if config in (orchestrator.CONFIG_NAME, FULL):
        return {"agent_budget": agent_budget, "review_budget": review_budget,
                **({"curator_budget": curator_budget} if config == FULL else {}), "fallback_budget": page_budget,
                "max_steps": max_steps, "max_tokens": max_tokens, "max_send_backs": orchestrator.MAX_SEND_BACKS}
    return {"page_budget": page_budget, "max_tokens": max_tokens}


def outcome_counts(outcomes: Sequence[Mapping]) -> dict[str, int]:
    """How often each result of `curator.RESULTS` occurs among a page's proposal outcomes (a `finalize` record's
    `outcomes`, or any list of {"result"}): every result is a key, those that did not occur are 0."""
    counted = Counter(o["result"] for o in outcomes)
    return {result: counted[result] for result in curator_rules.RESULTS}


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
    review_budget: int = reviewer.DEFAULT_REVIEW_BUDGET,
    curator_budget: int = curator_agent.DEFAULT_CURATOR_BUDGET,
    max_steps: int = DEFAULT_MAX_STEPS,
    kb_id: int | None = None,
) -> Outcome:
    """Run `config` (`baseline`, `translator`, `full_no_kb` or `full`) on one page and store the Run (and its
    Translations when it succeeds).

    `agent_budget` and `max_steps` are the Translator's; `page_budget` is `baseline`'s, which is the fallback's
    budget for the `translator`, `full_no_kb` and `full`; `review_budget` is each review's, in `full_no_kb` and `full`;
    `curator_budget` is the Curator's, in `full`. A `translator` Run that fell back to `baseline` is still a
    `translator` Run: its `attempts_json` says so. A `full_no_kb` Run is one Run for the whole page: its
    `attempts_json` has every stage, and a failure in any stage fails it (see `orchestrator.run_orchestrator`). An
    unknown `config` is a `ValueError`, raised before anything else happens.

    `full` is `full_no_kb` with the Knowledge Base `kb_id`, which must belong to the page's book (a `ValueError`,
    before any Run, when it is missing or another book's; a `kb_id` with any other config is one too). The KB is read
    once, before the page, as a snapshot: the Translator and the Reviewer see the KB as it was when the page started.
    When the Orchestrator's page succeeded, the Run finalizes (OUTLINE §3.6): the Curator Agent judges the page's term
    proposals on the Run's client, so its calls and cost are the Run's; then `curator.apply` writes the KB once, in a
    transaction of its own; then the Run is finished. A Curator that gives no answer does not fail the Run: its terms
    stay undecided, and `apply` still applies names, confirmations and conflicts by rule. A failure in `apply` fails the
    Run (`RunCrashed`), with nothing stored and nothing written to the KB. A page whose Orchestrator failed has no
    proposals, no Curator call and no write. The `attempts_json` is the Orchestrator's plus
      "finalize": null (the Orchestrator failed) | {"kb_id",
        "curator": {"status", "error", "asked": [proposal ids], "decisions": [{"id", "accept", "reason"}],
                    "attempts": <the Curator Agent's record>},
        "outcomes": [{"proposal_id", "kind", "source", "target", "result"}, ...]}   # in the order of the proposals

    `make_client` is a factory so that every Run gets its own client and Meter, and its usage is its own.
    It is called after the page is validated and before the Run row exists, so a client that cannot be
    built (missing .env) leaves no Run behind. A failed model call or unparseable output is not an
    exception: it is a failed Run, reported in `Outcome.result`. Once the Run row exists, any other exception
    marks that Run failed and is raised as `RunCrashed`. The one case that leaves a Run `running` is a
    database that fails the write marking it failed too: `RunCrashed` is still raised, with the `run_id`
    and a message that says the Run was not marked.

    With tracing on, the whole call is one trace (`tracing.page_span`); `Outcome.trace_id` and `RunCrashed.trace_id`
    name it. So does every other exception that leaves the call (a `TranslateError`, `LLMNotConfigured`, a failing
    `start_run`): it keeps its type and gains a `trace_id` attribute, None when tracing is off.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {CONFIGS}")
    _check_kb(conn, book_title, config, kb_id)
    with tracing.page_span(book=book_title, page_index=page_index, config=config) as span:
        try:
            return _translate_page(
                conn, book_title, page_index, span, make_client=make_client, dataset_root=dataset_root, config=config,
                page_budget=page_budget, max_tokens=max_tokens, agent_budget=agent_budget,
                review_budget=review_budget, curator_budget=curator_budget, max_steps=max_steps, kb_id=kb_id,
            )
        except Exception as e:
            try:
                e.trace_id = None if span is None else span.trace_id
            except AttributeError:  # an exception type that takes no attributes: same exception, no ID
                pass
            raise


def _translate_page(
    conn: sqlite3.Connection,
    book_title: str,
    page_index: int,
    span: tracing.PageSpan | None,
    *,
    make_client: Callable[[], LLMClient],
    dataset_root: Path,
    config: str,
    page_budget: int,
    max_tokens: int,
    agent_budget: int,
    review_budget: int,
    curator_budget: int,
    max_steps: int,
    kb_id: int | None,
) -> Outcome:
    trace_id = None if span is None else span.trace_id
    valid = validate_page(conn, book_title, page_index, dataset_root)
    page_row = valid.row
    segment_ids = {s["idx"]: s["id"] for s in valid.segment_rows}
    page = Page(
        book_title, page_index, valid.image_path,
        [Segment(s["idx"], s["x"], s["y"], s["w"], s["h"], s["text_ja"], s["text_en"], s["text_zh_hant"])
         for s in valid.segment_rows],
    )

    previous = previous_pages(conn, book_title, page_index, config) if config in AGENT_CONFIGS else []
    kb = load_snapshot(conn, kb_id) if config == FULL else None  # once, before the page: the whole page reads this
    client = make_client()
    run_id = store.start_run(
        conn, page_id=page_row["id"], config=config, model=client.config.model,
        params=run_params(config, page_budget=page_budget, max_tokens=max_tokens, agent_budget=agent_budget,
                          review_budget=review_budget, curator_budget=curator_budget, max_steps=max_steps),
    )
    if span is not None:
        span.set(run_id=run_id)

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
        elif config in (orchestrator.CONFIG_NAME, FULL):
            result = orchestrator.run_orchestrator(
                page, client, previous=previous, agent_budget=agent_budget, review_budget=review_budget,
                fallback_budget=page_budget, max_tokens=max_tokens, max_steps=max_steps, kb=kb,
            )
            if kb is not None:
                result.attempts["finalize"] = None if not result.ok else _finalize(
                    conn, page, kb, run_id, result, client, curator_budget=curator_budget, max_tokens=max_tokens,
                    max_steps=max_steps,
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
            _describe_run(span, client, "running", error)
            raise RunCrashed(
                run_id, f"{error} (marking the Run failed also failed: {type(recovery).__name__}: {recovery})",
                trace_id,
            ) from e
        _describe_run(span, client, "failed", error)
        raise RunCrashed(run_id, error, trace_id) from e
    _describe_run(span, client, "succeeded" if result.ok else "failed", result.error, result)
    return Outcome(run_id, result, trace_id)


def _check_kb(conn: sqlite3.Connection, book_title: str, config: str, kb_id: int | None) -> None:
    """`full` reads and writes one Knowledge Base, which must be the page's book's; no other config has one."""
    if config != FULL:
        if kb_id is not None:
            raise ValueError(f"config {config!r} has no Knowledge Base; `kb_id` goes with {FULL!r}")
        return
    if kb_id is None:
        raise ValueError(f"config {FULL!r} needs a `kb_id`: the Knowledge Base the page is translated with")
    kb = store.get_kb(conn, kb_id)
    if kb is None:
        raise ValueError(f"no knowledge base has id {kb_id}")
    if kb["book_title"] != book_title:
        raise ValueError(f"knowledge base {kb_id} is for the book {kb['book_title']!r}, not {book_title!r}")


def _finalize(
    conn: sqlite3.Connection, page: Page, kb: KBSnapshot, run_id: int, result: OrchestratorResult, client: LLMClient, *,
    curator_budget: int, max_tokens: int, max_steps: int,
) -> dict:
    """The page's finalize, in this order: the Curator Agent's one call on the Run's client, then `curator.apply` in its
    own transaction (none is open here), which is the only write to the KB. Returns the `finalize` part of the record
    (see `translate_page`). A Curator that gave no decisions is not an error: `apply` runs without them. An exception
    from either step propagates, and the caller fails the Run."""
    curated = curator_agent.run_curator(
        page, result.translations, result.proposals, client, kb=kb, curator_budget=curator_budget,
        max_tokens=max_tokens, max_steps=max_steps,
    )
    decisions = {d["id"]: curator_rules.Decision(d["accept"], d["reason"]) for d in curated.decisions}
    outcomes = ([] if not result.proposals else
                curator_rules.apply(conn, kb.kb_id, page.page_index, run_id, result.proposals, decisions))
    by_id = {p.proposal_id: p for p in result.proposals}
    records = []
    for outcome in outcomes:
        proposal = by_id[outcome.proposal_id]
        source, target = (proposal.payload[field] for field in KEY_FIELDS[proposal.kind])
        records.append({"proposal_id": outcome.proposal_id, "kind": proposal.kind, "source": source, "target": target,
                        "result": outcome.result})
    return {
        "kb_id": kb.kb_id,
        "curator": {"status": curated.status, "error": curated.error, "asked": list(curated.proposal_ids),
                    "decisions": curated.decisions, "attempts": curated.attempts},
        "outcomes": records,
    }


def _describe_run(
    span: tracing.PageSpan | None, client: LLMClient, status: str, error: str | None,
    result: BaselineResult | TranslatorResult | OrchestratorResult | None = None,
) -> None:
    """What the page's root span says about the Run: its status and calls, for a `translator` why it stopped, for
    `full_no_kb` how often the page was sent back and why the loop ended, and for `full` also the Curator's status and
    how often each result happened to a proposal."""
    if span is None:
        return
    span.set(run_status=status, calls=client.meter.summary()["calls"])
    if isinstance(result, TranslatorResult):
        span.set(stop_reason=result.attempts["agent"]["stop_reason"], fallback_reason=result.fallback_reason)
    elif isinstance(result, OrchestratorResult):
        span.set(send_backs=result.send_backs, final_stop=result.final_stop, fallback_reason=result.fallback_reason)
        if result.attempts.get("finalize") is not None:
            finalize = result.attempts["finalize"]
            span.set(curator_status=finalize["curator"]["status"],
                     **{f"proposals_{r}": n for r, n in outcome_counts(finalize["outcomes"]).items()})
    if error is not None:
        span.fail(error)
