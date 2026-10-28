"""Translate pages with `runs.translate_page` (the editor's code path) and score them against the references.

Outputs go to <out_root>/<eval id>/: manifest.json, segments.jsonl and scores.json.

For `full` the harness first creates the run's Knowledge Base (OUTLINE §3.3, §8: one KB per eval run, never shared with
another run), labelled with the eval id, and passes it to every page in the order of the pages, so that a page reads what
the earlier pages wrote.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .. import orchestrator, runs, store, tracing
from ..agents import curator as curator_agent
from ..agents import reviewer, translator
from ..agents.loop import DEFAULT_MAX_STEPS, refuse_tracing
from ..kb import curator as curator_rules
from ..llm.client import LLMClient, LLMNotConfigured
from ..pipeline import baseline
from . import manifest as manifest_facts
from .metrics import cost_summary, latency_summary, run_cost, score_corpus
from .references import OPENCC_CONFIG, to_zh_hans

CONFIGS = runs.CONFIGS
REVIEWED_CONFIGS = (orchestrator.CONFIG_NAME, runs.FULL)  # the configs that run the Orchestrator's review loop
KB_STATUSES = ("proposed", "approved", "rejected", "pending_human")  # the states an entry or a profile can be in
CURATOR_STATUSES = curator_agent.STATUSES


class EvalAborted(Exception):
    """The harness stopped before any model call; `problems` says why."""

    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


@dataclass
class EvalResult:
    out_dir: Path
    manifest: dict
    scores: dict
    failed_pages: list[dict] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        return 1 if self.failed_pages else 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_eval(
    conn: sqlite3.Connection,
    *,
    config: str,
    book: str,
    page_indexes: list[int],
    make_client: Callable[[], LLMClient],
    dataset_root: Path,
    out_root: Path,
    repo_root: Path,
    log: Callable[[str], None] = lambda _: None,
) -> EvalResult:
    """Translate the pages one after another, score the succeeded ones, and write the outputs.

    Raises `EvalAborted` before any model call when a page does not exist or its image is missing, and
    `TracingEnabled` before the first Run when a LangSmith tracing variable is on (prompts would leave the machine,
    and every `translator` page would fail). A page without text regions is skipped. A failed or crashed Run is
    listed under `failed_pages` and left out of the scores; its cost is reported apart, under `cost.failed_runs`.
    A `translator` Run that fell back to `baseline` is scored like any other, and flagged: per page (`fell_back`,
    `fallback_reason`) and as a count over the scored pages (`fallback_pages`). For `full_no_kb` that is the first pass;
    a revision never falls back. A `full_no_kb` page row also says how its review loop went (`send_backs`,
    `final_stop`, `final_reviewed`, and the status and verdict of each review), and `scores.json` counts the scored
    pages by `final_stop` (`final_stops`); `full` has the same. Each page row carries the `trace_id` of its trace (null
    when tracing is off), and the manifest says where the traces went.

    `full` makes one new KB for the eval run, after every page has been checked and before the first model call; the
    manifest records its id and label (`kb`). A scored page's row gives the Curator's status (`curator_status`) and the
    count of each result of `curator.RESULTS` among its proposals (`proposal_outcomes`, zeros included, so
    `conflict_across_kinds` is counted apart from `conflict`). `scores.json` gives the totals (`curator_statuses`,
    `proposal_outcomes`) and the KB's size at the end (`kb`: id, label, entries and profiles by status). Proposals the
    Curator left `undecided` stay so: an eval KB is used once. The client is built by the first page, so an
    `LLMNotConfigured` stop comes after the KB was made, and that empty KB stays in the database.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {CONFIGS}")
    if len(set(page_indexes)) != len(page_indexes):
        raise ValueError(f"duplicate pages in {page_indexes}")
    refuse_tracing()
    started_at = _now()
    params = {"page_budget": baseline.DEFAULT_PAGE_BUDGET, "max_tokens": baseline.DEFAULT_MAX_TOKENS}
    if config == translator.CONFIG_NAME:
        params |= {"agent_budget": translator.DEFAULT_PAGE_BUDGET, "max_steps": DEFAULT_MAX_STEPS}
    if config in REVIEWED_CONFIGS:
        params |= {"agent_budget": translator.DEFAULT_PAGE_BUDGET, "review_budget": reviewer.DEFAULT_REVIEW_BUDGET,
                   "max_steps": DEFAULT_MAX_STEPS}
    if config == runs.FULL:
        params |= {"curator_budget": curator_agent.DEFAULT_CURATOR_BUDGET}

    to_translate: list[int] = []
    skipped: list[dict] = []
    problems: list[str] = []
    for page_index in page_indexes:
        try:
            runs.validate_page(conn, book, page_index, dataset_root)
        except runs.NothingToTranslate as e:
            skipped.append({"book": book, "page_index": page_index, "reason": str(e)})
        except runs.TranslateError as e:
            problems.append(f"{book}/{page_index}: {e}")
        else:
            to_translate.append(page_index)
    if problems:
        raise EvalAborted(problems)

    kb_id = kb_label = None
    eval_id = None
    if config == runs.FULL:  # one KB for this eval run: its label is the eval id, which no other KB of the book has
        eval_id = _free_id(out_root, _eval_base(started_at, config),
                           taken=lambda label: store.find_kb(conn, book_title=book, purpose="eval", label=label))
        kb_id, kb_label = store.create_kb(conn, book_title=book, purpose="eval", label=eval_id), eval_id

    page_entries: dict[int, dict] = {}  # page_index -> manifest entry
    scored: list[dict] = []  # per-page score rows, in page order
    failed: list[dict] = []
    segment_lines: list[dict] = []
    run_rows: list[sqlite3.Row] = []  # every Run that was started and recorded, failed ones included
    scored_runs: list[sqlite3.Row] = []
    failed_runs: list[sqlite3.Row] = []  # failed or crashed; they cost money even though they are not scored
    for page_index in to_translate:
        log(f"translating {book}/{page_index} ...")
        start = time.monotonic()
        outcome = None
        trace_id = None
        try:
            outcome = runs.translate_page(
                conn, book, page_index, make_client=make_client, dataset_root=dataset_root, config=config,
                **({"kb_id": kb_id} if config == runs.FULL else {}), **params,
            )
        except LLMNotConfigured:
            raise  # raised before any Run exists, and it would be the same for every page
        except runs.RunCrashed as e:  # the Run exists: failed, or still running if the database failed too
            run_id, error, trace_id = e.run_id, e.error, e.trace_id
        except Exception as e:  # no Run row exists yet (make_client or start_run raised); keep the other pages
            run_id, error, trace_id = None, f"{type(e).__name__}: {e}", getattr(e, "trace_id", None)
        else:
            run_id, error, trace_id = outcome.run_id, outcome.result.error, outcome.trace_id
        latency = time.monotonic() - start
        run = None if run_id is None else store.get_run(conn, run_id)
        if run is not None:
            run_rows.append(run)
        cost = None if run is None else run_cost(run["cost_usd"], run["unknown_usage_calls"])
        ok = outcome is not None and outcome.result.ok
        fallback_reason = _fallback_reason(run)
        page_entries[page_index] = {"book": book, "page_index": page_index,
                                    "status": "scored" if ok else "failed", "run_id": run_id, "trace_id": trace_id}
        if not ok:
            failed.append({"book": book, "page_index": page_index, "run_id": run_id, "trace_id": trace_id,
                           "error": error, "fell_back": fallback_reason is not None, "fallback_reason": fallback_reason,
                           "latency_s": round(latency, 3), "cost_usd": cost})
            if run is not None:
                failed_runs.append(run)
            continue
        page_row = store.find_page(conn, book, page_index)
        rows = store.segments_with_translations(conn, page_row["id"], outcome.run_id)
        for r in rows:
            segment_lines.append({
                "book": book, "page_index": page_index, "idx": r["idx"], "run_id": outcome.run_id,
                "ja": r["text_ja"], "hyp": r["text_zh_hans"],
                "ref_zh_hans": to_zh_hans(r["text_zh_hant"]), "ref_zh_hant": r["text_zh_hant"],
            })
        scored_runs.append(run)
        scored.append({"book": book, "page_index": page_index, "run_id": outcome.run_id, "trace_id": trace_id,
                       "segments": len(rows), "calls": run["calls"], "unknown_usage_calls": run["unknown_usage_calls"],
                       "fell_back": fallback_reason is not None, "fallback_reason": fallback_reason,
                       **(_review_loop(run) if config in REVIEWED_CONFIGS else {}),
                       **(_curation(run) if config == runs.FULL else {}),
                       "cost_usd": cost, "latency_s": round(latency, 3)})
        log(f"  run {outcome.run_id}: {len(rows)} segments, {latency:.1f}s")
    for entry in skipped:
        page_entries[entry["page_index"]] = {"book": book, "page_index": entry["page_index"],
                                             "status": "skipped", "run_id": None, "trace_id": None}

    corpus = score_corpus([s["hyp"] for s in segment_lines], [s["ref_zh_hans"] for s in segment_lines])
    scores = {
        "config": config,
        "book": book,
        "pages_scored": len(scored),
        "segments_scored": len(segment_lines),
        "fallback_pages": sum(p["fell_back"] for p in scored),  # scored pages whose Run fell back to baseline
        **({"final_stops": {stop: sum(p["final_stop"] == stop for p in scored) for stop in orchestrator.FINAL_STOPS}}
           if config in REVIEWED_CONFIGS else {}),  # scored pages by why their review loop ended
        **({"curator_statuses": {status: sum(p["curator_status"] == status for p in scored)
                                 for status in CURATOR_STATUSES},
            "proposal_outcomes": {result: sum(p["proposal_outcomes"][result] for p in scored)
                                  for result in curator_rules.RESULTS},
            "kb": _kb_size(conn, kb_id, kb_label)}
           if config == runs.FULL else {}),  # the Curator's statuses, the results for the proposals, and the KB's size
        "chrf": {"score": corpus.chrf, "signature": corpus.chrf_signature},
        "bleu": {"score": corpus.bleu, "signature": corpus.bleu_signature},
        "cost": cost_summary([(r["cost_usd"], r["unknown_usage_calls"]) for r in scored_runs])
        | {"failed_runs": cost_summary([(r["cost_usd"], r["unknown_usage_calls"]) for r in failed_runs])},
        "latency": latency_summary([s["latency_s"] for s in scored]),
        "pages": scored,
        "failed_pages": failed,
        "skipped_pages": skipped,
    }
    models_requested = sorted({r["model"] for r in run_rows})
    models_response = sorted({m for r in run_rows if r["response_model"] for m in r["response_model"].split(", ")})
    git = manifest_facts.git_state(repo_root)
    finished_at = _now()
    if eval_id is None:
        eval_id = _free_id(out_root, _eval_base(started_at, config))
    manifest = {
        "eval_id": eval_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "config_name": config,
        "config_params": runs.run_params(config, **params),  # as the Runs record them
        "book": book,
        "pages": [page_entries[i] for i in page_indexes],
        "annotation_sha256": manifest_facts.file_sha256(dataset_root / "annotation.json"),
        "requested_models": models_requested,
        "response_models": models_response,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "sacrebleu_signatures": {"chrf": corpus.chrf_signature, "bleu": corpus.bleu_signature},
        "opencc_config": OPENCC_CONFIG,
        "tracing": tracing.describe(),  # whether the traces in `pages` exist, and where they went
        **({"kb": {"id": kb_id, "label": kb_label}} if config == runs.FULL else {}),  # this eval run's own KB
    }

    out_dir = out_root / eval_id
    out_dir.mkdir(parents=True)
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "scores.json", scores)
    with (out_dir / "segments.jsonl").open("w", encoding="utf-8") as f:
        for line in segment_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return EvalResult(out_dir, manifest, scores, failed)


def _attempts(run: sqlite3.Row | None) -> list | dict | None:
    """The Run's stored attempts record; None when it has none."""
    if run is None or not run["attempts_json"]:
        return None
    return json.loads(run["attempts_json"])


def _fallback_reason(run: sqlite3.Row | None) -> str | None:
    """Why the Run fell back to `baseline`, read from what it stored; None when it did not (or has no record)."""
    attempts = _attempts(run)
    if isinstance(attempts, dict) and "stages" in attempts:  # `full_no_kb`: only its first pass can fall back
        attempts = attempts["stages"][0]["attempts"]
    fallback = attempts.get("fallback") if isinstance(attempts, dict) else None  # baseline stores a list
    return None if fallback is None else fallback["reason"]


def _curation(run: sqlite3.Row) -> dict:
    """How a `full` Run's finalize went, read from what it stored: the Curator's status, and how often each result of the
    rules happened to the page's proposals (every result is a key)."""
    finalize = _attempts(run)["finalize"]
    return {"curator_status": finalize["curator"]["status"],
            "proposal_outcomes": runs.outcome_counts(finalize["outcomes"])}


def _kb_size(conn: sqlite3.Connection, kb_id: int, label: str) -> dict:
    """The KB as the eval left it: its entries and profiles, counted by status (every status is a key)."""
    def by_status(rows) -> dict[str, int]:
        return {status: sum(r["status"] == status for r in rows) for status in KB_STATUSES}

    return {"id": kb_id, "label": label, "entries": by_status(store.kb_entries(conn, kb_id, KB_STATUSES)),
            "profiles": by_status(store.kb_profiles(conn, kb_id, KB_STATUSES))}


def _review_loop(run: sqlite3.Row) -> dict:
    """How a `full_no_kb` Run's review loop went, read from what it stored: the send-backs, why the loop ended,
    whether the stored translation got a verdict, and the status and verdict of each review, in order."""
    attempts = _attempts(run)
    reviews = [stage for stage in attempts["stages"] if stage["kind"] == orchestrator.REVIEW]
    return {"send_backs": attempts["send_backs"], "final_stop": attempts["final"]["stop"],
            "final_reviewed": attempts["final"]["reviewed"],
            "review_statuses": [r["status"] for r in reviews], "review_verdicts": [r["verdict"] for r in reviews]}


def _eval_base(started_at: str, config: str) -> str:
    return datetime.fromisoformat(started_at).strftime("%Y%m%dT%H%M%SZ") + f"-{config}"


def _free_id(out_root: Path, base: str, taken: Callable[[str], object] = lambda _: None) -> str:
    """`base`, or `base-2`, `base-3`... so a second eval in the same second cannot clobber the first. `taken` says
    whether something else already has a candidate as its name (a KB's label): it must be falsy for the one chosen."""
    candidate, n = base, 1
    while (out_root / candidate).exists() or taken(candidate):
        n += 1
        candidate = f"{base}-{n}"
    return candidate


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def format_summary(result: EvalResult) -> str:
    s = result.scores
    cost, lat = s["cost"], s["latency"]

    def num(value, fmt):
        return "n/a" if value is None else format(value, fmt)

    def sig(value):
        return "" if value is None else f"   {value}"

    lines = [
        f"{result.manifest['eval_id']} -> {result.out_dir}",
        f"scored {s['pages_scored']} pages, {s['segments_scored']} segments",
        f"chrF  {num(s['chrf']['score'], '.2f')}{sig(s['chrf']['signature'])}",
        f"BLEU  {num(s['bleu']['score'], '.2f')}{sig(s['bleu']['signature'])}",
        f"cost  total ${num(cost['total_usd'], '.6f')}, ${num(cost['per_page_usd'], '.6f')} per page"
        f" (unknown-cost runs {cost['unknown_cost_runs']}, unknown_usage_calls {cost['unknown_usage_calls']})",
        f"latency per page  p50 {num(lat['p50_s'], '.1f')}s  p95 {num(lat['p95_s'], '.1f')}s",
    ]
    if s["config"] != baseline.CONFIG_NAME:
        lines.insert(2, f"fallback to baseline  {s['fallback_pages']} of {s['pages_scored']} scored pages")
    if s["config"] in REVIEWED_CONFIGS:
        stops = ", ".join(f"{stop} {n}" for stop, n in s["final_stops"].items())
        lines.insert(3, f"review loop  {stops}; send-backs {sum(p['send_backs'] for p in s['pages'])}")
        for p in s["pages"]:
            reviews = "/".join(verdict or status for verdict, status in zip(p["review_verdicts"], p["review_statuses"]))
            lines.append(f"REVIEW LOOP  {p['book']}/{p['page_index']} (run {p['run_id']}): {p['final_stop']}, "
                         f"{p['send_backs']} send-back(s), reviews {reviews}")
    if s["config"] == runs.FULL:
        kb = s["kb"]
        statuses = ", ".join(f"{status} {n}" for status, n in s["curator_statuses"].items())
        outcomes = ", ".join(f"{result} {n}" for result, n in s["proposal_outcomes"].items() if n)
        size = "; ".join(f"{kind} " + ", ".join(f"{status} {n}" for status, n in counts.items() if n)
                         for kind, counts in (("entries", kb["entries"]), ("profiles", kb["profiles"]))
                         if any(counts.values()))
        lines[4:4] = [f"curator  {statuses}", f"proposals  {outcomes or 'none'}",
                      f"knowledge base {kb['id']} ({kb['label']})  {size or 'empty'}"]
        for p in s["pages"]:
            made = ", ".join(f"{result} {n}" for result, n in p["proposal_outcomes"].items() if n)
            lines.append(f"CURATOR  {p['book']}/{p['page_index']} (run {p['run_id']}): {p['curator_status']}, "
                         f"proposals {made or 'none'}")
    for p in s["pages"]:
        if p["fell_back"]:
            lines.append(f"FELL BACK  {p['book']}/{p['page_index']} (run {p['run_id']}): {p['fallback_reason']}")
    for p in s["skipped_pages"]:
        lines.append(f"skipped {p['book']}/{p['page_index']}: {p['reason']}")
    for p in s["failed_pages"]:
        run = "no Run" if p["run_id"] is None else f"run {p['run_id']}"
        lines.append(f"FAILED  {p['book']}/{p['page_index']} ({run}): {p['error']}"
                     + (f"  (after falling back: {p['fallback_reason']})" if p["fell_back"] else ""))
    if s["failed_pages"]:
        failed = cost["failed_runs"]
        lines.append(f"failed runs cost  total ${num(failed['total_usd'], '.6f')}"
                     f" (unknown-cost runs {failed['unknown_cost_runs']},"
                     f" unknown_usage_calls {failed['unknown_usage_calls']})")
    return "\n".join(lines)
