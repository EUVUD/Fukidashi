"""Translate pages with `runs.translate_page` (the editor's code path) and score them against the references.

Outputs go to <out_root>/<eval id>/: manifest.json, segments.jsonl and scores.json.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .. import runs, store
from ..llm.client import LLMClient, LLMNotConfigured
from ..pipeline import baseline
from . import manifest as manifest_facts
from .metrics import cost_summary, latency_summary, run_cost, score_corpus
from .references import OPENCC_CONFIG, to_zh_hans

CONFIGS = (baseline.CONFIG_NAME,)  # `full` and `full_no_kb` arrive with the orchestrator


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

    Raises `EvalAborted` before any model call when a page does not exist or its image is missing. A page
    without text regions is skipped. A failed or crashed Run is listed under `failed_pages` and left out of
    the scores; its cost is reported apart, under `cost.failed_runs`.
    """
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {CONFIGS}")
    if len(set(page_indexes)) != len(page_indexes):
        raise ValueError(f"duplicate pages in {page_indexes}")
    started_at = _now()
    params = {"page_budget": baseline.DEFAULT_PAGE_BUDGET, "max_tokens": baseline.DEFAULT_MAX_TOKENS}

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
        try:
            outcome = runs.translate_page(
                conn, book, page_index, make_client=make_client, dataset_root=dataset_root, **params
            )
        except LLMNotConfigured:
            raise  # raised before any Run exists, and it would be the same for every page
        except runs.RunCrashed as e:  # the Run exists: failed, or still running if the database failed too
            run_id, error = e.run_id, e.error
        except Exception as e:  # no Run row exists yet (make_client or start_run raised); keep the other pages
            run_id, error = None, f"{type(e).__name__}: {e}"
        else:
            run_id, error = outcome.run_id, outcome.result.error
        latency = time.monotonic() - start
        run = None if run_id is None else store.get_run(conn, run_id)
        if run is not None:
            run_rows.append(run)
        cost = None if run is None else run_cost(run["cost_usd"], run["unknown_usage_calls"])
        ok = outcome is not None and outcome.result.ok
        page_entries[page_index] = {"book": book, "page_index": page_index,
                                    "status": "scored" if ok else "failed", "run_id": run_id}
        if not ok:
            failed.append({"book": book, "page_index": page_index, "run_id": run_id, "error": error,
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
        scored.append({"book": book, "page_index": page_index, "run_id": outcome.run_id, "segments": len(rows),
                       "calls": run["calls"], "unknown_usage_calls": run["unknown_usage_calls"],
                       "cost_usd": cost, "latency_s": round(latency, 3)})
        log(f"  run {outcome.run_id}: {len(rows)} segments, {latency:.1f}s")
    for entry in skipped:
        page_entries[entry["page_index"]] = {"book": book, "page_index": entry["page_index"],
                                             "status": "skipped", "run_id": None}

    corpus = score_corpus([s["hyp"] for s in segment_lines], [s["ref_zh_hans"] for s in segment_lines])
    scores = {
        "config": config,
        "book": book,
        "pages_scored": len(scored),
        "segments_scored": len(segment_lines),
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
    eval_id = _free_id(out_root, datetime.fromisoformat(started_at).strftime("%Y%m%dT%H%M%SZ") + f"-{config}")
    manifest = {
        "eval_id": eval_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "config_name": config,
        "config_params": params,
        "book": book,
        "pages": [page_entries[i] for i in page_indexes],
        "annotation_sha256": manifest_facts.file_sha256(dataset_root / "annotation.json"),
        "requested_models": models_requested,
        "response_models": models_response,
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "sacrebleu_signatures": {"chrf": corpus.chrf_signature, "bleu": corpus.bleu_signature},
        "opencc_config": OPENCC_CONFIG,
    }

    out_dir = out_root / eval_id
    out_dir.mkdir(parents=True)
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "scores.json", scores)
    with (out_dir / "segments.jsonl").open("w", encoding="utf-8") as f:
        for line in segment_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return EvalResult(out_dir, manifest, scores, failed)


def _free_id(out_root: Path, base: str) -> str:
    """`base`, or `base-2`, `base-3`... so a second eval in the same second cannot clobber the first."""
    candidate, n = base, 1
    while (out_root / candidate).exists():
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
    for p in s["skipped_pages"]:
        lines.append(f"skipped {p['book']}/{p['page_index']}: {p['reason']}")
    for p in s["failed_pages"]:
        run = "no Run" if p["run_id"] is None else f"run {p['run_id']}"
        lines.append(f"FAILED  {p['book']}/{p['page_index']} ({run}): {p['error']}")
    if s["failed_pages"]:
        failed = cost["failed_runs"]
        lines.append(f"failed runs cost  total ${num(failed['total_usd'], '.6f')}"
                     f" (unknown-cost runs {failed['unknown_cost_runs']},"
                     f" unknown_usage_calls {failed['unknown_usage_calls']})")
    return "\n".join(lines)
