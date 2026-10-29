"""Pool finished eval runs into one report (OUTLINE 4.3, 4.4, 4.8).

The inputs are eval runs' directories, one for each config and book. Each holds what the harness and the scorers wrote:
`manifest.json`, `scores.json`, `segments.jsonl`, `terms.json` (eval/terms.py) and `comet-wmt22-comet-da.json`
(eval/comet.py). The report reads those and the current `eval/gold_glossary/<book>.json` (for its hash) and nothing
else: no database, no model, no network. The command-line entry point is eval/report.py; the logic lives here so that
tests can import it.

- **Common pages.** A config's quality numbers cover only the pages that every config scored, for each book, so the
  configs are compared on the same pages (the later paired bootstrap resamples pages, and a page that failed in one
  config would otherwise shift the comparison). The pages left out are listed per config with their status. A page that
  every config skipped has no text and is not listed. Spend is the exception: it covers all of a config's Runs.
- **Pooling.** chrF and BLEU are recomputed from the pooled segments, never averaged over books. COMET is the mean of
  the segment scores, joined to `segments.jsonl` by (book, page_index, idx), never by position. Term accuracy pools hits
  and occurrences; its mean over entries is taken over every entry, of any book, that has an occurrence.
- **Unknown is not 0.** A field an older run's files do not have (`fell_back` before the `translator` config,
  `kb_reach` before #40) and a cost that `cost_summary` cannot total are `None`, here and in report.json.
- **Warnings, not refusals,** for provenance: a dirty tree, runs from different commits, different models.
  Everything that would make the numbers wrong is a refusal (`Refused`).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manifest import file_sha256
from .metrics import cost_summary, latency_summary, score_corpus
from .splits import DEV_BOOKS, TEST_BOOKS

CONFIG_ORDER = ("baseline", "translator", "full_no_kb", "full")  # the order of the columns; a test pins it to runs.CONFIGS
REVIEWED_CONFIGS = ("full_no_kb", "full")  # the configs that run the Orchestrator's review loop
KB_CONFIG = "full"
COMET_MODEL = "Unbabel/wmt22-comet-da"
COMET_FILE = "comet-wmt22-comet-da.json"
# What the page rows of scores.json count. These repeat the names of the Orchestrator, the Curator and the rules, which
# this module does not import (it reads files only); a test pins each to its source.
FINAL_STOPS = ("approved", "review_unusable", "revision_unusable", "send_back_limit")
CURATOR_STATUSES = ("ok", "unusable", "failed", "skipped")
PROPOSAL_RESULTS = ("duplicate", "confirmed", "merged", "conflict", "conflict_across_kinds", "pending_human",
                    "accepted", "rejected", "undecided")
LOOKUP_AGENTS = ("translator", "reviewer")
LOOKUP_TOOLS = ("lookup_term", "lookup_character")
LOOKUP_COUNTS = ("calls", "found", "not_found")
ROW_KEYS = ("cost_usd", "unknown_usage_calls", "latency_s")  # every scored page row must have these
SEGMENT_FIELDS = (("book", str), ("page_index", int), ("idx", int), ("hyp", str), ("ref_zh_hans", str))


class Refused(Exception):
    """The runs cannot be pooled; the message names the directory and the cause. Nothing has been written."""


@dataclass(frozen=True)
class Run:
    """One finished eval run, as read from its directory."""

    directory: Path
    manifest: dict
    eval_id: str
    config: str
    book: str
    statuses: dict[int, str]  # the manifest's page_index -> status
    rows: dict[int, dict]  # scores.json's page rows (the scored pages), by page_index
    failed: dict[int, dict]  # scores.json's failed_pages rows, by page_index
    segments: list[dict]
    term_entries: list[tuple[str, bool, list[tuple[int, int, bool]]]]  # (id, borderline, [(page_index, idx, hit)])
    comet: dict  # the COMET file without its segments
    comet_scores: dict[tuple[str, int, int], float]  # (book, page_index, idx) -> score
    terms: dict  # terms.json without its entries


# --- reading one run --------------------------------------------------------------------------------------------------

def _json_file(directory: Path, name: str) -> dict:
    path = directory / name
    if not path.is_file():
        raise Refused(f"{directory}: {name} is missing")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:  # a UnicodeDecodeError is a ValueError
        raise Refused(f"{directory}: {name} cannot be read as JSON: {e}")
    if not isinstance(data, dict):
        raise Refused(f"{directory}: {name} is not a JSON object")
    return data


def load_run(directory: Path, glossary_dir: Path) -> Run:
    """Read and check one eval run; raises `Refused`, having written nothing."""
    if not directory.is_dir():
        raise Refused(f"{directory}: it is not a directory")
    manifest = _json_file(directory, "manifest.json")
    eval_id, config, book = (manifest.get(key) for key in ("eval_id", "config_name", "book"))
    if not all(isinstance(value, str) and value for value in (eval_id, config, book)):
        raise Refused(f"{directory}: manifest.json has no `eval_id`, `config_name` and `book`")
    if manifest.get("finished_at") is None:
        raise Refused(f"{directory}: the eval run is not finished (`finished_at` is null in manifest.json): "
                      "its pages are not all there")
    statuses = _statuses(directory, manifest)
    scores = _json_file(directory, "scores.json")
    rows, failed = _rows(directory, scores)
    segments, segments_sha256 = _segments(directory, book)
    terms = _json_file(directory, "terms.json")
    comet = _json_file(directory, COMET_FILE)

    for name, document in (("terms.json", terms), (COMET_FILE, comet)):
        if document.get("eval_id") != eval_id:
            raise Refused(f"{directory}: {name} is of eval run {document.get('eval_id')!r}, "
                          f"the manifest's is {eval_id!r}")
    if comet.get("model") != COMET_MODEL:
        raise Refused(f"{directory}: {COMET_FILE} was scored with {comet.get('model')!r}; "
                      f"the report takes {COMET_MODEL!r}")
    if comet.get("segments_sha256") != segments_sha256:
        raise Refused(f"{directory}: {COMET_FILE} scored another segments.jsonl than the one there now "
                      "(its segments_sha256 differs): score the run again with eval/comet.py")
    glossary_path = glossary_dir / f"{book}.json"
    if not glossary_path.is_file():
        raise Refused(f"{directory}: there is no glossary for book {book!r} at {glossary_path}")
    if terms.get("glossary_sha256") != file_sha256(glossary_path):
        raise Refused(f"{directory}: terms.json was scored against another {glossary_path.name} than the one there "
                      "now (its glossary_sha256 differs): score the run again with eval/terms.py")

    for page_index, status in sorted(statuses.items()):
        if status != "scored":
            continue
        if page_index not in rows:
            raise Refused(f"{directory}: page {page_index} is scored in manifest.json but has no row in scores.json")
        if not any(s["page_index"] == page_index for s in segments):
            raise Refused(f"{directory}: page {page_index} is scored in manifest.json but has no line in segments.jsonl")
    comet_scores = _comet_scores(directory, comet)
    for s in segments:
        if (s["book"], s["page_index"], s["idx"]) not in comet_scores:
            raise Refused(f"{directory}: {COMET_FILE} has no score for segment {s['book']}/{s['page_index']} "
                          f"idx {s['idx']}")
    return Run(directory, manifest, eval_id, config, book, statuses, rows, failed, segments,
               _term_entries(directory, terms), {k: v for k, v in comet.items() if k != "segments"}, comet_scores,
               {k: v for k, v in terms.items() if k != "entries"})


def _statuses(directory: Path, manifest: dict) -> dict[int, str]:
    pages = manifest.get("pages")
    if not isinstance(pages, list):
        raise Refused(f"{directory}: manifest.json has no `pages`")
    statuses = {}
    for page in pages:
        if not isinstance(page, dict) or not _is_int(page.get("page_index")) or not isinstance(page.get("status"), str):
            raise Refused(f"{directory}: a page entry of manifest.json has no `page_index` and `status`")
        statuses[page["page_index"]] = page["status"]
    return statuses


def _rows(directory: Path, scores: dict) -> tuple[dict[int, dict], dict[int, dict]]:
    found = {}
    for name in ("pages", "failed_pages"):
        rows = scores.get(name)
        if not isinstance(rows, list) or not all(isinstance(r, dict) and _is_int(r.get("page_index")) for r in rows):
            raise Refused(f"{directory}: scores.json has no `{name}` list of rows with a `page_index`")
        found[name] = {r["page_index"]: r for r in rows}
    for page_index, row in found["pages"].items():
        for key in ROW_KEYS:
            if key not in row:
                raise Refused(f"{directory}: the row of page {page_index} in scores.json has no `{key}`")
        if not _is_int(row["unknown_usage_calls"]):
            raise Refused(f"{directory}: the row of page {page_index} in scores.json has an `unknown_usage_calls` "
                          "that is not an integer")
        if not _is_finite(row["latency_s"]):
            raise Refused(f"{directory}: the row of page {page_index} in scores.json has a `latency_s` "
                          "that is not a finite number")
    # a cost is a finite number, or null for "unknown"; NaN or Infinity would reach report.json, which cannot hold them
    for page_index, row in [*found["pages"].items(), *found["failed_pages"].items()]:
        if row.get("cost_usd") is not None and not _is_finite(row["cost_usd"]):
            raise Refused(f"{directory}: the row of page {page_index} in scores.json has a `cost_usd` "
                          "that is neither a finite number nor null")
    return found["pages"], found["failed_pages"]


def _segments(directory: Path, book: str) -> tuple[list[dict], str]:
    path = directory / "segments.jsonl"
    if not path.is_file():
        raise Refused(f"{directory}: segments.jsonl is missing")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise Refused(f"{directory}: segments.jsonl is not valid UTF-8: {e}")
    segments: list[dict] = []
    first_seen: dict[tuple[int, int], int] = {}
    # "\n" only: `splitlines()` also cuts at U+2028, U+2029 and U+0085, which the harness writes unescaped
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            segment = json.loads(line)
        except ValueError as e:
            raise Refused(f"{directory}: segments.jsonl line {number} is not valid JSON: {e}")
        if not isinstance(segment, dict):
            raise Refused(f"{directory}: segments.jsonl line {number} is not a JSON object")
        for field, kind in SEGMENT_FIELDS:
            value = segment.get(field)
            if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
                raise Refused(f"{directory}: segments.jsonl line {number} has no {kind.__name__} `{field}`")
        if segment["book"] != book:
            raise Refused(f"{directory}: segments.jsonl line {number} is of book {segment['book']!r}, "
                          f"the run is of {book!r}")
        place = (segment["page_index"], segment["idx"])
        if place in first_seen:
            raise Refused(f"{directory}: segments.jsonl line {number} repeats page_index {place[0]} idx {place[1]}, "
                          f"which line {first_seen[place]} has")
        first_seen[place] = number
        segments.append(segment)
    return segments, hashlib.sha256(raw).hexdigest()


def _comet_scores(directory: Path, comet: dict) -> dict[tuple[str, int, int], float]:
    entries = comet.get("segments")
    if not isinstance(entries, list):
        raise Refused(f"{directory}: {COMET_FILE} has no `segments` list")
    scores: dict[tuple[str, int, int], float] = {}
    for number, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or not all(k in entry for k in ("book", "page_index", "idx", "score")):
            raise Refused(f"{directory}: segment {number} of {COMET_FILE} has no book, page_index, idx and score")
        score = entry["score"]
        if not _is_finite(score):
            raise Refused(f"{directory}: segment {number} of {COMET_FILE} has a score that is not a finite number")
        key = (entry["book"], entry["page_index"], entry["idx"])
        if key in scores:
            raise Refused(f"{directory}: {COMET_FILE} scores segment {key[0]}/{key[1]} idx {key[2]} twice")
        scores[key] = score
    return scores


def _term_entries(directory: Path, terms: dict) -> list[tuple[str, bool, list[tuple[int, int, bool]]]]:
    try:
        return [(entry["id"], bool(entry["borderline"]),
                 [(s["page_index"], s["idx"], bool(s["hit"])) for s in entry["segments"]])
                for entry in terms["entries"]]
    except (KeyError, TypeError) as e:
        raise Refused(f"{directory}: terms.json is not shaped as eval/terms.py writes it ({type(e).__name__}: {e})")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# --- the set of runs --------------------------------------------------------------------------------------------------

def _check_set(runs: Sequence[Run]) -> None:
    seen: dict[tuple[str, str], Run] = {}
    for run in runs:
        earlier = seen.setdefault((run.config, run.book), run)
        if earlier is not run:
            raise Refused(f"{run.directory} and {earlier.directory} are both {run.config} on {run.book}: "
                          "the report takes one eval run for each config and book")
    covered: dict[str, set[str]] = {}
    for run in runs:
        covered.setdefault(run.config, set()).add(run.book)
    if len({frozenset(books) for books in covered.values()}) > 1:
        raise Refused("the configs do not cover the same books: " + "; ".join(
            f"{config} has {', '.join(sorted(books))} ("
            + ", ".join(str(r.directory) for r in runs if r.config == config) + ")"
            for config, books in sorted(covered.items(), key=lambda item: _config_order(item[0]))))


def _config_order(config: str) -> tuple[int, str]:
    return (CONFIG_ORDER.index(config), config) if config in CONFIG_ORDER else (len(CONFIG_ORDER), config)


def _book_order(book: str) -> tuple[int, str]:
    known = (*DEV_BOOKS, *TEST_BOOKS)
    return (known.index(book), book) if book in known else (len(known), book)


def _common_pages(runs: Sequence[Run]) -> dict[str, list[int]]:
    """For each book, the pages every config scored."""
    common = {}
    for book in sorted({r.book for r in runs}, key=_book_order):
        scored = [{p for p, status in r.statuses.items() if status == "scored"} for r in runs if r.book == book]
        common[book] = sorted(set.intersection(*scored))
    return common


def _left_out(runs: Sequence[Run], common: dict[str, list[int]]) -> dict[str, list[dict]]:
    """Per config, the pages its numbers do not cover, with its status for each and, for a failed page, the error."""
    left: dict[str, list[dict]] = {config: [] for config in dict.fromkeys(r.config for r in runs)}
    for book in common:
        group = [r for r in runs if r.book == book]
        for page in sorted(set().union(*(r.statuses for r in group)) - set(common[book])):
            if all(r.statuses.get(page, "skipped") == "skipped" for r in group):
                continue  # no text: every config skipped it
            for run in group:
                status = run.statuses.get(page, "absent")  # absent: the run was not asked to translate the page
                entry = {"book": book, "page_index": page, "status": status}
                if status == "failed":
                    entry["error"] = run.failed.get(page, {}).get("error")
                left[run.config].append(entry)
    return left


# --- the numbers -------------------------------------------------------------------------------------------------------

def _metrics(config: str, runs: Sequence[Run], common: dict[str, list[int]]) -> dict:
    """One config's numbers over the common pages of the books of `runs` (one run per book)."""
    wanted = {r.book: set(common[r.book]) for r in runs}
    rows = [r.rows[p] for r in runs for p in sorted(wanted[r.book])]
    segments = [(r, s) for r in runs for s in r.segments if s["page_index"] in wanted[r.book]]
    corpus = score_corpus([s["hyp"] for _, s in segments], [s["ref_zh_hans"] for _, s in segments])
    comet = [r.comet_scores[(s["book"], s["page_index"], s["idx"])] for r, s in segments]
    revisions = {r.comet.get("model_revision") for r in runs}
    metrics = {
        "pages": len(rows),
        "segments": len(segments),
        "chrf": {"score": corpus.chrf, "signature": corpus.chrf_signature},
        "bleu": {"score": corpus.bleu, "signature": corpus.bleu_signature},
        "comet": {"model": COMET_MODEL, "revision": revisions.pop() if len(revisions) == 1 else None,
                  "mean": math.fsum(comet) / len(comet) if comet else None, "n": len(comet)},
        "terms": _terms(runs, wanted),
        "cost": cost_summary([(row["cost_usd"], row["unknown_usage_calls"]) for row in rows]),
        "latency": latency_summary([row["latency_s"] for row in rows]),
        "fallback_pages": _count_if_all_have(rows, "fell_back"),
    }
    if config in REVIEWED_CONFIGS:
        metrics["review_loop"] = _review_loop(rows)
    if config == KB_CONFIG:
        metrics["curator"] = _curator(rows)
        metrics["kb_reach"] = _kb_reach(runs, wanted)
    return metrics


def _count_if_all_have(rows: Sequence[dict], key: str) -> int | None:
    """How many rows say yes to `key`; None (unknown) when any row lacks the field, never 0."""
    return None if any(key not in row for row in rows) else sum(bool(row[key]) for row in rows)


def _terms(runs: Sequence[Run], wanted: dict[str, set[int]]) -> dict:
    entries = []  # (borderline, occurrences, hits) on the common pages, for every entry of every book
    for run in runs:
        for _, borderline, occurrences in run.term_entries:
            kept = [hit for page, _, hit in occurrences if page in wanted[run.book]]
            entries.append((borderline, len(kept), sum(kept)))
    return {"primary": _term_set([e for e in entries if not e[0]]), "all_entries": _term_set(entries)}


def _term_set(entries: Sequence[tuple[bool, int, int]]) -> dict:
    occurrences, hits = sum(n for _, n, _ in entries), sum(h for _, _, h in entries)
    accuracies = [h / n for _, n, h in entries if n]
    return {
        "entries": len(entries),
        "entries_with_occurrences": len(accuracies),
        "occurrences": occurrences,
        "hits": hits,
        "accuracy": hits / occurrences if occurrences else None,
        "mean_over_entries": sum(accuracies) / len(accuracies) if accuracies else None,
    }


def _counts(values: Sequence[Any], keys: Sequence[str]) -> dict[str, int]:
    """How often each of `keys` occurs among `values`, and then any other value that does, so the counts add up."""
    return {key: sum(v == key for v in values) for key in (*keys, *sorted(set(values) - set(keys)))}


def _review_loop(rows: Sequence[dict]) -> dict:
    stops = [row.get("final_stop") for row in rows]
    send_backs = [row.get("send_backs") for row in rows]
    return {"final_stops": None if None in stops else _counts(stops, FINAL_STOPS),
            "send_backs": None if None in send_backs else sum(send_backs)}


def _curator(rows: Sequence[dict]) -> dict:
    statuses = [row.get("curator_status") for row in rows]
    outcomes = [row.get("proposal_outcomes") for row in rows]
    return {
        "statuses": None if None in statuses else _counts(statuses, CURATOR_STATUSES),
        "proposal_outcomes": None if None in outcomes else {
            result: sum(o.get(result, 0) for o in outcomes)
            for result in (*PROPOSAL_RESULTS, *sorted({r for o in outcomes for r in o} - set(PROPOSAL_RESULTS)))},
    }


def _kb_reach(runs: Sequence[Run], wanted: dict[str, set[int]]) -> dict:
    """The pages' `kb_reach` added up. A page without one (a Run stored before #40) is `unknown` among the snippet pages,
    and makes the lookups and the items reached unknown (None) rather than a count that leaves it out. Items reached are
    distinct within a book (its KB is its own) and summed over books."""
    snippet_pages = {"nonempty": 0, "empty": 0, "unknown": 0}
    lookups = {agent: {tool: dict.fromkeys(LOOKUP_COUNTS, 0) for tool in LOOKUP_TOOLS} for agent in LOOKUP_AGENTS}
    items_reached, complete = 0, True
    for run in runs:
        reached: set[tuple[str, str]] = set()
        for page in sorted(wanted[run.book]):
            reach = run.rows[page].get("kb_reach")
            if reach is None:
                snippet_pages["unknown"] += 1
                complete = False
                continue
            snippets = reach["snippets"]
            snippet_pages["unknown" if snippets is None else "nonempty" if snippets["items"] else "empty"] += 1
            for agent in LOOKUP_AGENTS:
                for tool in LOOKUP_TOOLS:
                    for count in LOOKUP_COUNTS:
                        lookups[agent][tool][count] += reach["lookups"][agent][tool][count]
            reached.update((item["kind"], item["key"]) for item in reach["reached"])
        items_reached += len(reached)
    totals = {count: sum(c[count] for tools in lookups.values() for c in tools.values()) for count in LOOKUP_COUNTS}
    return {"snippet_pages": snippet_pages,
            "lookups": {**totals, "by_agent": lookups} if complete else None,
            "items_reached": items_reached if complete else None}


def _spend(runs: Sequence[Run]) -> dict:
    """The cost of all of a config's Runs, the failed ones included, with `cost_summary`'s rule. A failed page row does
    not say how many calls had no usage, so its cost counts only through `cost_usd` (None when it cannot be trusted). A
    failed page with no Run recorded cost nothing the Meter saw and is not counted."""
    scored = [(row["cost_usd"], row["unknown_usage_calls"]) for r in runs for row in r.rows.values()]
    failed = [(row.get("cost_usd"), 0) for r in runs for row in r.failed.values() if row.get("run_id") is not None]
    return cost_summary(scored + failed) | {"runs": len(scored) + len(failed), "failed_runs": len(failed)}


# --- provenance --------------------------------------------------------------------------------------------------------

def _wall_seconds(manifest: dict) -> float | None:
    try:
        started = datetime.fromisoformat(manifest["started_at"])
        finished = datetime.fromisoformat(manifest["finished_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return (finished - started).total_seconds()


def _provenance(run: Run) -> dict:
    m = run.manifest
    return {
        "directory": str(run.directory),
        "eval_id": run.eval_id,
        "config": run.config,
        "book": run.book,
        "started_at": m.get("started_at"),
        "finished_at": m.get("finished_at"),
        "wall_s": _wall_seconds(m),
        "git_commit": m.get("git_commit"),
        "git_dirty": m.get("git_dirty"),
        "requested_models": m.get("requested_models"),
        "response_models": m.get("response_models"),
        "annotation_sha256": m.get("annotation_sha256"),
        "pages": {status: sum(s == status for s in run.statuses.values())
                  for status in ("scored", "failed", "skipped", "pending")},
        "comet": {key: run.comet.get(key) for key in ("model", "model_revision", "versions", "device")},
        "terms": {key: run.terms.get(key) for key in ("rules", "glossary_sha256")},
    }


def _warnings(runs: Sequence[Run]) -> list[str]:
    def name(run: Run) -> str:
        return f"{run.eval_id} ({run.config} on {run.book})"

    warnings = []
    for run in runs:
        dirty = run.manifest.get("git_dirty")
        if dirty is True:
            warnings.append(f"{name(run)} ran on a dirty working tree (git_dirty is true)")
        elif dirty is not False:
            warnings.append(f"{name(run)} does not say whether its working tree was dirty (git_dirty is unknown)")
    commits = {run.manifest.get("git_commit") for run in runs}
    if len(commits) > 1:
        warnings.append("the runs were made at different commits (or at an unknown one): " + ", ".join(
            f"{run.eval_id} {str(run.manifest.get('git_commit'))[:12]}" for run in runs))
    for key in ("requested_models", "response_models"):
        if len({tuple(run.manifest.get(key) or ()) for run in runs}) > 1:
            warnings.append(f"the runs differ in {key}: " + "; ".join(
                f"{run.eval_id} {', '.join(run.manifest.get(key) or ()) or 'none'}" for run in runs))
    if len({run.comet.get("model_revision") for run in runs}) > 1:
        warnings.append("the COMET scores come from different checkpoint revisions (or some do not say): " + ", ".join(
            f"{run.eval_id} {run.comet.get('model_revision')}" for run in runs))
    return warnings


# --- the report --------------------------------------------------------------------------------------------------------

def build_report(directories: Sequence[Path], glossary_dir: Path) -> dict:
    """The report of the eval runs in `directories` (what report.json holds). Raises `Refused`, having written nothing."""
    runs = [load_run(Path(d), glossary_dir) for d in directories]
    _check_set(runs)
    runs.sort(key=lambda r: (_config_order(r.config), _book_order(r.book)))
    configs = list(dict.fromkeys(r.config for r in runs))
    common = _common_pages(runs)
    by_config = {config: [r for r in runs if r.config == config] for config in configs}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "configs": configs,
        "books": list(common),
        "common_pages": common,
        "left_out": _left_out(runs, common),
        "pooled": {config: _metrics(config, by_config[config], common) for config in configs},
        "by_book": {book: {config: _metrics(config, [r for r in by_config[config] if r.book == book], common)
                           for config in configs} for book in common},
        "spend": {config: _spend(by_config[config]) for config in configs},
        "runs": [_provenance(r) for r in runs],
        "warnings": _warnings(runs),
    }


def write_report(report: dict, out_root: Path) -> Path:
    """Write report.json into a new directory `<out_root>/<UTC time>-report` (`-2`, `-3`... if that exists) and return
    it. The file is written under a temporary name and renamed, and an existing directory is never written into."""
    out_root.mkdir(parents=True, exist_ok=True)
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-report"
    candidate, n = base, 1
    while True:
        try:
            (out_root / candidate).mkdir()
            break
        except FileExistsError:
            n += 1
            candidate = f"{base}-{n}"
    out_dir = out_root / candidate
    temporary = out_dir / "report.json.tmp"
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, out_dir / "report.json")
    except BaseException:
        temporary.unlink(missing_ok=True)
        out_dir.rmdir()
        raise
    return out_dir


# --- the printed summary -----------------------------------------------------------------------------------------------

LABEL_WIDTH, CELL_WIDTH = 40, 14  # CELL_WIDTH: the least width of a column's cells


def _num(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def _row(label: str, cells: Sequence[str]) -> tuple[str, list[str]]:
    """A table row; `_layout` pads the cells of all rows to one width, once the widest is known."""
    return label, list(cells)


def _layout(lines: Sequence[str | tuple[str, list[str]]]) -> str:
    """The summary's text. Cells are right-aligned in columns as wide as the widest cell plus two spaces, so that no
    cell, however long (`n/a (16 unknown)`), touches the one before it."""
    width = max([len(cell) for line in lines if isinstance(line, tuple) for cell in line[1]] + [CELL_WIDTH]) + 2
    return "\n".join(line if isinstance(line, str)
                     else f"{line[0]:<{LABEL_WIDTH}}" + "".join(f"{cell:>{width}}" for cell in line[1])
                     for line in lines)


def _term_cell(score: dict) -> str:
    return f"{_pct(score['accuracy'])} ({score['hits']}/{score['occurrences']})"


def _mean_cell(score: dict) -> str:
    return f"{_pct(score['mean_over_entries'])} ({score['entries_with_occurrences']})"


def _cost_cell(cost: dict) -> str:
    unknown = f" ({cost['unknown_cost_runs']} unknown)" if cost["unknown_cost_runs"] else ""
    return _usd(cost["per_page_usd"]) + unknown


def render(report: dict) -> str:
    configs, pooled = report["configs"], report["pooled"]
    common = report["common_pages"]
    n_runs = len(report["runs"])
    lines = [f"Report of {n_runs} eval run{'s' * (n_runs != 1)} ({', '.join(report['books'])}), {report['generated_at']}",
             "common pages (scored by every config): " + ", ".join(f"{book} {len(pages)}" for book, pages in common.items()),
             "", _row("", configs)]

    def table(label: str, get) -> None:
        cells = []
        for config in configs:
            try:
                cells.append(get(pooled[config]))
            except KeyError:  # a section this config does not have
                cells.append("-")
        lines.append(_row(label, cells))

    table("pages", lambda m: str(m["pages"]))
    table("segments", lambda m: str(m["segments"]))
    table("chrF", lambda m: _num(m["chrf"]["score"], ".2f"))
    table("BLEU", lambda m: _num(m["bleu"]["score"], ".2f"))
    table("COMET (wmt22-comet-da)", lambda m: _num(m["comet"]["mean"], ".4f"))
    table("term accuracy, primary", lambda m: _term_cell(m["terms"]["primary"]))
    table("  mean over entries (n entries)", lambda m: _mean_cell(m["terms"]["primary"]))
    table("term accuracy, all entries", lambda m: _term_cell(m["terms"]["all_entries"]))
    table("  mean over entries (n entries)", lambda m: _mean_cell(m["terms"]["all_entries"]))
    table("cost per page", lambda m: _cost_cell(m["cost"]))
    table("latency p50 / p95 (s)", lambda m: f"{_num(m['latency']['p50_s'], '.1f')} / {_num(m['latency']['p95_s'], '.1f')}")
    table("fallback to baseline (pages)", lambda m: "n/a" if m["fallback_pages"] is None else str(m["fallback_pages"]))
    if any(c in REVIEWED_CONFIGS for c in configs):
        for stop in FINAL_STOPS:
            table(f"final_stop {stop}", lambda m, stop=stop: _num((m["review_loop"]["final_stops"] or {}).get(stop), "d"))
        table("send-backs", lambda m: _num(m["review_loop"]["send_backs"], "d"))
    if KB_CONFIG in configs:
        for status in CURATOR_STATUSES:
            table(f"curator {status}", lambda m, status=status: _num((m["curator"]["statuses"] or {}).get(status), "d"))
        for result in PROPOSAL_RESULTS:
            table(f"proposals {result}",
                  lambda m, result=result: _num((m["curator"]["proposal_outcomes"] or {}).get(result), "d"))
        for kind in ("nonempty", "empty", "unknown"):
            table(f"kb snippet pages {kind}", lambda m, kind=kind: str(m["kb_reach"]["snippet_pages"][kind]))
        table("kb lookups found", lambda m: _num((m["kb_reach"]["lookups"] or {}).get("found"), "d"))
        table("kb lookups not found", lambda m: _num((m["kb_reach"]["lookups"] or {}).get("not_found"), "d"))
        table("kb items reached", lambda m: _num(m["kb_reach"]["items_reached"], "d"))
    spend = report["spend"]
    lines.append(_row("spend, all runs", [_usd(spend[c]["total_usd"]) for c in configs]))
    lines.append(_row("  runs (failed)", [f"{spend[c]['runs']} ({spend[c]['failed_runs']})" for c in configs]))

    for book, by_config in report["by_book"].items():
        lines += ["", f"{book}  ({len(common[book])} common pages)", _row("", configs)]
        lines.append(_row("chrF", [_num(by_config[c]["chrf"]["score"], ".2f") for c in configs]))
        lines.append(_row("COMET (wmt22-comet-da)", [_num(by_config[c]["comet"]["mean"], ".4f") for c in configs]))
        lines.append(_row("term accuracy, primary", [_term_cell(by_config[c]["terms"]["primary"]) for c in configs]))

    lines.append("")
    left_out = [(config, entry) for config in configs for entry in report["left_out"][config]]
    if left_out:
        lines.append("pages left out of the numbers (not scored by every config):")
        for config, entry in left_out:
            error = f": {entry['error']}" if entry.get("error") else ""
            lines.append(f"  {config}: {entry['book']}/{entry['page_index']} {entry['status']}{error}")
    else:
        lines.append("pages left out of the numbers: none")
    signatures = {(m["chrf"]["signature"], m["bleu"]["signature"]) for m in pooled.values()}
    for chrf, bleu in sorted(signatures, key=str):
        lines.append(f"sacreBLEU  chrF {chrf}   BLEU {bleu}")
    if report["warnings"]:
        lines += ["", "warnings:", *(f"  {w}" for w in report["warnings"])]
    else:
        lines += ["", "warnings: none"]
    return _layout(lines)
