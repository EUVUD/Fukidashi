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
- **Term consistency is optional, but all or none** (OUTLINE 4.7). When every run has a `consistency-<judge model>.json`
  (eval/consistency.py) from the same judge model and rules, each config gets a `consistency` section: the Translations'
  consistency (the sums of the most common renderings' counts over the sums of the rendered occurrences, of the entries
  with 2 or more rendered occurrences on the common pages, both books together), the reference's, the extractor's
  accuracy and whether consistency is reportable, and the occurrences by status. report.json then also says that term
  accuracy is the KB ablation's primary metric and consistency its secondary one (OUTLINE 4.3, 8). When no run has
  such a file the report is what it was before; when only some have one, the report refuses.
- **GEMBA-MQM is optional too, and all or none, from one judge model** (OUTLINE 4.4, 4.5). When every run has an
  `mqm-<judge model>.json` (eval/mqm.py) from the same judge model, prompt and weights (with `mqm_judge`, the one of
  that model; without it, a run must have only one), each config gets an `mqm` section over the common pages: the mean
  MQM score of the scored segments, the failed and skipped segments, the error breakdown by severity, by category and by
  both (every error the judge marked, as the scorer counts them), and up to five segment keys for each category. The
  report's `mqm` section names the judge and says whether the breakdown is the main error analysis: only when an
  agreement file (eval/mqm_agreement.py, `mqm_agreement`) of the same judge model says the judge passed; otherwise it is
  for reference only. When no run has an MQM file the report is what it was before; when only some have one, or the
  files differ in judge model, prompt or weights, the report refuses.
- **Warnings, not refusals,** for provenance: a dirty tree, runs from different commits, different models.
  Everything that would make the numbers wrong is a refusal (`Refused`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
from collections import Counter
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
# Term consistency (eval/consistency.py). These repeat evaluation/consistency.py's rules, which this module does not
# import either; a test pins each to its source, and another checks that the report and the scorer pool one run alike.
CONSISTENCY_GLOB = "consistency-*.json"
CONSISTENCY_TEXTS = ("translation", "reference")
CONSISTENCY_STATUSES = ("rendered", "omitted", "not_this_term", "invalid")
CONSISTENCY_MIN_RENDERED = 2  # an entry counts in the pooled consistency with this many rendered occurrences
CONSISTENCY_REPORTABLE_PERCENT = 90  # the extractor's accuracy at which consistency is a result
KB_ABLATION = {"comparison": "full vs full_no_kb", "primary_metric": "term accuracy (terms.primary)",
               "secondary_metric": "term consistency (consistency.translation)"}  # OUTLINE 4.3, 8
# GEMBA-MQM (eval/mqm.py) and its agreement check (eval/mqm_agreement.py). These repeat evaluation/mqm.py's and
# evaluation/mqm_scorer.py's names, which this module does not import; a test pins each to its source.
MQM_GLOB = "mqm-*.json"
MQM_SEVERITIES = ("critical", "major", "minor")
MQM_STATUSES = ("scored", "skipped", "failed")
MQM_WORST = -25  # a segment's MQM score is between this and 0
MQM_EXAMPLES = 5  # segment keys kept for each error category
MQM_MAIN, MQM_REFERENCE = "main error analysis", "for reference only"  # OUTLINE 4.5


class Refused(Exception):
    """The runs cannot be pooled; the message names the directory and the cause. Nothing has been written."""


@dataclass(frozen=True)
class ConsistencyOccurrence:
    """One occurrence of a consistency file, as far as the report pools it."""

    page_index: int
    statuses: dict[str, str]  # by text (translation, reference)
    renderings: dict[str, str | None]  # by text: the normalised rendering of a `rendered` occurrence
    checked: bool  # the reference holds the target or a variant: the extractor's answer is known
    correct: bool  # the reference's extraction is correct (False where it is not checked)


@dataclass(frozen=True)
class Consistency:
    """A run's consistency-<model>.json: its name, the file without its entries, and each entry's occurrences."""

    name: str
    document: dict
    entries: list[tuple[str, list[ConsistencyOccurrence]]]  # (entry id, its occurrences)

    @property
    def judge_model(self) -> Any:
        return self.document["judge"]["model"]


@dataclass(frozen=True)
class MQM:
    """A run's mqm-<model>.json: its name, the file without its segments, and each segment's judgement."""

    name: str
    document: dict
    segments: dict[tuple[str, int, int], dict]  # (book, page_index, idx) -> {"status", "score", "errors"}

    @property
    def judge_model(self) -> str:
        return self.document["judge"]["model"]


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
    consistency: Consistency | None = None  # the run's consistency file, if it has one
    segments_sha256: str = ""  # of segments.jsonl
    mqm_names: tuple[str, ...] = ()  # the run's MQM files, by name
    mqm: MQM | None = None  # the MQM file the report pools, once build_report has chosen it


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
    glossary_sha256 = file_sha256(glossary_path)
    if terms.get("glossary_sha256") != glossary_sha256:
        raise Refused(f"{directory}: terms.json was scored against another {glossary_path.name} than the one there "
                      "now (its glossary_sha256 differs): score the run again with eval/terms.py")
    consistency = _consistency_file(directory, eval_id, book, segments_sha256, glossary_path.name, glossary_sha256)

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
               {k: v for k, v in terms.items() if k != "entries"}, consistency, segments_sha256,
               tuple(sorted(p.name for p in directory.glob(MQM_GLOB) if p.is_file())))


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


def _consistency_file(directory: Path, eval_id: str, book: str, segments_sha256: str, glossary_name: str,
                      glossary_sha256: str) -> Consistency | None:
    """The run's consistency file, checked against the run; None when it has none."""
    found = sorted(p.name for p in directory.glob(CONSISTENCY_GLOB) if p.is_file())
    if not found:
        return None
    if len(found) > 1:
        raise Refused(f"{directory}: it has more than one consistency file ({', '.join(found)}); the report pools one "
                      "judge's renderings")
    name = found[0]
    document = _json_file(directory, name)
    if document.get("eval_id") != eval_id:
        raise Refused(f"{directory}: {name} is of eval run {document.get('eval_id')!r}, the manifest's is {eval_id!r}")
    if document.get("book") != book:
        raise Refused(f"{directory}: {name} is of book {document.get('book')!r}, the run is of {book!r}")
    if document.get("segments_sha256") != segments_sha256:
        raise Refused(f"{directory}: {name} scored another segments.jsonl than the one there now (its segments_sha256 "
                      "differs): score the run again with eval/consistency.py")
    if document.get("glossary_sha256") != glossary_sha256:
        raise Refused(f"{directory}: {name} was scored against another {glossary_name} than the one there now (its "
                      "glossary_sha256 differs): score the run again with eval/consistency.py")
    judge = document.get("judge")
    if not isinstance(judge, dict) or not isinstance(judge.get("model"), str) or not judge["model"] \
            or not isinstance(document.get("rules"), str):
        raise Refused(f"{directory}: {name} has no judge `model` and `rules`")
    return Consistency(name, {k: v for k, v in document.items() if k != "entries"},
                       _consistency_entries(directory, name, document))


def _consistency_entries(directory: Path, name: str, document: dict) -> list[tuple[str, list[ConsistencyOccurrence]]]:
    def occurrence(segment: dict) -> ConsistencyOccurrence:
        statuses, renderings = {}, {}
        for which in CONSISTENCY_TEXTS:
            status, rendering = segment[which]["status"], segment[which]["normalised"]
            if status not in CONSISTENCY_STATUSES:
                raise ValueError(f"unknown status {status!r}")
            if status == "rendered" and not (isinstance(rendering, str) and rendering):
                raise ValueError("a rendered occurrence without its normalised rendering")
            statuses[which], renderings[which] = status, rendering if status == "rendered" else None
        checked, correct = segment["extractor"]["checked"], segment["extractor"]["correct"]
        if not isinstance(checked, bool) or (not isinstance(correct, bool) if checked else correct is not None):
            raise ValueError("an extractor check that is not true or false where it is checked, and null elsewhere")
        if not _is_int(segment["page_index"]):
            raise ValueError("a page_index that is not an integer")
        return ConsistencyOccurrence(segment["page_index"], statuses, renderings, checked, bool(correct))

    try:
        return [(entry["id"], [occurrence(s) for s in entry["segments"]]) for entry in document["entries"]]
    except (KeyError, TypeError, ValueError) as e:
        raise Refused(f"{directory}: {name} is not shaped as eval/consistency.py writes it ({type(e).__name__}: {e})")


def _bare_model(model: str) -> str:
    """The model without an organisation prefix, as eval/mqm.py names its file ("openai/gpt-4o-mini" -> "gpt-4o-mini")."""
    return model.strip().rsplit("/", 1)[-1]


def _mqm_file(run: Run, name: str) -> MQM:
    """The run's MQM file `name`, checked against the run."""
    directory = run.directory
    document = _json_file(directory, name)
    if document.get("eval_id") != run.eval_id:
        raise Refused(f"{directory}: {name} is of eval run {document.get('eval_id')!r}, the manifest's is {run.eval_id!r}")
    if document.get("book") != run.book:
        raise Refused(f"{directory}: {name} is of book {document.get('book')!r}, the run is of {run.book!r}")
    if document.get("segments_sha256") != run.segments_sha256:
        raise Refused(f"{directory}: {name} scored another segments.jsonl than the one there now (its segments_sha256 "
                      "differs): score the run again with eval/mqm.py")
    judge = document.get("judge")
    if not isinstance(judge, dict) or not isinstance(judge.get("model"), str) or not judge["model"].strip():
        raise Refused(f"{directory}: {name} has no judge `model`")
    if f"mqm-{_bare_model(judge['model'])}.json" != name:
        raise Refused(f"{directory}: {name} holds the judgements of {judge['model']!r}, whose file has another name")

    def judgement(segment: Any) -> tuple[tuple[str, int, int], dict]:
        if not isinstance(segment, dict) or not isinstance(segment.get("book"), str) \
                or not _is_int(segment.get("page_index")) or not _is_int(segment.get("idx")):
            raise ValueError("a segment without book, page_index and idx")
        status, score, errors = segment["status"], segment["score"], segment["errors"]
        if status not in MQM_STATUSES:
            raise ValueError(f"unknown status {status!r}")
        if not isinstance(errors, list) or not all(
                isinstance(e, dict) and e.get("severity") in MQM_SEVERITIES and isinstance(e.get("category"), str)
                for e in errors):
            raise ValueError("errors that are not a list of severities and categories")
        if status == "scored" and not (_is_finite(score) and MQM_WORST <= score <= 0):
            raise ValueError(f"a scored segment whose score is not a number between {MQM_WORST} and 0")
        if status != "scored" and (score is not None or errors):
            raise ValueError(f"a {status} segment with a score or errors")
        return ((segment["book"], segment["page_index"], segment["idx"]),
                {"status": status, "score": score, "errors": [(e["severity"], e["category"]) for e in errors]})

    segments: dict[tuple[str, int, int], dict] = {}
    try:
        if not isinstance(document["segments"], list):
            raise ValueError("`segments` is not a list")
        for segment in document["segments"]:
            key, value = judgement(segment)
            if key in segments:
                raise ValueError(f"segment {key[0]}/{key[1]} idx {key[2]} judged twice")
            segments[key] = value
    except (KeyError, TypeError, ValueError) as e:
        raise Refused(f"{directory}: {name} is not shaped as eval/mqm.py writes it ({type(e).__name__}: {e})")
    wanted = {(s["book"], s["page_index"], s["idx"]) for s in run.segments}
    for key in sorted(wanted - set(segments)):
        raise Refused(f"{directory}: {name} has no judgement for segment {key[0]}/{key[1]} idx {key[2]}")
    for key in sorted(set(segments) - wanted):
        raise Refused(f"{directory}: {name} judges segment {key[0]}/{key[1]} idx {key[2]}, which segments.jsonl does "
                      "not have")
    return MQM(name, {k: v for k, v in document.items() if k != "segments"}, segments)


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


def _check_consistency(runs: Sequence[Run]) -> dict | None:
    """All or none: what every run's consistency file shares (judge model, rules), or None when no run has one."""
    having = [r for r in runs if r.consistency is not None]
    if not having:
        return None
    first = having[0]
    for run in runs:
        if run.consistency is None:
            raise Refused(f"{run.directory}: it has no {CONSISTENCY_GLOB}, but {first.directory} has "
                          f"{first.consistency.name}: the report pools term consistency for every run or for none "
                          "(score this one with eval/consistency.py)")
    for key, label in (("model", "judge model"), ("rules", "rules")):
        def value(run: Run):
            return run.consistency.judge_model if key == "model" else run.consistency.document["rules"]

        for run in runs:
            if value(run) != value(first):
                raise Refused(f"{run.directory}: {run.consistency.name} has another {label} ({value(run)!r}) than "
                              f"{first.directory} ({value(first)!r}): the report pools one judge's renderings under "
                              "one set of rules")
    return {"judge_model": first.consistency.judge_model, "rules": first.consistency.document["rules"],
            "min_rendered": CONSISTENCY_MIN_RENDERED, "threshold": CONSISTENCY_REPORTABLE_PERCENT / 100}


def _select_mqm(runs: Sequence[Run], judge: str | None) -> list[Run]:
    """All or none, from one judge model: the runs with the MQM file the report pools set (`Run.mqm`), or the runs as
    they are when none has an MQM file and no `judge` is asked for."""
    if judge is not None:
        name = f"mqm-{_bare_model(judge)}.json"
        for run in runs:
            if name not in run.mqm_names:
                raise Refused(f"{run.directory}: it has no {name} (it has {', '.join(run.mqm_names) or 'no MQM file'}): "
                              "the report pools GEMBA-MQM from one judge for every run (score this one with "
                              "eval/mqm.py)")
        chosen = [dataclasses.replace(run, mqm=_mqm_file(run, name)) for run in runs]
    else:
        having = [r for r in runs if r.mqm_names]
        if not having:
            return list(runs)
        first = having[0]
        for run in runs:
            if not run.mqm_names:
                raise Refused(f"{run.directory}: it has no {MQM_GLOB}, but {first.directory} has {first.mqm_names[0]}: "
                              "the report pools GEMBA-MQM for every run or for none (score this one with eval/mqm.py)")
            if len(run.mqm_names) > 1:
                raise Refused(f"{run.directory}: it has more than one MQM file ({', '.join(run.mqm_names)}): name the "
                              "judge whose files the report pools (--mqm-judge)")
        chosen = [dataclasses.replace(run, mqm=_mqm_file(run, run.mqm_names[0])) for run in runs]
    first = chosen[0]
    for run in chosen[1:]:
        if _bare_model(run.mqm.judge_model) != _bare_model(first.mqm.judge_model):
            raise Refused(f"{run.directory}: {run.mqm.name} is by another judge model ({run.mqm.judge_model!r}) than "
                          f"{first.directory} ({first.mqm.judge_model!r}): the report pools one judge's MQM scores "
                          "(name it with --mqm-judge)")
        for key, label in (("gemba", "GEMBA prompt or commit"), ("weights", "weights"), ("score_rule", "score rule")):
            mine, theirs = run.mqm.document.get(key), first.mqm.document.get(key)
            if key == "gemba":
                mine, theirs = ({k: (d or {}).get(k) for k in ("commit", "template_sha256")} for d in (mine, theirs))
            if mine != theirs:
                raise Refused(f"{run.directory}: {run.mqm.name} has another {label} than {first.directory}: the report "
                              "pools MQM scores of one prompt and one set of weights")
    return chosen


def _mqm_header(runs: Sequence[Run], agreement_path: Path | None) -> dict | None:
    """The report's `mqm` section: the judge, the prompt, and whether its breakdown is the main error analysis."""
    if not runs or runs[0].mqm is None:
        if agreement_path is not None:
            raise Refused(f"{agreement_path}: an agreement file is given, but no run has an MQM file")
        return None
    document = runs[0].mqm.document
    model = runs[0].mqm.judge_model
    agreement = None if agreement_path is None else _agreement(agreement_path, model)
    if agreement is None:
        use, why = MQM_REFERENCE, "no agreement check (eval/mqm_agreement.py) was given for this judge"
    elif agreement["pass"]:
        use, why = MQM_MAIN, "the judge passed the agreement check (OUTLINE 4.5)"
    else:
        use, why = MQM_REFERENCE, "the judge failed the agreement check (OUTLINE 4.5)"
    return {"judge_model": model, "file": runs[0].mqm.name,
            "gemba": {k: (document.get("gemba") or {}).get(k) for k in ("repository", "commit", "template_sha256")},
            "weights": document.get("weights"), "score_rule": document.get("score_rule"),
            "agreement": agreement, "use": use, "why": why}


def _agreement(path: Path, model: str) -> dict:
    """An agreement file (eval/mqm_agreement.py), as far as the report quotes it; it must be the pooled judge's."""
    if not path.is_file():
        raise Refused(f"{path}: the agreement file is missing")
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
        judge, decision = document["judge"]["model"], document["decision"]
        summary = {
            "file": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "judge_model": judge,
            "pass": decision["pass"], "decided_by": decision["decided_by"],
            "tau_b": document["tau_b"]["value"], "recall": document["recall"]["value"],
            "items_with_major_or_critical": document["recall"]["items_with_major_or_critical"],
            "thresholds": document["thresholds"], "n_items": document["n_items"],
            "annotator": {"kind": document["annotator"]["kind"], "note": document["annotator"]["note"]},
            "sheet_sha256": document["sheet"]["sha256"], "computed_at": document["computed_at"]}
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise Refused(f"{path}: it is not an agreement file as eval/mqm_agreement.py writes it ({type(e).__name__}: {e})")
    if not isinstance(summary["pass"], bool) or not isinstance(judge, str):
        raise Refused(f"{path}: it is not an agreement file as eval/mqm_agreement.py writes it (no decision)")
    if _bare_model(judge) != _bare_model(model):
        raise Refused(f"{path}: the agreement check is of judge {judge!r}, the MQM files are by {model!r}")
    return summary


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
    if all(r.consistency is not None for r in runs):  # all or none, which build_report has checked
        metrics["consistency"] = _consistency(runs, wanted)
    if all(r.mqm is not None for r in runs):  # all or none, too
        metrics["mqm"] = _mqm(runs, wanted)
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


def _consistency(runs: Sequence[Run], wanted: dict[str, set[int]]) -> dict:
    """Term consistency over the common pages, pooled as eval/consistency.py pools one book: per entry and text, the
    most common rendering's count over the rendered occurrences; the entries with CONSISTENCY_MIN_RENDERED or more
    rendered occurrences are summed, over every book of `runs`. An entry's renderings are counted within its book."""
    texts = {which: {"entries": 0, "rendered": 0, "most_common_count": 0} for which in CONSISTENCY_TEXTS}
    statuses = {which: dict.fromkeys(CONSISTENCY_STATUSES, 0) for which in CONSISTENCY_TEXTS}
    checked = correct = 0
    for run in runs:
        for _, occurrences in run.consistency.entries:
            kept = [o for o in occurrences if o.page_index in wanted[run.book]]
            for which in CONSISTENCY_TEXTS:
                for o in kept:
                    statuses[which][o.statuses[which]] += 1
                renderings = Counter(o.renderings[which] for o in kept if o.statuses[which] == "rendered")
                rendered = sum(renderings.values())
                if rendered >= CONSISTENCY_MIN_RENDERED:
                    texts[which]["entries"] += 1
                    texts[which]["rendered"] += rendered
                    texts[which]["most_common_count"] += max(renderings.values())
            checked += sum(o.checked for o in kept)
            correct += sum(o.checked and o.correct for o in kept)
    for score in texts.values():
        score["consistency"] = score["most_common_count"] / score["rendered"] if score["rendered"] else None
    return {
        **texts,
        "statuses": statuses,
        "extractor": {"checked": checked, "correct": correct, "accuracy": correct / checked if checked else None,
                      "reportable": checked > 0 and correct * 100 >= checked * CONSISTENCY_REPORTABLE_PERCENT},
    }


def _mqm(runs: Sequence[Run], wanted: dict[str, set[int]]) -> dict:
    """GEMBA-MQM over the segments of the common pages: the mean score of the scored ones, the failed and skipped ones,
    and the errors of the scored ones counted as the scorer counts them (evaluation/mqm.py `breakdown`), with up to
    MQM_EXAMPLES segment keys for each category, in the order of the books and of segments.jsonl."""
    judged = [({"book": s["book"], "page_index": s["page_index"], "idx": s["idx"]},
               r.mqm.segments[(s["book"], s["page_index"], s["idx"])])
              for r in runs for s in r.segments if s["page_index"] in wanted[r.book]]
    scored = [(key, j) for key, j in judged if j["status"] == "scored"]
    by_severity = dict.fromkeys(MQM_SEVERITIES, 0)
    by_category: Counter[str] = Counter()
    both: dict[str, dict[str, int]] = {}
    examples: dict[str, list[dict]] = {}
    for key, j in scored:
        for severity, category in j["errors"]:
            by_severity[severity] += 1
            by_category[category] += 1
            both.setdefault(category, dict.fromkeys(MQM_SEVERITIES, 0))[severity] += 1
            kept = examples.setdefault(category, [])
            if key not in kept and len(kept) < MQM_EXAMPLES:
                kept.append(key)
    order = sorted(by_category, key=lambda c: (-by_category[c], c))
    scores = [j["score"] for _, j in scored]
    return {
        "segments": len(judged),
        "scored": len(scored),
        "failed": {"count": sum(j["status"] == "failed" for _, j in judged),
                   "segments": [key for key, j in judged if j["status"] == "failed"]},
        "skipped": {"count": sum(j["status"] == "skipped" for _, j in judged),
                    "segments": [key for key, j in judged if j["status"] == "skipped"]},
        "score": math.fsum(scores) / len(scores) if scores else None,
        "breakdown": {
            "errors": sum(by_severity.values()),
            "segments_with_errors": sum(1 for _, j in scored if j["errors"]),
            "by_severity": by_severity,
            "by_category": {c: by_category[c] for c in order},
            "by_category_and_severity": {c: both[c] for c in order},
        },
        "examples": {c: examples[c] for c in order},
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
    consistency = {} if run.consistency is None else {"consistency": {
        "file": run.consistency.name,
        **{key: run.consistency.document.get(key) for key in ("rules", "glossary_sha256", "segments_sha256")},
        **{key: run.consistency.document["judge"].get(key) for key in ("prefix", "model", "response_model")},
        "usage": run.consistency.document.get("usage")}}
    judged = {} if run.mqm is None else {"mqm": {
        "file": run.mqm.name,
        **{key: run.mqm.document.get(key) for key in ("segments_sha256", "scored_at")},
        **{key: run.mqm.document["judge"].get(key) for key in ("prefix", "model", "response_model", "temperature",
                                                                "reasoning_effort")},
        "usage": run.mqm.document.get("usage")}}
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
        **consistency,
        **judged,
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
    responses = [(run, json.dumps(run.consistency.document["judge"].get("response_model")))
                 for run in runs if run.consistency is not None]
    if len({response for _, response in responses}) > 1:
        warnings.append("the consistency files' judge responses came from different models (or some do not say): "
                        + ", ".join(f"{run.eval_id} {response}" for run, response in responses))
    judged = [(run, json.dumps(run.mqm.document["judge"].get("response_model")))
              for run in runs if run.mqm is not None]
    if len({response for _, response in judged}) > 1:
        warnings.append("the MQM files' judge responses came from different models (or some do not say): "
                        + ", ".join(f"{run.eval_id} {response}" for run, response in judged))
    return warnings


# --- the report --------------------------------------------------------------------------------------------------------

def build_report(directories: Sequence[Path], glossary_dir: Path, *, mqm_judge: str | None = None,
                 mqm_agreement: Path | None = None) -> dict:
    """The report of the eval runs in `directories` (what report.json holds). `mqm_judge` names the judge model whose
    MQM files are pooled (needed only when a run has files of more than one); `mqm_agreement` is that judge's agreement
    file (eval/mqm_agreement.py). Raises `Refused`, having written nothing."""
    runs = [load_run(Path(d), glossary_dir) for d in directories]
    _check_set(runs)
    consistency = _check_consistency(runs)
    runs = _select_mqm(runs, mqm_judge)
    mqm = _mqm_header(runs, None if mqm_agreement is None else Path(mqm_agreement))
    runs.sort(key=lambda r: (_config_order(r.config), _book_order(r.book)))
    configs = list(dict.fromkeys(r.config for r in runs))
    common = _common_pages(runs)
    by_config = {config: [r for r in runs if r.config == config] for config in configs}
    report = {
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
    if consistency is not None:  # without consistency files the report is what it was before
        report["consistency"] = consistency
        report["kb_ablation"] = dict(KB_ABLATION)
    if mqm is not None:  # and without MQM files, too
        report["mqm"] = mqm
    return report


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


def _ratio_cell(score: dict) -> str:
    return f"{_pct(score['consistency'])} ({score['most_common_count']}/{score['rendered']})"


def _consistency_heading(report: dict) -> str:
    c = report["consistency"]
    return (f"term consistency, the KB ablation's secondary metric (term accuracy is its primary one); renderings by "
            f"{c['judge_model']}, entries with {c['min_rendered']}+ rendered occurrences, reportable when the "
            f"extractor is right on {c['threshold']:.0%} or more:")


def _consistency_line(config: str, pooled: dict, by_book: dict[str, dict]) -> str:
    """One config's consistency on one line: pooled and per book, the reference's, the extractor and the statuses."""
    books = ", ".join(f"{book} {_ratio_cell(c['translation'])}" for book, c in by_book.items())
    extractor = pooled["extractor"]
    verdict = "reportable" if extractor["reportable"] else "for reference only"
    statuses = "; ".join(f"{which} " + " ".join(f"{s} {n}" for s, n in pooled["statuses"][which].items())
                         for which in CONSISTENCY_TEXTS)
    return (f"  {config}: {_ratio_cell(pooled['translation'])} [{books}]; reference {_ratio_cell(pooled['reference'])};"
            f" extractor {_pct(extractor['accuracy'])} ({extractor['correct']}/{extractor['checked']}), {verdict};"
            f" {statuses}")


def _mqm_heading(report: dict) -> list[str]:
    m = report["mqm"]
    lines = [f"GEMBA-MQM by {m['judge_model']}, the mean MQM score of the scored segments (0 is no error, "
             f"{MQM_WORST} the worst): {m['use']}, because {m['why']}"]
    a = m["agreement"]
    if a is not None:
        tau = "undefined" if a["tau_b"] is None else f"{a['tau_b']:.4f}"
        recall = "n/a" if a["recall"] is None else f"{a['recall']:.1%}"
        lines.append(f"  agreement check: {'PASS' if a['pass'] else 'FAIL'} ({a['decided_by']}); tau_b {tau}, recall "
                     f"{recall} over {a['n_items']} items; {a['annotator']['note']}")
    return lines


def _mqm_line(config: str, pooled: dict, by_book: dict[str, dict]) -> str:
    """One config's MQM on one line: pooled and per book, the segments, the errors and the most frequent categories."""
    books = ", ".join(f"{book} {_num(m['score'], '.3f')}" for book, m in by_book.items())
    b = pooled["breakdown"]
    severities = ", ".join(f"{s} {n}" for s, n in b["by_severity"].items())
    top = ", ".join(f"{c} {n}" for c, n in list(b["by_category"].items())[:5]) or "none"
    return (f"  {config}: {_num(pooled['score'], '.3f')} [{books}] over {pooled['scored']} scored segments (failed "
            f"{pooled['failed']['count']}, skipped {pooled['skipped']['count']}); errors {b['errors']} ({severities}) "
            f"in {b['segments_with_errors']} segments; most frequent: {top}")


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
    if "consistency" in report:
        lines += ["", _consistency_heading(report)]
        lines += [_consistency_line(config, pooled[config]["consistency"],
                                    {book: report["by_book"][book][config]["consistency"] for book in report["books"]})
                  for config in configs]
    if "mqm" in report:
        lines += ["", *_mqm_heading(report)]
        lines += [_mqm_line(config, pooled[config]["mqm"],
                            {book: report["by_book"][book][config]["mqm"] for book in report["books"]})
                  for config in configs]

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
