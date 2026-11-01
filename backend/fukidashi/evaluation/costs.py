"""Cost and latency of finished eval runs, from the translation ledger (OUTLINE 4, 8; eval/PROTOCOL.md, "Statistics").

The inputs are the eval runs that evaluation/report.py pools and, for each run, the range of lines of the translation
ledger (`runs/usage.jsonl`) that its calls wrote. Cost and latency are described, with no interval. The command-line
entry point is eval/costs.py; the logic lives here so that tests can import it. It reads files only: no database, no
model, no network.

- **The runs** are read and refused by the report's own code (`report.build_report` runs first, so the same inputs are
  refused with the same messages), and the common pages come from the report.
- **A run's lines.** The ledger has one line per call and no timestamp or run id, so each run is given its range: the
  first and last line, counted from 1, both included. The range must split into maximal runs of one `page_id`, one for
  each page the eval run attempted (scored or failed), in the manifest's order and named `<book>/<page_index:03d>`; each
  scored page's run must be as long as its `calls` in scores.json; every line must have been made with one of the
  manifest's requested models; and no two ranges may overlap. Anything else is refused: a wrong range would put another
  run's calls into the numbers. A failed page has no `calls` in scores.json, so its calls are the lines of its run.
- **The prices** are recovered from the lines, never read from `.env`: the price per input token and per output token that
  fit `cost_usd = input_tokens * a + output_tokens * b` over the lines of the given ranges that have a cost. The fit is
  exact, or the tool refuses: no line may be off by more than `RESIDUAL_TOLERANCE`, relatively.
- **Two sets of pages,** each figure computed on both. `common`: the pages every config scored, as the report has them
  (its quality numbers and its cost and latency rows). `attempted`: every page a run attempted, the failed ones
  included, so their calls count. Both are given for each config, pooled over the books and for each book.
- **Cost is a range.** `LLMClient._cost` prices every input token at the input price, although the provider may bill
  cached input lower, so the ledger's cost is an upper end. The lower end prices cached input at $0. Given a price for
  cached input (`cached_input_price_per_m`), a value at that price is added. The tokens are split into uncached input
  (`input_tokens` minus `cached_input_tokens`), cached input and output.
- **Latency** is each page's `latency_s` of scores.json (the wall time of the page, whatever its calls were), summarised
  by `metrics.latency_summary` (p50, p95) and the mean. A failed page's latency is its row's in `failed_pages`.
- **Unknown is not 0.** A line without usage has no tokens and no cost; a line with usage and no price has no cost. A
  figure that needs what a line lacks is `None` for every scope that has the line, as `cost_summary` makes a cost `None`
  and never a partial sum; the scope lists the line numbers. A run's own totals (`tokens_of_lines_with_usage`,
  `cost_usd_of_lines_with_cost`) are the sums over the lines that have them, named so, beside the lines that do not.
  Usage without a `cached_input_tokens` count is refused: its split is unknown and the tool does not guess it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from itertools import groupby
from pathlib import Path

from ..config import ROOT
from . import report
from .manifest import git_state
from .metrics import latency_summary

Refused = report.Refused  # the tool refuses what the report refuses, and a few things more

ATTEMPTED = ("scored", "failed")  # the manifest statuses of the pages that made calls
SCOPES = ("common", "attempted")  # the two sets of pages a figure is computed on
RESIDUAL_TOLERANCE = 1e-9  # the price fit is exact: no line may be off by more than this, relatively
PER_M = 1_000_000  # tokens
RANGE = re.compile(r"(?P<eval_id>[^=\s]+)=(?P<first>[0-9]+)-(?P<last>[0-9]+)")  # --ledger-range <eval id>=<first>-<last>
USAGE_COUNTS = ("input_tokens", "output_tokens", "cached_input_tokens")


@dataclass(frozen=True)
class Line:
    """One call of the translation ledger."""

    number: int  # of the line in the ledger, from 1
    page_id: str
    requested_model: str
    input_tokens: int | None  # None for all three counts: the provider returned no usage (unknown, not 0)
    cached_input_tokens: int | None  # a part of input_tokens
    output_tokens: int | None
    cost_usd: float | None  # None: no usage, or no price when the call was made
    error: str | None


@dataclass(frozen=True)
class PageCalls:
    """The calls one eval run made for one page it attempted, with the page's row of scores.json."""

    page_index: int
    status: str  # "scored" or "failed"
    lines: tuple[Line, ...]
    latency_s: float
    error: str | None  # of a failed page


@dataclass(frozen=True)
class RunCalls:
    """One eval run and the ledger lines it wrote."""

    run: report.Run
    first: int
    last: int
    lines_sha256: str  # of the range's lines, each with its newline: `sed -n 'F,Lp' usage.jsonl | shasum -a 256`
    pages: tuple[PageCalls, ...]  # the attempted pages, in the manifest's order


@dataclass(frozen=True)
class Prices:
    """What the ledger's costs say a token costs."""

    input_per_token: float
    output_per_token: float
    lines: int  # the lines it was fitted on
    max_relative_residual: float
    worst_line: int  # the line with that residual


# --- the ranges and the ledger ------------------------------------------------------------------------------------------

def parse_ranges(options: Sequence[str]) -> dict[str, tuple[int, int]]:
    """`--ledger-range` options, `<eval id>=<first>-<last>` (lines counted from 1, both included), as {eval id: (first,
    last)}. Raises `Refused` for a form that is not that, a range that is empty, or an eval run given two ranges."""
    ranges: dict[str, tuple[int, int]] = {}
    for option in options:
        found = RANGE.fullmatch(option.strip())
        if found is None:
            raise Refused(f"{option!r} is not a ledger range: write it as <eval id>=<first line>-<last line>")
        eval_id, first, last = found["eval_id"], int(found["first"]), int(found["last"])
        if not 1 <= first <= last:
            raise Refused(f"the ledger range {option!r} is empty or starts before line 1 (lines are counted from 1)")
        if eval_id in ranges:
            raise Refused(f"{eval_id} has two ledger ranges ({ranges[eval_id][0]}-{ranges[eval_id][1]} and "
                          f"{first}-{last}): give one for each eval run")
        ranges[eval_id] = (first, last)
    return ranges


def read_ledger(path: Path) -> list[str]:
    """The ledger's lines. Split on "\\n" only: an error message may hold U+2028 or U+0085, which `splitlines()` would cut
    at, and the line numbers of the ranges are those of `sed -n` and `wc -l`."""
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except OSError as e:
        raise Refused(f"{path}: the ledger cannot be read: {e}")
    except UnicodeDecodeError as e:
        raise Refused(f"{path}: the ledger is not valid UTF-8: {e}")
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()  # the file ends with a newline
    return lines


def parse_line(number: int, text: str, ledger: Path) -> Line:
    """One line of a range, checked for what the tool needs. Raises `Refused` naming the line."""

    def bad(why: str) -> Refused:
        return Refused(f"{ledger}: line {number} {why}")

    try:
        data = json.loads(text)
    except ValueError as e:
        raise bad(f"is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise bad("is not a JSON object")
    page_id, model, usage, cost, error = (data.get(key) for key in ("page_id", "requested_model", "usage", "cost_usd",
                                                                    "error"))
    if not (isinstance(page_id, str) and page_id and isinstance(model, str) and model):
        raise bad("has no `page_id` and `requested_model`")
    if error is not None and not isinstance(error, str):
        raise bad("has an `error` that is not text")
    if cost is not None and not (report._is_finite(cost) and cost >= 0):
        raise bad("has a `cost_usd` that is neither a non-negative number nor null")
    if usage is None:
        if cost is not None:
            raise bad("has a cost but no usage")
        return Line(number, page_id, model, None, None, None, None, error)
    if not isinstance(usage, dict):
        raise bad("has a `usage` that is neither null nor an object")
    for key in USAGE_COUNTS:
        count = usage.get(key)
        if key == "cached_input_tokens" and count is None:
            raise bad("has usage without a `cached_input_tokens` count: its cached share is unknown and this tool does "
                      "not guess it")
        if not (report._is_int(count) and count >= 0):
            raise bad(f"has a `usage` whose `{key}` is not a non-negative whole number")
    if usage["cached_input_tokens"] > usage["input_tokens"]:
        raise bad("has more cached input tokens than input tokens")
    return Line(number, page_id, model, usage["input_tokens"], usage["cached_input_tokens"], usage["output_tokens"],
                cost, error)


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'s' * (n != 1)}"


def _span(first: int, last: int) -> str:
    return f"line {first}" if first == last else f"lines {first}-{last}"


def _check_overlaps(ranges: Mapping[str, tuple[int, int]]) -> None:
    ordered = sorted((first, last, eval_id) for eval_id, (first, last) in ranges.items())
    for (first, last, eval_id), (next_first, next_last, next_id) in zip(ordered, ordered[1:]):
        if next_first <= last:  # sorted by start: if any two overlap, two neighbours do
            raise Refused(f"the ledger ranges of {eval_id} ({first}-{last}) and {next_id} ({next_first}-{next_last}) "
                          "overlap: a call belongs to one eval run")


# --- one run's lines, split into its pages -------------------------------------------------------------------------------

def read_run_calls(run: report.Run, first: int, last: int, ledger: Sequence[str], ledger_path: Path) -> RunCalls:
    """The lines `first` to `last` of the ledger, split into the pages the eval run attempted. Raises `Refused`, naming
    the run, when they are not the lines of its calls: they must form one run of one `page_id` for each attempted page,
    in the manifest's order, as long as `calls` for a scored page, made with the models the run asked for."""
    where = str(run.directory)
    if not 1 <= first <= last:
        raise Refused(f"{where}: its ledger range {first}-{last} is empty or starts before line 1 (lines are counted "
                      "from 1)")
    if last > len(ledger):
        raise Refused(f"{where}: its ledger range {first}-{last} goes past the end of {ledger_path}, which has "
                      f"{len(ledger)} lines")
    models = run.manifest.get("requested_models")
    if not isinstance(models, list) or not models or not all(isinstance(m, str) and m for m in models):
        raise Refused(f"{where}: manifest.json has no `requested_models`, so the ledger's lines cannot be checked "
                      "against the model the run asked for")
    lines = [parse_line(n, ledger[n - 1], ledger_path) for n in range(first, last + 1)]
    for line in lines:
        if line.requested_model not in models:
            raise Refused(f"{where}: ledger line {line.number} was made with model {line.requested_model!r}, which the "
                          f"eval run did not ask for (manifest.json: {', '.join(models)}): the range holds another "
                          "run's calls")
    attempted = [(page, status) for page, status in run.statuses.items() if status in ATTEMPTED]
    groups = [list(group) for _, group in groupby(lines, key=lambda line: line.page_id)]
    if len(groups) != len(attempted):
        raise Refused(f"{where}: ledger lines {first}-{last} form {_plural(len(groups), 'run')} of one page_id, but the "
                      f"eval run attempted {_plural(len(attempted), 'page')} (scored or failed): the range is not this "
                      "run's calls")
    pages = []
    for position, (group, (page_index, status)) in enumerate(zip(groups, attempted), start=1):
        page_id = f"{run.book}/{page_index:03d}"
        where_lines = _span(group[0].number, group[-1].number)
        if group[0].page_id != page_id:
            raise Refused(f"{where}: page {page_index} is number {position} of the pages the eval run attempted, so its "
                          f"calls should be {page_id!r}, but the ledger's number {position} ({where_lines}) is "
                          f"{group[0].page_id!r}")
        if status == "scored":
            calls = run.rows[page_index].get("calls")
            if not report._is_int(calls):
                raise Refused(f"{where}: the row of page {page_index} in scores.json has no whole-number `calls`")
            if calls != len(group):
                raise Refused(f"{where}: page {page_index} made {_plural(calls, 'call')} (scores.json), but the ledger "
                              f"has {_plural(len(group), 'line')} for {page_id!r} ({where_lines})")
            row = run.rows[page_index]
        else:
            row = run.failed.get(page_index)
            if row is None:
                raise Refused(f"{where}: page {page_index} is failed in manifest.json but has no row in `failed_pages` "
                              "of scores.json")
        if not report._is_finite(row.get("latency_s")):
            raise Refused(f"{where}: the row of page {page_index} in scores.json has a `latency_s` that is not a finite "
                          "number")
        error = row.get("error") if status == "failed" else None
        pages.append(PageCalls(page_index, status, tuple(group), row["latency_s"], error))
    text = "".join(ledger[n - 1] + "\n" for n in range(first, last + 1))
    return RunCalls(run, first, last, hashlib.sha256(text.encode("utf-8")).hexdigest(), tuple(pages))


# --- the prices ---------------------------------------------------------------------------------------------------------

def fit_prices(lines: Sequence[Line], ledger: Path | None = None) -> Prices:
    """The price per input token and per output token that the lines with a cost fit exactly.

    A least squares fit of `cost_usd = input_tokens * a + output_tokens * b`, with no intercept, solved in exact
    fractions (the costs are floats, which fractions hold exactly), so that the result does not depend on rounding in
    the solution. Every priced line must then fit within `RESIDUAL_TOLERANCE`, relatively, or the fit is refused: the
    lines were not priced by one pair of prices. `ledger` only names the file in a refusal.
    """
    where = f"{ledger}: " if ledger is not None else ""
    priced = [line for line in lines if line.cost_usd is not None]
    if not priced:
        raise Refused(f"{where}no line of the given ledger ranges has a cost, so the prices cannot be recovered")
    sxx = sxy = syy = 0
    sxc = syc = Fraction(0)
    for line in priced:
        cost = Fraction(line.cost_usd)
        sxx += line.input_tokens ** 2
        sxy += line.input_tokens * line.output_tokens
        syy += line.output_tokens ** 2
        sxc += line.input_tokens * cost
        syc += line.output_tokens * cost
    determinant = sxx * syy - sxy * sxy
    if determinant == 0:
        raise Refused(f"{where}the prices cannot be recovered from {_plural(len(priced), 'priced line')}: their input "
                      "and output token counts are proportional, so one pair of prices is not told from another")
    price_in = float((sxc * syy - syc * sxy) / determinant)
    price_out = float((syc * sxx - sxc * sxy) / determinant)
    worst, worst_line, worst_predicted = 0.0, priced[0], 0.0
    for line in priced:
        predicted = line.input_tokens * price_in + line.output_tokens * price_out
        if line.cost_usd:
            residual = abs(predicted - line.cost_usd) / line.cost_usd
        else:
            residual = 0.0 if predicted == 0 else math.inf
        if residual > worst:
            worst, worst_line, worst_predicted = residual, line, predicted
    if worst > RESIDUAL_TOLERANCE:
        raise Refused(f"{where}the costs of the given ranges do not fit one pair of prices ({price_in * PER_M:.6g} and "
                      f"{price_out * PER_M:.6g} dollars per million input and output tokens): line {worst_line.number} "
                      f"costs {worst_line.cost_usd!r}, those prices give {worst_predicted!r}, off by {worst:.3g} "
                      f"relatively where {RESIDUAL_TOLERANCE:g} is allowed: the ranges hold calls that were priced "
                      "differently")
    return Prices(price_in, price_out, len(priced), worst, worst_line.number)


# --- the numbers ---------------------------------------------------------------------------------------------------------

def _ratio(total: float | int, pages: int, digits: int) -> float | None:
    return round(total / pages, digits) if pages else None


def _tokens(lines: Sequence[Line]) -> dict:
    """The lines' tokens; every line has usage."""
    total_in = sum(line.input_tokens for line in lines)
    cached = sum(line.cached_input_tokens for line in lines)
    return {"input": total_in, "uncached_input": total_in - cached, "cached_input": cached,
            "output": sum(line.output_tokens for line in lines)}


def _cost(lines: Sequence[Line], prices: Prices, cached_price_per_m: float | None) -> dict:
    """The lines' cost: the ledger's (the upper end), cached input at $0 (the lower end), and, when a price for cached
    input is given, at that price. Every line has a cost."""
    upper = math.fsum(line.cost_usd for line in lines)
    lower = math.fsum((line.input_tokens - line.cached_input_tokens) * prices.input_per_token
                      + line.output_tokens * prices.output_per_token for line in lines)
    cached = sum(line.cached_input_tokens for line in lines)
    at_price = None if cached_price_per_m is None else lower + cached * cached_price_per_m / PER_M
    return {"upper": upper, "lower": lower, "at_cached_price": at_price}


def scope_figures(pages: Sequence[PageCalls], prices: Prices, cached_price_per_m: float | None) -> dict:
    """The figures of one config on a set of pages (one book's, or all books'). A figure that needs usage or a cost that
    a line of the set lacks is None, and the line numbers are listed."""
    lines = [line for page in pages for line in page.lines]
    n = len(pages)
    without_usage = [line.number for line in lines if line.input_tokens is None]
    without_cost = [line.number for line in lines if line.cost_usd is None]
    tokens = None if without_usage else _tokens(lines)
    cost = _cost(lines, prices, cached_price_per_m) if not without_cost else dict.fromkeys(
        ("upper", "lower", "at_cached_price"))
    return {
        "pages": n,
        "calls": len(lines),
        "calls_per_page": _ratio(len(lines), n, 3),
        "tokens": tokens,
        "tokens_per_page": None if tokens is None else {
            key: _ratio(value, n, 3) for key, value in tokens.items() if key != "input"},
        "cached_share_of_input": None if tokens is None or not tokens["input"]
        else round(tokens["cached_input"] / tokens["input"], 6),
        "cost_usd": {key: None if value is None else round(value, 6) for key, value in cost.items()},
        "cost_per_page_usd": {key: None if value is None else _ratio(value, n, 6) for key, value in cost.items()},
        "latency_s": _latency([page.latency_s for page in pages]),
        "lines_without_usage": without_usage,
        "lines_without_cost": without_cost,
    }


def _latency(seconds: Sequence[float]) -> dict:
    summary = latency_summary(seconds)
    return {**summary, "mean_s": round(math.fsum(seconds) / len(seconds), 3) if seconds else None}


def _run_record(calls: RunCalls) -> dict:
    """A run's range and what its lines add up to: tokens over the lines that have usage, cost over the lines that have a
    cost, and the lines that do not, listed (never counted as 0)."""
    run = calls.run
    lines = [line for page in calls.pages for line in page.lines]
    with_usage = [line for line in lines if line.input_tokens is not None]
    with_cost = [line for line in lines if line.cost_usd is not None]
    return {
        "directory": str(run.directory),
        "eval_id": run.eval_id,
        "config": run.config,
        "book": run.book,
        "requested_models": run.manifest.get("requested_models"),
        "ledger_range": {"first": calls.first, "last": calls.last, "lines": calls.last - calls.first + 1},
        "lines_sha256": calls.lines_sha256,
        "pages": {"attempted": len(calls.pages), **{status: sum(p.status == status for p in calls.pages)
                                                      for status in ATTEMPTED}},
        "calls": len(lines),
        "tokens_of_lines_with_usage": _tokens(with_usage),
        "cost_usd_of_lines_with_cost": round(math.fsum(line.cost_usd for line in with_cost), 6),
        "lines_without_usage": [line.number for line in lines if line.input_tokens is None],
        "lines_without_cost": [line.number for line in lines if line.cost_usd is None],
    }


def _failed_pages(runs: Sequence[RunCalls]) -> list[dict]:
    return [{
        "eval_id": calls.run.eval_id, "config": calls.run.config, "book": calls.run.book,
        "page_index": page.page_index, "page_id": page.lines[0].page_id, "calls": len(page.lines),
        "ledger_lines": [page.lines[0].number, page.lines[-1].number], "latency_s": page.latency_s,
        "lines_without_usage": [line.number for line in page.lines if line.input_tokens is None],
        "lines_without_cost": [line.number for line in page.lines if line.cost_usd is None],
        "error": page.error,
    } for calls in runs for page in calls.pages if page.status == "failed"]


def _lines_without_cost(runs: Sequence[RunCalls]) -> list[dict]:
    found = [{
        "line": line.number, "eval_id": calls.run.eval_id, "config": calls.run.config, "book": calls.run.book,
        "page_index": page.page_index, "page_status": page.status, "has_usage": line.input_tokens is not None,
        "error": line.error,
    } for calls in runs for page in calls.pages for line in page.lines if line.cost_usd is None]
    return sorted(found, key=lambda entry: entry["line"])


def _method(cached_price_per_m: float | None) -> dict:
    return {
        "protocol": "eval/PROTOCOL.md, Statistics: cost and latency are described, without an interval",
        "scopes": {"common": "the pages every config scored, per book (the report's common pages)",
                   "attempted": "every page a run attempted, scored or failed; a failed page's calls count"},
        "prices": "recovered from the ledger lines of the given ranges that have a cost: a least squares fit of "
                  "cost_usd = input_tokens x a + output_tokens x b, solved exactly, every line within the tolerance",
        "cost_usd": {
            "upper": "the ledger's cost: every input token at the input price (LLMClient._cost)",
            "lower": "cached input at $0: (input_tokens - cached_input_tokens) x a + output_tokens x b",
            "at_cached_price": None if cached_price_per_m is None else (
                f"cached input at ${cached_price_per_m:g} per million tokens: lower + cached_input_tokens x that price"),
        },
        "tokens": "input is input_tokens (cached ones included); uncached_input is input minus cached_input",
        "latency_s": "each page's latency_s of scores.json (failed pages: of failed_pages); p50 and p95 as "
                     "metrics.latency_summary; the mean is the plain mean of the pages",
        "unknown": "a figure that needs usage or a cost that a line of its scope lacks is null for that scope, never a "
                   "partial sum; the lines are listed. A run's own totals are sums over the lines that have them.",
    }


def build_costs(directories: Sequence[Path], glossary_dir: Path, ledger_path: Path,
                ledger_ranges: Mapping[str, tuple[int, int]], *, mqm_judge: str | None = None,
                cached_input_price_per_m: float | None = None) -> dict:
    """The cost and latency of the eval runs in `directories` (what costs.json holds). `ledger_ranges` maps each eval
    id to the first and last line of the ledger at `ledger_path` that the run's calls wrote (`parse_ranges`).
    `mqm_judge` is the report's (needed only when a run has MQM files of more than one judge). Raises `Refused`, having
    written nothing."""
    directories = list(directories)
    pooled_report = report.build_report(directories, glossary_dir, mqm_judge=mqm_judge)  # the report's refusals, first
    runs = [report.load_run(Path(d), glossary_dir) for d in directories]
    runs.sort(key=lambda r: (report._config_order(r.config), report._book_order(r.book)))
    if cached_input_price_per_m is not None and not (
            report._is_finite(cached_input_price_per_m) and cached_input_price_per_m >= 0):
        raise Refused(f"the price of cached input, {cached_input_price_per_m!r} dollars per million tokens, is not a "
                      "non-negative number")
    known = {run.eval_id for run in runs}
    for eval_id in ledger_ranges:
        if eval_id not in known:
            raise Refused(f"a ledger range is given for {eval_id}, which is not one of the eval runs (they are "
                          f"{', '.join(sorted(known))})")
    for run in runs:
        if run.eval_id not in ledger_ranges:
            raise Refused(f"{run.directory}: it has no ledger range: give --ledger-range {run.eval_id}=<first line>-"
                          "<last line>")
    _check_overlaps(ledger_ranges)
    ledger = read_ledger(ledger_path)
    calls = [read_run_calls(run, *ledger_ranges[run.eval_id], ledger, ledger_path) for run in runs]
    prices = fit_prices([line for c in calls for page in c.pages for line in page.lines], ledger_path)
    if cached_input_price_per_m is not None and cached_input_price_per_m > prices.input_per_token * PER_M * (
            1 + RESIDUAL_TOLERANCE):
        raise Refused(f"the price of cached input, {cached_input_price_per_m:g} dollars per million tokens, is above "
                      f"the input price recovered from the ledger ({prices.input_per_token * PER_M:.6g}): the range "
                      "would not hold the value")

    common, configs = pooled_report["common_pages"], pooled_report["configs"]

    def pages_of(config: str, book: str | None, which: str) -> list[PageCalls]:
        """The pages of `config` (of one book, or of all) that the set of pages `which` holds."""
        return [page for c in calls if c.run.config == config and (book is None or c.run.book == book)
                for page in c.pages
                if which == "attempted" or (page.status == "scored" and page.page_index in common[c.run.book])]

    def both(config: str, book: str | None) -> dict:
        return {which: scope_figures(pages_of(config, book, which), prices, cached_input_price_per_m)
                for which in SCOPES}

    state = git_state(ROOT)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": {"git_commit": state["commit"], "git_dirty": state["dirty"]},
        "inputs": {
            "ledger": {"path": str(ledger_path), "lines": len(ledger)},
            "glossary_dir": str(glossary_dir),
            "mqm_judge": mqm_judge,
            "cached_input_price_per_m": cached_input_price_per_m,
        },
        "prices": {
            "requested_models": sorted({line.requested_model for c in calls for page in c.pages for line in page.lines}),
            "input_usd_per_m_tokens": round(prices.input_per_token * PER_M, 9),
            "output_usd_per_m_tokens": round(prices.output_per_token * PER_M, 9),
            "input_usd_per_token": prices.input_per_token,
            "output_usd_per_token": prices.output_per_token,
            "fitted_lines": prices.lines,
            "max_relative_residual": prices.max_relative_residual,
            "worst_line": prices.worst_line,
            "tolerance": RESIDUAL_TOLERANCE,
        },
        "method": _method(cached_input_price_per_m),
        "configs": configs,
        "books": pooled_report["books"],
        "common_pages": common,
        "runs": [_run_record(c) for c in calls],
        "failed_pages": _failed_pages(calls),
        "lines_without_cost": _lines_without_cost(calls),
        "pooled": {config: both(config, None) for config in configs},
        "by_book": {book: {config: both(config, book) for config in configs} for book in pooled_report["books"]},
        "warnings": pooled_report["warnings"],
    }


def write_costs(document: dict, out_root: Path) -> Path:
    """Write costs.json into a new directory `<out_root>/<UTC time>-costs` (`-2`, `-3`... if that exists) and return it.
    The file is written under a temporary name and renamed, and an existing directory is never written into."""
    out_root.mkdir(parents=True, exist_ok=True)
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-costs"
    candidate, n = base, 1
    while True:
        try:
            (out_root / candidate).mkdir()
            break
        except FileExistsError:
            n += 1
            candidate = f"{base}-{n}"
    out_dir = out_root / candidate
    temporary = out_dir / "costs.json.tmp"
    try:
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, out_dir / "costs.json")
    except BaseException:
        temporary.unlink(missing_ok=True)
        out_dir.rmdir()
        raise
    return out_dir


# --- the printed table ---------------------------------------------------------------------------------------------------

ERROR_WIDTH = 100  # an error message is cut to this in the printed table (costs.json has it whole)
LATENCY_LABEL = "latency p50 / p95 (s)"


def _num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def _latency_cell(latency: dict) -> str:
    return f"{_num(latency['p50_s'])} / {_num(latency['p95_s'])}"


def _short(text: str | None) -> str:
    text = " ".join((text or "no error recorded").split())
    return text if len(text) <= ERROR_WIDTH else text[:ERROR_WIDTH - 1] + "\u2026"


def _table_rows(cached_price_per_m: float | None) -> list[tuple[str, object, object, str | None]]:
    """(label, a scope's value, how to show it, the scope's list of lines that leave the value unknown, if it has one).
    A value that is None is shown as n/a, with the number of those lines."""
    rows = [
        ("pages", lambda s: s["pages"], str, None),
        ("calls per page", lambda s: s["calls_per_page"], lambda v: f"{v:.2f}", None),
        ("tokens per page: uncached input", lambda s: (s["tokens_per_page"] or {}).get("uncached_input"), _num,
         "lines_without_usage"),
        ("tokens per page: cached input", lambda s: (s["tokens_per_page"] or {}).get("cached_input"), _num,
         "lines_without_usage"),
        ("tokens per page: output", lambda s: (s["tokens_per_page"] or {}).get("output"), _num, "lines_without_usage"),
        ("cached share of input", lambda s: s["cached_share_of_input"], report._pct, "lines_without_usage"),
        ("cost per page, lower (cached input $0)", lambda s: s["cost_per_page_usd"]["lower"], report._usd,
         "lines_without_cost"),
    ]
    if cached_price_per_m is not None:
        rows.append((f"cost per page, cached at ${cached_price_per_m:g}/M",
                     lambda s: s["cost_per_page_usd"]["at_cached_price"], report._usd, "lines_without_cost"))
    rows += [
        ("cost per page, upper (the ledger's)", lambda s: s["cost_per_page_usd"]["upper"], report._usd,
         "lines_without_cost"),
        (LATENCY_LABEL, lambda s: s["latency_s"] if s["latency_s"]["n"] else None, _latency_cell, None),
        ("latency mean (s)", lambda s: s["latency_s"]["mean_s"], _num, None),
    ]
    return rows


def _block(document: dict, heading: str, scopes: Mapping[str, dict]) -> list:
    """A block of the table: a heading, the configs as columns, one row for each figure."""
    configs = document["configs"]
    lines: list = [heading, report._row("", configs)]
    for label, get, show, unknown in _table_rows(document["inputs"]["cached_input_price_per_m"]):
        cells = []
        for config in configs:
            value = get(scopes[config])
            if value is None:
                missing = len(scopes[config][unknown]) if unknown else 0
                cells.append("n/a" + (f" ({missing} unknown)" if missing else ""))
            else:
                cells.append(show(value))
        lines.append(report._row(label, cells))
    return lines


def render(document: dict) -> str:
    """The table that eval/costs.py prints."""
    configs, books, common = document["configs"], document["books"], document["common_pages"]
    prices, ledger = document["prices"], document["inputs"]["ledger"]
    lines: list = [
        f"Cost and latency of {len(document['runs'])} eval runs ({', '.join(books)}), {document['generated_at']}",
        f"ledger {ledger['path']} ({ledger['lines']} lines); each run's range is in costs.json",
        f"prices recovered from {prices['fitted_lines']} lines: input ${prices['input_usd_per_m_tokens']:g} and output "
        f"${prices['output_usd_per_m_tokens']:g} per million tokens; largest relative residual "
        f"{prices['max_relative_residual']:.2g}",
        "cost per page is a range: the upper end is the ledger's cost, which prices every input token at the input "
        "price; the lower end prices cached input at $0",
    ]
    titles = {"common": "common pages (scored by every config): " + ", ".join(f"{b} {len(common[b])}" for b in books),
              "attempted": "all attempted pages (scored or failed)"}
    for which in SCOPES:
        lines += ["", titles[which], *_block(document, "pooled over the books",
                                              {c: document["pooled"][c][which] for c in configs})]
        for book in books:
            lines += ["", *_block(document, book, {c: document["by_book"][book][c][which] for c in configs})]
    lines += ["", "runs (the ledger lines of each, its calls, and the ledger's cost summed over the lines that have one):"]
    for run in document["runs"]:
        first, last = run["ledger_range"]["first"], run["ledger_range"]["last"]
        lines.append(f"  {run['eval_id']}  {run['config']} on {run['book']}: {_span(first, last)}, "
                     f"{_plural(run['calls'], 'call')}, {report._usd(run['cost_usd_of_lines_with_cost'])}"
                     + (f" (+ {len(run['lines_without_cost'])} without a cost)" if run["lines_without_cost"] else ""))
    lines.append("")
    if document["failed_pages"]:
        lines.append("failed pages (outside the common pages; their calls count in the attempted pages):")
        for page in document["failed_pages"]:
            first, last = page["ledger_lines"]
            span = _span(first, last)
            unknown = len(page["lines_without_usage"])
            lines.append(f"  {page['config']}: {page['book']}/{page['page_index']}: {_plural(page['calls'], 'call')} "
                         f"(ledger {span}{f', {unknown} without usage' if unknown else ''}): {_short(page['error'])}")
    else:
        lines.append("failed pages: none")
    without = document["lines_without_cost"]
    if without:
        lines.append("ledger lines without a cost (never counted as 0): " + ", ".join(
            f"{e['line']} ({e['config']} {e['book']}/{e['page_index']})" for e in without))
    else:
        lines.append("ledger lines without a cost: none")
    if document["warnings"]:
        lines += ["", "warnings:", *(f"  {w}" for w in document["warnings"])]
    else:
        lines += ["", "warnings: none"]
    return report._layout(lines)
