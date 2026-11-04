# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib==3.11.2",
# ]
# ///
"""Draw M8's charts from the pooled report, the paired bootstrap's intervals and the cost / latency summary.

Usage:
  uv run eval/charts.py --report runs/report/<id>/report.json --bootstrap runs/bootstrap/<id>/bootstrap.json \\
      --costs runs/analysis/<id>/costs.json [--out-root runs/analysis]

This script is the one entry point in eval/ that draws. It has an environment of its own (the inline metadata above;
`uv run` builds it and leaves the project's alone) and imports nothing from `fukidashi`, so `matplotlib` and its
dependencies never enter `uv sync`, the CI or the lock. Like `eval/comet.py`, it imports only the standard library at
the top: `matplotlib` is imported inside `render`, and the tests run everything else without it.

It reads three summary files and nothing else: no segment-level file, no database, no model, no network. It writes
`<out-root>/<UTC time>-charts/` with five charts, each as `<name>.svg` and `<name>.png`, and `charts.json`:

  intervals       both comparisons' point estimates and 95% intervals, one panel per metric, with a band of ± the
                  pooled run-to-run spread and each metric's reading (OUTLINE §8, "Reading the intervals next to the
                  run-to-run spread")
  by_book         COMET and primary term accuracy of each config on each test book and pooled (common pages)
  mqm_breakdown   GEMBA-MQM errors by severity, and by category and severity, for each config
  cost_vs_comet   cost per page as a range, against pooled COMET (OUTLINE §8, "Cost per page with cached input")
  latency         p50, p95 and mean latency per page of each config

`build_charts` is a pure function: it turns the three parsed files into everything the charts draw, and every number in
that result carries the file and the JSON Pointer (RFC 6901) it came from. `render` only draws what `build_charts`
returned, so a number on a chart is a number in `charts.json`. A number that is null in its source is left out and
listed under `missing`; it is never drawn as 0. The only numbers that are in no file are the run-to-run spreads, which
are constants here (SPREADS) and are listed apart under `constants`.

`charts.json` holds each chart's numbers (value, source file, pointer, and the conversion applied: term accuracy and
term consistency are fractions in the sources and percentage points on the charts), each metric's reading, the SHA-256
of the three inputs and of this file, and the `matplotlib` version. It has no timestamp: the time is in the directory's
name. The same inputs give the same bytes (SVG, PNG and `charts.json`): `svg.hashsalt` is fixed, SVG text is paths,
neither format carries a date, and the font is `matplotlib`'s bundled DejaVu Sans.

Exit codes: 0 the charts were written; 2 nothing was written because an input cannot be charted (a file is missing or is
not the expected JSON, the three files do not belong together, a comparison or metric is absent, a breakdown does not
add up; the message names the file and the cause) or the arguments are wrong; 1 drawing or writing failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import textwrap
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

CONFIGS = ("baseline", "full_no_kb", "full")  # the order of every chart; the color of a config never depends on it
FULL = "full"

# The three configs take the first three slots of the reference categorical palette, in order. They validate on all
# pairs on the light surface (worst CVD ΔE 9.2, worst normal-vision ΔE 24.0); the aqua slot is below 3:1 on the surface,
# so every mark in that color is direct-labeled and every number is in `charts.json` (the table view).
CONFIG_COLORS = {"baseline": "#2a78d6", "full_no_kb": "#eb6834", "full": "#1baf7a"}
# Severity is an ordered scale, so it is a neutral ramp (light = minor); a hue would collide with the configs'.
SEVERITY_COLORS = {"critical": "#2b2a28", "major": "#75746f", "minor": "#b3b2a8"}
SEVERITIES = ("critical", "major", "minor")  # the stacking order, from the baseline of the bar
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#e6e5de"
MINUS = "−"

READING_ZERO = "interval includes 0"
READING_NOISE = "not separated from run-to-run variation"
READINGS = (READING_ZERO, READING_NOISE, "higher", "lower")
USES = ("result", "for reference only")


class Metric(NamedTuple):
    label: str
    unit: str  # on the charts
    scale: int  # multiplied into the source's number: a fraction in the source is percentage points on the chart
    decimals: int
    spread: float  # the pooled run-to-run spread, in the chart's unit


# OUTLINE §8, "Reading the intervals next to the run-to-run spread": M5's one sample of the spread, a second `baseline`
# pass on the dev books, pooled. They are in no output file.
SPREAD_SOURCE = ("OUTLINE §8, 'Reading the intervals next to the run-to-run spread': M5's second baseline pass on the "
                 "dev books, pooled")
METRICS = {
    "comet": Metric("COMET", "score", 1, 4, 0.0007),
    "chrf": Metric("chrF", "points", 1, 2, 0.55),
    "bleu": Metric("BLEU", "points", 1, 2, 0.81),
    "term_accuracy": Metric("Term accuracy", "pp", 100, 1, 1.7),
    "term_consistency": Metric("Term consistency", "pp", 100, 1, 7.0),
    "mqm": Metric("MQM", "score", 1, 2, 0.24),
}
# (name in bootstrap.json, the other config, the metrics drawn, in panel order)
COMPARISONS = (
    ("full vs baseline", "baseline", ("comet", "chrf", "bleu", "term_accuracy", "term_consistency", "mqm")),
    ("full vs full_no_kb", "full_no_kb", ("term_accuracy", "term_consistency")),
)
LATENCY_STATS = (("p50_s", "p50"), ("p95_s", "p95"), ("mean_s", "mean"))


class InputRejected(Exception):
    """An input cannot be charted; the message names the file and the cause. Nothing has been written."""


class Source(NamedTuple):
    name: str  # "report", "bootstrap" or "costs": the key in `charts.json`
    file: str  # as the messages and `charts.json` name it
    data: dict
    sha256: str = ""


def pointer(*tokens: str | int) -> str:
    """RFC 6901 JSON Pointer: `/pooled/baseline/comet/mean`; a list index is its number."""
    return "".join("/" + str(t).replace("~", "~0").replace("/", "~1") for t in tokens)


def dig(source: Source, *tokens: str | int) -> Any:
    node = source.data
    for token in tokens:
        if isinstance(node, dict) and isinstance(token, str) and token in node:
            node = node[token]
        elif isinstance(node, list) and isinstance(token, int) and 0 <= token < len(node):
            node = node[token]
        else:
            raise InputRejected(f"{source.file}: {pointer(*tokens)} is missing")
    return node


def dig_dict(source: Source, *tokens: str | int) -> dict:
    node = dig(source, *tokens)
    if not isinstance(node, dict):
        raise InputRejected(f"{source.file}: {pointer(*tokens)} is not an object")
    return node


def dig_list(source: Source, *tokens: str | int) -> list:
    node = dig(source, *tokens)
    if not isinstance(node, list):
        raise InputRejected(f"{source.file}: {pointer(*tokens)} is not a list")
    return node


def dig_str(source: Source, *tokens: str | int) -> str:
    node = dig(source, *tokens)
    if not isinstance(node, str):
        raise InputRejected(f"{source.file}: {pointer(*tokens)} is not a string")
    return node


class Numbers:
    """The numbers one chart draws. `num` is the only way a number gets from a source into a chart: it records the value,
    the file and the pointer, and a null goes to `missing` and comes back as None, never as 0."""

    def __init__(self, chart: str):
        self.chart = chart
        self.numbers: list[dict] = []
        self.missing: list[dict] = []

    def num(self, label: str, source: Source, *tokens: str | int, scale: int = 1, unit: str | None = None):
        raw = dig(source, *tokens)
        if raw is None:
            self.missing.append({"chart": self.chart, "label": label, "source": source.name, "path": pointer(*tokens)})
            return None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise InputRejected(f"{source.file}: {pointer(*tokens)} is not a number")
        value = raw * scale if scale != 1 else raw
        self.numbers.append({"label": label, "value": value, "unit": unit, "source": source.name,
                             "path": pointer(*tokens), "conversion": f"x{scale}" if scale != 1 else None})
        return value


def agreed(source: Source, what: str, values: dict[str, Any]):
    """The one value the configs share, or None when any is unknown; the configs must not disagree."""
    if any(v is None for v in values.values()):
        return None
    if len(set(values.values())) != 1:
        raise InputRejected(f"{source.file}: {what} differs between the configs ({values}), so the configs do not "
                            "cover the same pages")
    return next(iter(values.values()))


def load_source(name: str, path: Path) -> Source:
    if not path.is_file():
        raise InputRejected(f"{path} (the {name} file) is missing")
    raw = path.read_bytes()
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise InputRejected(f"{path} is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise InputRejected(f"{path} is not a JSON object")
    return Source(name, f"{path.parent.name}/{path.name}" if path.parent.name else path.name, data,
                  hashlib.sha256(raw).hexdigest())


def signed(value: float, decimals: int, *, minus: str = MINUS) -> str:
    text = f"{abs(value):.{decimals}f}"
    return ("+" if value > 0 else minus if value < 0 else "") + text


def plain(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}".replace("-", MINUS)


def read_interval(point: float, low: float, high: float, spread: float) -> str:
    """OUTLINE §8, applied to a difference in the chart's unit. An interval that touches 0 includes it; a difference equal
    to the spread is 'at least' the spread. Both sides are rounded to 9 places so that converting a fraction to
    percentage points (0.29 x 100 is 28.999999999999996) cannot decide a tie."""
    if low <= 0 <= high:
        return READING_ZERO
    if round(abs(point), 9) < round(spread, 9):
        return READING_NOISE
    return "higher" if point > 0 else "lower"


def check_use(source: Source, *tokens: str | int) -> str:
    use = dig_str(source, *tokens)
    if use not in USES:
        raise InputRejected(f"{source.file}: {pointer(*tokens)} is {use!r}, not one of {list(USES)}")
    return use


# ---------------------------------------------------------------------------------------------------------------------
# What the charts draw: a pure function of the three parsed files, with no matplotlib.
# ---------------------------------------------------------------------------------------------------------------------

def eval_ids(source: Source, *tokens: str | int) -> list[str]:
    ids = []
    for i, run in enumerate(dig_list(source, *tokens)):
        if not isinstance(run, dict) or not isinstance(run.get("eval_id"), str):
            raise InputRejected(f"{source.file}: {pointer(*tokens, i)} has no eval_id")
        ids.append(run["eval_id"])
    return sorted(ids)


def check_same_runs(report: Source, bootstrap: Source, costs: Source) -> list[str]:
    """The eval runs the three files name (the report's `runs`, the bootstrap's `inputs.runs`, the cost summary's `runs`)
    must be the same ones, each once."""
    named = {report.file: eval_ids(report, "runs"), bootstrap.file: eval_ids(bootstrap, "inputs", "runs"),
             costs.file: eval_ids(costs, "runs")}
    for file, ids in named.items():
        if len(ids) != len(set(ids)):
            raise InputRejected(f"{file} names an eval run more than once")
    union = set().union(*map(set, named.values()))
    gaps = {file: sorted(union - set(ids)) for file, ids in named.items() if set(ids) != union}
    if gaps:
        raise InputRejected("the three files do not name the same eval runs: "
                            + "; ".join(f"{file} lacks {ids}" for file, ids in gaps.items()))
    return sorted(union)


def find_comparison(bootstrap: Source, name: str, other: str, metrics: tuple[str, ...]) -> int:
    comparisons = dig_list(bootstrap, "comparisons")
    found = [i for i, c in enumerate(comparisons) if isinstance(c, dict) and c.get("name") == name]
    if len(found) != 1:
        raise InputRejected(f"{bootstrap.file}: the comparison {name!r} is "
                            f"{'absent' if not found else 'listed more than once'}")
    (index,) = found
    comparison = comparisons[index]
    if comparison.get("config") != FULL or comparison.get("other") != other:
        raise InputRejected(f"{bootstrap.file}: {pointer('comparisons', index)} is not {FULL} minus {other}")
    for metric in metrics:
        if not isinstance(comparison.get("metrics"), dict) or metric not in comparison["metrics"]:
            raise InputRejected(f"{bootstrap.file}: the comparison {name!r} has no metric {metric!r}")
    return index


def build_intervals(bootstrap: Source) -> dict:
    chart = Numbers("intervals")
    readings, comparisons, whys = [], [], []
    resamples = chart.num("bootstrap resamples", bootstrap, "method", "resamples", unit="resamples")
    for name, other, metrics in COMPARISONS:
        index = find_comparison(bootstrap, name, other, metrics)
        n_pages = chart.num(f"{name}: pages", bootstrap, "comparisons", index, "n_pages", unit="pages")
        panels = []
        for key in metrics:
            metric = METRICS[key]
            base = ("comparisons", index, "metrics", key)
            point = chart.num(f"{name}: {metric.label} point estimate", bootstrap, *base, "point_estimate",
                              scale=metric.scale, unit=metric.unit)
            low = chart.num(f"{name}: {metric.label} interval low", bootstrap, *base, "interval", "low",
                            scale=metric.scale, unit=metric.unit)
            high = chart.num(f"{name}: {metric.label} interval high", bootstrap, *base, "interval", "high",
                             scale=metric.scale, unit=metric.unit)
            use = check_use(bootstrap, *base, "label", "use")
            why = dig_dict(bootstrap, *base, "label").get("why")
            if use != "result" and isinstance(why, str) and why not in whys:
                whys.append(why)
            known = None not in (point, low, high)
            reading = read_interval(point, low, high, metric.spread) if known else None
            panels.append({"metric": key, "label": metric.label, "unit": metric.unit, "decimals": metric.decimals,
                           "point": point, "low": low, "high": high, "spread": metric.spread, "reading": reading,
                           "use": use, "why": why})
            readings.append({"comparison": name, "metric": key, "label": metric.label, "unit": metric.unit,
                             "point_estimate": point, "interval": [low, high], "spread": metric.spread,
                             "reading": reading, "use": use, "why": why})
        comparisons.append({"name": name, "n_pages": n_pages, "panels": panels})
    return {"numbers": chart.numbers, "missing": chart.missing, "readings": readings, "whys": whys,
            "resamples": resamples, "comparisons": comparisons}


def build_by_book(report: Source) -> dict:
    chart = Numbers("by_book")
    books = dig_list(report, "books")
    rows = []
    for book in [*books, None]:
        where = ("pooled",) if book is None else ("by_book", book)
        label = "Pooled" if book is None else book
        pages = agreed(report, f"{label}: pages", {
            c: chart.num(f"{label}: {c} pages", report, *where, c, "pages", unit="pages") for c in CONFIGS})
        occurrences = agreed(report, f"{label}: term occurrences", {
            c: chart.num(f"{label}: {c} term occurrences", report, *where, c, "terms", "primary", "occurrences",
                         unit="occurrences") for c in CONFIGS})
        rows.append({"key": "pooled" if book is None else book, "label": label, "pages": pages,
                     "occurrences": occurrences, "comet": {}, "term_accuracy": {}})
        for c in CONFIGS:
            rows[-1]["comet"][c] = chart.num(f"{label}: {c} COMET", report, *where, c, "comet", "mean", unit="score")
            rows[-1]["term_accuracy"][c] = chart.num(f"{label}: {c} term accuracy", report, *where, c, "terms",
                                                     "primary", "accuracy", scale=100, unit="percent")
    return {"numbers": chart.numbers, "missing": chart.missing, "rows": rows, "pages": rows[-1]["pages"]}


def build_mqm(report: Source) -> dict:
    chart = Numbers("mqm_breakdown")
    use = check_use(report, "mqm", "use")
    why = dig(report, "mqm", "why")
    configs = {}
    for c in CONFIGS:
        base = ("pooled", c, "mqm", "breakdown")
        errors = chart.num(f"{c}: errors", report, *base, "errors", unit="errors")
        by_category = dig_dict(report, *base, "by_category")
        by_both = dig_dict(report, *base, "by_category_and_severity")
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in by_category.values()):
            raise InputRejected(f"{report.file}: {pointer(*base, 'by_category')} holds a count that is not an integer")
        if errors is not None and sum(by_category.values()) != errors:
            raise InputRejected(f"{report.file}: {pointer(*base, 'by_category')} sums to {sum(by_category.values())}, "
                                f"not to {pointer(*base, 'errors')} ({errors})")
        if set(by_both) != set(by_category):
            raise InputRejected(f"{report.file}: {pointer(*base, 'by_category_and_severity')} does not list the same "
                                f"categories as {pointer(*base, 'by_category')}")
        for category in by_category:
            cells = dig_dict(report, *base, "by_category_and_severity", category)
            counts = [cells.get(s) for s in SEVERITIES]
            if not all(isinstance(n, int) for n in counts) or sum(counts) != by_category[category]:
                raise InputRejected(f"{report.file}: {pointer(*base, 'by_category_and_severity', category)} does not "
                                    f"sum to {pointer(*base, 'by_category', category)}")
        configs[c] = {
            "errors": errors,
            "segments_with_errors": chart.num(f"{c}: segments with errors", report, *base, "segments_with_errors",
                                              unit="segments"),
            "scored": chart.num(f"{c}: segments scored", report, "pooled", c, "mqm", "scored", unit="segments"),
            "by_severity": {s: chart.num(f"{c}: {s} errors", report, *base, "by_severity", s, unit="errors")
                            for s in SEVERITIES},
            "categories": {category: {
                "total": chart.num(f"{c}: {category} errors", report, *base, "by_category", category, unit="errors"),
                "by_severity": {s: chart.num(f"{c}: {category}, {s} errors", report, *base,
                                             "by_category_and_severity", category, s, unit="errors")
                                for s in SEVERITIES}} for category in by_category}}
    names = {category for c in CONFIGS for category in configs[c]["categories"]}
    order = sorted(names, key=lambda n: (-sum(configs[c]["categories"].get(n, {}).get("total") or 0 for c in CONFIGS), n))
    return {"numbers": chart.numbers, "missing": chart.missing, "use": use, "why": why, "configs": configs,
            "categories": order}


def build_cost_vs_comet(report: Source, costs: Source) -> dict:
    chart = Numbers("cost_vs_comet")
    pages = agreed(costs, "pooled common pages", {
        c: chart.num(f"{c}: common pages", costs, "pooled", c, "common", "pages", unit="pages") for c in CONFIGS})
    points = []
    for c in CONFIGS:
        base = ("pooled", c, "common", "cost_per_page_usd")
        points.append({
            "config": c,
            "lower": chart.num(f"{c}: cost per page, cached input at $0", costs, *base, "lower", unit="USD per page"),
            "upper": chart.num(f"{c}: cost per page, the ledger's cost", costs, *base, "upper", unit="USD per page"),
            "comet": chart.num(f"{c}: pooled COMET", report, "pooled", c, "comet", "mean", unit="score")})
    return {"numbers": chart.numbers, "missing": chart.missing, "pages": pages, "points": points}


def build_latency(costs: Source) -> dict:
    chart = Numbers("latency")
    pages = agreed(costs, "pooled common pages", {
        c: chart.num(f"{c}: common pages", costs, "pooled", c, "common", "pages", unit="pages") for c in CONFIGS})
    panels = [{"stat": key, "label": label, "values": {
        c: chart.num(f"{c}: latency {label}", costs, "pooled", c, "common", "latency_s", key, unit="s")
        for c in CONFIGS}} for key, label in LATENCY_STATS]
    return {"numbers": chart.numbers, "missing": chart.missing, "pages": pages, "panels": panels}


def build_charts(report: Source, bootstrap: Source, costs: Source) -> dict:
    """Everything the five charts draw, from the three parsed files. Raises `InputRejected` (nothing is drawn or written).
    Each chart's `numbers` list is every number it draws, with its source and pointer."""
    eval_runs = check_same_runs(report, bootstrap, costs)
    if dig_dict(report, "common_pages") != dig_dict(costs, "common_pages"):
        raise InputRejected(f"{report.file} and {costs.file} do not name the same common pages "
                            f"({pointer('common_pages')} differs)")
    charts = {"intervals": build_intervals(bootstrap), "by_book": build_by_book(report),
              "mqm_breakdown": build_mqm(report), "cost_vs_comet": build_cost_vs_comet(report, costs),
              "latency": build_latency(costs)}
    return {"eval_runs": eval_runs, "charts": charts, "readings": charts["intervals"]["readings"],
            "constants": [{"label": f"{m.label} run-to-run spread", "metric": key, "value": m.spread, "unit": m.unit,
                           "cites": SPREAD_SOURCE} for key, m in METRICS.items()],
            "missing": [item for chart in charts.values() for item in chart["missing"]]}


def describe_reading(r: dict) -> str:
    """One line per metric: what the script prints."""
    metric = METRICS[r["metric"]]
    use = "" if r["use"] == "result" else f"  [{r['use']}]"
    if r["reading"] is None:
        return f"{r['comparison']:<20} {r['label']:<18} n/a (a number is null in bootstrap.json){use}"
    low, high = r["interval"]
    d = metric.decimals
    return (f"{r['comparison']:<20} {r['label']:<18} {signed(r['point_estimate'], d, minus='-'):>8} "
            f"[{signed(low, d, minus='-')}, {signed(high, d, minus='-')}] {metric.unit:<6} {r['reading']}{use}")


# ---------------------------------------------------------------------------------------------------------------------
# Drawing: only what `build_charts` returned.
# ---------------------------------------------------------------------------------------------------------------------

def comma(n: float | int | None) -> str:
    return "n/a" if n is None else f"{n:,.0f}"


def render(charts: dict, out_dir: Path) -> str:
    """Draw the five charts into `out_dir` as SVG and PNG. The only place that imports `matplotlib`; returns its version."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    rc = {
        "svg.hashsalt": "fukidashi-charts", "svg.fonttype": "path", "font.family": ["DejaVu Sans"], "font.size": 9,
        "axes.unicode_minus": True, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "text.color": INK,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 150,
        "figure.dpi": 100, "lines.solid_capstyle": "round", "xtick.major.size": 3, "ytick.major.size": 0,
        "text.parse_math": False,  # a label with two $ would otherwise be set as math
    }

    class Canvas:
        """A figure laid out in inches from its top left, so a panel never depends on a tight-layout pass."""

        def __init__(self, width: float, height: float):
            self.w, self.h = width, height
            self.fig = plt.figure(figsize=(width, height))

        def axes(self, left: float, top: float, w: float, h: float):
            return self.fig.add_axes([left / self.w, 1 - (top + h) / self.h, w / self.w, h / self.h])

        def text(self, x: float, y: float, s: str, **kw):
            kw.setdefault("va", "baseline")
            return self.fig.text(x / self.w, 1 - y / self.h, s, **kw)

        def header(self, title: str, lines: list[str], x: float = 0.5, *, bold_first: bool = False) -> float:
            """The title and wrapped lines under it; returns the y of the last line."""
            self.text(x, 0.42, title, size=14, weight="bold")
            y = 0.42
            for i, line in enumerate(lines):
                strong = bold_first and i == 0
                for part in textwrap.wrap(line, int((self.w - 2 * x) * 13.4)):
                    y += 0.24
                    self.text(x, y, part, size=9.5, color=INK if strong else INK2, weight="bold" if strong else "normal")
            return y

        def footer(self, notes: list[str], x: float = 0.5) -> None:
            lines = []
            for note in notes:
                lines += textwrap.wrap(note, int((self.w - 2 * x) * 14), subsequent_indent="   ") + [""]
            y = self.h - 0.28 - 0.175 * (len(lines) - 1)
            for line in lines:
                self.text(x, y, line, size=8.5, color=INK2)
                y += 0.175

        def config_legend(self, right: float, top: float) -> None:
            handles = [Line2D([], [], marker="o", ms=8, color=CONFIG_COLORS[c], mec=SURFACE, mew=1.5, ls="")
                       for c in CONFIGS]
            self.fig.legend(handles, CONFIGS, loc="upper right", ncol=3, frameon=False, fontsize=9.5,
                            bbox_to_anchor=(right / self.w, 1 - top / self.h), handletextpad=0.2, columnspacing=1.4)

        def save(self, name: str) -> None:
            self.fig.savefig(out_dir / f"{name}.svg", format="svg", metadata={"Date": None})
            self.fig.savefig(out_dir / f"{name}.png", format="png", metadata={"Software": None})
            plt.close(self.fig)

    def quiet_axes(ax, *, left: bool = False) -> None:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_visible(left)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(axis="x", labelsize=8, length=3, color=AXIS, pad=2)
        ax.tick_params(axis="y", length=0)

    def hairlines(ax) -> None:
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, color=GRID, lw=0.8)
        ax.yaxis.grid(False)

    def plain_ticks(ax, decimals: int | None = None) -> None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if decimals is None:
            ax.xaxis.get_major_formatter().set_useOffset(False)
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: plain(v, decimals)))

    def dot(ax, x, y, color, *, hollow: bool = False, size: float = 8, **kw):
        ax.plot([x], [y], "o", ms=size, mfc=SURFACE if hollow else color, mec=color if hollow else SURFACE,
                mew=1.8 if hollow else 1.5, zorder=4, clip_on=False, **kw)

    # ---------------------------------------------------------------- intervals
    def draw_intervals(spec: dict) -> None:
        width, col_w, gap, left = 13.2, 3.55, 0.55, 0.6
        axes_h, header_h, row_pitch = 0.55, 0.78, 1.82
        canvas = Canvas(width, 9.3)
        title_end = canvas.header(
            "Paired bootstrap intervals of the difference: full minus baseline, and full minus full_no_kb",
            [f"Dot: the difference on the pages both configs scored. Line: the 95% interval of the paired bootstrap by "
             f"page ({comma(spec['resamples'])} resamples). Gray band: ± the pooled run-to-run spread.",
             "Each config ran once, so an interval covers which pages were drawn, not how much a rerun would change."])
        y = title_end + 0.45
        for comparison in spec["comparisons"]:
            canvas.text(left, y, f"{comparison['name']}: {comma(comparison['n_pages'])} pages scored by both configs",
                        size=11.5, weight="bold")
            y += 0.28
            for i, p in enumerate(comparison["panels"]):
                x = left + (i % 3) * (col_w + gap)
                top = y + (i // 3) * row_pitch
                muted = p["use"] != "result"
                color = MUTED if muted else INK
                d = p["decimals"]
                tag = f"  ·  {p['use']}" if muted else ""
                canvas.text(x, top + 0.17, f"{p['label']} ({p['unit']}){tag}", size=10, weight="bold")
                if p["reading"] is None:
                    canvas.text(x, top + 0.40, "not available (null in the source)", size=9)
                else:
                    canvas.text(x, top + 0.40, f"{signed(p['point'], d)}   [{signed(p['low'], d)}, {signed(p['high'], d)}]",
                                size=9)
                    canvas.text(x, top + 0.60, f"Reading: {p['reading']}", size=9, color=INK2)
                ax = canvas.axes(x, top + header_h, col_w, axes_h)
                quiet_axes(ax)
                ax.set_yticks([])
                ax.set_ylim(0, 1)
                s = p["spread"]
                lo, hi = -s, s
                if p["reading"] is not None:
                    lo, hi = min(lo, p["low"]), max(hi, p["high"])
                pad = 0.1 * (hi - lo)
                ax.set_xlim(lo - pad, hi + pad)
                ax.axvspan(-s, s, color=BAND, lw=0, zorder=0)
                ax.text(s, 0.98, f" ±{s:.{d}f}", transform=ax.get_xaxis_transform(), va="top", ha="left", size=7.5,
                        color=MUTED)
                ax.axvline(0, color=INK2, lw=1.0, zorder=1)
                if p["reading"] is not None:
                    ax.plot([p["low"], p["high"]], [0.38, 0.38], color=color, lw=2, zorder=3)
                    ax.vlines([p["low"], p["high"]], 0.24, 0.52, color=color, lw=2, zorder=3)
                    dot(ax, p["point"], 0.38, color, hollow=muted)
                plain_ticks(ax)
            y += row_pitch * ((len(comparison["panels"]) + 2) // 3) + 0.12
        notes = ["Reading (OUTLINE §8). If the interval includes 0 (touching 0 counts), the reading is 'interval "
                 "includes 0'. If it excludes 0 but the difference is smaller than the band, it is 'not separated from "
                 "run-to-run variation'. If it excludes 0 and the difference is at least the band, it is 'higher' or "
                 "'lower'.",
                 "The band is M5's one sample of the run-to-run spread, a second baseline pass on the dev books "
                 "(OUTLINE §8). pp: percentage points. Each comparison covers the pages both of its configs scored, so a "
                 "point estimate is not the difference of the pooled report's numbers."]
        notes += [f"for reference only: {why}." for why in spec["whys"]]
        canvas.footer(notes)
        canvas.save("intervals")

    # ---------------------------------------------------------------- by_book
    def draw_by_book(spec: dict) -> None:
        canvas = Canvas(13.6, 6.2)
        canvas.header("COMET and primary term accuracy of each config, on each test book and pooled",
                      [f"Common pages: the pages every config scored ({comma(spec['pages'])} in all). One run per config, "
                       "no interval. Term accuracy: hits over occurrences of the gold glossary's entries that are not "
                       "borderline (OUTLINE §4.7)."])
        canvas.config_legend(12.4, 1.2)
        rows = spec["rows"]
        n = len(rows)
        panels = [("comet", "COMET (mean over segments)", 4, 1), ("term_accuracy", "Primary term accuracy (%)", 0, 100)]
        offsets = {"baseline": -0.22, "full_no_kb": 0.0, "full": 0.22}
        for k, (metric, label, decimals, _) in enumerate(panels):
            left = 2.75 + k * 5.0
            ax = canvas.axes(left, 1.75, 4.5, 3.5)
            quiet_axes(ax)
            hairlines(ax)
            ax.set_ylim(n - 0.5, -0.5)
            ax.set_yticks(range(n))
            ax.set_yticklabels([])
            values = [r[metric][c] for r in rows for c in CONFIGS if r[metric][c] is not None]
            lo, hi = (min(values), max(values)) if values else (0, 1)
            pad = 0.12 * (hi - lo or 1)
            ax.set_xlim(lo - pad, hi + pad)
            for y in range(n - 1):
                ax.axhline(y + 0.5, color=INK2 if y == n - 2 else GRID, lw=1.0 if y == n - 2 else 0.8, zorder=1)
            for y, row in enumerate(rows):
                for c in CONFIGS:
                    v = row[metric][c]
                    if v is None:
                        continue
                    dot(ax, v, y + offsets[c], CONFIG_COLORS[c])
                    if row["key"] == "pooled":
                        ax.annotate(plain(v, decimals if metric == "comet" else 1), (v, y + offsets[c]),
                                    xytext=(8, 0), textcoords="offset points", va="center", size=8.5, color=INK)
                if metric == "term_accuracy":
                    ax.annotate(f"{comma(row['occurrences'])} occurrences", (1.0, y), xycoords=ax.get_yaxis_transform(),
                                xytext=(8, 0), textcoords="offset points", va="center", size=8, color=MUTED)
            plain_ticks(ax, 3 if metric == "comet" else 0)
            canvas.text(left, 1.62, label, size=10, weight="bold")
            if k == 0:
                for y, row in enumerate(rows):
                    py = 1.75 + 3.5 * (y + 0.5) / n
                    canvas.text(left - 0.15, py - 0.02, row["label"], size=9.5, ha="right", weight="bold")
                    canvas.text(left - 0.15, py + 0.17, f"{comma(row['pages'])} pages", size=8.5, ha="right", color=MUTED)
        canvas.footer(["The COMET axis does not start at 0. Exact values are in charts.json."])
        canvas.save("by_book")

    # ---------------------------------------------------------------- mqm_breakdown
    def draw_mqm(spec: dict) -> None:
        canvas = Canvas(14.2, 10.6)
        canvas.header("GEMBA-MQM errors by severity and by category, for each config",
                      [f"{spec['use'].capitalize()}: {spec['why']}." if isinstance(spec["why"], str) else
                       f"{spec['use'].capitalize()}.",
                       "Common pages. Every category name in the report is shown; the names are the judge's own labels. "
                       "A blank row: the judge named no error of that category for that config."],
                      bold_first=True)
        # severity panel
        ax = canvas.axes(3.0, 1.75, 6.4, 1.25)
        quiet_axes(ax)
        hairlines(ax)
        ax.set_ylim(len(CONFIGS) - 0.5, -0.5)
        ax.set_yticks(range(len(CONFIGS)))
        ax.set_yticklabels(CONFIGS, size=9.5)
        top = max(spec["configs"][c]["errors"] or 0 for c in CONFIGS)
        ax.set_xlim(0, top * 1.12)
        for y, c in enumerate(CONFIGS):
            start = 0
            for s in SEVERITIES:
                v = spec["configs"][c]["by_severity"][s]
                if v is None:
                    continue
                ax.barh(y, v, left=start, height=0.5, color=SEVERITY_COLORS[s], edgecolor=SURFACE, linewidth=1.5,
                        zorder=3)
                label = ax.text(start + v / 2, y, comma(v), ha="center", va="center", size=8.5, zorder=5,
                                color="#ffffff" if s != "minor" else INK)
                canvas.fig.canvas.draw()
                box = label.get_window_extent().transformed(ax.transData.inverted())
                if box.width > v * 0.9:
                    label.remove()
                start += v
            if spec["configs"][c]["errors"] is not None:
                ax.text(start, y, "  " + comma(spec["configs"][c]["errors"]), ha="left", va="center", size=8.5,
                        weight="bold")
        plain_ticks(ax, 0)
        canvas.text(3.0, 1.62, "Errors by severity", size=10, weight="bold")
        for i, s in enumerate(SEVERITIES):
            x = 10.0 + i * 1.3
            canvas.fig.add_artist(Line2D([x / canvas.w], [1 - 1.9 / canvas.h], marker="s", ms=9,
                                         color=SEVERITY_COLORS[s], mec=SURFACE, ls="", transform=canvas.fig.transFigure))
            canvas.text(x + 0.14, 1.94, s, size=9.5)
        # category panels
        cats = spec["categories"]
        panel_top, panel_h, panel_w = 4.1, 5.45, 3.3
        xmax = max((spec["configs"][c]["categories"][n]["total"] or 0) for c in CONFIGS
                   for n in spec["configs"][c]["categories"]) * 1.16
        for k, c in enumerate(CONFIGS):
            left = 3.0 + k * (panel_w + 0.45)
            info = spec["configs"][c]
            canvas.fig.add_artist(Line2D([(left) / canvas.w], [1 - 3.65 / canvas.h], marker="o", ms=8,
                                         color=CONFIG_COLORS[c], mec=SURFACE, ls="", transform=canvas.fig.transFigure))
            canvas.text(left + 0.14, 3.69, c, size=10.5, weight="bold")
            canvas.text(left, 3.91, f"{comma(info['errors'])} errors in {comma(info['segments_with_errors'])} of "
                                    f"{comma(info['scored'])} scored segments", size=8.5, color=INK2)
            ax = canvas.axes(left, panel_top, panel_w, panel_h)
            quiet_axes(ax)
            hairlines(ax)
            ax.set_ylim(len(cats) - 0.5, -0.5)
            ax.set_yticks(range(len(cats)))
            ax.set_yticklabels(cats if k == 0 else [], size=8.5)
            ax.set_xlim(0, xmax)
            for y, name in enumerate(cats):
                cell = info["categories"].get(name)
                if cell is None:
                    continue
                start = 0
                for s in SEVERITIES:
                    v = cell["by_severity"][s]
                    if v:
                        ax.barh(y, v, left=start, height=0.6, color=SEVERITY_COLORS[s], edgecolor=SURFACE,
                                linewidth=1.5, zorder=3)
                        start += v
                if cell["total"] is not None:
                    ax.text(start, y, "  " + comma(cell["total"]), ha="left", va="center", size=8.5, color=INK)
            plain_ticks(ax, 0)
        canvas.footer(["Bars stack critical, major and minor from the left; the number at the end of a bar is the "
                       "category's total. Counts are the errors the judge named on the common pages; they are not "
                       "weighted."])
        canvas.save("mqm_breakdown")

    # ---------------------------------------------------------------- cost_vs_comet
    def draw_cost(spec: dict) -> None:
        canvas = Canvas(10.0, 7.0)
        canvas.header("Cost per page and pooled COMET of each config",
                      [f"Common pages ({comma(spec['pages'])} in all). Cost is a range from the ledger, not the invoice: "
                       "the left end takes cached input at $0, the right end is the ledger's cost (OUTLINE §8). "
                       "No line connects the configs."])
        canvas.config_legend(9.6, 0.2)
        ax = canvas.axes(1.25, 1.5, 7.9, 4.2)
        quiet_axes(ax, left=True)
        hairlines(ax)
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        pts = spec["points"]
        ends = [v for p in pts for v in (p["lower"], p["upper"]) if v is not None]
        comets = [p["comet"] for p in pts if p["comet"] is not None]
        ax.set_xlim(0, (max(ends) if ends else 1) * 1.28)
        lo, hi = (min(comets), max(comets)) if comets else (0, 1)
        pad = 0.6 * (hi - lo or 0.01)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Cost per page (USD)", size=9, labelpad=6)
        ax.set_ylabel("Pooled COMET (mean over segments)", size=9, labelpad=8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.3f}"))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: plain(v, 3)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(axis="y", labelsize=8, length=3, color=AXIS)
        for p in pts:
            if p["comet"] is None:
                continue
            color = CONFIG_COLORS[p["config"]]
            if p["lower"] is not None and p["upper"] is not None:
                ax.plot([p["lower"], p["upper"]], [p["comet"]] * 2, color=color, lw=2, zorder=3)
            for end in (p["lower"], p["upper"]):
                if end is not None:
                    dot(ax, end, p["comet"], color)
            known = [v for v in (p["lower"], p["upper"]) if v is not None]
            if not known:
                continue
            cost = (f"${p['lower']:.4f} to ${p['upper']:.4f}" if len(known) == 2 and p["lower"] != p["upper"]
                    else f"${known[0]:.4f}" + ("" if len(known) == 2 else " (one end is null in the source)"))
            ax.annotate(p["config"], (min(known), p["comet"]), xytext=(0, 22), textcoords="offset points",
                        ha="left", va="bottom", size=10, weight="bold", color=INK)
            ax.annotate(f"{cost} per page  ·  COMET {plain(p['comet'], 4)}", (min(known), p["comet"]), xytext=(0, 9),
                        textcoords="offset points", ha="left", va="bottom", size=8.5, color=INK2)
        canvas.footer(["The y axis does not start at 0. Where the two ends coincide (baseline has no cached input) the "
                       "range is one dot. Exact values are in charts.json."])
        canvas.save("cost_vs_comet")

    # ---------------------------------------------------------------- latency
    def draw_latency(spec: dict) -> None:
        canvas = Canvas(11.6, 4.5)
        canvas.header("Latency per page of each config",
                      [f"Common pages ({comma(spec['pages'])} in all). Seconds per page: p50, p95 and mean over the pages "
                       "(a failed page is not among them)."])
        canvas.config_legend(11.2, 0.2)
        for k, panel in enumerate(spec["panels"]):
            left = 1.85 + k * 3.25
            ax = canvas.axes(left, 1.5, 2.65, 2.2)
            quiet_axes(ax)
            hairlines(ax)
            ax.set_ylim(len(CONFIGS) - 0.5, -0.5)
            ax.set_yticks(range(len(CONFIGS)))
            ax.set_yticklabels(CONFIGS if k == 0 else [], size=9.5)
            values = [v for v in panel["values"].values() if v is not None]
            ax.set_xlim(0, (max(values) if values else 1) * 1.3)
            for y, c in enumerate(CONFIGS):
                v = panel["values"][c]
                if v is None:
                    continue
                ax.barh(y, v, height=0.5, color=CONFIG_COLORS[c], zorder=3)
                ax.text(v, y, f"  {v:.1f} s", va="center", ha="left", size=8.5, color=INK)
            plain_ticks(ax, 0)
            canvas.text(left, 1.38, panel["label"], size=10, weight="bold")
        canvas.footer(["Each panel has its own axis, starting at 0. Exact values are in charts.json."])
        canvas.save("latency")

    with matplotlib.rc_context(rc):
        draw_intervals(charts["intervals"])
        draw_by_book(charts["by_book"])
        draw_mqm(charts["mqm_breakdown"])
        draw_cost(charts["cost_vs_comet"])
        draw_latency(charts["latency"])
    return matplotlib.__version__


# ---------------------------------------------------------------------------------------------------------------------
# Writing.
# ---------------------------------------------------------------------------------------------------------------------

CHART_NAMES = ("intervals", "by_book", "mqm_breakdown", "cost_vs_comet", "latency")


def build_record(built: dict, sources: list[Source], script_sha256: str, matplotlib_version: str) -> dict:
    """`charts.json`: no timestamp, no path of this machine."""
    return {
        "path_syntax": "RFC 6901 JSON Pointer into the source file; conversion 'xN' means the source's number times N",
        "inputs": {s.name: {"file": s.file, "sha256": s.sha256} for s in sources},
        "eval_runs": built["eval_runs"],
        "script": {"file": "eval/charts.py", "sha256": script_sha256, "matplotlib": matplotlib_version},
        "readings": built["readings"],
        "constants": built["constants"],
        "charts": {name: {"files": [f"{name}.svg", f"{name}.png"], "numbers": built["charts"][name]["numbers"]}
                   for name in CHART_NAMES},
        "missing": built["missing"],
    }


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def utc_stamp(now: datetime) -> str:
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Draw M8's charts from the pooled report, the paired bootstrap's intervals "
                                             "and the cost / latency summary.")
    ap.add_argument("--report", type=Path, required=True, help="the pooled report: runs/report/<id>/report.json")
    ap.add_argument("--bootstrap", type=Path, required=True,
                    help="the paired bootstrap: runs/bootstrap/<id>/bootstrap.json")
    ap.add_argument("--costs", type=Path, required=True, help="the cost / latency summary: runs/analysis/<id>/costs.json")
    ap.add_argument("--out-root", type=Path, default=Path("runs/analysis"),
                    help="where <UTC time>-charts/ goes (default: runs/analysis)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None, *, draw: Callable[[dict, Path], str] = render,
         now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> int:
    """`draw` renders the charts after the inputs have been checked; tests pass one that needs no `matplotlib`."""
    args = parse_args(argv)
    try:
        sources = [load_source("report", args.report), load_source("bootstrap", args.bootstrap),
                   load_source("costs", args.costs)]
        built = build_charts(*sources)
    except InputRejected as e:
        print(f"Nothing was written: {e}", file=sys.stderr)
        return 2

    final = args.out_root / f"{utc_stamp(now())}-charts"
    temporary = args.out_root / f".{final.name}.tmp"
    try:
        if final.exists():
            raise FileExistsError(f"{final} exists already")
        args.out_root.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir()
        version = draw({name: built["charts"][name] for name in CHART_NAMES}, temporary)
        script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        write_json(temporary / "charts.json", build_record(built, sources, script_sha256, version))
        os.replace(temporary, final)
    except BaseException as e:
        shutil.rmtree(temporary, ignore_errors=True)
        if not isinstance(e, Exception):
            raise  # Ctrl-C: nothing is left behind, and the interrupt goes on
        print(f"Nothing was written: drawing or writing failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    for reading in built["readings"]:
        print(describe_reading(reading))
    if built["missing"]:
        print(f"{len(built['missing'])} number(s) are null in their source and left out (listed under 'missing')")
    print(f"charts -> {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
