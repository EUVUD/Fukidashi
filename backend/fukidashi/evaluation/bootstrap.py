"""The paired bootstrap by page, for the two comparisons (OUTLINE 4.3, 8; eval/PROTOCOL.md, "Statistics").

The inputs are the eval runs that evaluation/report.py pools: one finished, scored eval run for each of `baseline`,
`full_no_kb` and `full` and each book, every run with its consistency file and its MQM file. They are read and refused
by the report's own code (`report.build_report` runs first, so the same inputs are refused with the same messages), and
nothing else is read: no database, no model, no network. The command-line entry point is eval/bootstrap.py; the logic
lives here so that tests can import it. The standard library and sacreBLEU only.

The protocol fixes the resamples (10,000), the seed (2027) and the pairing, not every detail of the implementation.
These details were written down before any test page was translated (OUTLINE 8, "Bootstrap code before the test
runs"), so that none of them can follow a result:

- **What gets an interval.** `full` vs `baseline`: COMET (the primary metric), chrF, BLEU, term accuracy, term
  consistency and GEMBA-MQM. `full` vs `full_no_kb`: term accuracy (the primary metric) and term consistency. Cost and
  latency get none, and nothing else does. Term accuracy is the primary score (the gold glossary's entries that are not
  borderline); term consistency is the Translations', strict.
- **The pages.** For each comparison, the pages that both of its configs scored, per book, pooled over the books as one
  list sorted by the report's book order and then by page index. This can differ from the report's common pages, which
  every config must have scored: a page that only `full_no_kb` failed stays in `full` vs `baseline`.
- **The draws.** Each comparison has its own `random.Random(SEED)`; each of the `RESAMPLES` resamples is
  `rng.choices(pages, k=len(pages))`, drawn in order. Both configs of the comparison, and all of its metrics, are
  recomputed on the same drawn pages.
- **A page drawn more than once counts each time,** with all of its segments and occurrences, in every metric. Term
  consistency is recomputed on that multiset as the report computes it: per book and entry, the most common rendering's
  count over the rendered occurrences, and an entry counts once it has `CONSISTENCY_MIN_RENDERED` rendered occurrences
  in the resample (so an entry rendered once on a page drawn twice counts).
- **Each metric** on a set of pages is the report's: chrF and BLEU are corpus scores over the segments (sacreBLEU, the
  settings of `metrics.score_corpus`), COMET is the mean of the segment scores, term accuracy pools hits over
  occurrences, consistency is as above, and MQM is the mean score of the `scored` segments (failed and skipped ones are
  left out). On the comparison's own pages, so without a resample, they equal the report's pooled values when the
  comparison's pages are the report's common pages.
- **The difference** is `full` minus the other config, on each resample. **The interval** is the 2.5th and the 97.5th
  percentile of the resamples' differences (`metrics.percentile`: linear interpolation between ranks). **The point
  estimate** is the difference on the comparison's pages, without resampling.
- **A resample on which a metric is undefined** (no occurrence, no rendered entry, no scored segment, in either config)
  gives no difference for that metric: the output counts those resamples and the interval uses the others.
- **Labels.** The MQM interval is for reference only, with the judge's model, because neither judge passed the agreement
  check (OUTLINE 4.5). The consistency interval of a comparison is for reference only when the extractor's accuracy on
  the references is under the threshold (the report's `reportable`) in any of its runs, a run being one config on one
  book, taken over the comparison's pages (a run with no page in the comparison does not count). The report decides
  `reportable` on a config's counts pooled over its books; here a run decides, which is the stricter reading.

**Speed.** The sufficient statistics of every metric are summed once for each page (sacreBLEU's own statistics for chrF
and BLEU, sums and counts for COMET and MQM, hits and occurrences for term accuracy, the renderings for consistency);
a resample only adds up the drawn pages' numbers.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

from ..config import ROOT
from . import report
from .manifest import git_state
from .metrics import percentile

Refused = report.Refused  # the bootstrap refuses what the report refuses, and a few things more

RESAMPLES = 10_000  # eval/PROTOCOL.md, Statistics (OUTLINE 8)
SEED = 2027
PERCENTILES = (2.5, 97.5)  # the 95% interval
CONFIGS = ("baseline", "full_no_kb", "full")  # the three configs of the input, in the report's column order
COMPARISONS = (("full", "baseline"), ("full", "full_no_kb"))  # (config, other): the difference is config minus other
METRICS = {  # what each comparison gives an interval, in the order of the printed table
    ("full", "baseline"): ("comet", "chrf", "bleu", "term_accuracy", "term_consistency", "mqm"),
    ("full", "full_no_kb"): ("term_accuracy", "term_consistency"),
}
PRIMARY = {("full", "baseline"): "comet", ("full", "full_no_kb"): "term_accuracy"}  # eval/PROTOCOL.md's table
TEXT = "translation"  # the text whose renderings the consistency interval uses (not the reference's)
RESULT, REFERENCE_ONLY = "result", "for reference only"  # a label's `use`
MQM_WHY = "neither GEMBA-MQM judge passed the agreement check (OUTLINE 4.5)"
DEFINITIONS = {
    "comet": f"the mean of the segments' COMET scores ({report.COMET_MODEL})",
    "chrf": "sacreBLEU's corpus chrF over the segments, as metrics.score_corpus computes it",
    "bleu": "sacreBLEU's corpus BLEU (tokenize=zh) over the segments, as metrics.score_corpus computes it",
    "term_accuracy": "hits over occurrences of the gold glossary's entries that are not borderline (OUTLINE 4.7)",
    "term_consistency": (
        "the Translations' term consistency, strict (OUTLINE 4.7): per book and entry, the most common rendering's "
        f"count over the rendered occurrences, summed over the entries with {report.CONSISTENCY_MIN_RENDERED} or more "
        "rendered occurrences, over those occurrences"),
    "mqm": "the mean GEMBA-MQM score of the scored segments (failed and skipped segments left out)",
}


# --- the numbers of one page, and the sums of a sample ----------------------------------------------------------------

class SacreBLEU:
    """sacreBLEU's own metric objects, set as `metrics.score_corpus` sets them, used through the statistics they sum.

    A corpus's statistics are the sums of its segments', so a page's are computed once and a sample's are added up; the
    score is then sacreBLEU's own function of the sums. On the same segments the result is `score_corpus`'s, bit for
    bit.
    """

    def __init__(self) -> None:
        self.chrf, self.bleu = CHRF(), BLEU(tokenize="zh")

    def page_statistics(self, hypotheses: Sequence[str],
                        references: Sequence[str]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The (chrF, BLEU) statistics of one page: the sums over its segments, one reference for each."""
        chrf = self.chrf._extract_corpus_statistics(list(hypotheses), [list(references)])
        bleu = self.bleu._extract_corpus_statistics(list(hypotheses), [list(references)])
        return tuple(_column_sums(chrf)), tuple(_column_sums(bleu))

    def chrf_score(self, statistics: Sequence[int]) -> float:
        return self.chrf._compute_score_from_stats(list(statistics)).score

    def bleu_score(self, statistics: Sequence[int]) -> float:
        return self.bleu._compute_score_from_stats(list(statistics)).score

    def signatures(self) -> dict[str, str]:
        """Known once the metrics have seen references, so after the first page's statistics."""
        return {"chrf": str(self.chrf.get_signature()), "bleu": str(self.bleu.get_signature())}


@dataclass(frozen=True)
class PageNumbers:
    """One config's numbers for one page: the sufficient statistics of every metric."""

    segments: int
    chrf: tuple[int, ...]  # sacreBLEU's statistics, summed over the page's segments
    bleu: tuple[int, ...]
    comet: float  # the sum of the segments' COMET scores
    term_hits: int  # over the entries that are not borderline
    term_occurrences: int
    mqm_sum: float  # the sum of the scored segments' MQM scores
    mqm_scored: int


@dataclass(frozen=True)
class Material:
    """One config's numbers over a comparison's pages."""

    pages: list[PageNumbers]  # by position in the comparison's page list
    renderings: list[dict[int, Counter[str]]]  # for each (book, entry): position -> its rendered occurrences


def _column_sums(rows: Sequence[Sequence[int]]) -> list[int]:
    return [sum(column) for column in zip(*rows)]


def comparison_pages(runs: Sequence[report.Run], config: str, other: str) -> list[tuple[str, int]]:
    """The pages that both configs scored, per book, pooled over the books as one list sorted by the report's book order
    and then by page index."""
    pages: list[tuple[str, int]] = []
    for book in sorted({r.book for r in runs}, key=report._book_order):
        scored = [{p for p, status in r.statuses.items() if status == "scored"}
                  for r in runs if r.book == book and r.config in (config, other)]
        pages += [(book, p) for p in sorted(set.intersection(*scored))]
    return pages


def build_material(runs: Sequence[report.Run], pages: Sequence[tuple[str, int]], scorer: SacreBLEU) -> Material:
    """One config's numbers over `pages`; `runs` are the config's eval runs, one for each book."""
    by_book = {run.book: run for run in runs}
    segments = {book: _segments_by_page(run) for book, run in by_book.items()}
    terms = {book: _primary_term_counts(run) for book, run in by_book.items()}
    numbers = []
    for book, page_index in pages:
        run, page_segments = by_book[book], segments[book][page_index]
        chrf, bleu = scorer.page_statistics([s["hyp"] for s in page_segments],
                                            [s["ref_zh_hans"] for s in page_segments])
        comet = [run.comet_scores[(book, page_index, s["idx"])] for s in page_segments]
        judged = [run.mqm.segments[(book, page_index, s["idx"])] for s in page_segments]
        scored = [j["score"] for j in judged if j["status"] == "scored"]
        hits, occurrences = terms[book].get(page_index, (0, 0))
        numbers.append(PageNumbers(len(page_segments), chrf, bleu, math.fsum(comet), hits, occurrences,
                                   math.fsum(scored), len(scored)))
    position = {page: i for i, page in enumerate(pages)}
    renderings = []
    for run in runs:
        for _, occurrences in run.consistency.entries:
            by_position: dict[int, Counter[str]] = {}
            for o in occurrences:
                i = position.get((run.book, o.page_index))
                if i is not None and o.statuses[TEXT] == "rendered":
                    by_position.setdefault(i, Counter())[o.renderings[TEXT]] += 1
            if by_position:
                renderings.append(by_position)
    return Material(numbers, renderings)


def _segments_by_page(run: report.Run) -> dict[int, list[dict]]:
    by_page: dict[int, list[dict]] = {}
    for segment in run.segments:
        by_page.setdefault(segment["page_index"], []).append(segment)
    return by_page


def _primary_term_counts(run: report.Run) -> dict[int, tuple[int, int]]:
    """For each page, the hits and the occurrences of the entries that are not borderline."""
    counts: dict[int, list[int]] = {}
    for _, borderline, occurrences in run.term_entries:
        if borderline:
            continue
        for page_index, _, hit in occurrences:
            count = counts.setdefault(page_index, [0, 0])
            count[0] += hit
            count[1] += 1
    return {page: (hits, occurrences) for page, (hits, occurrences) in counts.items()}


def sample_sums(material: Material, drawn: Sequence[int], keys: Sequence[str]) -> dict[str, tuple]:
    """What the metrics in `keys` add up over the drawn pages (positions in the page list; a page drawn twice is added
    twice)."""
    pages = [material.pages[i] for i in drawn]
    sums: dict[str, tuple] = {"segments": (sum(p.segments for p in pages),)}
    if "chrf" in keys:
        sums["chrf"] = tuple(_column_sums([p.chrf for p in pages]))
    if "bleu" in keys:
        sums["bleu"] = tuple(_column_sums([p.bleu for p in pages]))
    if "comet" in keys:
        sums["comet"] = (math.fsum(p.comet for p in pages),)
    if "term_accuracy" in keys:
        sums["term_accuracy"] = (sum(p.term_hits for p in pages), sum(p.term_occurrences for p in pages))
    if "mqm" in keys:
        sums["mqm"] = (math.fsum(p.mqm_sum for p in pages), sum(p.mqm_scored for p in pages))
    if "term_consistency" in keys:
        times = [0] * len(material.pages)
        for i in drawn:
            times[i] += 1
        sums["term_consistency"] = _consistency_sums(material.renderings, times)
    return sums


def _consistency_sums(renderings: Sequence[dict[int, Counter[str]]], times: Sequence[int]) -> tuple[int, int, int]:
    """(entries, rendered occurrences, most common renderings' counts) of the entries with enough rendered occurrences,
    each page counted as often as it was drawn."""
    entries = rendered_total = most_common_total = 0
    for by_position in renderings:
        merged: Counter[str] = Counter()
        for i, counts in by_position.items():
            if times[i]:
                for rendering, n in counts.items():
                    merged[rendering] += times[i] * n
        rendered = sum(merged.values())
        if rendered >= report.CONSISTENCY_MIN_RENDERED:
            entries += 1
            rendered_total += rendered
            most_common_total += max(merged.values())
    return entries, rendered_total, most_common_total


def sample_values(sums: dict[str, tuple], scorer: SacreBLEU) -> dict[str, float | None]:
    """The metrics of a sample from its sums; None where the metric is undefined on the sample."""
    values: dict[str, float | None] = {}
    segments = sums["segments"][0]
    for key, total in sums.items():
        if key == "chrf":
            values[key] = scorer.chrf_score(total)
        elif key == "bleu":
            values[key] = scorer.bleu_score(total)
        elif key == "comet":
            values[key] = total[0] / segments if segments else None
        elif key == "term_accuracy":
            values[key] = total[0] / total[1] if total[1] else None
        elif key == "mqm":
            values[key] = total[0] / total[1] if total[1] else None
        elif key == "term_consistency":
            values[key] = total[2] / total[1] if total[1] else None
    return values


def values_on(material: Material, drawn: Sequence[int], scorer: SacreBLEU,
              keys: Sequence[str]) -> dict[str, float | None]:
    """The metrics in `keys` on the drawn pages."""
    return sample_values(sample_sums(material, drawn, keys), scorer)


def draws(size: int, resamples: int, seed: int) -> Iterator[list[int]]:
    """The resamples of a comparison with `size` pages: `rng.choices(pages, k=len(pages))` with a new
    `random.Random(seed)`, drawn in order, each as positions in the page list (the same draw as of the pages
    themselves)."""
    rng = random.Random(seed)
    positions = list(range(size))
    for _ in range(resamples):
        yield rng.choices(positions, k=size)


def _support(key: str, sums: tuple, segments: int) -> dict:
    """What a full-sample value rests on, in the counts the report shows."""
    if key in ("chrf", "bleu", "comet"):
        return {"segments": segments}
    if key == "term_accuracy":
        return {"hits": sums[0], "occurrences": sums[1]}
    if key == "mqm":
        return {"scored": sums[1], "segments": segments}
    return {"entries": sums[0], "rendered": sums[1], "most_common_count": sums[2]}


# --- the labels -------------------------------------------------------------------------------------------------------

def extractor_checks(runs: Sequence[report.Run], config: str, other: str,
                     pages: Sequence[tuple[str, int]]) -> list[dict]:
    """For each run of the comparison (one config on one book), the extractor's accuracy on the references over the
    comparison's pages, and whether consistency is reportable there (the report's rule). A run with no page in the
    comparison adds nothing to the interval and is not listed."""
    wanted = set(pages)
    books = {book for book, _ in pages}
    rows = []
    for run in runs:
        if run.config not in (config, other) or run.book not in books:
            continue
        kept = [o for _, occurrences in run.consistency.entries for o in occurrences
                if (run.book, o.page_index) in wanted]
        checked, correct = sum(o.checked for o in kept), sum(o.checked and o.correct for o in kept)
        rows.append({"config": run.config, "book": run.book, "checked": checked, "correct": correct,
                     "accuracy": correct / checked if checked else None,
                     "reportable": checked > 0 and correct * 100 >= checked * report.CONSISTENCY_REPORTABLE_PERCENT})
    return rows


def label_of(key: str, runs: Sequence[report.Run], config: str, other: str,
             pages: Sequence[tuple[str, int]]) -> dict:
    """Whether the interval of `key` is a result or for reference only, and why."""
    if key == "mqm":
        return {"use": REFERENCE_ONLY, "why": MQM_WHY, "judge_model": runs[0].mqm.judge_model}
    if key == "term_consistency":
        extractor = extractor_checks(runs, config, other, pages)
        low = [f"{row['config']} on {row['book']}" for row in extractor if not row["reportable"]]
        why = None if not low else (
            f"the extractor's accuracy on the references is under {report.CONSISTENCY_REPORTABLE_PERCENT}% in "
            f"{', '.join(low)} (OUTLINE 4.7)")
        return {"use": RESULT if not low else REFERENCE_ONLY, "why": why,
                "judge_model": runs[0].consistency.judge_model, "extractor": extractor}
    return {"use": RESULT, "why": None}


# --- one comparison ---------------------------------------------------------------------------------------------------

def compare(runs: Sequence[report.Run], config: str, other: str, scorer: SacreBLEU, resamples: int) -> dict:
    """The bootstrap of one comparison: its pages, and for each metric the values, the point estimate and the
    interval."""
    name = f"{config} vs {other}"
    keys = METRICS[config, other]
    pages = comparison_pages(runs, config, other)
    if not pages:
        raise Refused(f"{name}: no page was scored by both {config} and {other}")
    materials = {c: build_material([r for r in runs if r.config == c], pages, scorer) for c in (config, other)}
    everything = list(range(len(pages)))
    sums = {c: sample_sums(m, everything, keys) for c, m in materials.items()}
    full_sample = {c: sample_values(sums[c], scorer) for c in materials}
    differences: dict[str, list[float]] = {key: [] for key in keys}
    for drawn in draws(len(pages), resamples, SEED):
        mine, theirs = (values_on(materials[c], drawn, scorer, keys) for c in (config, other))
        for key in keys:
            if mine[key] is not None and theirs[key] is not None:
                differences[key].append(mine[key] - theirs[key])
    metrics = {}
    for key in keys:
        low, high = (percentile(differences[key], q) for q in PERCENTILES)
        mine, theirs = full_sample[config][key], full_sample[other][key]
        metrics[key] = {
            "definition": DEFINITIONS[key],
            "role": "primary" if PRIMARY[config, other] == key else "secondary",
            "values": {c: full_sample[c][key] for c in (config, other)},
            "support": {c: _support(key, sums[c][key], sums[c]["segments"][0]) for c in (config, other)},
            "point_estimate": None if mine is None or theirs is None else mine - theirs,
            "interval": None if low is None else {"low": low, "high": high},
            "defined_resamples": len(differences[key]),
            "undefined_resamples": resamples - len(differences[key]),
            "label": label_of(key, runs, config, other, pages),
        }
    by_book = {book: [page_index for b, page_index in pages if b == book]
               for book in sorted({r.book for r in runs}, key=report._book_order)}
    return {"name": name, "config": config, "other": other, "n_pages": len(pages), "pages": by_book, "metrics": metrics}


# --- the inputs -------------------------------------------------------------------------------------------------------

def load_runs(directories: Sequence[Path], glossary_dir: Path, *,
              mqm_judge: str | None = None) -> tuple[list[report.Run], dict]:
    """The eval runs, read and checked as the report reads and checks them, then the bootstrap's own requirements: the
    three configs, each run with its consistency file and its MQM file. Also the report's own document, for its
    warnings. Raises `Refused`, having written nothing."""
    directories = list(directories)
    document = report.build_report(directories, glossary_dir, mqm_judge=mqm_judge)  # the report's refusals, first
    runs = report._select_mqm([report.load_run(Path(d), glossary_dir) for d in directories], mqm_judge)
    configs = {run.config for run in runs}
    if configs != set(CONFIGS):
        got = "; ".join(f"{name} ({', '.join(str(r.directory) for r in runs if r.config == name)})"
                        for name in sorted(configs, key=report._config_order))
        raise Refused(f"the bootstrap takes the configs {', '.join(CONFIGS)}, one eval run for each book; it got {got}")
    for run in runs:
        if run.consistency is None:
            raise Refused(f"{run.directory}: it has no {report.CONSISTENCY_GLOB}: the bootstrap gives term consistency "
                          "an interval, so every run needs one (score this one with eval/consistency.py)")
        if run.mqm is None:
            raise Refused(f"{run.directory}: it has no {report.MQM_GLOB}: the bootstrap gives GEMBA-MQM an interval, "
                          "so every run needs one (score this one with eval/mqm.py)")
    runs.sort(key=lambda r: (report._config_order(r.config), report._book_order(r.book)))
    return runs, document


def _inputs(runs: Sequence[report.Run], glossary_dir: Path, mqm_judge: str | None) -> dict:
    first = runs[0]
    return {
        "glossary_dir": str(glossary_dir),
        "mqm_judge": mqm_judge,
        "consistency": {"judge_model": first.consistency.judge_model, "rules": first.consistency.document["rules"],
                        "min_rendered": report.CONSISTENCY_MIN_RENDERED,
                        "threshold": report.CONSISTENCY_REPORTABLE_PERCENT / 100},
        "mqm": {"judge_model": first.mqm.judge_model, "file": first.mqm.name},
        "runs": [{
            "directory": str(run.directory), "eval_id": run.eval_id, "config": run.config, "book": run.book,
            "segments_sha256": run.segments_sha256, "glossary_sha256": run.terms.get("glossary_sha256"),
            "files": {"comet": report.COMET_FILE, "terms": "terms.json", "consistency": run.consistency.name,
                      "mqm": run.mqm.name}} for run in runs],
    }


def _method(resamples: int, scorer: SacreBLEU) -> dict:
    return {
        "protocol": "eval/PROTOCOL.md, Statistics (OUTLINE 4.3, 8)",
        "resamples": resamples,
        "seed": SEED,
        "generator": "random.Random(seed), a new one for each comparison; resample r is "
                     "rng.choices(pages, k=len(pages)) on the comparison's page list, drawn in order",
        "pages": "for each comparison, the pages both of its configs scored, per book, pooled over the books as one "
                 "list sorted by the report's book order and then by page index",
        "duplicated_pages": "a page drawn more than once counts each time it is drawn, with all of its segments and "
                            "occurrences, in every metric",
        "difference": "full minus the other config, on each resample",
        "percentiles": list(PERCENTILES),
        "interval": "the 2.5th and the 97.5th percentile of the resamples' differences (metrics.percentile: linear "
                    "interpolation between ranks)",
        "point_estimate": "the difference on the comparison's pages, without resampling",
        "undefined": "a resample on which a metric is undefined (no occurrence, no rendered entry, no scored segment, "
                     "in either config) gives no difference for that metric; the interval uses the others",
        "sacrebleu": scorer.signatures(),
    }


def build_bootstrap(directories: Sequence[Path], glossary_dir: Path, *, mqm_judge: str | None = None,
                    resamples: int = RESAMPLES) -> dict:
    """The bootstrap of the eval runs in `directories` (what bootstrap.json holds). `mqm_judge` names the judge model
    whose MQM files are used (needed only when a run has files of more than one). `resamples` is the protocol's number
    except in tests. Raises `Refused`, having written nothing."""
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    runs, base = load_runs(directories, glossary_dir, mqm_judge=mqm_judge)
    scorer = SacreBLEU()
    comparisons = [compare(runs, config, other, scorer, resamples) for config, other in COMPARISONS]
    state = git_state(ROOT)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code": {"git_commit": state["commit"], "git_dirty": state["dirty"]},
        "inputs": _inputs(runs, glossary_dir, mqm_judge),
        "method": _method(resamples, scorer),
        "comparisons": comparisons,
        "warnings": base["warnings"],
    }


def write_bootstrap(document: dict, out_root: Path) -> Path:
    """Write bootstrap.json into a new directory `<out_root>/<UTC time>-bootstrap` (`-2`, `-3`... if that exists) and
    return it. The file is written under a temporary name and renamed, and an existing directory is never written
    into."""
    out_root.mkdir(parents=True, exist_ok=True)
    base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-bootstrap"
    candidate, n = base, 1
    while True:
        try:
            (out_root / candidate).mkdir()
            break
        except FileExistsError:
            n += 1
            candidate = f"{base}-{n}"
    out_dir = out_root / candidate
    temporary = out_dir / "bootstrap.json.tmp"
    try:
        text = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, out_dir / "bootstrap.json")
    except BaseException:
        temporary.unlink(missing_ok=True)
        out_dir.rmdir()
        raise
    return out_dir


# --- the printed summary ----------------------------------------------------------------------------------------------

TITLES = {  # key -> (title, how a value is shown, how a difference is shown)
    "comet": ("COMET (wmt22-comet-da)", ".4f", "+.4f"),
    "chrf": ("chrF", ".2f", "+.2f"),
    "bleu": ("BLEU", ".2f", "+.2f"),
    "term_accuracy": ("term accuracy (non-borderline)", "%", "pp"),
    "term_consistency": ("term consistency (strict)", "%", "pp"),
    "mqm": ("GEMBA-MQM", ".3f", "+.3f"),
}
LABEL_WIDTH = 32


def _show(key: str, value: float | None, *, difference: bool = False) -> str:
    if value is None:
        return "n/a"
    _, shown, change = TITLES[key]
    if shown == "%":
        return f"{value * 100:+.1f} pp" if difference else f"{value:.1%}"
    return format(value, change if difference else shown)


def _interval(key: str, interval: dict | None) -> str:
    if interval is None:
        return "n/a"
    if TITLES[key][1] == "%":
        return f"[{interval['low'] * 100:+.1f}, {interval['high'] * 100:+.1f}] pp"
    return f"[{format(interval['low'], TITLES[key][2])}, {format(interval['high'], TITLES[key][2])}]"


def render(document: dict) -> str:
    """The short table that eval/bootstrap.py prints."""
    method, code = document["method"], document["code"]
    runs = document["inputs"]["runs"]
    books = list(dict.fromkeys(run["book"] for run in runs))
    state = "dirty tree" if code["git_dirty"] else "clean tree" if code["git_dirty"] is False else "unknown tree"
    lines = [f"Paired bootstrap by page of {len(runs)} eval runs ({', '.join(books)}), {document['generated_at']}",
             f"{method['resamples']:,} resamples, seed {method['seed']}; the interval is the "
             f"{PERCENTILES[0]}th to {PERCENTILES[1]}th percentile of the difference, full minus the other config; "
             f"code {str(code['git_commit'])[:12]} ({state})"]
    for comparison in document["comparisons"]:
        config, other = comparison["config"], comparison["other"]
        pages = ", ".join(f"{book} {len(p)}" for book, p in comparison["pages"].items())
        rows: list[list[str]] = [["", config, other, "difference", "95% interval", "defined", ""]]
        for key, metric in comparison["metrics"].items():
            label = metric["label"]
            notes = ["primary metric"] if metric["role"] == "primary" else []
            if label["use"] != RESULT:
                notes.append(label["use"] + (f" ({label['judge_model']})" if key == "mqm" else ""))
            rows.append([TITLES[key][0], _show(key, metric["values"][config]), _show(key, metric["values"][other]),
                         _show(key, metric["point_estimate"], difference=True), _interval(key, metric["interval"]),
                         f"{metric['defined_resamples']}/{method['resamples']}", "; ".join(notes)])
        widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
        widths[0] = max(widths[0], LABEL_WIDTH)
        lines += ["", f"{comparison['name']}: {comparison['n_pages']} pages ({pages})"]
        for row in rows:
            cells = [row[0].ljust(widths[0])] + [cell.rjust(widths[i]) for i, cell in enumerate(row[1:-1], start=1)]
            lines.append("  ".join(cells + [row[-1]]).rstrip())
    sacrebleu = method["sacrebleu"]
    lines += ["", f"sacreBLEU  chrF {sacrebleu['chrf']}   BLEU {sacrebleu['bleu']}"]
    if document["warnings"]:
        lines += ["", "warnings:", *(f"  {w}" for w in document["warnings"])]
    else:
        lines += ["", "warnings: none"]
    return "\n".join(lines)
