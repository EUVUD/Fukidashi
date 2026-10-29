"""The gold glossary: the ground truth for term accuracy (OUTLINE 4.7). Terms only, never sentences from the dataset.

An assistant drafts it from the Japanese text and the OpenCC-converted reference translations alone, and the author
reviews every entry. This module reads those two fields of the annotation file and nothing else: it imports nothing
that can reach a stored Run or Translation. The command-line entry point is eval/glossary.py; the logic lives here so
that tests can import it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from opencc import OpenCC

from ..datasets.openmantra import Page
from .references import OPENCC_CONFIG, to_zh_hans
from .splits import split_of

CATEGORIES = ("character", "term")  # mirror Character Profile and Glossary entry (OUTLINE 3.3)
MAX_TERM_CHARS = 20  # a guard: a longer "term" is probably a sentence, and sentences must not enter git (CC BY-NC)
MIN_SEGMENTS = 2  # an entry that occurs once cannot show consistency

_HEADER_KEYS = ("book_title", "split", "opencc_config", "drafted_by", "reviewed_by_author", "entries")
_ENTRY_REQUIRED = ("id", "category", "source", "target")
_ENTRY_OPTIONAL = ("source_aliases", "target_variants", "note")
_ENTRY_FLAGS = ("borderline",)  # required, and a boolean


@dataclass(frozen=True)
class Problem:
    code: str  # which rule failed, stable enough for tests to key on
    where: str  # "header", "file" or an entry id
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message} [{self.code}]"


@dataclass(frozen=True)
class Entry:
    id: str
    category: str
    source: str  # the Japanese form as it appears in `text_ja`
    target: str  # the zh-Hans rendering the converted reference uses most often
    source_aliases: tuple[str, ...] = ()  # other Japanese forms of the same thing; none contains `source`
    target_variants: tuple[str, ...] = ()  # the other renderings the reference uses, and the target in mainland forms
    # (which `t2s` never produces, so `check` does not look for it in the reference); all of them are also correct
    note: str | None = None
    borderline: bool = False  # an ordinary word: term accuracy leaves it out of its primary score (OUTLINE 4.7)

    @property
    def sources(self) -> tuple[str, ...]:
        return (self.source, *self.source_aliases)

    @property
    def targets(self) -> tuple[str, ...]:
        return (self.target, *self.target_variants)


@dataclass(frozen=True)
class Glossary:
    book_title: str
    split: str
    opencc_config: str
    drafted_by: str
    reviewed_by_author: str | None  # an ISO date once the author has reviewed every entry
    entries: tuple[Entry, ...]
    problems: tuple[Problem, ...] = ()  # what was wrong with the file's shape; such entries are not in `entries`


@dataclass(frozen=True)
class Occurrence:
    """One segment whose `text_ja` contains an entry's source or alias."""

    page_index: int
    region_id: int  # the segment's `idx`, the id the prompt, the image badge and the row order use
    text_ja: str
    text_zh: str  # the reference converted to zh-Hans
    covered: bool  # the converted reference contains the target or a variant


@dataclass(frozen=True)
class Summary:
    entries: int
    by_category: dict[str, int]
    multi_page: int  # entries that occur on 2 or more pages
    mean_coverage: float | None  # mean of the entries' coverage
    pooled_coverage: float | None  # covered segments over all entries' segments


# --- loading -----------------------------------------------------------------------------------------------------

def load_glossary(path: Path) -> Glossary:
    """Read a glossary file. Whatever is wrong with its shape ends up in `Glossary.problems`, not in an exception."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as e:
        return _empty(Problem("shape", "file", f"cannot read {path}: {e}"))
    except json.JSONDecodeError as e:
        return _empty(Problem("shape", "file", f"{path} is not valid JSON: {e}"))
    return parse_glossary(raw)


def _empty(problem: Problem) -> Glossary:
    return Glossary("", "", "", "", None, (), (problem,))


def parse_glossary(raw: Any) -> Glossary:
    if not isinstance(raw, dict):
        return _empty(Problem("shape", "file", "the top level must be a JSON object"))
    problems: list[Problem] = []
    for key in sorted(set(raw) - set(_HEADER_KEYS)):
        problems.append(Problem("shape", "header", f"unknown field `{key}`"))

    def header_text(key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            problems.append(Problem("shape", "header", f"`{key}` must be a non-empty string"))
            return ""
        return value

    book_title, split, opencc_config, drafted_by = (
        header_text(k) for k in ("book_title", "split", "opencc_config", "drafted_by"))

    reviewed = raw.get("reviewed_by_author")
    if "reviewed_by_author" not in raw or not (reviewed is None or _is_iso_date(reviewed)):
        problems.append(Problem("shape", "header", "`reviewed_by_author` must be null or a date like 2026-10-28"))
        reviewed = None

    entries: list[Entry] = []
    raw_entries = raw.get("entries")
    if not isinstance(raw_entries, list):
        problems.append(Problem("shape", "header", "`entries` must be a list"))
        raw_entries = []
    for position, item in enumerate(raw_entries):
        entry, entry_problems = _parse_entry(item, position)
        problems.extend(entry_problems)
        if entry is not None:
            entries.append(entry)
    return Glossary(book_title, split, opencc_config, drafted_by, reviewed, tuple(entries), tuple(problems))


def _is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _parse_entry(raw: Any, position: int) -> tuple[Entry | None, list[Problem]]:
    where = f"entries[{position}]"
    if not isinstance(raw, dict):
        return None, [Problem("shape", where, "an entry must be a JSON object")]
    if isinstance(raw.get("id"), str) and raw["id"].strip():
        where = raw["id"]
    problems: list[Problem] = []
    for key in sorted(set(raw) - set(_ENTRY_REQUIRED) - set(_ENTRY_OPTIONAL) - set(_ENTRY_FLAGS)):
        problems.append(Problem("shape", where, f"unknown field `{key}`"))
    for key in _ENTRY_REQUIRED:
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            problems.append(Problem("shape", where, f"`{key}` must be a non-empty string"))
    for key in _ENTRY_FLAGS:
        if not isinstance(raw.get(key), bool):
            problems.append(Problem("shape", where, f"`{key}` must be true or false"))
    lists: dict[str, tuple[str, ...]] = {}
    for key in ("source_aliases", "target_variants"):
        value = raw.get(key, [])
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            problems.append(Problem("shape", where, f"`{key}` must be a list of strings"))
            value = []
        lists[key] = tuple(value)
    note = raw.get("note")
    if note is not None and not isinstance(note, str):
        problems.append(Problem("shape", where, "`note` must be a string or null"))
    if problems:
        return None, problems
    return Entry(raw["id"], raw["category"], raw["source"], raw["target"], lists["source_aliases"],
                 lists["target_variants"], note, raw["borderline"]), []


# --- segments and occurrences ------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _hans(text_zh_hant: str) -> str:
    return to_zh_hans(text_zh_hant)


@lru_cache(maxsize=1)
def _taiwan_to_mainland() -> OpenCC:
    return OpenCC("tw2s")


def mainland_form(target: str) -> str:
    """`target` with the Taiwan character forms that `t2s` keeps written the mainland way (封著眼 -> 封着眼).

    A glossary lists that form as a variant: the reference cannot contain it, but a system's translation will.
    """
    return _taiwan_to_mainland().convert(target)


@dataclass(frozen=True)
class _Segment:
    page_index: int
    region_id: int
    ja: str
    zh: str  # converted to zh-Hans


def _segments(pages: Sequence[Page]) -> list[_Segment]:
    return [_Segment(p.page_index, s.index, s.ja, _hans(s.zh_hant))
            for p in sorted(pages, key=lambda p: p.page_index) for s in p.segments]


def book_pages(pages: Sequence[Page], book_title: str) -> list[Page]:
    return [p for p in pages if p.book_title == book_title]


def occurrences(entry: Entry, pages: Sequence[Page]) -> list[Occurrence]:
    """The segments of `pages` (one book's) whose `text_ja` contains the source or an alias, in reading order.

    Matching is by substring, on the Japanese as it is and on the reference after OpenCC `t2s`.
    """
    sources = [s for s in entry.sources if s]
    targets = [t for t in entry.targets if t]
    return [
        Occurrence(seg.page_index, seg.region_id, seg.ja, seg.zh, any(t in seg.zh for t in targets))
        for seg in _segments(pages) if any(s in seg.ja for s in sources)
    ]


def coverage(occs: Sequence[Occurrence]) -> float | None:
    """The share of the segments whose converted reference contains the target or a variant; None with no segment."""
    return sum(o.covered for o in occs) / len(occs) if occs else None


# --- validation --------------------------------------------------------------------------------------------------

def check(glossary: Glossary, pages: Sequence[Page]) -> list[Problem]:
    """Every problem found, empty when the glossary is valid. `pages` may hold several books."""
    problems = list(glossary.problems)
    book = book_pages(pages, glossary.book_title)

    if glossary.book_title and not book:
        problems.append(Problem("book", "header", f"the dataset has no book {glossary.book_title!r}"))
    if glossary.book_title and glossary.split:
        expected = split_of(glossary.book_title)
        if expected is None:
            problems.append(Problem("split", "header", f"splits.py lists {glossary.book_title!r} in neither split"))
        elif glossary.split != expected:
            problems.append(Problem(
                "split", "header", f"split is {glossary.split!r} but splits.py puts {glossary.book_title!r} in "
                f"{expected!r}"))
    if glossary.opencc_config and glossary.opencc_config != OPENCC_CONFIG:
        problems.append(Problem(
            "opencc", "header", f"opencc_config is {glossary.opencc_config!r}, the references use {OPENCC_CONFIG!r}"))

    ids: dict[str, int] = {}
    owners: dict[str, list[int]] = {}  # term -> positions of the entries that list it as a source or alias
    for position, entry in enumerate(glossary.entries):
        where = entry.id
        if entry.id in ids:
            problems.append(Problem(
                "duplicate_id", where, f"id is also used by the entry at position {ids[entry.id]}"))
        ids.setdefault(entry.id, position)
        if entry.category not in CATEGORIES:
            problems.append(Problem(
                "category", where, f"category {entry.category!r} is not one of {', '.join(CATEGORIES)}"))

        for term in (*entry.sources, *entry.targets):
            if not term.strip():
                problems.append(Problem("term_empty", where, "a source, alias, target or variant is empty"))
            elif len(term) > MAX_TERM_CHARS:
                problems.append(Problem(
                    "term_too_long", where, f"{term!r} is {len(term)} characters, the limit is {MAX_TERM_CHARS}"))
        for alias in entry.source_aliases:
            if entry.source and entry.source in alias:
                problems.append(Problem(
                    "alias_contains_source", where, f"alias {alias!r} contains the source {entry.source!r}"))
        for variant in entry.target_variants:
            if variant == entry.target:
                problems.append(Problem("variant_repeats_target", where, f"variant {variant!r} is the target"))
        for term in dict.fromkeys(t for t in entry.sources if t.strip()):
            owners.setdefault(term, []).append(position)
        if len(set(entry.sources)) != len(entry.sources):
            problems.append(Problem("term_reused", where, "lists the same source or alias twice"))

        if not book:
            continue
        occs = occurrences(entry, book)
        for source in dict.fromkeys(s for s in entry.sources if s.strip()):
            if not any(source in o.text_ja for o in occs):
                problems.append(Problem("source_not_found", where, f"{source!r} occurs in no segment"))
        if len(occs) < MIN_SEGMENTS:
            problems.append(Problem(
                "too_few_segments", where, f"occurs in {len(occs)} segment(s), at least {MIN_SEGMENTS} are needed"))
        mainland = mainland_form(entry.target)
        for target in dict.fromkeys(t for t in entry.targets if t.strip()):
            if target == mainland and target != entry.target:
                continue  # the target in mainland forms: the converted reference cannot contain it, a translation can
            if not any(target in o.text_zh for o in occs):
                problems.append(Problem(
                    "target_not_found", where,
                    f"{target!r} is in no converted reference of the segments that contain its source"))

    for term, positions in owners.items():
        if len(positions) > 1:
            names = ", ".join(glossary.entries[p].id for p in positions)
            problems.append(Problem("term_reused", names, f"{term!r} is a source or alias of more than one entry"))
    return problems


def summarize(glossary: Glossary, pages: Sequence[Page]) -> Summary:
    book = book_pages(pages, glossary.book_title)
    by_category: dict[str, int] = {}
    multi_page = covered = total = 0
    per_entry: list[float] = []
    for entry in glossary.entries:
        by_category[entry.category] = by_category.get(entry.category, 0) + 1
        occs = occurrences(entry, book)
        multi_page += len({o.page_index for o in occs}) >= 2
        if occs:
            per_entry.append(coverage(occs))
            covered += sum(o.covered for o in occs)
            total += len(occs)
    return Summary(len(glossary.entries), by_category, multi_page,
                   sum(per_entry) / len(per_entry) if per_entry else None, covered / total if total else None)


# --- review sheet ------------------------------------------------------------------------------------------------

EVIDENCE_COVERED = 3  # segments quoted per entry where the reference has the target or a variant
EVIDENCE_UNCOVERED = 2  # and where it does not: those are the ones a reviewer needs to see


def render_sheet(glossary: Glossary, pages: Sequence[Page]) -> str:
    """Markdown for the author's review: one table row per entry, then each entry's evidence. Quotes the dataset."""
    book = book_pages(pages, glossary.book_title)
    summary = summarize(glossary, book)
    lines = [
        f"# Review sheet: {glossary.book_title} ({glossary.split})",
        "",
        f"Drafted by: {glossary.drafted_by}",
        f"Reviewed by author: {glossary.reviewed_by_author or 'not yet'}",
        "",
        f"{summary.entries} entries ({_by_category(summary)}); {summary.multi_page} occur on 2 or more pages; "
        f"mean coverage {_percent(summary.mean_coverage)}, pooled {_percent(summary.pooled_coverage)}.",
        "",
        "Coverage is the share of an entry's segments whose converted reference (OpenCC "
        f"`{OPENCC_CONFIG}`) contains the target or a variant. Low coverage means the reference leaves the term out "
        "or words it another way; the evidence below shows which. Segments are written `page#region`.",
        "",
        "| # | id | category | source (aliases) | target (variants) | segments | pages | coverage |",
        "|---|---|---|---|---|---|---|---|",
    ]
    per_entry = [(entry, occurrences(entry, book)) for entry in glossary.entries]
    for number, (entry, occs) in enumerate(per_entry, start=1):
        lines.append(
            f"| {number} | {entry.id} | {entry.category} | {_forms(entry.source, entry.source_aliases)} | "
            f"{_forms(entry.target, entry.target_variants)} | {len(occs)} | {len({o.page_index for o in occs})} | "
            f"{_percent(coverage(occs))} |")
    lines += ["", "## Evidence"]
    for entry, occs in per_entry:
        lines += ["", f"### {entry.id}: {entry.source} → {entry.target}", ""]
        lines.append("Segments containing each rendering: " + ", ".join(
            f"{t} {sum(t in o.text_zh for o in occs)}/{len(occs)}" for t in entry.targets if t))
        if entry.note:
            lines.append(f"Note: {entry.note}")
        covered = _spread([o for o in occs if o.covered], EVIDENCE_COVERED)
        uncovered = _spread([o for o in occs if not o.covered], EVIDENCE_UNCOVERED)
        lines.append("")
        lines += [_quote(o) for o in sorted(covered + uncovered, key=lambda o: (o.page_index, o.region_id))]
    return "\n".join(lines) + "\n"


def _by_category(summary: Summary) -> str:
    return ", ".join(f"{name} {n}" for name, n in sorted(summary.by_category.items())) or "none"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _forms(main: str, others: Sequence[str]) -> str:
    return main + (f" ({', '.join(others)})" if others else "")


def _quote(o: Occurrence) -> str:
    mark = "covered" if o.covered else "NOT covered"
    return f"- {o.page_index}#{o.region_id} [{mark}] ja: {_oneline(o.text_ja)} / zh: {_oneline(o.text_zh)}"


def _oneline(text: str) -> str:
    return text.replace("\n", " ⏎ ")


# --- candidates --------------------------------------------------------------------------------------------------

_KANJI = "一-鿿々〆"  # CJK ideographs, 々, 〆
_KATAKANA = "ァ-ヺー・"  # katakana, the long vowel mark, the middle dot
_HONORIFICS = "さん|ちゃん|くん|君|様|さま|先輩|先生|殿"

# (kind, pattern with the candidate in group 1), in the order the sections are written
_PATTERNS = (
    ("honorific", re.compile(rf"([{_KANJI}{_KATAKANA}]{{1,8}}?)(?:{_HONORIFICS})")),
    ("bracket", re.compile(r"[「『]([^「」『』\n]{1,12})[」』]")),
    ("katakana", re.compile(rf"[{_KATAKANA}]{{2,}}")),
    ("kanji", re.compile(rf"[{_KANJI}]{{2,}}")),
)
CANDIDATE_EXAMPLES = 3  # segments quoted per candidate


@dataclass(frozen=True)
class Candidate:
    text: str
    kinds: tuple[str, ...]  # which patterns found it, in the order of `_PATTERNS`
    segments: int
    pages: int
    examples: tuple[Occurrence, ...]  # a few of its segments; there is no target yet, so `covered` is always False


def find_candidates(pages: Sequence[Page], *, min_segments: int = MIN_SEGMENTS) -> list[Candidate]:
    """Strings of one book's `text_ja` that may be names or terms, found by pattern and nothing else (no model).

    Only strings that occur in at least `min_segments` segments are listed, most frequent first.
    """
    segments = _segments(pages)
    kinds: dict[str, set[str]] = {}
    for seg in segments:
        for kind, pattern in _PATTERNS:
            for match in pattern.finditer(seg.ja):
                kinds.setdefault(match.group(1) if pattern.groups else match.group(0), set()).add(kind)
    order = {kind: i for i, (kind, _) in enumerate(_PATTERNS)}
    found = []
    for text, text_kinds in kinds.items():
        hits = [s for s in segments if text in s.ja]
        if len(hits) < min_segments:
            continue
        picked = _spread(hits, CANDIDATE_EXAMPLES)
        found.append(Candidate(
            text, tuple(sorted(text_kinds, key=order.__getitem__)), len(hits), len({s.page_index for s in hits}),
            tuple(Occurrence(s.page_index, s.region_id, s.ja, s.zh, False) for s in picked)))
    return sorted(found, key=lambda c: (-c.segments, -c.pages, c.text))


def _spread[T](items: Sequence[T], n: int) -> list[T]:
    """Up to n items spread over the list, first and last included, so the examples come from different pages."""
    if len(items) <= n:
        return list(items)
    if n < 2:
        return list(items[:n])
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def render_candidates(book_title: str, pages: Sequence[Page]) -> str:
    """Markdown listing find_candidates' result by kind. Quotes the dataset, so it goes under runs/, not into git."""
    candidates = find_candidates(book_pages(pages, book_title))
    lines = [
        f"# Glossary candidates: {book_title}",
        "",
        "Found by pattern in `text_ja` and listed when they occur in at least 2 segments. Most are ordinary words: "
        "pick the names and recurring in-world terms. Segments are written `page#region`; `zh` is the reference "
        f"converted with OpenCC `{OPENCC_CONFIG}`.",
    ]
    for kind, _ in _PATTERNS:
        section = [c for c in candidates if c.kinds[0] == kind]
        lines += ["", f"## {kind} ({len(section)})"]
        for c in section:
            also = f", also {', '.join(c.kinds[1:])}" if len(c.kinds) > 1 else ""
            lines += ["", f"### {c.text}: {c.segments} segments, {c.pages} pages{also}", ""]
            lines += [f"- {o.page_index}#{o.region_id} ja: {_oneline(o.text_ja)} / zh: {_oneline(o.text_zh)}"
                      for o in c.examples]
    return "\n".join(lines) + "\n"
