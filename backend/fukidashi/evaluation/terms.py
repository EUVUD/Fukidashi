"""Term accuracy: how often the system's translation uses the gold glossary's rendering (OUTLINE 4.7, v0.22).

A scorer over a finished eval run, like COMET: its inputs are a glossary and the `segments.jsonl` lines of one book,
and nothing else. It reads no database and no Run, and of a line it reads only `page_index`, `idx`, `ja`, `hyp` and
`ref_zh_hans`, so a change of rules can score old runs again without a model call. The command-line entry point is
eval/terms.py; the logic lives here so that tests can import it.

The rules, per glossary entry:

- **Occurrence.** A segment counts when its Japanese contains the entry's source or an alias *and* its converted
  reference contains the target or a variant. Where the reference does not use the term it cannot say what is right,
  so the segment is *left out* (and listed, so the author can see what was dropped).
- **Hit.** An occurrence whose translation (`hyp`) contains the target or a variant.
- **Matching** is NFKC plus substring, on the text and on the glossary's terms, so a full-width / half-width
  difference does not matter. Character forms that `t2s` keeps (著 / 着) are variants in the glossary, not code here.

Scores, for the entries that are not borderline (`primary`) and for all entries: the pooled accuracy (hits over
occurrences) and the mean of the per-entry accuracies over the entries that have an occurrence. A ratio with no
occurrence is None, never 0.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .glossary import Entry, Glossary

RULES = "OUTLINE §4.7 (v0.22)"  # what terms.json says its numbers follow

_LINE_FIELDS = (("page_index", int), ("idx", int), ("ja", str), ("hyp", str), ("ref_zh_hans", str))


class TermsError(ValueError):
    """A segment line the scorer cannot read."""


@dataclass(frozen=True)
class OccurrenceScore:
    page_index: int
    idx: int  # the segment's `idx`, the region id
    hit: bool  # the translation contains the target or a variant


@dataclass(frozen=True)
class EntryScore:
    id: str
    borderline: bool
    occurrences: tuple[OccurrenceScore, ...]  # in reading order
    left_out: tuple[tuple[int, int], ...]  # (page_index, idx) of the segments the reference leaves the term out of

    @property
    def hits(self) -> int:
        return sum(o.hit for o in self.occurrences)

    @property
    def accuracy(self) -> float | None:
        return self.hits / len(self.occurrences) if self.occurrences else None


@dataclass(frozen=True)
class SetScore:
    """The score of a set of entries: the primary set (no borderline entries) or all of them."""

    entries: int  # entries in the set
    entries_with_occurrences: int  # the ones the mean is over
    occurrences: int
    hits: int
    accuracy: float | None  # pooled: hits / occurrences
    mean_over_entries: float | None  # mean of the entries' accuracies


@dataclass(frozen=True)
class TermScores:
    primary: SetScore
    all_entries: SetScore
    entries: tuple[EntryScore, ...]  # in the glossary's order

    def to_dict(self) -> dict:
        """What terms.json holds below the run's identity: the two scores and every entry's evidence."""
        return {
            "scores": {"primary": _set_dict(self.primary), "all_entries": _set_dict(self.all_entries)},
            "entries": [{
                "id": e.id,
                "borderline": e.borderline,
                "occurrences": len(e.occurrences),
                "hits": e.hits,
                "accuracy": e.accuracy,
                "left_out": [{"page_index": p, "idx": i} for p, i in e.left_out],
                "segments": [{"page_index": o.page_index, "idx": o.idx, "hit": o.hit} for o in e.occurrences],
            } for e in self.entries],
        }


def _set_dict(score: SetScore) -> dict:
    return {
        "entries": score.entries,
        "entries_with_occurrences": score.entries_with_occurrences,
        "occurrences": score.occurrences,
        "hits": score.hits,
        "accuracy": score.accuracy,
        "mean_over_entries": score.mean_over_entries,
    }


@dataclass(frozen=True)
class _Line:  # one segment, normalised for matching
    page_index: int
    idx: int
    ja: str
    hyp: str
    ref: str


def normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _lines(raw: Iterable[Mapping[str, Any]]) -> list[_Line]:
    lines = []
    for position, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise TermsError(f"segment line {position + 1} is not a JSON object")
        for key, kind in _LINE_FIELDS:
            value = item.get(key)
            if not isinstance(value, kind) or isinstance(value, bool):
                raise TermsError(f"segment line {position + 1} has no {kind.__name__} `{key}`")
        lines.append(_Line(item["page_index"], item["idx"], normalise(item["ja"]), normalise(item["hyp"]),
                           normalise(item["ref_zh_hans"])))
    return sorted(lines, key=lambda line: (line.page_index, line.idx))


def _forms(terms: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(normalise(t) for t in terms if t))


def _score_entry(entry: Entry, lines: Sequence[_Line]) -> EntryScore:
    sources, targets = _forms(entry.sources), _forms(entry.targets)
    occurrences: list[OccurrenceScore] = []
    left_out: list[tuple[int, int]] = []
    for line in lines:
        if not any(s in line.ja for s in sources):
            continue
        if not any(t in line.ref for t in targets):
            left_out.append((line.page_index, line.idx))
            continue
        occurrences.append(OccurrenceScore(line.page_index, line.idx, any(t in line.hyp for t in targets)))
    return EntryScore(entry.id, entry.borderline, tuple(occurrences), tuple(left_out))


def _set_score(entries: Sequence[EntryScore]) -> SetScore:
    occurrences = sum(len(e.occurrences) for e in entries)
    hits = sum(e.hits for e in entries)
    accuracies = [e.accuracy for e in entries if e.accuracy is not None]
    return SetScore(
        entries=len(entries), entries_with_occurrences=len(accuracies), occurrences=occurrences, hits=hits,
        accuracy=hits / occurrences if occurrences else None,
        mean_over_entries=sum(accuracies) / len(accuracies) if accuracies else None)


def score_terms(glossary: Glossary, segment_lines: Iterable[Mapping[str, Any]]) -> TermScores:
    """Term accuracy of one book's `segments.jsonl` lines against its glossary (rules in the module docstring).

    Raises TermsError for a line without the fields it reads. A glossary with problems is the caller's to refuse.
    """
    lines = _lines(segment_lines)
    scored = tuple(_score_entry(entry, lines) for entry in glossary.entries)
    return TermScores(_set_score([e for e in scored if not e.borderline]), _set_score(scored), scored)
