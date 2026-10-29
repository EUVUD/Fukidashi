"""Term consistency: whether a text keeps one rendering of a term across a book (OUTLINE 4.7, v0.25).

The KB ablation's secondary metric (term accuracy is its primary one, OUTLINE 8). Term accuracy asks whether the system
uses the reference translator's rendering of a term; consistency asks whether it keeps one rendering of it, which is
what the KB is for. This module holds the rules: finding the occurrences, the judge's extraction request and the
reading of its answer, the normalisation, the scores and the extractor's own check. It makes no call and reads no file:
the command-line entry point, eval/consistency.py, reads the eval run, sends the requests through the judge and writes
`consistency-<judge model>.json`.

The rules (`RULES`), per gold glossary entry:

- **Entries.** Only the entries that are not borderline (names and coined terms). An ordinary word may change with its
  context, and one rendering for it is not a goal.
- **Occurrence.** A segment whose Japanese text contains the entry's source or an alias, matched as term accuracy
  matches it (NFKC on both sides, then substring). The reference is not consulted. Of a `segments.jsonl` line only
  `page_index`, `idx`, `ja`, `hyp` and `ref_zh_hans` are read.
- **Rendering.** For each occurrence the judge extracts, in one call per occurrence and text, the part of that text that
  renders the term, without honorifics; or it answers that the term was left out (`omitted`) or that the match is not
  this term (`not_this_term`). A call sees the Japanese text, the form of the term found in it, whether the entry is a
  name or a term, and one Chinese text: the Translation or the converted reference, never both, never another
  occurrence and never the page image. An answer is usable only if it is a JSON object with a known status and, for
  `rendered`, a rendering that occurs in the text (both normalised as below). An unusable answer gets one retry, which
  shows the judge its answer and why it cannot be used; after that the occurrence is `invalid`: unknown, never counted
  as consistent or not.
- **Comparison.** Renderings are compared after NFKC, with whitespace removed and the middle-dot separators unified
  (`normalise_rendering`).
- **Score.** Per entry and text, the share of the rendered occurrences that use the entry's most common rendering. The
  pooled score takes the entries with at least `MIN_RENDERED` rendered occurrences, sums the counts of each one's most
  common rendering and divides by their rendered occurrences. `omitted`, `not_this_term` and `invalid` occurrences are
  counted and reported, never scored. A ratio with nothing to divide by is None, never 0.
- **The extractor's check.** The same extraction runs on the converted reference. Where the reference contains the
  entry's target or a variant (normalised as above), the answer is known: the extraction is correct when it is
  `rendered` and its rendering contains, or is contained in, the target or a variant (so an honorific or a particle at
  the edge is allowed, and nothing else). Every other answer there, `invalid` included, is not correct. Consistency is
  `reportable` as a result when the share of correct extractions is at least `REPORTABLE_PERCENT`; otherwise it is
  reported for reference only.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .glossary import Entry, Glossary
from .terms import normalise

RULES = "OUTLINE §4.7 term consistency (v0.25)"  # what consistency-<model>.json says its numbers follow
TEXTS = ("translation", "reference")  # the Run's stored Translation (`hyp`) and the converted reference (`ref_zh_hans`)
ANSWERS = ("rendered", "omitted", "not_this_term")  # what the judge may answer
STATUSES = (*ANSWERS, "invalid")  # invalid: no usable answer after the retry
MIN_RENDERED = 2  # an entry enters the pooled score with this many rendered occurrences or more
REPORTABLE_PERCENT = 90  # the extractor's accuracy at which consistency is reported as a result
ATTEMPTS = 2  # an unusable answer gets one retry
# U+00B7 middle dot, U+0387 Greek ano teleia, U+2022 bullet, U+2027 hyphenation point, U+2219 bullet operator,
# U+22C5 dot operator, U+2E31 word separator middle dot, U+30FB katakana middle dot and U+FF65 halfwidth katakana
# middle dot all become U+00B7, the separator of names in Simplified Chinese
MIDDLE_DOTS = "\u00b7\u0387\u2022\u2027\u2219\u22c5\u2e31\u30fb\uff65"
_UNIFY_DOTS = str.maketrans(dict.fromkeys(MIDDLE_DOTS, "\u00b7"))

_LINE_FIELDS = (("page_index", int), ("idx", int), ("ja", str), ("hyp", str), ("ref_zh_hans", str))


class ConsistencyError(ValueError):
    """A segment line the scorer cannot read."""


class Unusable(ValueError):
    """An answer that cannot be used. The message says why, in words the judge is shown on the retry."""


# --- the occurrences --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    page_index: int
    idx: int  # the segment's `idx`, the region id
    ja: str  # the texts as the harness wrote them: the judge sees them unchanged
    translation: str
    reference: str

    def text(self, which: str) -> str:
        if which not in TEXTS:
            raise ValueError(f"unknown text {which!r}")
        return self.translation if which == "translation" else self.reference


@dataclass(frozen=True)
class Occurrence:
    entry: Entry
    form: str  # the first of `Entry.sources` that the Japanese text contains, as the glossary writes it
    segment: Segment

    @property
    def key(self) -> dict:
        return {"page_index": self.segment.page_index, "idx": self.segment.idx}


def read_segments(raw: Iterable[Mapping[str, Any]]) -> list[Segment]:
    """The segments of `segments.jsonl` lines, in reading order. Raises ConsistencyError for a line without the fields
    it reads and for a repeated (page_index, idx), which would count its occurrences twice."""
    segments = []
    first_seen: dict[tuple[int, int], int] = {}
    for position, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ConsistencyError(f"segment line {position + 1} is not a JSON object")
        for key, kind in _LINE_FIELDS:
            value = item.get(key)
            if not isinstance(value, kind) or isinstance(value, bool):
                raise ConsistencyError(f"segment line {position + 1} has no {kind.__name__} `{key}`")
        place = (item["page_index"], item["idx"])
        if place in first_seen:
            raise ConsistencyError(f"segment line {position + 1} repeats page_index {place[0]} idx {place[1]}, "
                                   f"which line {first_seen[place]} has")
        first_seen[place] = position + 1
        segments.append(Segment(item["page_index"], item["idx"], item["ja"], item["hyp"], item["ref_zh_hans"]))
    return sorted(segments, key=lambda s: (s.page_index, s.idx))


def scored_entries(glossary: Glossary) -> tuple[Entry, ...]:
    """The entries consistency covers: the ones that are not borderline, in the glossary's order."""
    return tuple(e for e in glossary.entries if not e.borderline)


def find_occurrences(glossary: Glossary, segments: Sequence[Segment]) -> list[Occurrence]:
    """Every (entry, segment) whose Japanese text contains the entry's source or an alias, entry by entry in the
    glossary's order and then in reading order. Only `ja` is read: the texts the judge will see do not decide it."""
    occurrences = []
    for entry in scored_entries(glossary):
        forms = [(form, normalise(form)) for form in dict.fromkeys(entry.sources) if form and normalise(form)]
        for segment in segments:
            ja = normalise(segment.ja)
            found = next((form for form, normalised in forms if normalised in ja), None)
            if found is not None:
                occurrences.append(Occurrence(entry, found, segment))
    return occurrences


# --- the extraction request and its answer ---------------------------------------------------------------------------

SYSTEM_PROMPT = ("You help evaluate translations of Japanese manga into Simplified Chinese. You are given a Japanese "
                 "text, a term that occurs in it and a Chinese translation of the text, and you find the part of the "
                 "translation that renders the term. You answer with one JSON object and nothing else.")

USER_TEMPLATE = """Japanese text:
```{ja}```
Chinese translation:
```{text}```

The Japanese text contains 「{form}」, {kind}. Which part of the Chinese translation renders 「{form}」 here? Answer with one JSON object:
- {{"status": "rendered", "rendering": "..."}} when the translation renders it. Copy the rendering exactly as the translation writes it, character for character, and leave out honorifics and titles (such as 小姐, 先生, 君, 酱, 大人) and any particle or punctuation around it.
- {{"status": "omitted", "rendering": null}} when the translation leaves it out.
- {{"status": "not_this_term", "rendering": null}} when 「{form}」 in this Japanese text is not {this} (for example, when it is part of another word)."""  # noqa: E501

RETRY_TEMPLATE = "That answer cannot be used: {problem}. Answer again with one JSON object, as described above."

_KINDS = {"character": ("the name of a character of the story", "this name"),
          "term": ("a term of the story", "this term")}


def build_messages(occurrence: Occurrence, which: str) -> list[dict]:
    """The request for one occurrence in one text (`which`: translation or reference). It holds the Japanese text, the
    form found in it, the entry's kind and that one text: no other text, no other occurrence, no glossary target."""
    kind, this = _KINDS.get(occurrence.entry.category, _KINDS["term"])
    user = USER_TEMPLATE.format(ja=occurrence.segment.ja, text=occurrence.segment.text(which), form=occurrence.form,
                                kind=kind, this=this)
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def retry_messages(messages: Sequence[dict], answer: str, problem: str) -> list[dict]:
    """The retry: the same conversation, the answer that could not be used and why."""
    return [*messages, {"role": "assistant", "content": answer},
            {"role": "user", "content": RETRY_TEMPLATE.format(problem=problem)}]


@dataclass(frozen=True)
class Answer:
    status: str  # one of ANSWERS
    rendering: str | None = None  # as the judge wrote it; only for `rendered`
    normalised: str | None = None


def read_answer(answer: str, text: str) -> Answer:
    """A usable answer to the request on `text`; raises Unusable. A rendering must occur in `text`, both normalised."""
    try:
        data = json.loads(answer)
    except ValueError:
        raise Unusable("it is not valid JSON")
    if not isinstance(data, dict):
        raise Unusable("it is not a JSON object")
    status = data.get("status")
    if not isinstance(status, str) or status not in ANSWERS:
        raise Unusable(f"its status {status!r} is not \"rendered\", \"omitted\" or \"not_this_term\"")
    if status != "rendered":
        return Answer(status)
    rendering = data.get("rendering")
    if not isinstance(rendering, str) or not normalise_rendering(rendering):
        raise Unusable("a \"rendered\" answer needs the rendering as a string that is not empty")
    normalised = normalise_rendering(rendering)
    if normalised not in normalise_rendering(text):
        raise Unusable(f"the rendering {json.dumps(rendering, ensure_ascii=False)} does not occur in the Chinese "
                       "translation; copy it from the translation exactly")
    return Answer(status, rendering, normalised)


def normalise_rendering(text: str) -> str:
    """NFKC, no whitespace, and every middle-dot separator as U+00B7."""
    return "".join(ch for ch in normalise(text) if not ch.isspace()).translate(_UNIFY_DOTS)


@dataclass
class Extraction:
    """What the judge's calls gave for one occurrence in one text."""

    status: str = "invalid"  # one of STATUSES
    rendering: str | None = None
    normalised: str | None = None
    attempts: int = 0
    problems: list[str] = field(default_factory=list)  # why each unusable attempt was unusable
    answer: str | None = None  # the answer that was used, or the last one that could not be

    def to_dict(self) -> dict:
        return {"status": self.status, "rendering": self.rendering, "normalised": self.normalised,
                "attempts": self.attempts, "problems": list(self.problems), "answer": self.answer}


# --- the scores ------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TextScore:
    """One entry's occurrences in one text."""

    statuses: dict[str, int]  # every status in STATUSES order, 0 included
    renderings: dict[str, int]  # normalised rendering -> occurrences, most common first (ties in reading order)
    rendered: int
    most_common: str | None
    most_common_count: int
    consistency: float | None  # most_common_count / rendered; None when nothing was rendered
    pooled: bool  # rendered >= MIN_RENDERED: the entry counts in the pooled score

    def to_dict(self) -> dict:
        return {"rendered": self.rendered, "most_common": self.most_common,
                "most_common_count": self.most_common_count, "consistency": self.consistency, "pooled": self.pooled,
                "renderings": dict(self.renderings), "statuses": dict(self.statuses)}


def score_text(extractions: Sequence[Extraction]) -> TextScore:
    """The score of one entry's extractions in one text, given in reading order."""
    statuses = dict.fromkeys(STATUSES, 0)
    for e in extractions:
        statuses[e.status] += 1  # a status outside STATUSES is a bug, not data: KeyError
    counts = Counter(e.normalised for e in extractions if e.status == "rendered")
    ordered = dict(counts.most_common())  # ties keep the order in which the renderings were first seen
    rendered = sum(counts.values())
    most_common, most_common_count = next(iter(ordered.items()), (None, 0))
    return TextScore(statuses, ordered, rendered, most_common, most_common_count,
                     most_common_count / rendered if rendered else None, rendered >= MIN_RENDERED)


@dataclass(frozen=True)
class PooledScore:
    entries: int  # the entries with MIN_RENDERED or more rendered occurrences
    rendered: int  # their rendered occurrences
    most_common_count: int  # the sum of their most common renderings' counts
    consistency: float | None  # most_common_count / rendered

    def to_dict(self) -> dict:
        return {"entries": self.entries, "rendered": self.rendered, "most_common_count": self.most_common_count,
                "consistency": self.consistency}


def pool(scores: Iterable[TextScore]) -> PooledScore:
    kept = [s for s in scores if s.pooled]
    rendered, most_common = sum(s.rendered for s in kept), sum(s.most_common_count for s in kept)
    return PooledScore(len(kept), rendered, most_common, most_common / rendered if rendered else None)


def reference_has_target(entry: Entry, reference: str) -> bool:
    """Whether the converted reference contains the entry's target or a variant: there the extractor's answer is
    known."""
    text = normalise_rendering(reference)
    return any(t in text for t in _targets(entry))


def extraction_correct(entry: Entry, extraction: Extraction) -> bool:
    """A `rendered` extraction whose rendering contains, or is contained in, the target or a variant."""
    if extraction.status != "rendered" or not extraction.normalised:
        return False
    return any(t in extraction.normalised or extraction.normalised in t for t in _targets(entry))


def _targets(entry: Entry) -> tuple[str, ...]:
    return tuple(t for t in dict.fromkeys(normalise_rendering(t) for t in entry.targets) if t)


def is_reportable(correct: int, checked: int) -> bool:
    """Consistency is a result when at least REPORTABLE_PERCENT of the checked extractions are correct; with none
    checked the accuracy is unknown and the rule is not met."""
    return checked > 0 and correct * 100 >= checked * REPORTABLE_PERCENT


@dataclass(frozen=True)
class ExtractorCheck:
    checked: int  # occurrences whose reference contains the target or a variant
    correct: int
    accuracy: float | None
    reportable: bool

    def to_dict(self) -> dict:
        return {"checked": self.checked, "correct": self.correct, "accuracy": self.accuracy,
                "threshold": REPORTABLE_PERCENT / 100, "reportable": self.reportable}


def check_extractor(results: Iterable[tuple[bool, bool]]) -> ExtractorCheck:
    """From (checked, correct) per occurrence."""
    pairs = [(checked, correct) for checked, correct in results if checked]
    checked, correct = len(pairs), sum(correct for _, correct in pairs)
    return ExtractorCheck(checked, correct, correct / checked if checked else None, is_reportable(correct, checked))


@dataclass(frozen=True)
class OccurrenceResult:
    occurrence: Occurrence
    extractions: Mapping[str, Extraction]  # by text: translation and reference

    @property
    def checked(self) -> bool:
        return reference_has_target(self.occurrence.entry, self.occurrence.segment.reference)

    @property
    def correct(self) -> bool | None:
        """Whether the reference's extraction is correct; None where the answer is not known."""
        return extraction_correct(self.occurrence.entry, self.extractions["reference"]) if self.checked else None

    def to_dict(self) -> dict:
        return {**self.occurrence.key, "form": self.occurrence.form,
                **{which: self.extractions[which].to_dict() for which in TEXTS},
                "extractor": {"checked": self.checked, "correct": self.correct}}


@dataclass(frozen=True)
class EntryScore:
    entry: Entry
    results: tuple[OccurrenceResult, ...]  # in reading order
    texts: Mapping[str, TextScore]

    def to_dict(self) -> dict:
        check = check_extractor((r.checked, bool(r.correct)) for r in self.results)
        return {"id": self.entry.id, "category": self.entry.category, "occurrences": len(self.results),
                **{which: self.texts[which].to_dict() for which in TEXTS},
                "extractor": {"checked": check.checked, "correct": check.correct, "accuracy": check.accuracy},
                "segments": [r.to_dict() for r in self.results]}


@dataclass(frozen=True)
class BookScores:
    entries: tuple[EntryScore, ...]  # the scored entries, in the glossary's order
    left_out: tuple[str, ...]  # the ids of the borderline entries
    pooled: Mapping[str, PooledScore]  # by text
    statuses: Mapping[str, dict[str, int]]  # by text: occurrences by status
    extractor: ExtractorCheck

    @property
    def occurrences(self) -> int:
        return sum(len(e.results) for e in self.entries)

    def to_dict(self) -> dict:
        """What consistency-<model>.json holds below the run's identity and the judge."""
        return {
            "occurrences": self.occurrences,
            "entries_scored": len(self.entries),
            "entries_left_out": list(self.left_out),
            "pooled": {which: self.pooled[which].to_dict() for which in TEXTS},
            "statuses": {which: dict(self.statuses[which]) for which in TEXTS},
            "extractor": self.extractor.to_dict(),
            "entries": [e.to_dict() for e in self.entries],
        }


def score_book(glossary: Glossary, results: Sequence[OccurrenceResult]) -> BookScores:
    """The scores of one book's occurrences (as `find_occurrences` gives them) and their extractions."""
    entries = []
    for entry in scored_entries(glossary):
        mine = tuple(sorted((r for r in results if r.occurrence.entry.id == entry.id),
                            key=lambda r: (r.occurrence.segment.page_index, r.occurrence.segment.idx)))
        entries.append(EntryScore(entry, mine, {which: score_text([r.extractions[which] for r in mine])
                                                for which in TEXTS}))
    statuses = {which: {status: sum(e.texts[which].statuses[status] for e in entries) for status in STATUSES}
                for which in TEXTS}
    return BookScores(
        entries=tuple(entries),
        left_out=tuple(e.id for e in glossary.entries if e.borderline),
        pooled={which: pool(e.texts[which] for e in entries) for which in TEXTS},
        statuses=statuses,
        extractor=check_extractor((r.checked, bool(r.correct)) for e in entries for r in e.results))
