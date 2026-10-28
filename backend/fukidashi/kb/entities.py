"""Plain types of the Knowledge Base (OUTLINE 3.3): entries, profiles, proposals, and the proposal id.

Nothing here touches a database or a model. An entry or a profile is an immutable value, so a snapshot made of them can
be shared with worker threads. A proposal's payload is validated and normalised (NFKC, trimmed) when the proposal is
made, and every string in it must be one line (no newline, tab or other control character), because a stored entry
reaches an agent's prompt as one snippet line. Its id is a hash of the KB, the page index, the kind and the normalised
pair, so the same proposal on the same page always has the same id (a retry never writes twice) and the same pair on
another page, or in another KB, has another.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Any

from pydantic import (
    AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, StringConstraints, ValidationError, ValidationInfo,
    field_validator,
)

APPROVED = "approved"
PENDING_HUMAN = "pending_human"

TERM = "term"
CHARACTER = "character"
KINDS = (TERM, CHARACTER)

MAX_NAME_CHARS = 60  # a source or target term, a name, an alias
MAX_CATEGORY_CHARS = 30
MAX_ALIASES = 8
MAX_NOTES_CHARS = 300


class KBNotFound(LookupError):
    """No KB has the id."""


def normalise(text: str) -> str:
    """NFKC, trimmed: the form in which terms and names are stored, compared and hashed."""
    return unicodedata.normalize("NFKC", text).strip()


def _nfkc(value: Any) -> Any:
    return normalise(value) if isinstance(value, str) else value  # anything else: pydantic complains


def _trim(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


# What breaks a line: control characters (category Cc: \n, \r, \t, ...) and the Unicode line and paragraph separators,
# which `str.splitlines` also splits on.
_LINE_BREAKING = ("Cc", "Zl", "Zp")


def _one_line(value: str) -> str:
    """Refuse a control character inside a value. A stored entry reaches an agent's prompt as exactly one snippet line,
    so a value must not be able to end that line and start another. (Leading and trailing whitespace is already
    trimmed, so this is about a character in the middle.)"""
    bad = next((c for c in value if unicodedata.category(c) in _LINE_BREAKING), None)
    if bad is not None:
        raise ValueError(f"must be one line of text, without a newline, a tab or another control character "
                         f"(found U+{ord(bad):04X})")
    return value


# Trimmed by the BeforeValidator, not by StringConstraints(strip_whitespace=True): behind a BeforeValidator that let
# a blank string through as an empty one (pydantic 2.14), past min_length.
Name = Annotated[str, BeforeValidator(_nfkc), StringConstraints(min_length=1, max_length=MAX_NAME_CHARS),
                 AfterValidator(_one_line)]
Category = Annotated[str, BeforeValidator(_nfkc), StringConstraints(max_length=MAX_CATEGORY_CHARS),
                     AfterValidator(_one_line)]
# Notes are prose in the target language: NFKC would turn its full-width punctuation into ASCII, so only trim.
Notes = Annotated[str, BeforeValidator(_trim), StringConstraints(max_length=MAX_NOTES_CHARS), AfterValidator(_one_line)]


class TermPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_term: Name = Field(description="The term as it is written in the Japanese text.")
    target_term: Name = Field(description="Its rendering in Simplified Chinese.")
    category: Category = Field(default="", description="A short label such as place or organisation; optional.")


class CharacterPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name_src: Name = Field(description="The character's name as it is written in the Japanese text.")
    name_tgt: Name = Field(description="Its rendering in Simplified Chinese.")
    aliases: list[Name] = Field(default_factory=list, max_length=MAX_ALIASES,
                                description="Other Japanese names the character is called by.")
    notes: Notes = Field(default="", description="Speech style and anything else worth keeping; optional.")

    @field_validator("aliases")
    @classmethod
    def _distinct(cls, aliases: list[str], info: ValidationInfo) -> list[str]:
        # an alias that repeats the name or another alias says nothing new
        seen = {info.data.get("name_src")}
        return [a for a in aliases if not (a in seen or seen.add(a))]


PAYLOADS: Mapping[str, type[BaseModel]] = MappingProxyType({TERM: TermPayload, CHARACTER: CharacterPayload})
# The two fields of a payload that identify the proposal: the source and its rendering.
KEY_FIELDS: Mapping[str, tuple[str, str]] = MappingProxyType(
    {TERM: ("source_term", "target_term"), CHARACTER: ("name_src", "name_tgt")}
)


def _freeze(value: Any) -> Any:
    """A JSON value (as json.loads returns it) as an immutable one."""
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def frozen_json(text: str) -> Any:
    return _freeze(json.loads(text))


@dataclass(frozen=True)
class GlossaryEntry:
    id: int
    source_term: str
    target_term: str
    category: str
    status: str  # approved or pending_human in a snapshot
    provenance: str  # agent or human
    evidence_pages: tuple[int, ...]
    history: tuple[Mapping[str, Any], ...]

    @property
    def keys(self) -> tuple[str, ...]:
        """The strings that, found in a page's text, make the entry relevant to the page."""
        return (self.source_term,)


@dataclass(frozen=True)
class CharacterProfile:
    id: int
    name_src: str
    name_tgt: str
    aliases: tuple[str, ...]
    notes: str
    status: str
    evidence_pages: tuple[int, ...]
    history: tuple[Mapping[str, Any], ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return (self.name_src, *self.aliases)


@dataclass(frozen=True)
class Proposal:
    """One proposal as the Translator makes it. `payload` is the normalised payload as a dict; do not modify it
    (`curator.apply` recomputes the id and refuses a proposal that no longer matches)."""

    proposal_id: str
    kb_id: int
    page_index: int
    kind: str
    payload: dict[str, Any]
    proposer: str


def proposal_id(kb_id: int, page_index: int, kind: str, payload: Mapping[str, Any]) -> str:
    """Hash of the KB, the page index, the kind and the normalised source and target of the payload."""
    source, target = KEY_FIELDS[kind]
    blob = json.dumps([kb_id, page_index, kind, normalise(payload[source]), normalise(payload[target])],
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def make_proposal(kb_id: int, page_index: int, kind: str, payload: Mapping[str, Any], proposer: str) -> Proposal:
    """Validate and normalise `payload` and give the proposal its id. Raises ValueError, with a message the model can
    act on, for an unknown kind or a payload that does not fit."""
    if kind not in PAYLOADS:
        raise ValueError(f"unknown kind {kind!r}; use one of {', '.join(KINDS)}")
    if not isinstance(payload, Mapping):
        raise ValueError(f"invalid {kind} proposal: payload: Input should be an object with the fields of a {kind}, "
                         f"not {type(payload).__name__}")
    try:
        clean = PAYLOADS[kind].model_validate(dict(payload)).model_dump()
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'payload'}: {e['msg']}" for e in exc.errors())
        raise ValueError(f"invalid {kind} proposal: {problems}") from None
    return Proposal(proposal_id(kb_id, page_index, kind, clean), kb_id, page_index, kind, clean, proposer)
