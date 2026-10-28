"""A read-only, in-memory copy of one KB: the only view of the KB an agent gets (OUTLINE 3.2, 3.3).

`kb.loader.load_snapshot` reads the KB's entries and profiles in the readable states (`approved` and `pending_human`,
each marked with its status; never `proposed` or `rejected`) and returns plain, immutable values. The snapshot holds no
database connection, so it is safe in worker threads, cannot write, and does not change when the KB does: a run sees the
KB as it was when the run started. This module imports no database, so an agent can import the type.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .entities import APPROVED, PENDING_HUMAN, CharacterProfile, GlossaryEntry, normalise


def _rank(item: GlossaryEntry | CharacterProfile) -> tuple[bool, int]:
    """Approved before pending, then the earlier one."""
    return (item.status != APPROVED, item.id)


@dataclass(frozen=True)
class KBSnapshot:
    kb_id: int
    book_title: str
    entries: tuple[GlossaryEntry, ...]  # in the order they were created
    profiles: tuple[CharacterProfile, ...]
    _terms: Mapping[str, GlossaryEntry] = field(init=False, repr=False, compare=False)
    _names: Mapping[str, CharacterProfile] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        # Where one source has several entries, the approved one wins, then the earlier one. A profile is found by its
        # name before it is found by an alias that another profile happens to share.
        terms: dict[str, GlossaryEntry] = {}
        for entry in sorted(self.entries, key=_rank):
            terms.setdefault(entry.source_term, entry)
        names: dict[str, CharacterProfile] = {}
        for profile in sorted(self.profiles, key=_rank):
            names.setdefault(profile.name_src, profile)
        for profile in sorted(self.profiles, key=_rank):
            for alias in profile.aliases:
                names.setdefault(alias, profile)
        object.__setattr__(self, "_terms", MappingProxyType(terms))
        object.__setattr__(self, "_names", MappingProxyType(names))

    def term(self, source: str) -> GlossaryEntry | None:
        """The entry whose source term is exactly `source` (after NFKC and trimming), or None."""
        return self._terms.get(normalise(source))

    def character(self, name: str) -> CharacterProfile | None:
        """The profile with `name` as its name or as one of its aliases, or None."""
        return self._names.get(normalise(name))

    def snippet_items(self, text_ja: str, char_budget: int) -> tuple[GlossaryEntry | CharacterProfile, ...]:
        """The entries and profiles whose source term, name or alias occurs in `text_ja`, approved before pending, then
        by where each first occurs, cut after the last whole line that fits in `char_budget` characters (a line that
        would take the text past it ends the list). `snippets` is these items' lines, so what is recorded as shown to a
        page is what its prompt carried."""
        text = normalise(text_ja)
        ranked = []
        for kind, items in ((0, self.entries), (1, self.profiles)):
            for item in items:
                found = [at for key in item.keys if key and (at := text.find(key)) >= 0]
                if found:
                    ranked.append(((item.status != APPROVED, min(found), kind, item.id), item))
        kept: list[GlossaryEntry | CharacterProfile] = []
        used = 0
        for _, item in sorted(ranked, key=lambda pair: pair[0]):
            used += len(item_line(item)) + bool(kept)  # the newline before every line but the first
            if used > char_budget:
                break
            kept.append(item)
        return tuple(kept)

    def snippets(self, text_ja: str, char_budget: int) -> str:
        """One line for each of `snippet_items`, joined by newlines: never longer than `char_budget` characters and
        never with half a line."""
        return "\n".join(item_line(item) for item in self.snippet_items(text_ja, char_budget))


def entry_line(entry: GlossaryEntry) -> str:
    line = f"term: {entry.source_term} → {entry.target_term}"
    if entry.category:
        line += f" ({entry.category})"
    return line + (" [pending]" if entry.status == PENDING_HUMAN else "")


def profile_line(profile: CharacterProfile) -> str:
    line = f"character: {profile.name_src} → {profile.name_tgt}"
    if profile.aliases:
        line += f" (also {', '.join(profile.aliases)})"
    if profile.notes:
        line += f"; {profile.notes}"
    return line + (" [pending]" if profile.status == PENDING_HUMAN else "")


def item_line(item: GlossaryEntry | CharacterProfile) -> str:
    return entry_line(item) if isinstance(item, GlossaryEntry) else profile_line(item)
