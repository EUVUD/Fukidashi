"""What both agents read from the Knowledge Base (OUTLINE §3.2, §3.3): the snippets part and the two lookup tools.

The Translator and the Reviewer both start their user message with the KB's entries for the page and both can look a
term or a character up, so the one definition of each is here, and neither agent module imports the other. The
Translator alone can propose, and `propose_entry` stays in `translator.py`.

Everything here reads a `KBSnapshot`, which is immutable and holds no database connection, so the handlers are safe in
the loop's worker threads. This module imports no LangChain and no database.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..datasets.openmantra import Page
from ..kb.entities import APPROVED, CharacterProfile, GlossaryEntry
from ..kb.snapshot import KBSnapshot
from .loop import Tool

KB_SNIPPET_CHARS = 1_000  # the character budget of the KB's entries at the start of a user message (OUTLINE §3.3)


def kb_snippets_part(page: Page, kb: KBSnapshot) -> dict:
    """The text part that starts a user message when the KB is on: the entries and profiles found in the page's Japanese
    text, cut to `KB_SNIPPET_CHARS`, or a line saying there are none.

    The regions are matched one per line, so that no entry can match across two regions.
    """
    snippets = kb.snippets("\n".join(s.ja for s in page.segments), KB_SNIPPET_CHARS)
    if not snippets:
        return {"type": "text", "text": "The knowledge base has no entry for this page's text yet."}
    return {"type": "text", "text": "Knowledge base entries that occur in this page's text:\n" + snippets}


def _status(item: GlossaryEntry | CharacterProfile) -> str:
    return "approved" if item.status == APPROVED else "pending"  # a snapshot holds nothing but these two


class LookupTermArgs(BaseModel):
    source_term: str = Field(description="The term exactly as it is written in the Japanese text.")


class LookupCharacterArgs(BaseModel):
    name: str = Field(description="The character's name, or another name they are called by, as written in the "
                                  "Japanese text.")


def lookup_tools(kb: KBSnapshot) -> list[Tool]:
    """`lookup_term` and `lookup_character` over `kb`, the Reviewer's tools as well as the Translator's.

    The handlers only read the snapshot, which is immutable and holds no database connection, so they are safe in worker
    threads. An exact lookup: a term must be written as the entry has it (after NFKC and trimming).

    The status a lookup returns is `approved` (the item is in the knowledge base) or `pending` (it was proposed on an
    earlier page and no person has confirmed it yet): the words of the system prompts and of the descriptions below.
    """

    def lookup_term(source_term: str) -> dict:
        entry = kb.term(source_term)
        if entry is None:
            return {"found": False, "note": "The knowledge base has no entry for this term."}
        return {"found": True, "source_term": entry.source_term, "target_term": entry.target_term,
                "category": entry.category, "status": _status(entry)}

    def lookup_character(name: str) -> dict:
        profile = kb.character(name)
        if profile is None:
            return {"found": False, "note": "The knowledge base has no character with this name."}
        return {"found": True, "name_src": profile.name_src, "name_tgt": profile.name_tgt,
                "aliases": list(profile.aliases), "notes": profile.notes, "status": _status(profile)}

    return [
        Tool(name="lookup_term",
             description="Look up one Japanese term in the knowledge base of this book. Returns its Chinese rendering "
                         "and its status: approved (in the knowledge base) or pending (proposed on an earlier page, "
                         "not yet confirmed by a person), or says there is none.",
             args=LookupTermArgs, handler=lookup_term),
        Tool(name="lookup_character",
             description="Look up a character in the knowledge base of this book by their name or by another name they "
                         "are called by. Returns the Chinese name, the other names, notes on their speech style and "
                         "the profile's status: approved (in the knowledge base) or pending (proposed on an earlier "
                         "page, not yet confirmed by a person), or says there is none.",
             args=LookupCharacterArgs, handler=lookup_character),
    ]
