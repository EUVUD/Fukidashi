"""The Knowledge Base (OUTLINE 3.3): the glossary and the character profiles of one book.

What an agent may use is exported here: a read-only snapshot and a proposal collector, both without a database
connection. The Curator (`kb.curator`) is the only code that writes entries, and it is not exported on purpose: whoever
needs it imports it by name.
"""

from .collector import ProposalCollector
from .entities import CharacterProfile, GlossaryEntry, KBNotFound, Proposal
from .snapshot import KBSnapshot, load_snapshot

__all__ = [
    "CharacterProfile", "GlossaryEntry", "KBNotFound", "KBSnapshot", "Proposal", "ProposalCollector", "load_snapshot",
]
