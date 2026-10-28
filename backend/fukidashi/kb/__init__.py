"""The Knowledge Base (OUTLINE 3.3): the glossary and the character profiles of one book.

What an agent may use is exported here: a read-only snapshot and a proposal collector, both without a database
connection, and nothing in this package's `__init__` touches the database, so importing it (as the agents do) does not
load the store. The two modules that do are not exported on purpose: `kb.loader` (`load_snapshot`, which reads a KB out
of the database) and `kb.curator` (the only code that writes entries). Whoever needs one imports it by name.
"""

from .collector import ProposalCollector
from .entities import CharacterProfile, GlossaryEntry, KBNotFound, Proposal
from .snapshot import KBSnapshot

__all__ = ["CharacterProfile", "GlossaryEntry", "KBNotFound", "KBSnapshot", "Proposal", "ProposalCollector"]
