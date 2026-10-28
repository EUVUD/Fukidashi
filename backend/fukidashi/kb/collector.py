"""The Proposal Queue's in-memory side: the proposals one page's run collects (OUTLINE 3.3).

The Translator can only propose. It gets a collector, which holds no database connection, and `curator.apply` later
writes what it collected. The calls of one model reply run in parallel worker threads, so every access is guarded by a
lock.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from .entities import Proposal, make_proposal

DEFAULT_PROPOSER = "translator"


class ProposalCollector:
    """The proposals made for one page of one KB, each kept once: what `propose_entry` validates and answers with.

    A proposal made twice on the page (the same kind and the same source and target) has the same id and is kept once,
    the first time it arrives. `proposals` lists them in the order they arrived. The calls of one model reply run side
    by side, so for them arrival is a race: when two of them give one id other details (a category, aliases, notes),
    which payload this collector holds depends on timing. That is fine for the tool's answer ("recorded"), which does
    not show the details, but it is not what gets stored. The page's proposals in the order of the calls, with the first
    call's payload for each id, are `translator.step_proposals` of the Translator's steps, and `OrchestratorResult`
    carries them.
    """

    def __init__(self, kb_id: int, page_index: int):
        self.kb_id = kb_id
        self.page_index = page_index
        self._lock = threading.Lock()
        self._by_id: dict[str, Proposal] = {}  # a dict keeps the order of arrival

    def add(self, kind: str, payload: Mapping[str, Any], *, proposer: str = DEFAULT_PROPOSER) -> tuple[Proposal, bool]:
        """Validate and keep a proposal. Returns it and whether it is new (False: the page had it already, and the
        proposal returned is the one that arrived first). Raises ValueError, whose message says what is wrong, for an
        unknown kind or a payload that does not fit; nothing is kept then."""
        proposal = make_proposal(self.kb_id, self.page_index, kind, payload, proposer)
        with self._lock:
            kept = self._by_id.get(proposal.proposal_id)
            if kept is not None:
                return kept, False
            self._by_id[proposal.proposal_id] = proposal
            return proposal, True

    @property
    def proposals(self) -> tuple[Proposal, ...]:
        with self._lock:
            return tuple(self._by_id.values())
