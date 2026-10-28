"""The Curator's rules, and the only code that writes Knowledge Base entries (OUTLINE 3.3, 8).

The Curator Agent's model call decides, once per page and without tools, whether each new term is worth keeping. Its
answer is an input here (`decisions`). The rules are code, so they hold whatever the model answered, and
`store.KB_WRITES` are called from this module only (a test scans the code for any other caller). This module makes no
model call and imports no LangChain.

`apply` takes one page's proposals, in order, and handles each in one transaction (all of them are stored or none):

1. A proposal whose id is already queued is ignored, so a retried page writes nothing the second time.
2. The same source and the same target as an existing entry (profile): the page joins its evidence, the history gets a
   line, and the proposal is approved. No judgment is needed.
3. The same source and another target: a conflict. The proposal goes to the Human Review Queue (`pending_human`) and the
   existing entry stays as it is, so the KB keeps its first rendering. On one page the first proposal wins and the later
   ones conflict, because by then the first one is an entry.
   A name under the wrong kind is a conflict too, across kinds: a term whose source is the name, or an alias, of a
   profile; a character whose name is the source of an entry. The proposal goes to `pending_human`, nothing is written
   for it, the existing profile or entry stays as it is, and the proposal's reason says it is about the kinds (the result
   is `CONFLICT_ACROSS_KINDS`, apart from `CONFLICT` so that a count of them needs no reading of reasons). A person
   decides which kind it is, whatever the decision says. This catches only a collision with what
   the KB already holds, and on one page the earlier proposal counts; whether a term that collides with nothing is really
   a name is the Curator Agent's judgment.
4. A new name (kind `character`) becomes a profile with status `pending_human`, whatever the decision says.
5. A new term becomes an `approved` entry (provenance `agent`) when accepted; a rejected one writes no entry and its
   proposal keeps the reason; one without a decision stays `proposed`, to be decided later.

An existing entry or profile is one in a readable state (`approved` or `pending_human`): a `proposed` or `rejected` row is
not there for these rules. A term is compared with the entries and a name with the profiles (rules 2 and 3), and a
same-kind match comes first. Only the cross-kind conflict of rule 3 looks at the other kind, and only when the proposal
would otherwise create a new entry or profile.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .. import store
from .entities import APPROVED, PENDING_HUMAN, TERM, KBNotFound, Proposal, proposal_id

# What happened to a proposal.
DUPLICATE = "duplicate"
CONFIRMED = "confirmed"
CONFLICT = "conflict"
CONFLICT_ACROSS_KINDS = "conflict_across_kinds"  # a name under the wrong kind; what is written is the same as for CONFLICT
PENDING = PENDING_HUMAN  # a new name, now a pending profile
ACCEPTED = "accepted"
REJECTED = "rejected"
UNDECIDED = "undecided"

RESULTS = (DUPLICATE, CONFIRMED, CONFLICT, CONFLICT_ACROSS_KINDS, PENDING, ACCEPTED, REJECTED, UNDECIDED)


@dataclass(frozen=True)
class Decision:
    """The Curator Agent's judgment of one proposal."""

    accept: bool
    reason: str


@dataclass(frozen=True)
class Outcome:
    proposal_id: str
    result: str  # one of RESULTS


def apply(
    conn: sqlite3.Connection, kb_id: int, page_index: int, run_id: int, proposals: Sequence[Proposal],
    decisions: Mapping[str, Decision] | None = None,
) -> list[Outcome]:
    """Apply one page's proposals to the KB, in order and in one transaction, and say what happened to each.

    `decisions` maps a proposal id to the Curator Agent's judgment; a proposal without one is undecided. Raises
    KBNotFound for an unknown KB and ValueError, before anything is written, for a decision whose id is not among the
    proposals or a proposal that is not this page's or whose id does not match its content. When a write fails, the
    exception propagates and nothing of the call is stored.
    """
    decisions = decisions or {}
    if store.get_kb(conn, kb_id) is None:
        raise KBNotFound(f"no knowledge base has id {kb_id}")
    unknown = sorted(set(decisions) - {p.proposal_id for p in proposals})
    if unknown:
        raise ValueError(f"decisions for proposals that are not among this page's: {', '.join(unknown)}")
    for p in proposals:
        genuine = p.proposal_id == proposal_id(kb_id, page_index, p.kind, p.payload)
        if (p.kb_id, p.page_index) != (kb_id, page_index) or not genuine:
            raise ValueError(f"proposal {p.proposal_id} is not a proposal of page {page_index} of KB {kb_id}")
    with store.transaction(conn):
        return [_apply_one(conn, p, run_id, decisions.get(p.proposal_id)) for p in proposals]


def _apply_one(conn: sqlite3.Connection, p: Proposal, run_id: int, decision: Decision | None) -> Outcome:
    queued = store.insert_proposal(
        conn, proposal_id=p.proposal_id, kb_id=p.kb_id, kind=p.kind, payload=p.payload, proposer=p.proposer,
        page_index=p.page_index, run_id=run_id,
    )
    if not queued:
        return Outcome(p.proposal_id, DUPLICATE)
    return Outcome(p.proposal_id, (_term if p.kind == TERM else _name)(conn, p, run_id, decision))


def _event(kind: str, p: Proposal, run_id: int, **extra) -> dict:
    return {"event": kind, "page": p.page_index, "run_id": run_id, "proposal_id": p.proposal_id, **extra}


def _term(conn: sqlite3.Connection, p: Proposal, run_id: int, decision: Decision | None) -> str:
    source, target = p.payload["source_term"], p.payload["target_term"]
    existing = store.find_entry(conn, p.kb_id, source)
    if existing is not None:
        if existing["target_term"] != target:
            return _conflict(conn, p, f"entry {existing['id']}", existing["target_term"])
        store.add_entry_evidence(conn, existing["id"], p.page_index, _event("confirmed", p, run_id))
        store.set_proposal_status(conn, p.proposal_id, status=APPROVED, reason=f"confirms entry {existing['id']}")
        return CONFIRMED
    if (profile := _profile_called(conn, p.kb_id, source)) is not None:
        called = "the name" if source == profile["name_src"] else "an alias"
        return _wrong_kind(conn, p, f"{source} is proposed as a term, but it is {called} of profile {profile['id']} "
                                    f"({profile['name_tgt']}), a character")
    if decision is None:
        return UNDECIDED  # stays 'proposed'
    if not decision.accept:
        store.set_proposal_status(conn, p.proposal_id, status="rejected", reason=decision.reason)
        return REJECTED
    store.insert_entry(
        conn, kb_id=p.kb_id, source_term=source, target_term=target, category=p.payload["category"],
        status=APPROVED, provenance="agent", page_index=p.page_index,
        event=_event("created", p, run_id, reason=decision.reason),
    )
    store.set_proposal_status(conn, p.proposal_id, status=APPROVED, reason=decision.reason)
    return ACCEPTED


def _name(conn: sqlite3.Connection, p: Proposal, run_id: int, decision: Decision | None) -> str:
    name, rendering = p.payload["name_src"], p.payload["name_tgt"]
    existing = store.find_profile(conn, p.kb_id, name)
    if existing is not None:
        if existing["name_tgt"] != rendering:
            return _conflict(conn, p, f"profile {existing['id']}", existing["name_tgt"])
        store.add_profile_evidence(conn, existing["id"], p.page_index, _event("confirmed", p, run_id))
        store.set_proposal_status(conn, p.proposal_id, status=APPROVED, reason=f"confirms profile {existing['id']}")
        return CONFIRMED
    if (entry := store.find_entry(conn, p.kb_id, name)) is not None:
        return _wrong_kind(conn, p, f"{name} is proposed as a character, but it is the source of entry {entry['id']} "
                                    f"({entry['target_term']}), a term")
    reason = "a new name waits for a human"
    if decision is not None:  # kept for the human, but it does not change the rule
        reason += f"; the Curator Agent would {'accept' if decision.accept else 'reject'} it: {decision.reason}"
    store.insert_profile(
        conn, kb_id=p.kb_id, name_src=name, name_tgt=rendering, aliases=p.payload["aliases"],
        notes=p.payload["notes"], status=PENDING_HUMAN, page_index=p.page_index,
        event=_event("created", p, run_id, reason=reason),
    )
    store.set_proposal_status(conn, p.proposal_id, status=PENDING_HUMAN, reason=reason)
    return PENDING


def _conflict(conn: sqlite3.Connection, p: Proposal, existing: str, rendering: str) -> str:
    """The existing entry is not touched; the proposal waits for a human."""
    store.set_proposal_status(
        conn, p.proposal_id, status=PENDING_HUMAN, reason=f"conflicts with {existing}, which is {rendering}",
    )
    return CONFLICT


def _profile_called(conn: sqlite3.Connection, kb_id: int, name: str) -> sqlite3.Row | None:
    """The readable profile that has `name` as its name or as one of its aliases: an approved one before a pending one,
    then the lowest id, as `store.find_profile` picks. (That one looks at the name only; the aliases are in a JSON
    column, so they are read here, from the KB's profiles.)"""
    called = [row for row in store.kb_profiles(conn, kb_id, store.READABLE_STATUSES)
              if name == row["name_src"] or name in json.loads(row["aliases"])]
    return min(called, key=lambda row: (row["status"] != APPROVED, row["id"]), default=None)


def _wrong_kind(conn: sqlite3.Connection, p: Proposal, what: str) -> str:
    """A conflict across kinds: `p`'s source is already something else in the KB. Nothing is written for it and the
    existing row is not touched; the proposal waits for a human, and its reason says `what` is wrong."""
    store.set_proposal_status(conn, p.proposal_id, status=PENDING_HUMAN, reason=f"conflicts across kinds: {what}")
    return CONFLICT_ACROSS_KINDS
