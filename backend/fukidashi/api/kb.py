"""A book's Editor KB over HTTP: read it, and let a person decide its proposals, entries and profiles (OUTLINE §8,
"A person's decisions on KB items"; SCOPE.md #5).

This layer parses requests, maps errors to status codes and shapes responses. The decisions are `fukidashi.kb.curator`'s,
each one transaction that writes nothing when it is refused, and the SQL is `fukidashi.store`'s. Only a book's Editor KB
is read or decided here: an eval run's KB has no route, so a decision aimed at one through a book's route is a 404.

  GET  /api/books/{book_title}/kb                                       the Editor KB, both queues and the counts
  POST /api/books/{book_title}/kb/proposals/{proposal_id}/decision      answers the outcome and the KB
  POST /api/books/{book_title}/kb/entries/{entry_id}/decision           answers the KB
  POST /api/books/{book_title}/kb/profiles/{profile_id}/decision        answers the KB

Errors, all of which write nothing: 404 for a book that is not there, a decision on a book with no Editor KB, and an id
the book's Editor KB does not have; 409 for a decision the rules refuse, with the reason in `detail`; 422 for a body that
is not exactly `{"decision": "approve" | "reject"}`.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from .. import store
from ..kb import KBNotFound, curator
from ..kb.entities import PENDING_HUMAN
from .deps import get_db

router = APIRouter(prefix="/api")


class KBEntryOut(BaseModel):
    """A glossary entry of the Editor KB, in any status: a rejected one stays listed, so a person can approve it again."""

    id: int
    source_term: str  # as written in the Japanese text
    target_term: str  # its rendering in Simplified Chinese
    category: str  # a short label such as place or organisation; may be empty
    status: str  # "approved" or "pending_human" (the agents can find it), "rejected" or "proposed" (they cannot)
    provenance: str  # "agent", or "human" once a person has decided the entry (approved, rejected, or made it)
    evidence_pages: list[int]  # the pages that show it, ascending
    # one object per change, oldest first; each has "at" (UTC) and "event", and a person's decision has "by": "human"
    history: list[dict[str, Any]]


class KBProfileOut(BaseModel):
    """A character profile of the Editor KB, in any status."""

    id: int
    name_src: str  # the name as written in the Japanese text
    name_tgt: str  # its rendering in Simplified Chinese
    aliases: list[str]  # other Japanese names the character is called by
    notes: str  # speech style and anything else worth keeping; may be empty
    status: str  # as for an entry
    provenance: str  # derived, a profile has no such column: "human" when a history line has "by": "human", else "agent"
    evidence_pages: list[int]
    history: list[dict[str, Any]]  # as for an entry


class KBProposalOut(BaseModel):
    """A proposal that waits in the Proposal Queue (status "proposed": a term nobody decided yet)."""

    proposal_id: str
    kind: str  # "term" or "character"
    payload: dict[str, Any]  # a term: source_term, target_term, category; a character: name_src, name_tgt, aliases, notes
    proposer: str  # who proposed it ("translator")
    page_index: int  # the page it was proposed on
    run_id: int  # the Run that proposed it
    reason: str | None  # null while the proposal is undecided; later what was said when it was decided or sent to a person


class ReviewProposalOut(KBProposalOut):
    """A proposal that waits in the Human Review Queue (status "pending_human"), and what it waits for."""

    waits_for: str  # "new_name": a profile was made and waits for a person; "conflict": nothing was written for it
    profile_id: int | None  # the new name's profile (`profiles[].id`); null for a conflict


class ProposalCounts(BaseModel):
    """How many of the KB's proposals have each status; every status is there, 0 when none."""

    proposed: int
    approved: int
    rejected: int
    pending_human: int


class KBOut(BaseModel):
    """What the panel shows: the book's Editor KB, as it is now."""

    kb_id: int | None  # null when the book has no Editor KB yet (then every list is empty and every count 0)
    entries: list[KBEntryOut]  # every status, by id
    profiles: list[KBProfileOut]  # every status, by id
    proposal_queue: list[KBProposalOut]  # the Proposal Queue, in the order the proposals were queued
    human_review_queue: list[ReviewProposalOut]  # the Human Review Queue, in the same order
    proposal_counts: ProposalCounts  # over all of the KB's proposals, decided ones included


class DecisionRequest(BaseModel):
    """The body of a decision: exactly one field. Anything else, or another word, is a 422."""

    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject"]


class ProposalDecisionOut(BaseModel):
    outcome: str  # what the decision did: one of `curator.HUMAN_OUTCOMES`, listed there
    kb: KBOut  # the KB after the decision, as the GET gives it


def _kb_out(conn: sqlite3.Connection, kb_id: int | None) -> KBOut:
    """The KB as the panel shows it, read in one snapshot so that the lists agree with each other."""
    if kb_id is None:
        return KBOut(kb_id=None, entries=[], profiles=[], proposal_queue=[], human_review_queue=[],
                     proposal_counts=ProposalCounts(**dict.fromkeys(store.ALL_STATUSES, 0)))
    with store.reading(conn):
        entries = store.kb_entries(conn, kb_id, store.ALL_STATUSES)
        profiles = store.kb_profiles(conn, kb_id, store.ALL_STATUSES)
        waiting = store.kb_proposals(conn, kb_id, PENDING_HUMAN)
        undecided = store.kb_proposals(conn, kb_id, "proposed")
        counts = store.proposal_counts(conn, kb_id)
    new_names = {pid: p["id"] for p in profiles if (pid := curator.made_from_proposal(p)) is not None}
    return KBOut(
        kb_id=kb_id,
        entries=[
            KBEntryOut(
                id=e["id"], source_term=e["source_term"], target_term=e["target_term"], category=e["category"],
                status=e["status"], provenance=e["provenance"], evidence_pages=json.loads(e["evidence_pages"]),
                history=json.loads(e["history"]),
            ) for e in entries
        ],
        profiles=[
            KBProfileOut(
                id=p["id"], name_src=p["name_src"], name_tgt=p["name_tgt"], aliases=json.loads(p["aliases"]),
                notes=p["notes"], status=p["status"], provenance=_profile_provenance(json.loads(p["history"])),
                evidence_pages=json.loads(p["evidence_pages"]), history=json.loads(p["history"]),
            ) for p in profiles
        ],
        proposal_queue=[KBProposalOut(**_proposal_fields(r)) for r in undecided],
        human_review_queue=[
            ReviewProposalOut(
                **_proposal_fields(r), waits_for="new_name" if r["proposal_id"] in new_names else "conflict",
                profile_id=new_names.get(r["proposal_id"]),
            ) for r in waiting
        ],
        proposal_counts=ProposalCounts(**counts),
    )


def _proposal_fields(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "proposal_id": row["proposal_id"], "kind": row["kind"], "payload": json.loads(row["payload"]),
        "proposer": row["proposer"], "page_index": row["page_index"], "run_id": row["run_id"], "reason": row["reason"],
    }


def _profile_provenance(history: list[dict[str, Any]]) -> str:
    """A profile has no provenance column: it is a person's once a history line says so (`by` is "human")."""
    return curator.HUMAN if any(line.get("by") == curator.HUMAN for line in history) else curator.AGENT


def _book_or_404(conn: sqlite3.Connection, book_title: str) -> None:
    if store.find_book(conn, book_title) is None:
        raise HTTPException(status_code=404, detail=f"no book {book_title}")


def _editor_kb_id_or_404(conn: sqlite3.Connection, book_title: str) -> int:
    """The book's Editor KB. A decision never makes one: a book that has none has nothing to decide."""
    _book_or_404(conn, book_title)
    kb = store.find_editor_kb(conn, book_title)
    if kb is None:
        raise HTTPException(
            status_code=404, detail=f"book {book_title} has no Editor KB yet, so there is nothing to decide")
    return kb["id"]


def _decide(conn: sqlite3.Connection, book_title: str, row_id: int | str, decide: Callable[[int], str]) -> tuple[int, str]:
    """Run one decision in the book's Editor KB: the KB's id and what `decide` (given that id) did, errors mapped."""
    kb_id = _editor_kb_id_or_404(conn, book_title)
    if isinstance(row_id, int) and not -(2**63) <= row_id < 2**63:  # SQLite's integers are 64-bit: no row has this id
        raise HTTPException(status_code=404, detail=f"no row {row_id} in the book's Editor KB")
    try:
        return kb_id, decide(kb_id)
    except (KBNotFound, curator.ItemNotFound) as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except curator.DecisionRefused as e:
        raise HTTPException(status_code=409, detail=str(e)) from e


@router.get("/books/{book_title}/kb", response_model=KBOut)
def get_book_kb(book_title: str, conn: sqlite3.Connection = Depends(get_db)) -> KBOut:
    """The book's Editor KB. A book that has none yet answers `kb_id` null and empty lists: a GET writes nothing."""
    _book_or_404(conn, book_title)
    kb = store.find_editor_kb(conn, book_title)
    return _kb_out(conn, None if kb is None else kb["id"])


@router.post("/books/{book_title}/kb/proposals/{proposal_id}/decision", response_model=ProposalDecisionOut)
def decide_kb_proposal(
    book_title: str, proposal_id: str, body: DecisionRequest, conn: sqlite3.Connection = Depends(get_db),
) -> ProposalDecisionOut:
    """Approve or reject a proposal of the Proposal Queue or the Human Review Queue. A conflict can only be rejected, a
    new name is decided through its profile, and an approved undecided term goes through the term rules as the KB is
    now: `outcome` says which of those happened."""
    kb_id, outcome = _decide(
        conn, book_title, proposal_id,
        lambda kb: curator.decide_proposal(conn, kb, proposal_id, approve=body.decision == "approve"),
    )
    return ProposalDecisionOut(outcome=outcome, kb=_kb_out(conn, kb_id))


@router.post("/books/{book_title}/kb/entries/{entry_id}/decision", response_model=KBOut)
def decide_kb_entry(
    book_title: str, entry_id: int, body: DecisionRequest, conn: sqlite3.Connection = Depends(get_db),
) -> KBOut:
    kb_id, _ = _decide(
        conn, book_title, entry_id,
        lambda kb: curator.decide_entry(conn, kb, entry_id, approve=body.decision == "approve"),
    )
    return _kb_out(conn, kb_id)


@router.post("/books/{book_title}/kb/profiles/{profile_id}/decision", response_model=KBOut)
def decide_kb_profile(
    book_title: str, profile_id: int, body: DecisionRequest, conn: sqlite3.Connection = Depends(get_db),
) -> KBOut:
    kb_id, _ = _decide(
        conn, book_title, profile_id,
        lambda kb: curator.decide_profile(conn, kb, profile_id, approve=body.decision == "approve"),
    )
    return _kb_out(conn, kb_id)
