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
4. A new name (kind `character`) becomes a profile with status `pending_human`, whatever the decision says, unless
   rule 6 finds that it is the name of a character the KB already holds.
5. A new term becomes an `approved` entry (provenance `agent`) when accepted; a rejected one writes no entry and its
   proposal keeps the reason; one without a decision stays `proposed`, to be decided later.
6. One character, one profile. Before rule 4 makes a profile, the name is matched against the readable profiles. A
   proposal matches a profile when one's name contains the other's (a given name inside a full name, a title inside
   the same title with an honorific), or when a name or alias of one equals a name or alias of the other. Aliases are
   compared for equality only, never for containment, so a family name used as an alias does not tie two sisters
   together.
   - Exactly one profile matches and one rendering contains the other (燕 in 樱野燕): the proposal is that profile's
     character. Its name and aliases join the profile's aliases (no duplicates, never the profile's own name), the
     page joins the evidence, the history gets a `merged` line, and the proposal is approved (`MERGED`). The profile
     keeps its status: a person still confirms names (rule 4).
   - A rendering that agrees with neither (铃 against 樱野雀), or more than one matching profile: a conflict. The
     proposal goes to `pending_human`, nothing is written for it, the profiles stay as they are, and its reason names
     the profiles and the names that matched. So a second rendering of one name never becomes a readable profile that
     spreads.
   Two characters who share an alias exactly (both called by their family name) are taken for one, and the proposal
   waits for a person: the safe side. Aliases are not checked against term sources here.

An existing entry or profile is one in a readable state (`approved` or `pending_human`): a `proposed` or `rejected` row is
not there for these rules. A term is compared with the entries and a name with the profiles (rules 2 and 3), and a
same-kind match comes first. Only the cross-kind conflict of rule 3 looks at the other kind, and only when the proposal
would otherwise create a new entry or profile. Rule 6 comes after rules 2 and 3, so the same name is still a
confirmation or a conflict, and a name that is a term's source is still a conflict across kinds.

A person's decisions (OUTLINE 8, "A person's decisions on KB items") are the other entry points: `decide_proposal`,
`decide_entry` and `decide_profile`, each approve or reject, in one transaction that writes nothing when the decision is
refused. A person decides only in a book's Editor KB (never in an eval run's KB). They add no rule of their own for
proposals and change none of the six above:

- An undecided term (status `proposed`) that a person approves goes through rules 2, 3 and 5 as the KB is now, as an
  accepted term would: a new `approved` entry made by a person (provenance `human`), a confirmation, a conflict, or a
  conflict across kinds. Rejecting it sets it `rejected`.
- A new name (a `pending_human` proposal with the profile rule 4 made) is decided through that profile: the profile and
  the proposal become `approved` or `rejected` together. A conflict (`pending_human`, nothing written) can be rejected,
  and its approval is refused: the KB keeps what it holds. A term's conflicting rendering is taken in place of the entry's
  by renaming the entry (`rename_entry`, below), which settles the conflict.
- An entry or a profile can be approved or rejected whatever its state (this stops a wrong entry), unless that changes
  nothing, or approving would leave two readable rows for one source term (one `name_src`), or a term whose source is a
  readable profile's name or alias (a name that is a readable entry's source): the collisions the rules above refuse.
  Deciding a profile also settles the proposal it was made from, while that is still `pending_human`.

What a person decided is recorded: a history line `{"event": "approved" | "rejected", "by": "human", "from": <previous
status>}` (plus `proposal_id` when the decision came through a proposal), an entry's provenance `human`, and a proposal's
reason that says a person decided it. A person's outcomes (`HUMAN_OUTCOMES`) are not `RESULTS`, which count what `apply` did.

A person's renames (OUTLINE 8, "Rename → whole-book sync") are the last entry points. `preview_entry_rename` and
`preview_profile_rename` say what renaming an entry's target term, or a profile's Chinese name, would do to the book, in one
read-only view of the database; `rename_entry` and `rename_profile` do it, in one transaction that writes nothing when the
rename is refused or fails partway. Which boxes follow is `kb.rename`'s rule (a box changes when its Japanese holds one of the
item's keys and its displayed translation holds the old rendering). What is written, all or nothing:

- the row's new rendering, one history line `{"event": "renamed", "by": "human", "from": <previous status>, "before": <old>,
  "after": <new>}`, and the status `approved` (a rename of a row that waits for a person approves it, so the checks of an
  approval apply, and a profile settles the new-name proposal it was made from, as deciding it does); an entry's provenance
  becomes `human`;
- for an entry, each term conflict in the Human Review Queue on the same source whose proposed rendering is now the entry's:
  it goes through the term rules again as a person's approval, so it becomes a confirmation (the page joins the entry's
  evidence, the proposal is `approved`). A conflict with another rendering, an undecided term and every character conflict
  stay as they are;
- one Human edit (a `human` Run) for each page with a changed box, based on that page's latest succeeded Run, with an
  `attempts_json` of `{"config", "base_run_id", "edits": [{"idx", "before", "after"}], "rename": {"kind", "id", "kb_id",
  "before", "after"}}`: `review_notes` reads its `edits` as the edited regions, and `rename` says why.

A rename is refused, with nothing written, for a KB that is not an Editor KB (KBNotFound), an id the KB does not have
(ItemNotFound), a rendering the KB would not store (InvalidRendering), a row that is not readable, a rendering equal to the
row's, an approval's collision, and a book with a `running` Run (DecisionRefused): a `full` Run writes the KB when it ends.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .. import store
from .entities import APPROVED, PENDING_HUMAN, TERM, KBNotFound, Proposal, proposal_id
from .rename import Box, BoxChange, plan_boxes, rendering as checked_rendering

# Who made an entry: the Curator's rules from an agent's proposal, or a person (a decision of theirs, `decide_*`).
AGENT = "agent"
HUMAN = "human"

# What happened to a proposal.
DUPLICATE = "duplicate"
CONFIRMED = "confirmed"
MERGED = "merged"  # a new name of a character the KB already holds: it joins that profile (rule 6)
CONFLICT = "conflict"
CONFLICT_ACROSS_KINDS = "conflict_across_kinds"  # a name under the wrong kind; what is written is the same as for CONFLICT
PENDING = PENDING_HUMAN  # a new name, now a pending profile
ACCEPTED = "accepted"
REJECTED = "rejected"
UNDECIDED = "undecided"

RESULTS = (DUPLICATE, CONFIRMED, MERGED, CONFLICT, CONFLICT_ACROSS_KINDS, PENDING, ACCEPTED, REJECTED, UNDECIDED)

# What a person's decision did, and what the API answers with. Not `RESULTS`: those count what `apply` did with a page's
# proposals, and the eval harness counts each of the nine.
ENTRY_CREATED = "entry_created"  # an undecided term approved: a new entry, made by a person
ENTRY_CONFIRMED = "entry_confirmed"  # an undecided term approved: it confirms the entry that holds this pair now
HELD_AS_CONFLICT = "held_as_conflict"  # approved, but the KB holds another rendering: it waits for a person again
HELD_ACROSS_KINDS = "held_as_conflict_across_kinds"  # approved, but its source is a profile's name or alias: same
PROPOSAL_REJECTED = "proposal_rejected"  # an undecided term, or a conflict, rejected: only the proposal is written
ENTRY_APPROVED = "entry_approved"
ENTRY_REJECTED = "entry_rejected"
PROFILE_APPROVED = "profile_approved"  # a profile decided as an item, or through its new-name proposal: both change
PROFILE_REJECTED = "profile_rejected"

HUMAN_OUTCOMES = (
    ENTRY_CREATED, ENTRY_CONFIRMED, HELD_AS_CONFLICT, HELD_ACROSS_KINDS, PROPOSAL_REJECTED, ENTRY_APPROVED,
    ENTRY_REJECTED, PROFILE_APPROVED, PROFILE_REJECTED,
)

_PROPOSED = "proposed"  # a proposal's status while it is undecided
_REJECTED_STATUS = "rejected"
_APPROVED_BY_PERSON = "approved by a person"


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


def _term(
    conn: sqlite3.Connection, p: Proposal, run_id: int, decision: Decision | None, provenance: str = AGENT,
) -> str:
    """Rules 2, 3 and 5 for a term. `provenance` is `agent` when `apply` applies a page, and `human` when a person approves
    an undecided term (`decide_proposal`): the rules are the same, the entry made is the person's, and what is written says
    so (the history lines carry `"by": "human"`, and a reason starts with the person's approval)."""
    source, target = p.payload["source_term"], p.payload["target_term"]
    by = {"by": HUMAN} if provenance == HUMAN else {}
    lead = f"{_APPROVED_BY_PERSON}; " if by else ""
    existing = store.find_entry(conn, p.kb_id, source)
    if existing is not None:
        if existing["target_term"] != target:
            return _conflict(conn, p, f"entry {existing['id']}", existing["target_term"], lead)
        store.add_entry_evidence(conn, existing["id"], p.page_index, _event("confirmed", p, run_id, **by))
        store.set_proposal_status(conn, p.proposal_id, status=APPROVED, reason=f"{lead}confirms entry {existing['id']}")
        return CONFIRMED
    if (profile := _profile_called(conn, p.kb_id, source)) is not None:
        called = "the name" if source == profile["name_src"] else "an alias"
        return _wrong_kind(conn, p, f"{source} is proposed as a term, but it is {called} of profile {profile['id']} "
                                    f"({profile['name_tgt']}), a character", lead)
    if decision is None:
        return UNDECIDED  # stays 'proposed'
    if not decision.accept:
        store.set_proposal_status(conn, p.proposal_id, status=_REJECTED_STATUS, reason=decision.reason)
        return REJECTED
    store.insert_entry(
        conn, kb_id=p.kb_id, source_term=source, target_term=target, category=p.payload["category"],
        status=APPROVED, provenance=provenance, page_index=p.page_index,
        event=_event("created", p, run_id, reason=decision.reason, **by),
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
    if matched := _matching_profiles(conn, p):
        return _known_character(conn, p, run_id, matched)
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


def _matches(payload: Mapping, profile: sqlite3.Row) -> list[str]:
    """How a character proposal matches a profile (rule 6), in words, or an empty list when it does not: one's name
    contains the other's, or a name or alias of one equals a name or alias of the other. Aliases are compared for
    equality only."""
    name, other = payload["name_src"], profile["name_src"]
    how = []
    if name in other:
        how.append(f"{name} is part of {other}")
    elif other and other in name:  # a profile a person wrote could have an empty name, which every name contains
        how.append(f"{other} is part of {name}")
    theirs = {other, *json.loads(profile["aliases"])}
    how += [f"both are called {called}" for called in dict.fromkeys((name, *payload["aliases"])) if called in theirs]
    return how


def _matching_profiles(conn: sqlite3.Connection, p: Proposal) -> list[tuple[sqlite3.Row, list[str]]]:
    """The readable profiles that `p` matches (rule 6), each with how it matches, in id order."""
    matched = [(row, _matches(p.payload, row)) for row in store.kb_profiles(conn, p.kb_id, store.READABLE_STATUSES)]
    return [(row, how) for row, how in matched if how]


def _profile_label(profile: sqlite3.Row) -> str:
    return f"profile {profile['id']} ({profile['name_src']} → {profile['name_tgt']})"


def _known_character(
    conn: sqlite3.Connection, p: Proposal, run_id: int, matched: list[tuple[sqlite3.Row, list[str]]],
) -> str:
    """Rule 6: `p` matches the profiles in `matched`. One profile whose rendering agrees takes in `p`'s names; anything
    else waits for a person and writes nothing."""
    rendering = p.payload["name_tgt"]
    if len(matched) > 1:
        found = "; ".join(f"{_profile_label(row)}: {', '.join(how)}" for row, how in matched)
        return _conflict_reason(conn, p, f"conflicts with more than one profile, so a person decides which character "
                                         f"it is: {found}")
    ((profile, how),) = matched
    if rendering not in profile["name_tgt"] and profile["name_tgt"] not in rendering:
        return _conflict_reason(conn, p, f"conflicts with {_profile_label(profile)}: {', '.join(how)}, but {rendering} "
                                         f"does not agree with {profile['name_tgt']}")
    known = {profile["name_src"], *json.loads(profile["aliases"])}
    new_aliases = [a for a in dict.fromkeys((p.payload["name_src"], *p.payload["aliases"])) if a not in known]
    reason = (f"the same character as {_profile_label(profile)}: {', '.join(how)}, and {rendering} agrees with "
              f"{profile['name_tgt']}; " + (f"new aliases {', '.join(new_aliases)}" if new_aliases else "no new alias"))
    store.add_profile_aliases(conn, profile["id"], new_aliases, p.page_index,
                              _event("merged", p, run_id, aliases=new_aliases, reason=reason))
    store.set_proposal_status(conn, p.proposal_id, status=APPROVED, reason=reason)
    return MERGED


def _conflict(conn: sqlite3.Connection, p: Proposal, existing: str, rendering: str, lead: str = "") -> str:
    """The existing entry is not touched; the proposal waits for a human."""
    return _conflict_reason(conn, p, f"conflicts with {existing}, which is {rendering}", lead)


def _conflict_reason(conn: sqlite3.Connection, p: Proposal, reason: str, lead: str = "") -> str:
    store.set_proposal_status(conn, p.proposal_id, status=PENDING_HUMAN, reason=lead + reason)
    return CONFLICT


def _profile_called(conn: sqlite3.Connection, kb_id: int, name: str) -> sqlite3.Row | None:
    """The readable profile that has `name` as its name or as one of its aliases: an approved one before a pending one,
    then the lowest id, as `store.find_profile` picks. (That one looks at the name only; the aliases are in a JSON
    column, so they are read here, from the KB's profiles.)"""
    called = [row for row in store.kb_profiles(conn, kb_id, store.READABLE_STATUSES)
              if name == row["name_src"] or name in json.loads(row["aliases"])]
    return min(called, key=lambda row: (row["status"] != APPROVED, row["id"]), default=None)


def _wrong_kind(conn: sqlite3.Connection, p: Proposal, what: str, lead: str = "") -> str:
    """A conflict across kinds: `p`'s source is already something else in the KB. Nothing is written for it and the
    existing row is not touched; the proposal waits for a human, and its reason says `what` is wrong."""
    store.set_proposal_status(conn, p.proposal_id, status=PENDING_HUMAN, reason=f"{lead}conflicts across kinds: {what}")
    return CONFLICT_ACROSS_KINDS


# --- a person's decisions (OUTLINE 8, "A person's decisions on KB items") ---


class ItemNotFound(LookupError):
    """The KB has no proposal, entry or profile with that id (the id of another KB's does not count)."""


class DecisionRefused(ValueError):
    """The rules do not allow this decision, whoever asks. Nothing was written; the message says why."""


# What the rules did with an approved undecided term, as a person's outcome.
_FROM_RULES = {
    ACCEPTED: ENTRY_CREATED, CONFIRMED: ENTRY_CONFIRMED, CONFLICT: HELD_AS_CONFLICT,
    CONFLICT_ACROSS_KINDS: HELD_ACROSS_KINDS,
}
_APPROVAL = Decision(True, _APPROVED_BY_PERSON)


def made_from_proposal(profile: sqlite3.Row) -> str | None:
    """The id of the proposal that made the profile (rule 4), or None for a profile no proposal made. The profile's first
    history line is `created` and carries the proposal's id, which is how a new name's proposal and its profile find each
    other."""
    history = json.loads(profile["history"])
    first = history[0] if history else {}
    return first.get("proposal_id") if first.get("event") == "created" else None


def decide_proposal(conn: sqlite3.Connection, kb_id: int, pid: str, *, approve: bool) -> str:
    """A person approves or rejects proposal `pid` of the Editor KB `kb_id`, in one transaction. Returns what it did, one
    of `HUMAN_OUTCOMES`.

    - An undecided term (`proposed`): rejected, it becomes `rejected`. Approved, it goes through the term rules as the KB
      is now, as an accepted term would (the KB may have changed since its page), and a new entry is made by a person.
    - A new name (`pending_human`, with the profile rule 4 made): decided through that profile, so the profile and the
      proposal become `approved` or `rejected` together.
    - A conflict (`pending_human`, nothing written): rejected, it becomes `rejected` and the KB keeps what it holds. Its
      approval is refused: a term's rendering is taken by renaming the entry (`rename_entry`), which settles the conflict.
    - A decided proposal (`approved`, `rejected`) is refused.

    Raises KBNotFound for a KB that is not an Editor KB, ItemNotFound for a proposal the KB does not have, and
    DecisionRefused; in each case nothing is written. Must be called with no transaction open."""
    with store.transaction(conn):
        _require_editor_kb(conn, kb_id)
        row = store.get_proposal(conn, kb_id, pid)
        if row is None:
            raise ItemNotFound(f"no proposal {pid} in the KB")
        if row["status"] == _PROPOSED:
            return _decide_undecided(conn, row, approve)
        if row["status"] == PENDING_HUMAN:
            return _decide_waiting(conn, row, approve)
        raise DecisionRefused(f"proposal {pid} is already {row['status']}: only a proposal that waits in the Proposal "
                              f"Queue or the Human Review Queue can be decided")


def decide_entry(conn: sqlite3.Connection, kb_id: int, entry_id: int, *, approve: bool) -> str:
    """A person approves or rejects entry `entry_id` of the Editor KB `kb_id`, in one transaction. Its status becomes
    `approved` or `rejected`, its provenance `human`, and its history gets the person's line. Returns `ENTRY_APPROVED` or
    `ENTRY_REJECTED`.

    Refused (DecisionRefused) when it changes nothing, and when approving would leave two readable entries for one source
    term, or an entry whose source is a readable profile's name or alias. Otherwise raises what `decide_proposal` does.
    Must be called with no transaction open."""
    with store.transaction(conn):
        _require_editor_kb(conn, kb_id)
        entry = store.get_entry(conn, kb_id, entry_id)
        if entry is None:
            raise ItemNotFound(f"no entry {entry_id} in the KB")
        status = APPROVED if approve else _REJECTED_STATUS
        if entry["status"] == status:
            raise DecisionRefused(f"entry {entry_id} is already {status}")
        if approve:
            _refuse_entry_collision(conn, entry)
        store.set_entry_status(conn, entry_id, status=status, provenance=HUMAN, event=_decided(approve, entry["status"]))
    return ENTRY_APPROVED if approve else ENTRY_REJECTED


def decide_profile(conn: sqlite3.Connection, kb_id: int, profile_id: int, *, approve: bool) -> str:
    """A person approves or rejects profile `profile_id` of the Editor KB `kb_id`, in one transaction. Its status becomes
    `approved` or `rejected` and its history gets the person's line (a profile has no provenance column: the line's `by`
    is the record). When the profile was made from a proposal that still waits in the Human Review Queue, that proposal
    is settled the same way. Returns `PROFILE_APPROVED` or `PROFILE_REJECTED`.

    Refused when it changes nothing, and when approving would leave two readable profiles with one `name_src`, or a
    profile whose `name_src` is a readable entry's source. Otherwise raises what `decide_proposal` does. Must be called
    with no transaction open."""
    with store.transaction(conn):
        _require_editor_kb(conn, kb_id)
        profile = store.get_profile(conn, kb_id, profile_id)
        if profile is None:
            raise ItemNotFound(f"no profile {profile_id} in the KB")
        return _decide_profile(conn, profile, approve)


def _require_editor_kb(conn: sqlite3.Connection, kb_id: int) -> sqlite3.Row:
    """A person decides in a book's Editor KB only: an eval run's KB is an eval artefact, and nothing here changes one. The
    KB's row, for a caller that wants its book."""
    kb = store.get_kb(conn, kb_id)
    if kb is None or kb["purpose"] != store.EDITOR:
        raise KBNotFound(f"no Editor KB has id {kb_id}")
    return kb


def _person_reason(approve: bool, earlier: str | None, as_profile: int | None = None) -> str:
    """A proposal's reason once a person has decided it: who decided what, then what it said before (what it was waiting
    for), which would otherwise be lost."""
    reason = _APPROVED_BY_PERSON if approve else "rejected by a person"
    if as_profile is not None:
        reason += f", as profile {as_profile}"
    return reason + (f" (earlier: {earlier})" if earlier else "")


def _decided(approve: bool, before: str, proposal: str | None = None) -> dict:
    """The history line of a person's decision on an entry or a profile: what they did, that a person did it, and the
    status it replaced; the proposal's id when the decision came through one."""
    event = {"event": APPROVED if approve else _REJECTED_STATUS, "by": HUMAN, "from": before}
    return event if proposal is None else {**event, "proposal_id": proposal}


def _stored_proposal(row: sqlite3.Row) -> Proposal:
    return Proposal(row["proposal_id"], row["kb_id"], row["page_index"], row["kind"], json.loads(row["payload"]),
                    row["proposer"])


def _decide_undecided(conn: sqlite3.Connection, row: sqlite3.Row, approve: bool) -> str:
    """A proposal that is still `proposed`: only a term can be (`apply` makes a name a pending profile at once)."""
    p = _stored_proposal(row)
    if not approve:
        store.set_proposal_status(conn, p.proposal_id, status=_REJECTED_STATUS, reason=_person_reason(False, row["reason"]))
        return PROPOSAL_REJECTED
    if p.kind != TERM:
        raise DecisionRefused(f"proposal {p.proposal_id} is a name that waits undecided, which the rules never leave: only "
                              f"an undecided term can be approved")
    return _FROM_RULES[_term(conn, p, row["run_id"], _APPROVAL, HUMAN)]


def _decide_waiting(conn: sqlite3.Connection, row: sqlite3.Row, approve: bool) -> str:
    """A proposal in the Human Review Queue: a new name, whose profile `apply` made, or a conflict, for which `apply` wrote
    nothing."""
    profile = next((r for r in store.kb_profiles(conn, row["kb_id"], store.ALL_STATUSES)
                    if made_from_proposal(r) == row["proposal_id"]), None)
    if profile is not None:
        return _decide_profile(conn, profile, approve, via=row)
    if approve:
        raise DecisionRefused(f"proposal {row['proposal_id']} is a conflict, and a conflict cannot be approved: the KB "
                              f"keeps what it holds. To take a term's proposed rendering, rename the entry that holds its "
                              f"source term to it, which settles this proposal; otherwise reject the proposal to close it")
    store.set_proposal_status(conn, row["proposal_id"], status=_REJECTED_STATUS,
                              reason=_person_reason(False, row["reason"]))
    return PROPOSAL_REJECTED


def _decide_profile(conn: sqlite3.Connection, profile: sqlite3.Row, approve: bool, via: sqlite3.Row | None = None) -> str:
    """Decide a profile, as an item or `via` the new-name proposal it was made from, and settle that proposal: a new name is
    one decision, whichever way it is reached, so the two queues never disagree."""
    status = APPROVED if approve else _REJECTED_STATUS
    if profile["status"] == status:
        raise DecisionRefused(f"profile {profile['id']} is already {status}")
    if approve:
        _refuse_profile_collision(conn, profile)
    store.set_profile_status(
        conn, profile["id"], status=status,
        event=_decided(approve, profile["status"], None if via is None else via["proposal_id"]),
    )
    queued = via
    if queued is None and (pid := made_from_proposal(profile)) is not None:
        queued = store.get_proposal(conn, profile["kb_id"], pid)
    if queued is not None and queued["status"] == PENDING_HUMAN:
        store.set_proposal_status(
            conn, queued["proposal_id"], status=status,
            reason=_person_reason(approve, queued["reason"], None if via is not None else profile["id"]),
        )
    return PROFILE_APPROVED if approve else PROFILE_REJECTED


def _refuse_entry_collision(conn: sqlite3.Connection, entry: sqlite3.Row) -> None:
    """Approving `entry` makes it readable: refuse when that would leave two readable entries for its source term, or when
    its source is a readable profile's name or alias (the collisions `apply` refuses)."""
    source = entry["source_term"]
    rival = next((e for e in store.kb_entries(conn, entry["kb_id"], store.READABLE_STATUSES)
                  if e["source_term"] == source and e["id"] != entry["id"]), None)
    if rival is not None:
        raise DecisionRefused(f"entry {entry['id']} cannot be approved: entry {rival['id']} ({rival['target_term']}) is "
                              f"{rival['status']} for the source term {source}, and two readable entries for one term "
                              f"would leave the agents a choice")
    if (profile := _profile_called(conn, entry["kb_id"], source)) is not None:
        called = "the name" if source == profile["name_src"] else "an alias"
        raise DecisionRefused(f"entry {entry['id']} cannot be approved: {source} is {called} of profile {profile['id']} "
                              f"({profile['name_tgt']}), a character")


def _refuse_profile_collision(conn: sqlite3.Connection, profile: sqlite3.Row) -> None:
    """Approving `profile` makes it readable: refuse when that would leave two readable profiles with its `name_src`, or
    when that name is a readable entry's source."""
    name = profile["name_src"]
    rival = next((r for r in store.kb_profiles(conn, profile["kb_id"], store.READABLE_STATUSES)
                  if r["name_src"] == name and r["id"] != profile["id"]), None)
    if rival is not None:
        raise DecisionRefused(f"profile {profile['id']} cannot be approved: profile {rival['id']} ({rival['name_tgt']}) is "
                              f"{rival['status']} for the name {name}, and two readable profiles for one name would leave "
                              f"the agents a choice")
    if (entry := store.find_entry(conn, profile["kb_id"], name)) is not None:
        raise DecisionRefused(f"profile {profile['id']} cannot be approved: {name} is the source of entry {entry['id']} "
                              f"({entry['target_term']}), a term")


# --- a person's renames (OUTLINE 8, "Rename → whole-book sync") ---

# What a rename renames (the `kind` of a plan and of the `rename` record a Human edit keeps).
ENTRY = "entry"
PROFILE = "profile"

_RENAMED = "renamed"


@dataclass(frozen=True)
class RenamePlan:
    """What renaming one row does, as the preview says it and as the write says it after it: the same plan, made the same
    way. `changes` and `listed` are boxes of the book (`kb.rename`); `settles` are the ids of the proposals the write settles
    together with the row, in the order they were queued."""

    kind: str  # ENTRY or PROFILE
    id: int  # the row's id
    before: str  # the row's rendering now
    after: str  # the new rendering, as the KB stores it
    changes: tuple[BoxChange, ...]
    listed: tuple[BoxChange, ...]
    settles: tuple[str, ...]


@dataclass(frozen=True)
class PageRun:
    """The Human edit a rename wrote on one page."""

    page_index: int
    run_id: int


@dataclass(frozen=True)
class RenameResult:
    plan: RenamePlan
    runs: tuple[PageRun, ...]  # one for each page with a changed box, in page order


@dataclass(frozen=True)
class _Page:
    """A page of the book as a rename reads it: the segments with the translation the page displays, and the Run that
    shows it (None when the page has no succeeded Run), which is the base of a Human edit."""

    page_id: int
    page_index: int
    run_id: int | None
    rows: list[sqlite3.Row]  # `store.segments_with_translations`, by `idx`


@dataclass(frozen=True)
class _Prepared:
    """A rename that passed every check, read in one view of the database: its plan, the row, the book's pages and the
    proposals the write settles with it."""

    plan: RenamePlan
    row: sqlite3.Row
    pages: list[_Page]
    settling: list[sqlite3.Row]  # proposal rows


def preview_entry_rename(conn: sqlite3.Connection, kb_id: int, entry_id: int, target: str) -> RenamePlan:
    """What renaming entry `entry_id` of the Editor KB `kb_id` to `target` would do: which boxes of the book change, which
    are only listed, and which proposals are settled. Reads one view of the database and writes nothing. Raises what
    `rename_entry` raises. Must be called with no transaction open."""
    with store.reading(conn):
        return _prepare_entry(conn, kb_id, entry_id, target).plan


def preview_profile_rename(conn: sqlite3.Connection, kb_id: int, profile_id: int, target: str) -> RenamePlan:
    """`preview_entry_rename` for a profile: renaming its Chinese name (`name_tgt`) to `target`."""
    with store.reading(conn):
        return _prepare_profile(conn, kb_id, profile_id, target).plan


def rename_entry(conn: sqlite3.Connection, kb_id: int, entry_id: int, target: str) -> RenameResult:
    """A person renames entry `entry_id` of the Editor KB `kb_id`: its target term becomes `target`, as the book's boxes do
    (`kb.rename`), in one transaction. The entry becomes `approved` with provenance `human` and a `renamed` history line; each
    term conflict in the Human Review Queue on its source whose proposal has the new rendering is confirmed as a person's
    approval; each page with a changed box gets a Human edit. Returns the plan and the Human edits' Runs.

    Raises KBNotFound for a KB that is not an Editor KB, ItemNotFound for an entry the KB does not have, InvalidRendering for
    a `target` the KB would not store (NFKC, trimmed, 1 to 60 characters, one line), and DecisionRefused for an entry that
    is not readable (`approved` or `pending_human`: approve it first), a `target` equal to its rendering, an approval's
    collision (for an entry that was `pending_human`) and a book with a `running` Run. In each case, and when any write
    fails, nothing is written. Must be called with no transaction open."""
    with store.transaction(conn):
        prepared = _prepare_entry(conn, kb_id, entry_id, target)
        plan, entry = prepared.plan, prepared.row
        store.set_entry_rendering(conn, entry_id, target_term=plan.after, status=APPROVED, provenance=HUMAN,
                                  event=_renamed(entry["status"], plan))
        for conflict in prepared.settling:  # the entry holds the new rendering now, so each of these confirms it
            _term(conn, _stored_proposal(conflict), conflict["run_id"], _APPROVAL, HUMAN)
        return RenameResult(plan, _write_human_edits(conn, prepared))


def rename_profile(conn: sqlite3.Connection, kb_id: int, profile_id: int, target: str) -> RenameResult:
    """`rename_entry` for a profile: its Chinese name (`name_tgt`) becomes `target`, as the boxes whose Japanese holds its name
    or one of its aliases do. The profile becomes `approved` with a `renamed` history line, and the new-name proposal it was
    made from is settled with it while that waits in the Human Review Queue, as deciding the profile does. No conflict is
    settled: a character conflict is rule 6's. Raises what `rename_entry` raises, for a profile."""
    with store.transaction(conn):
        prepared = _prepare_profile(conn, kb_id, profile_id, target)
        plan, profile = prepared.plan, prepared.row
        store.set_profile_rendering(conn, profile_id, name_tgt=plan.after, status=APPROVED,
                                    event=_renamed(profile["status"], plan))
        for queued in prepared.settling:
            store.set_proposal_status(conn, queued["proposal_id"], status=APPROVED,
                                      reason=_person_reason(True, queued["reason"], profile_id))
        return RenameResult(plan, _write_human_edits(conn, prepared))


def _prepare_entry(conn: sqlite3.Connection, kb_id: int, entry_id: int, target: str) -> _Prepared:
    """Everything a rename of an entry needs, after every check. The preview and the write both start here, so they refuse
    the same and plan the same."""
    new = checked_rendering(target)
    kb = _require_editor_kb(conn, kb_id)
    entry = store.get_entry(conn, kb_id, entry_id)
    if entry is None:
        raise ItemNotFound(f"no entry {entry_id} in the KB")
    _refuse_rename(entry, f"entry {entry_id}", entry["target_term"], new)
    if entry["status"] == PENDING_HUMAN:  # the rename approves it
        _refuse_entry_collision(conn, entry)
    _refuse_while_running(conn, kb["book_title"])
    pages = _book_pages(conn, kb["book_title"])
    boxes = plan_boxes(_boxes(pages), (entry["source_term"],), entry["target_term"], new)
    agreeing = _agreeing_conflicts(conn, entry, new)
    plan = RenamePlan(ENTRY, entry_id, entry["target_term"], new, boxes.changes, boxes.listed,
                      tuple(row["proposal_id"] for row in agreeing))
    return _Prepared(plan, entry, pages, agreeing)


def _prepare_profile(conn: sqlite3.Connection, kb_id: int, profile_id: int, target: str) -> _Prepared:
    new = checked_rendering(target)
    kb = _require_editor_kb(conn, kb_id)
    profile = store.get_profile(conn, kb_id, profile_id)
    if profile is None:
        raise ItemNotFound(f"no profile {profile_id} in the KB")
    _refuse_rename(profile, f"profile {profile_id}", profile["name_tgt"], new)
    if profile["status"] == PENDING_HUMAN:  # the rename approves it
        _refuse_profile_collision(conn, profile)
    _refuse_while_running(conn, kb["book_title"])
    pages = _book_pages(conn, kb["book_title"])
    keys = (profile["name_src"], *json.loads(profile["aliases"]))
    boxes = plan_boxes(_boxes(pages), keys, profile["name_tgt"], new)
    waiting = _new_name_waiting(conn, profile)
    plan = RenamePlan(PROFILE, profile_id, profile["name_tgt"], new, boxes.changes, boxes.listed,
                      tuple(row["proposal_id"] for row in waiting))
    return _Prepared(plan, profile, pages, waiting)


def _refuse_rename(row: sqlite3.Row, label: str, rendering_now: str, new: str) -> None:
    """Only a readable row is renamed (a `rejected` one is approved first: a rename would otherwise bring it back without
    anyone deciding that), and only to another rendering."""
    if row["status"] not in store.READABLE_STATUSES:
        raise DecisionRefused(f"{label} is {row['status']}, so it cannot be renamed: approve it first")
    if new == rendering_now:
        raise DecisionRefused(f"{label} already has the rendering {new}: a rename needs another one")


def _refuse_while_running(conn: sqlite3.Connection, book_title: str) -> None:
    """A `full` Run writes the book's KB when it finishes, so a rename that read the KB before it and wrote after it could
    meet half of what it wrote: nothing is renamed while any page of the book is running."""
    running = store.running_pages(conn, book_title)
    if running:
        where = ", ".join(f"page {page} (Run {run_id})" for page, run_id in running)
        raise DecisionRefused(f"a Run is still running on {where} of {book_title}, and a Run that finishes writes the KB: "
                              f"rename after it ends")


def _book_pages(conn: sqlite3.Connection, book_title: str) -> list[_Page]:
    """Every page of the book, in page order, with the translations its latest succeeded Run displays."""
    pages = []
    for page_index in store.book_page_indexes(conn, book_title):
        page = store.find_page(conn, book_title, page_index)
        run = store.latest_succeeded_run(conn, page["id"])
        run_id = None if run is None else run["id"]
        pages.append(_Page(page["id"], page_index, run_id, store.segments_with_translations(conn, page["id"], run_id)))
    return pages


def _boxes(pages: list[_Page]) -> list[Box]:
    return [Box(page.page_index, row["idx"], row["text_ja"], row["text_zh_hans"]) for page in pages for row in page.rows]


def _agreeing_conflicts(conn: sqlite3.Connection, entry: sqlite3.Row, new: str) -> list[sqlite3.Row]:
    """The term proposals in the Human Review Queue for the entry's source whose proposed rendering is `new`, in the order
    they were queued: once the entry has `new`, none of them conflicts with it any more. A conflict with another rendering
    is not one of them, and neither is a character conflict or an undecided term."""
    agreeing = []
    for row in store.kb_proposals(conn, entry["kb_id"], PENDING_HUMAN):
        if row["kind"] != TERM:
            continue
        payload = json.loads(row["payload"])
        if payload["source_term"] == entry["source_term"] and payload["target_term"] == new:
            agreeing.append(row)
    return agreeing


def _new_name_waiting(conn: sqlite3.Connection, profile: sqlite3.Row) -> list[sqlite3.Row]:
    """The proposal that made the profile (rule 4), as a list of at most one, while it still waits in the Human Review
    Queue: renaming the profile settles it, as deciding the profile does."""
    pid = made_from_proposal(profile)
    queued = None if pid is None else store.get_proposal(conn, profile["kb_id"], pid)
    return [queued] if queued is not None and queued["status"] == PENDING_HUMAN else []


def _renamed(before: str, plan: RenamePlan) -> dict:
    """The history line of a rename: that a person renamed the row, the two renderings, and the status it replaced."""
    return {"event": _RENAMED, "by": HUMAN, "from": before, "before": plan.before, "after": plan.after}


def _write_human_edits(conn: sqlite3.Connection, prepared: _Prepared) -> tuple[PageRun, ...]:
    """One Human edit for each page with a changed box (as `runs.edit_translation` makes one for a box): a `human` Run
    based on the page's latest succeeded Run, with that Run's Translations and the changed boxes replaced. Its record lists
    every box it changed and names the rename. Writes no commit: the caller's transaction owns all-or-nothing."""
    plan, kb_id = prepared.plan, prepared.row["kb_id"]
    rename = {"kind": plan.kind, "id": plan.id, "kb_id": kb_id, "before": plan.before, "after": plan.after}
    runs = []
    for page in prepared.pages:
        changes = [change for change in plan.changes if change.page_index == page.page_index]
        if not changes:
            continue
        segment_ids = {row["idx"]: row["id"] for row in page.rows}
        translations = {row["id"]: row["text_zh_hans"] for row in page.rows if row["text_zh_hans"] is not None}
        for change in changes:
            translations[segment_ids[change.idx]] = change.after
        attempts = {"config": store.HUMAN, "base_run_id": page.run_id,
                    "edits": [{"idx": c.idx, "before": c.before, "after": c.after} for c in changes], "rename": rename}
        runs.append(PageRun(page.page_index, store.insert_human_run(
            conn, page_id=page.page_id, attempts=attempts, translations=translations)))
    return tuple(runs)
