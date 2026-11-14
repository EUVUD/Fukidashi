"""A person's decisions on the Editor KB (task 040; OUTLINE §8, "A person's decisions on KB items"): `decide_proposal`,
`decide_entry` and `decide_profile` of `kb/curator.py`, one transaction each, and what they leave in the tables."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

import pytest
from kb_support import accept, dump, name, new_kb, new_run, reject, term

from fukidashi import store
from fukidashi.datasets.importer import import_openmantra
from fukidashi.kb import KBNotFound
from fukidashi.kb.curator import (
    CONFLICT, CONFLICT_ACROSS_KINDS, ENTRY_APPROVED, ENTRY_CONFIRMED, ENTRY_CREATED, ENTRY_REJECTED, HELD_ACROSS_KINDS,
    HELD_AS_CONFLICT, HUMAN_OUTCOMES, PENDING, PROFILE_APPROVED, PROFILE_REJECTED, PROPOSAL_REJECTED, RESULTS, UNDECIDED,
    DecisionRefused, ItemNotFound, apply, decide_entry, decide_profile, decide_proposal, made_from_proposal,
)
from fukidashi.kb.loader import load_snapshot


@pytest.fixture
def imported(conn, dataset):
    import_openmantra(conn, dataset)
    return conn


@pytest.fixture
def kb(imported):
    """The Editor KB of book_a: the only kind of KB a person decides in."""
    return store.editor_kb(imported, "book_a")


@pytest.fixture
def run_id(imported):
    return new_run(imported)


# --- what the tests need to read and to set up ---


def results(outcomes):
    return [o.result for o in outcomes]


def accept_all(proposals):
    return {p.proposal_id: accept() for p in proposals}


def proposal_row(conn, proposal_or_id):
    pid = getattr(proposal_or_id, "proposal_id", proposal_or_id)
    return conn.execute("SELECT * FROM proposals WHERE proposal_id = ?", (pid,)).fetchone()


def entries(conn, kb_id):
    return store.kb_entries(conn, kb_id, store.ALL_STATUSES)


def profiles(conn, kb_id):
    return store.kb_profiles(conn, kb_id, store.ALL_STATUSES)


def pages(row):
    return json.loads(row["evidence_pages"])


def history(row):
    return json.loads(row["history"])


def without_time(line: dict) -> dict:
    """A history line as the tests compare it: its time must be there and be a time, and is left out."""
    datetime.fromisoformat(line["at"])
    return {k: v for k, v in line.items() if k != "at"}


def undecided_term(conn, kb, run_id, source="封印", target="封印", page=1, **extra):
    """A term the Curator's call left undecided: `proposed`, in the Proposal Queue."""
    proposal = term(kb, page, source, target, **extra)
    assert results(apply(conn, kb, page, run_id, [proposal])) == [UNDECIDED]
    return proposal


def new_name(conn, kb, run_id, source="ユキ", target="小雪", page=1, **extra):
    """A new name: a `pending_human` profile and its `pending_human` proposal (rule 4)."""
    proposal = name(kb, page, source, target, **extra)
    assert results(apply(conn, kb, page, run_id, [proposal])) == [PENDING]
    return proposal


def accepted_term(conn, kb, run_id, source="封印", target="封印", page=1):
    """An `approved` entry made by the agent (rule 5): returns its row."""
    proposal = term(kb, page, source, target)
    apply(conn, kb, page, run_id, [proposal], accept_all([proposal]))
    return store.find_entry(conn, kb, source)


def insert_entry(conn, kb_id, source, target="封印", status="approved", provenance="agent"):
    """An entry in any state, as an older Curator or a person left it. Committed: a decision wants no open transaction."""
    entry_id = store.insert_entry(
        conn, kb_id=kb_id, source_term=source, target_term=target, category="", status=status, provenance=provenance,
        page_index=1, event={"event": "created"})
    conn.commit()
    return entry_id


def insert_profile(conn, kb_id, name_src, name_tgt="小雪", aliases=(), status="approved"):
    profile_id = store.insert_profile(
        conn, kb_id=kb_id, name_src=name_src, name_tgt=name_tgt, aliases=list(aliases), notes="", status=status,
        page_index=1, event={"event": "created"})
    conn.commit()
    return profile_id


def entry_row(conn, entry_id):
    return conn.execute("SELECT * FROM glossary_entries WHERE id = ?", (entry_id,)).fetchone()


def profile_row(conn, profile_id):
    return conn.execute("SELECT * FROM character_profiles WHERE id = ?", (profile_id,)).fetchone()


def refused(conn, call, match=None):
    """The decision is refused, and nothing is written, not even by a transaction left open."""
    before = dump(conn)
    with pytest.raises(DecisionRefused, match=match):
        call()
    assert dump(conn) == before
    assert not conn.in_transaction


# --- the outcomes are a person's own ---


def test_a_persons_outcomes_are_their_own_and_the_nine_results_are_the_nine():
    assert set(RESULTS) == {"duplicate", "confirmed", "merged", "conflict", "conflict_across_kinds", "pending_human",
                            "accepted", "rejected", "undecided"}
    assert len(HUMAN_OUTCOMES) == len(set(HUMAN_OUTCOMES)) == 9
    assert not set(HUMAN_OUTCOMES) & set(RESULTS)  # a count of RESULTS can never absorb one of them


# --- an undecided term approved: the term rules as of now, made by a person ---


def test_an_undecided_term_approved_becomes_an_entry_made_by_a_person(imported, kb, run_id):
    proposal = undecided_term(imported, kb, run_id, "封印", "封印", page=3, category="magic")

    outcome = decide_proposal(imported, kb, proposal.proposal_id, approve=True)

    assert outcome == ENTRY_CREATED
    (entry,) = entries(imported, kb)
    assert (entry["source_term"], entry["target_term"], entry["category"]) == ("封印", "封印", "magic")
    assert (entry["status"], entry["provenance"], pages(entry)) == ("approved", "human", [3])
    (created,) = history(entry)
    assert without_time(created) == {
        "event": "created", "by": "human", "page": 3, "run_id": run_id, "proposal_id": proposal.proposal_id,
        "reason": "approved by a person"}
    row = proposal_row(imported, proposal)
    assert (row["status"], row["reason"], row["decided_at"] is not None) == ("approved", "approved by a person", True)
    assert load_snapshot(imported, kb).term("封印").status == "approved"  # the agents read it from now on
    assert not imported.in_transaction


def test_an_undecided_term_approved_confirms_the_entry_that_holds_its_pair_by_now(imported, kb, run_id):
    proposal = undecided_term(imported, kb, run_id, "封印", "封印", page=2)
    entry = accepted_term(imported, kb, run_id, "封印", "封印", page=1)  # the KB changed while it waited

    outcome = decide_proposal(imported, kb, proposal.proposal_id, approve=True)

    assert outcome == ENTRY_CONFIRMED
    (after,) = entries(imported, kb)
    assert (after["id"], after["provenance"], pages(after)) == (entry["id"], "agent", [1, 2])  # not decided by a person
    assert without_time(history(after)[-1]) == {
        "event": "confirmed", "by": "human", "page": 2, "run_id": run_id, "proposal_id": proposal.proposal_id}
    row = proposal_row(imported, proposal)
    assert (row["status"], row["reason"]) == ("approved", f"approved by a person; confirms entry {entry['id']}")


def test_an_undecided_term_approved_after_the_kb_took_another_rendering_waits_for_a_person_again(imported, kb, run_id):
    proposal = undecided_term(imported, kb, run_id, "封印", "封缄", page=2)
    entry = accepted_term(imported, kb, run_id, "封印", "封印", page=1)
    before = dump(imported)

    outcome = decide_proposal(imported, kb, proposal.proposal_id, approve=True)

    assert outcome == HELD_AS_CONFLICT
    after = dump(imported)
    assert (after["glossary_entries"], after["character_profiles"]) == (before["glossary_entries"], before["character_profiles"])
    row = proposal_row(imported, proposal)  # the Human Review Queue now, with a reason that says how it got there
    assert (row["status"], row["decided_at"] is not None) == ("pending_human", True)
    assert row["reason"] == f"approved by a person; conflicts with entry {entry['id']}, which is 封印"
    assert load_snapshot(imported, kb).term("封印").target_term == "封印"


def test_an_undecided_term_approved_after_the_kb_took_its_source_as_a_name_waits_for_a_person_again(imported, kb, run_id):
    proposal = undecided_term(imported, kb, run_id, "ユキ", "雪", page=2)
    new_name(imported, kb, run_id, "ユキ", "小雪", page=1)
    before = dump(imported)

    outcome = decide_proposal(imported, kb, proposal.proposal_id, approve=True)

    assert outcome == HELD_ACROSS_KINDS
    after = dump(imported)
    assert (after["glossary_entries"], after["character_profiles"]) == (before["glossary_entries"], before["character_profiles"])
    row = proposal_row(imported, proposal)
    assert row["status"] == "pending_human"
    assert row["reason"].startswith("approved by a person; conflicts across kinds: ユキ is proposed as a term")
    assert entries(imported, kb) == []


def test_an_undecided_term_rejected_is_rejected_and_writes_no_entry(imported, kb, run_id):
    proposal = undecided_term(imported, kb, run_id, "封印", "封印")
    before = dump(imported)

    outcome = decide_proposal(imported, kb, proposal.proposal_id, approve=False)

    assert outcome == PROPOSAL_REJECTED
    after = dump(imported)
    assert (after["glossary_entries"], after["character_profiles"]) == (before["glossary_entries"], before["character_profiles"])
    row = proposal_row(imported, proposal)
    assert (row["status"], row["reason"], row["decided_at"] is not None) == ("rejected", "rejected by a person", True)
    assert load_snapshot(imported, kb).term("封印") is None
    assert store.kb_proposals(imported, kb, "proposed") == []  # it left the Proposal Queue


def test_a_proposal_that_is_a_name_and_still_undecided_can_be_rejected_but_not_approved(imported, kb, run_id):
    """`apply` never leaves a name undecided; a row that someone wrote by hand still gets a clean answer."""
    proposal = name(kb, 1, "ユキ", "小雪")
    store.insert_proposal(imported, proposal_id=proposal.proposal_id, kb_id=kb, kind="character", payload=proposal.payload,
                          proposer="translator", page_index=1, run_id=run_id)
    imported.commit()

    refused(imported, lambda: decide_proposal(imported, kb, proposal.proposal_id, approve=True), "only an undecided term")
    assert decide_proposal(imported, kb, proposal.proposal_id, approve=False) == PROPOSAL_REJECTED
    assert proposal_row(imported, proposal)["status"] == "rejected" and profiles(imported, kb) == []


# --- a new name: the profile and the proposal change together ---


def test_a_new_name_approved_makes_its_profile_and_its_proposal_approved_together(imported, kb, run_id):
    proposal = new_name(imported, kb, run_id, "ユキ", "小雪", page=4, aliases=["ゆき"])
    (pending,) = profiles(imported, kb)
    earlier = proposal_row(imported, proposal)["reason"]
    assert earlier == "a new name waits for a human"

    outcome = decide_proposal(imported, kb, proposal.proposal_id, approve=True)

    assert outcome == PROFILE_APPROVED
    (profile,) = profiles(imported, kb)
    assert (profile["id"], profile["status"], pages(profile)) == (pending["id"], "approved", [4])
    assert without_time(history(profile)[-1]) == {
        "event": "approved", "by": "human", "from": "pending_human", "proposal_id": proposal.proposal_id}
    assert len(history(profile)) == 2 and history(profile)[0]["event"] == "created"
    row = proposal_row(imported, proposal)
    assert (row["status"], row["decided_at"] is not None) == ("approved", True)
    assert row["reason"] == f"approved by a person (earlier: {earlier})"  # what it waited for is not lost
    assert load_snapshot(imported, kb).character("ゆき").status == "approved"
    assert entries(imported, kb) == []


def test_a_new_name_rejected_rejects_its_profile_and_its_proposal_together(imported, kb, run_id):
    proposal = new_name(imported, kb, run_id, "ユキ", "小雪", aliases=["ゆき"])

    outcome = decide_proposal(imported, kb, proposal.proposal_id, approve=False)

    assert outcome == PROFILE_REJECTED
    (profile,) = profiles(imported, kb)
    assert profile["status"] == "rejected"
    assert without_time(history(profile)[-1]) == {
        "event": "rejected", "by": "human", "from": "pending_human", "proposal_id": proposal.proposal_id}
    row = proposal_row(imported, proposal)
    assert (row["status"], row["reason"].startswith("rejected by a person (earlier: a new name waits")) == ("rejected", True)
    assert load_snapshot(imported, kb).character("ユキ") is None  # not readable any more
    assert store.kb_proposals(imported, kb, "pending_human") == []


def test_approving_a_new_name_is_refused_when_it_breaks_a_rule_of_item_decisions(imported, kb, run_id):
    proposal = new_name(imported, kb, run_id, "ユキ", "小雪")
    insert_entry(imported, kb, "ユキ", "雪")  # a readable entry whose source is the name: a conflict across kinds

    refused(imported, lambda: decide_proposal(imported, kb, proposal.proposal_id, approve=True), "a term")
    (profile,) = profiles(imported, kb)
    assert (profile["status"], proposal_row(imported, proposal)["status"]) == ("pending_human", "pending_human")


# --- a conflict: only its rejection is allowed ---


def conflict_of_a_term(conn, kb, run_id):
    first = term(kb, 1, "封印", "封印")
    apply(conn, kb, 1, run_id, [first], accept_all([first]))
    rival = term(kb, 2, "封印", "封缄")
    assert results(apply(conn, kb, 2, run_id, [rival], accept_all([rival]))) == [CONFLICT]
    return rival


def conflict_of_a_term_across_kinds(conn, kb, run_id):
    new_name(conn, kb, run_id, "ユキ", "小雪")
    word = term(kb, 2, "ユキ", "雪")
    assert results(apply(conn, kb, 2, run_id, [word], accept_all([word]))) == [CONFLICT_ACROSS_KINDS]
    return word


def conflict_of_a_name(conn, kb, run_id):
    new_name(conn, kb, run_id, "ユキ", "小雪")
    rival = name(kb, 2, "ユキ", "雪儿")
    assert results(apply(conn, kb, 2, run_id, [rival], accept_all([rival]))) == [CONFLICT]
    return rival


def conflict_of_a_name_across_kinds(conn, kb, run_id):
    accepted_term(conn, kb, run_id, "ユキ", "雪")
    person = name(kb, 2, "ユキ", "小雪")
    assert results(apply(conn, kb, 2, run_id, [person])) == [CONFLICT_ACROSS_KINDS]
    return person


def conflict_of_rule_6(conn, kb, run_id):
    new_name(conn, kb, run_id, "桜野ツバメ", "樱野燕")
    other = name(kb, 2, "ツバメ", "铃")  # inside the name, with a rendering that agrees with neither
    assert results(apply(conn, kb, 2, run_id, [other])) == [CONFLICT]
    return other


CONFLICTS = [conflict_of_a_term, conflict_of_a_term_across_kinds, conflict_of_a_name, conflict_of_a_name_across_kinds,
             conflict_of_rule_6]


@pytest.mark.parametrize("make", CONFLICTS, ids=lambda f: f.__name__.removeprefix("conflict_of_"))
def test_a_conflict_can_be_rejected_and_the_kb_keeps_what_it_holds(imported, kb, run_id, make):
    proposal = make(imported, kb, run_id)
    row = proposal_row(imported, proposal)
    assert row["status"] == "pending_human"
    before = dump(imported)

    outcome = decide_proposal(imported, kb, proposal.proposal_id, approve=False)

    assert outcome == PROPOSAL_REJECTED
    after = dump(imported)
    assert (after["glossary_entries"], after["character_profiles"]) == (before["glossary_entries"], before["character_profiles"])
    decided = proposal_row(imported, proposal)
    assert decided["status"] == "rejected" and decided["decided_at"] is not None
    assert decided["reason"] == f"rejected by a person (earlier: {row['reason']})"  # why it was waiting stays readable
    waiting = [r["proposal_id"] for r in store.kb_proposals(imported, kb, "pending_human")]
    assert proposal.proposal_id not in waiting  # it left the Human Review Queue


@pytest.mark.parametrize("make", CONFLICTS, ids=lambda f: f.__name__.removeprefix("conflict_of_"))
def test_approving_a_conflict_is_refused_and_writes_nothing(imported, kb, run_id, make):
    proposal = make(imported, kb, run_id)

    refused(imported, lambda: decide_proposal(imported, kb, proposal.proposal_id, approve=True), "cannot be approved")

    assert proposal_row(imported, proposal)["status"] == "pending_human"  # still waiting, and the rendering was not taken


def test_approving_a_conflict_never_replaces_the_rendering_the_kb_holds(imported, kb, run_id):
    rival = conflict_of_a_term(imported, kb, run_id)
    (held,) = entries(imported, kb)

    refused(imported, lambda: decide_proposal(imported, kb, rival.proposal_id, approve=True))

    (still,) = entries(imported, kb)
    assert (tuple(still), still["target_term"]) == (tuple(held), "封印")


# --- a decided proposal is in no queue ---


@pytest.mark.parametrize("approve", [True, False], ids=["approve", "reject"])
@pytest.mark.parametrize("how", ["accepted", "rejected", "confirmed"])
def test_a_proposal_in_no_queue_is_refused(imported, kb, run_id, how, approve):
    if how == "accepted":
        proposal = term(kb, 1, "封印", "封印")
        apply(imported, kb, 1, run_id, [proposal], accept_all([proposal]))
    elif how == "rejected":
        proposal = term(kb, 1, "王国", "王国")
        apply(imported, kb, 1, run_id, [proposal], {proposal.proposal_id: reject()})
    else:
        accepted_term(imported, kb, run_id, "封印", "封印", page=1)
        proposal = term(kb, 2, "封印", "封印")
        assert results(apply(imported, kb, 2, run_id, [proposal])) == ["confirmed"]
    status = proposal_row(imported, proposal)["status"]

    refused(imported, lambda: decide_proposal(imported, kb, proposal.proposal_id, approve=approve), f"already {status}")


def test_a_proposal_a_person_decided_cannot_be_decided_again(imported, kb, run_id):
    proposal = undecided_term(imported, kb, run_id)
    decide_proposal(imported, kb, proposal.proposal_id, approve=False)

    refused(imported, lambda: decide_proposal(imported, kb, proposal.proposal_id, approve=True), "already rejected")
    refused(imported, lambda: decide_proposal(imported, kb, proposal.proposal_id, approve=False), "already rejected")


# --- an entry or a profile: approved or rejected, with the person's line ---


def test_an_entry_rejected_by_a_person_is_no_longer_readable_and_is_the_persons(imported, kb, run_id):
    entry = accepted_term(imported, kb, run_id, "封印", "封印", page=5)
    assert (entry["provenance"], entry["status"]) == ("agent", "approved")

    outcome = decide_entry(imported, kb, entry["id"], approve=False)

    assert outcome == ENTRY_REJECTED
    after = entry_row(imported, entry["id"])
    assert (after["status"], after["provenance"], pages(after)) == ("rejected", "human", [5])
    assert [without_time(line)["event"] for line in history(after)] == ["created", "rejected"]
    assert without_time(history(after)[-1]) == {"event": "rejected", "by": "human", "from": "approved"}
    assert load_snapshot(imported, kb).term("封印") is None  # a wrong entry is stopped: the agents no longer read it
    assert proposal_row(imported, history(after)[0]["proposal_id"])["status"] == "approved"  # the proposal is decided history


def test_an_entry_a_person_rejected_can_be_approved_again(imported, kb, run_id):
    entry = accepted_term(imported, kb, run_id, "封印", "封印")
    decide_entry(imported, kb, entry["id"], approve=False)

    outcome = decide_entry(imported, kb, entry["id"], approve=True)

    assert outcome == ENTRY_APPROVED
    after = entry_row(imported, entry["id"])
    assert (after["status"], after["provenance"]) == ("approved", "human")
    assert [without_time(line).get("from") for line in history(after)] == [None, "approved", "rejected"]
    assert without_time(history(after)[-1]) == {"event": "approved", "by": "human", "from": "rejected"}
    assert load_snapshot(imported, kb).term("封印").status == "approved"


def test_a_pending_entry_is_approved_by_a_person(imported, kb):
    entry_id = insert_entry(imported, kb, "封印", status="pending_human")

    assert decide_entry(imported, kb, entry_id, approve=True) == ENTRY_APPROVED

    after = entry_row(imported, entry_id)
    assert (after["status"], after["provenance"]) == ("approved", "human")
    assert without_time(history(after)[-1])["from"] == "pending_human"


def test_a_profile_decided_by_a_person_has_their_line_and_no_provenance_column(imported, kb, run_id):
    new_name(imported, kb, run_id, "ユキ", "小雪")
    (profile,) = profiles(imported, kb)
    assert "provenance" not in profile.keys()  # the schema is as it was: the history line is the record

    assert decide_profile(imported, kb, profile["id"], approve=True) == PROFILE_APPROVED
    assert decide_profile(imported, kb, profile["id"], approve=False) == PROFILE_REJECTED

    after = profile_row(imported, profile["id"])
    assert after["status"] == "rejected"
    assert [(without_time(line).get("event"), without_time(line).get("by"), without_time(line).get("from"))
            for line in history(after)] == [("created", None, None), ("approved", "human", "pending_human"),
                                            ("rejected", "human", "approved")]
    assert all("proposal_id" not in line for line in history(after)[1:])  # not decided through a proposal


def test_a_decision_changes_the_status_and_the_history_and_nothing_else(imported, kb, run_id):
    entry = accepted_term(imported, kb, run_id, "封印", "封印", page=5)
    decide_entry(imported, kb, entry["id"], approve=False)

    after = entry_row(imported, entry["id"])

    for column in ("id", "kb_id", "source_term", "target_term", "category", "evidence_pages", "created_at"):
        assert after[column] == entry[column], column


# --- the refusals of item decisions: nothing is written ---


@pytest.mark.parametrize("status, approve", [("approved", True), ("rejected", False)], ids=["approve-approved", "reject-rejected"])
def test_a_decision_that_changes_nothing_is_refused_for_an_entry(imported, kb, status, approve):
    entry_id = insert_entry(imported, kb, "封印", status=status)

    refused(imported, lambda: decide_entry(imported, kb, entry_id, approve=approve), f"already {status}")


@pytest.mark.parametrize("status, approve", [("approved", True), ("rejected", False)], ids=["approve-approved", "reject-rejected"])
def test_a_decision_that_changes_nothing_is_refused_for_a_profile(imported, kb, status, approve):
    profile_id = insert_profile(imported, kb, "ユキ", status=status)

    refused(imported, lambda: decide_profile(imported, kb, profile_id, approve=approve), f"already {status}")


@pytest.mark.parametrize("rival_status", ["approved", "pending_human"])
def test_an_entry_is_not_approved_when_that_leaves_two_readable_rows_for_one_source_term(imported, kb, rival_status):
    rival = insert_entry(imported, kb, "封印", "封印", status=rival_status)
    rejected = insert_entry(imported, kb, "封印", "封缄", status="rejected", provenance="human")

    refused(imported, lambda: decide_entry(imported, kb, rejected, approve=False), "already rejected")
    refused(imported, lambda: decide_entry(imported, kb, rejected, approve=True), f"entry {rival} \\(封印\\) is {rival_status}")
    assert entry_row(imported, rejected)["status"] == "rejected"


def test_an_entry_is_approved_when_the_other_rows_for_its_source_are_not_readable(imported, kb):
    insert_entry(imported, kb, "封印", "封印", status="rejected")
    insert_entry(imported, kb, "封印", "封印术", status="proposed")
    wanted = insert_entry(imported, kb, "封印", "封缄", status="rejected")

    assert decide_entry(imported, kb, wanted, approve=True) == ENTRY_APPROVED
    assert entry_row(imported, wanted)["status"] == "approved"


@pytest.mark.parametrize("rival_status", ["approved", "pending_human"])
def test_a_profile_is_not_approved_when_that_leaves_two_readable_rows_for_one_name(imported, kb, rival_status):
    rival = insert_profile(imported, kb, "ユキ", "小雪", status=rival_status)
    rejected = insert_profile(imported, kb, "ユキ", "雪儿", status="rejected")

    refused(imported, lambda: decide_profile(imported, kb, rejected, approve=True), f"profile {rival} \\(小雪\\) is {rival_status}")


def test_a_pending_row_is_not_approved_while_another_readable_row_holds_its_source(imported, kb):
    """Approving a `pending_human` row only changes its status, but two readable rows would stay for one source: refused."""
    insert_entry(imported, kb, "封印", "封印")
    pending_entry = insert_entry(imported, kb, "封印", "封缄", status="pending_human")
    insert_profile(imported, kb, "ユキ", "小雪")
    pending_profile = insert_profile(imported, kb, "ユキ", "雪儿", status="pending_human")

    refused(imported, lambda: decide_entry(imported, kb, pending_entry, approve=True), "two readable entries")
    refused(imported, lambda: decide_profile(imported, kb, pending_profile, approve=True), "two readable profiles")


def test_a_profile_is_approved_when_the_other_rows_for_its_name_are_not_readable(imported, kb):
    insert_profile(imported, kb, "ユキ", "小雪", status="rejected")
    wanted = insert_profile(imported, kb, "ユキ", "雪儿", status="rejected")

    assert decide_profile(imported, kb, wanted, approve=True) == PROFILE_APPROVED


@pytest.mark.parametrize("profile_status", ["approved", "pending_human"])
@pytest.mark.parametrize("called, aliases", [("ユキ", []), ("ゆき", ["ゆき"])], ids=["name", "alias"])
def test_an_entry_is_not_approved_when_its_source_is_a_readable_profiles_name_or_alias(
        imported, kb, profile_status, called, aliases):
    insert_profile(imported, kb, "ユキ", "小雪", aliases=aliases, status=profile_status)
    entry_id = insert_entry(imported, kb, called, "雪", status="rejected")

    refused(imported, lambda: decide_entry(imported, kb, entry_id, approve=True), "a character")


@pytest.mark.parametrize("entry_status", ["approved", "pending_human"])
def test_a_profile_is_not_approved_when_its_name_is_a_readable_entrys_source(imported, kb, entry_status):
    entry = insert_entry(imported, kb, "ユキ", "雪", status=entry_status)
    profile_id = insert_profile(imported, kb, "ユキ", "小雪", status="rejected")

    refused(imported, lambda: decide_profile(imported, kb, profile_id, approve=True), f"source of entry {entry} \\(雪\\), a term")


def test_an_entry_is_approved_when_the_profile_with_its_name_is_not_readable(imported, kb):
    insert_profile(imported, kb, "ユキ", "小雪", status="rejected")
    entry_id = insert_entry(imported, kb, "ユキ", "雪", status="rejected")

    assert decide_entry(imported, kb, entry_id, approve=True) == ENTRY_APPROVED


def test_rejecting_is_never_refused_for_a_collision(imported, kb):
    insert_entry(imported, kb, "封印", "封印")
    second = insert_entry(imported, kb, "封印", "封缄", status="pending_human")  # two readable rows already
    profile_id = insert_profile(imported, kb, "封印", "印", status="pending_human")

    assert decide_entry(imported, kb, second, approve=False) == ENTRY_REJECTED
    assert decide_profile(imported, kb, profile_id, approve=False) == PROFILE_REJECTED


# --- deciding a profile settles the proposal it was made from ---


@pytest.mark.parametrize("approve", [True, False], ids=["approve", "reject"])
def test_a_profile_decided_as_an_item_settles_its_new_name_proposal(imported, kb, run_id, approve):
    proposal = new_name(imported, kb, run_id, "ユキ", "小雪")
    (profile,) = profiles(imported, kb)
    earlier = proposal_row(imported, proposal)["reason"]

    outcome = decide_profile(imported, kb, profile["id"], approve=approve)

    status = "approved" if approve else "rejected"
    assert outcome == (PROFILE_APPROVED if approve else PROFILE_REJECTED)
    assert profile_row(imported, profile["id"])["status"] == status
    row = proposal_row(imported, proposal)  # the two queues never disagree
    assert (row["status"], row["decided_at"] is not None) == (status, True)
    assert row["reason"] == f"{status} by a person, as profile {profile['id']} (earlier: {earlier})"
    assert store.kb_proposals(imported, kb, "pending_human") == []
    assert "proposal_id" not in without_time(history(profile_row(imported, profile["id"]))[-1])  # it did not come through it


def test_a_profile_decided_later_leaves_a_proposal_that_was_decided_already(imported, kb, run_id):
    proposal = new_name(imported, kb, run_id, "ユキ", "小雪")
    decide_proposal(imported, kb, proposal.proposal_id, approve=True)
    decided = tuple(proposal_row(imported, proposal))
    (profile,) = profiles(imported, kb)

    assert decide_profile(imported, kb, profile["id"], approve=False) == PROFILE_REJECTED

    assert tuple(proposal_row(imported, proposal)) == decided  # the person's earlier decision stands as it was written


def test_a_profile_that_no_proposal_made_is_decided_alone(imported, kb):
    profile_id = insert_profile(imported, kb, "ユキ", "小雪", status="pending_human")
    assert made_from_proposal(profile_row(imported, profile_id)) is None

    assert decide_profile(imported, kb, profile_id, approve=True) == PROFILE_APPROVED
    assert store.kb_proposals(imported, kb) == []


def test_a_profile_is_found_from_the_proposal_that_made_it(imported, kb, run_id):
    proposal = new_name(imported, kb, run_id)
    (profile,) = profiles(imported, kb)

    assert made_from_proposal(profile) == proposal.proposal_id
    decide_proposal(imported, kb, proposal.proposal_id, approve=True)
    assert made_from_proposal(profile_row(imported, profile["id"])) == proposal.proposal_id  # the first line stays the first


def test_a_decision_does_not_touch_the_proposal_of_a_profile_that_merged_a_name_in(imported, kb, run_id):
    first = new_name(imported, kb, run_id, "桜野ツバメ", "樱野燕", page=1)
    joined = name(kb, 2, "ツバメ", "燕")
    assert results(apply(imported, kb, 2, run_id, [joined])) == ["merged"]
    (profile,) = profiles(imported, kb)

    decide_profile(imported, kb, profile["id"], approve=True)

    assert proposal_row(imported, first)["status"] == "approved"  # the new name's proposal is settled
    row = proposal_row(imported, joined)
    assert row["status"] == "approved" and not row["reason"].startswith("approved by a person")  # the merge is the agent's


# --- the Editor KB only, and only what the KB has ---


def test_an_eval_runs_kb_is_never_changed_by_a_decision(imported, run_id):
    eval_kb = new_kb(imported, "run-1")  # purpose eval
    waiting = new_name(imported, eval_kb, run_id, "ユキ", "小雪")
    undecided = undecided_term(imported, eval_kb, run_id, "封印", "封印")
    entry = accepted_term(imported, eval_kb, run_id, "魔法", "魔法")
    (profile,) = profiles(imported, eval_kb)
    before = dump(imported)

    for call in (
        lambda: decide_proposal(imported, eval_kb, waiting.proposal_id, approve=True),
        lambda: decide_proposal(imported, eval_kb, undecided.proposal_id, approve=False),
        lambda: decide_entry(imported, eval_kb, entry["id"], approve=False),
        lambda: decide_profile(imported, eval_kb, profile["id"], approve=True),
    ):
        with pytest.raises(KBNotFound, match="no Editor KB"):
            call()
        assert dump(imported) == before and not imported.in_transaction


def test_an_unknown_kb_is_not_found(imported):
    for call in (lambda: decide_proposal(imported, 99, "x", approve=True), lambda: decide_entry(imported, 99, 1, approve=True),
                 lambda: decide_profile(imported, 99, 1, approve=True)):
        with pytest.raises(KBNotFound):
            call()


def test_an_id_the_kb_does_not_have_is_not_found_and_nothing_is_written(imported, kb, run_id):
    imported.execute("INSERT INTO books (book_title) VALUES ('book_b')")
    imported.commit()
    other = store.editor_kb(imported, "book_b")  # another book's Editor KB has rows of its own
    other_entry = accepted_term(imported, other, run_id, "魔法", "魔法")
    other_proposal = new_name(imported, other, run_id, "ユキ", "小雪")
    (other_profile,) = profiles(imported, other)
    before = dump(imported)

    for call in (
        lambda: decide_proposal(imported, kb, "no-such-proposal", approve=True),
        lambda: decide_proposal(imported, kb, other_proposal.proposal_id, approve=True),
        lambda: decide_entry(imported, kb, 12345, approve=False),
        lambda: decide_entry(imported, kb, other_entry["id"], approve=False),
        lambda: decide_profile(imported, kb, 12345, approve=True),
        lambda: decide_profile(imported, kb, other_profile["id"], approve=True),
    ):
        with pytest.raises(ItemNotFound):
            call()
        assert dump(imported) == before and not imported.in_transaction


# --- all or nothing, one transaction ---


def scenario_undecided_term_approved(conn, kb, run_id):
    proposal = undecided_term(conn, kb, run_id)
    return lambda: decide_proposal(conn, kb, proposal.proposal_id, approve=True)


def scenario_undecided_term_approved_as_a_confirmation(conn, kb, run_id):
    proposal = undecided_term(conn, kb, run_id, page=2)
    accepted_term(conn, kb, run_id, page=1)
    return lambda: decide_proposal(conn, kb, proposal.proposal_id, approve=True)


def scenario_undecided_term_approved_as_a_conflict(conn, kb, run_id):
    proposal = undecided_term(conn, kb, run_id, "封印", "封缄", page=2)
    accepted_term(conn, kb, run_id, "封印", "封印", page=1)
    return lambda: decide_proposal(conn, kb, proposal.proposal_id, approve=True)


def scenario_undecided_term_rejected(conn, kb, run_id):
    proposal = undecided_term(conn, kb, run_id)
    return lambda: decide_proposal(conn, kb, proposal.proposal_id, approve=False)


def scenario_new_name_approved(conn, kb, run_id):
    proposal = new_name(conn, kb, run_id)
    return lambda: decide_proposal(conn, kb, proposal.proposal_id, approve=True)


def scenario_new_name_rejected(conn, kb, run_id):
    proposal = new_name(conn, kb, run_id)
    return lambda: decide_proposal(conn, kb, proposal.proposal_id, approve=False)


def scenario_conflict_rejected(conn, kb, run_id):
    rival = conflict_of_a_term(conn, kb, run_id)
    return lambda: decide_proposal(conn, kb, rival.proposal_id, approve=False)


def scenario_entry_rejected(conn, kb, run_id):
    entry = accepted_term(conn, kb, run_id)
    return lambda: decide_entry(conn, kb, entry["id"], approve=False)


def scenario_profile_decided_as_an_item(conn, kb, run_id):
    new_name(conn, kb, run_id)
    (profile,) = profiles(conn, kb)
    return lambda: decide_profile(conn, kb, profile["id"], approve=True)


FAULTS = [
    (scenario_undecided_term_approved, "insert_entry"),
    (scenario_undecided_term_approved, "set_proposal_status"),
    (scenario_undecided_term_approved_as_a_confirmation, "add_entry_evidence"),
    (scenario_undecided_term_approved_as_a_confirmation, "set_proposal_status"),
    (scenario_undecided_term_approved_as_a_conflict, "set_proposal_status"),
    (scenario_undecided_term_rejected, "set_proposal_status"),
    (scenario_new_name_approved, "set_profile_status"),
    (scenario_new_name_approved, "set_proposal_status"),
    (scenario_new_name_rejected, "set_profile_status"),
    (scenario_new_name_rejected, "set_proposal_status"),
    (scenario_conflict_rejected, "set_proposal_status"),
    (scenario_entry_rejected, "set_entry_status"),
    (scenario_profile_decided_as_an_item, "set_profile_status"),
    (scenario_profile_decided_as_an_item, "set_proposal_status"),
]


@pytest.mark.parametrize("scenario, function", FAULTS, ids=[f"{s.__name__.removeprefix('scenario_')}:{f}" for s, f in FAULTS])
def test_when_a_write_fails_partway_nothing_of_the_decision_is_stored(imported, kb, run_id, monkeypatch, scenario, function):
    decide = scenario(imported, kb, run_id)
    before = dump(imported)
    real = getattr(store, function)
    calls = []

    def failing(*args, **kwargs):
        calls.append(function)
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, function, failing)

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        decide()

    assert len(calls) == 1  # it did get that far, and what was written before it is gone
    assert dump(imported) == before
    assert not imported.in_transaction
    monkeypatch.setattr(store, function, real)  # once the fault is gone the decision goes through, whole
    decide()
    assert dump(imported) != before


@pytest.mark.parametrize("decide", [
    lambda conn, kb: decide_proposal(conn, kb, "x", approve=True),
    lambda conn, kb: decide_entry(conn, kb, 1, approve=True),
    lambda conn, kb: decide_profile(conn, kb, 1, approve=True),
], ids=["proposal", "entry", "profile"])
def test_a_decision_does_not_commit_what_the_caller_has_pending(imported, kb, decide):
    imported.execute("INSERT INTO books (book_title) VALUES ('pending_book')")  # an open transaction of the caller's

    with pytest.raises(sqlite3.OperationalError, match="within a transaction"):
        decide(imported, kb)

    imported.rollback()
    assert imported.execute("SELECT COUNT(*) FROM books WHERE book_title = 'pending_book'").fetchone()[0] == 0


# --- `apply` is as it was ---


def test_apply_still_writes_what_the_agent_decided_and_never_says_a_person_did(imported, kb, run_id):
    accepted = term(kb, 1, "封印", "封印")
    rejected = term(kb, 1, "王国", "王国")
    known = term(kb, 2, "封印", "封印")
    rival = term(kb, 2, "封印", "封缄")
    across = term(kb, 2, "ユキ", "雪")
    apply(imported, kb, 1, run_id, [accepted, rejected, name(kb, 1, "ユキ", "小雪")],
          {accepted.proposal_id: accept("keep"), rejected.proposal_id: reject("generic")})
    apply(imported, kb, 2, run_id, [known, rival, across], accept_all([known, rival, across]))

    assert [e["provenance"] for e in entries(imported, kb)] == ["agent"]
    lines = [line for row in (*entries(imported, kb), *profiles(imported, kb)) for line in history(row)]
    assert lines and all("by" not in line for line in lines)  # a person's mark is theirs alone
    assert [proposal_row(imported, p)["reason"] for p in (accepted, rejected, known, rival)] == [
        "keep", "generic", f"confirms entry {entries(imported, kb)[0]['id']}",
        f"conflicts with entry {entries(imported, kb)[0]['id']}, which is 封印"]
    assert proposal_row(imported, across)["reason"].startswith("conflicts across kinds: ユキ is proposed as a term")
