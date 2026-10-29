"""The Curator's rules, one test each, and that `apply` is all or nothing."""

from __future__ import annotations

import dataclasses
import json
import sqlite3

import pytest
from kb_support import accept, dump, name, new_kb, new_run, reject, term

from fukidashi import store
from fukidashi.datasets.importer import import_openmantra
from fukidashi.kb import KBNotFound
from fukidashi.kb.curator import (
    ACCEPTED, CONFIRMED, CONFLICT, CONFLICT_ACROSS_KINDS, DUPLICATE, PENDING, REJECTED, RESULTS, UNDECIDED, Outcome,
    apply,
)
from fukidashi.kb.loader import load_snapshot

ALL_STATUSES = ("proposed", "approved", "rejected", "pending_human")


@pytest.fixture
def imported(conn, dataset):
    import_openmantra(conn, dataset)
    return conn


@pytest.fixture
def kb(imported):
    return new_kb(imported)


@pytest.fixture
def run_id(imported):
    return new_run(imported)


def results(outcomes):
    return [o.result for o in outcomes]


def accept_all(proposals):
    return {p.proposal_id: accept() for p in proposals}


def proposal_row(conn, proposal):
    return conn.execute("SELECT * FROM proposals WHERE proposal_id = ?", (proposal.proposal_id,)).fetchone()


def entries(conn, kb_id):
    return store.kb_entries(conn, kb_id, ALL_STATUSES)


def profiles(conn, kb_id):
    return store.kb_profiles(conn, kb_id, ALL_STATUSES)


def pages(row):
    return json.loads(row["evidence_pages"])


def history(row):
    return json.loads(row["history"])


def test_the_results_are_the_nine_the_rules_can_give():
    assert set(RESULTS) == {"duplicate", "confirmed", "merged", "conflict", "conflict_across_kinds", "pending_human",
                            "accepted", "rejected", "undecided"}


# --- rule 1: a duplicate id is ignored ---


def test_a_retried_page_proposes_the_same_ids_and_writes_nothing_the_second_time(imported, kb, run_id):
    def page(n):  # built afresh each time, as a retried page would
        return [term(kb, n, "封印", "封印"), term(kb, n, "魔法", "魔法"), name(kb, n, "ユキ", "小雪", aliases=["ゆき"])]

    first, second = page(1), page(2)
    assert results(apply(imported, kb, 1, run_id, first, accept_all(first))) == [ACCEPTED, ACCEPTED, PENDING]
    assert results(apply(imported, kb, 2, run_id, second, accept_all(second))) == [CONFIRMED] * 3
    before = dump(imported)
    assert pages(entries(imported, kb)[0]) == [1, 2] and len(history(profiles(imported, kb)[0])) == 2

    for n, proposals in [(2, page(2)), (1, page(1))]:
        again = apply(imported, kb, n, new_run(imported), proposals, accept_all(proposals))
        assert results(again) == [DUPLICATE] * 3
        assert [o.proposal_id for o in again] == [p.proposal_id for p in proposals]
        assert dump(imported) == before  # entries, profiles, evidence, history and proposals: nothing written


def test_a_proposal_listed_twice_in_one_call_is_applied_once(imported, kb, run_id):
    proposal = term(kb, 1, "封印", "封印")

    outcomes = apply(imported, kb, 1, run_id, [proposal, proposal], accept_all([proposal]))

    assert results(outcomes) == [ACCEPTED, DUPLICATE]
    assert len(entries(imported, kb)) == 1 and len(store.kb_proposals(imported, kb)) == 1


def test_a_proposal_that_stayed_undecided_is_a_duplicate_when_its_page_comes_again(imported, kb, run_id):
    proposal = term(kb, 1, "封印", "封印")
    apply(imported, kb, 1, run_id, [proposal])  # the Curator's call failed: no decision
    before = dump(imported)

    again = apply(imported, kb, 1, run_id, [term(kb, 1, "封印", "封印")], accept_all([proposal]))

    assert results(again) == [DUPLICATE] and dump(imported) == before  # rule 1 comes first, as written


# --- rule 2: a confirmation joins the evidence ---


def test_a_confirmation_adds_the_page_to_the_evidence_once(imported, kb, run_id):
    first = term(kb, 1, "封印", "封印", category="magic")
    apply(imported, kb, 1, run_id, [first], accept_all([first]))
    second_run = new_run(imported)
    second = term(kb, 2, "封印", "封印")

    outcomes = apply(imported, kb, 2, second_run, [second])  # no decision needed

    assert results(outcomes) == [CONFIRMED]
    (entry,) = entries(imported, kb)
    assert (entry["target_term"], entry["category"], entry["status"], entry["provenance"]) == (
        "封印", "magic", "approved", "agent")
    assert pages(entry) == [1, 2]
    created, confirmed = history(entry)
    assert (created["event"], created["page"], created["run_id"]) == ("created", 1, run_id)
    assert (confirmed["event"], confirmed["page"], confirmed["run_id"], confirmed["proposal_id"]) == (
        "confirmed", 2, second_run, second.proposal_id)
    row = proposal_row(imported, second)
    assert (row["status"], row["decided_at"] is not None) == ("approved", True) and "entry" in row["reason"]
    assert results(apply(imported, kb, 2, second_run, [second])) == [DUPLICATE]
    assert pages(entries(imported, kb)[0]) == [1, 2] and len(history(entries(imported, kb)[0])) == 2


def test_a_decision_does_not_change_a_confirmation(imported, kb, run_id):
    first = term(kb, 1, "封印", "封印")
    apply(imported, kb, 1, run_id, [first], accept_all([first]))
    second = term(kb, 2, "封印", "封印")

    outcomes = apply(imported, kb, 2, run_id, [second], {second.proposal_id: reject("not needed")})

    assert results(outcomes) == [CONFIRMED] and entries(imported, kb)[0]["status"] == "approved"


def test_a_confirmed_name_keeps_its_status_and_gains_the_page(imported, kb, run_id):
    first = name(kb, 1, "ユキ", "小雪", aliases=["ゆき"], notes="轻声细语")
    apply(imported, kb, 1, run_id, [first])
    second = name(kb, 2, "ユキ", "小雪", aliases=["雪ちゃん"], notes="大声说话")

    outcomes = apply(imported, kb, 2, run_id, [second])

    assert results(outcomes) == [CONFIRMED]
    (profile,) = profiles(imported, kb)
    assert (profile["status"], pages(profile)) == ("pending_human", [1, 2])
    assert (json.loads(profile["aliases"]), profile["notes"]) == (["ゆき"], "轻声细语")  # the profile is as it was
    assert [h["event"] for h in history(profile)] == ["created", "confirmed"]
    assert proposal_row(imported, second)["status"] == "approved"


def test_evidence_never_lists_a_page_twice(imported, kb, run_id):
    first = term(kb, 1, "封印", "封印")
    apply(imported, kb, 1, run_id, [first], accept_all([first]))
    entry_id = entries(imported, kb)[0]["id"]
    before = dump(imported)

    assert store.add_entry_evidence(imported, entry_id, 1, {"event": "confirmed"}) is False

    assert dump(imported) == before


# --- rule 3: a conflict goes to a human, the entry stays ---


def test_a_conflicting_term_goes_to_the_human_review_queue_and_the_entry_stays(imported, kb, run_id):
    first = term(kb, 1, "封印", "封印")
    apply(imported, kb, 1, run_id, [first], accept_all([first]))
    entry_before = tuple(entries(imported, kb)[0])
    rival = term(kb, 2, "封印", "封缄")

    outcomes = apply(imported, kb, 2, run_id, [rival], accept_all([rival]))  # even an "accept" cannot override it

    assert results(outcomes) == [CONFLICT]
    assert [tuple(e) for e in entries(imported, kb)] == [entry_before]
    row = proposal_row(imported, rival)
    assert (row["status"], row["decided_at"] is not None) == ("pending_human", True)
    assert "conflicts with entry" in row["reason"] and "封印" in row["reason"]
    assert load_snapshot(imported, kb).term("封印").target_term == "封印"  # later pages keep the first rendering


def test_of_two_conflicting_proposals_on_one_page_the_first_wins(imported, kb, run_id):
    first, second = term(kb, 1, "封印", "封印"), term(kb, 1, "封印", "封缄")

    outcomes = apply(imported, kb, 1, run_id, [first, second], accept_all([first, second]))

    assert results(outcomes) == [ACCEPTED, CONFLICT]
    (entry,) = entries(imported, kb)
    assert (entry["target_term"], pages(entry)) == ("封印", [1])
    assert (proposal_row(imported, first)["status"], proposal_row(imported, second)["status"]) == (
        "approved", "pending_human")


def test_the_order_of_the_proposals_decides_which_conflicting_one_wins(imported, kb, run_id):
    first, second = term(kb, 1, "封印", "封印"), term(kb, 1, "封印", "封缄")

    outcomes = apply(imported, kb, 1, run_id, [second, first], accept_all([first, second]))

    assert results(outcomes) == [ACCEPTED, CONFLICT] and entries(imported, kb)[0]["target_term"] == "封缄"


def test_a_conflict_needs_an_entry_so_a_rejected_first_proposal_does_not_win(imported, kb, run_id):
    first, second = term(kb, 1, "封印", "封印"), term(kb, 1, "封印", "封缄")

    outcomes = apply(imported, kb, 1, run_id, [first, second], {first.proposal_id: reject(), second.proposal_id: accept()})

    assert results(outcomes) == [REJECTED, ACCEPTED] and entries(imported, kb)[0]["target_term"] == "封缄"


def test_a_conflicting_name_goes_to_the_human_review_queue_and_the_profile_stays(imported, kb, run_id):
    apply(imported, kb, 1, run_id, [name(kb, 1, "ユキ", "小雪")])
    profile_before = tuple(profiles(imported, kb)[0])
    rival = name(kb, 2, "ユキ", "雪儿")

    outcomes = apply(imported, kb, 2, run_id, [rival], accept_all([rival]))

    assert results(outcomes) == [CONFLICT]
    assert [tuple(p) for p in profiles(imported, kb)] == [profile_before]
    row = proposal_row(imported, rival)
    assert (row["status"], "conflicts with profile" in row["reason"]) == ("pending_human", True)


def test_a_term_and_a_name_with_the_same_source_meet_as_a_conflict_across_kinds(imported, kb, run_id):
    # Task 015: this used to assert [ACCEPTED], "terms are compared with entries, names with profiles". A name proposed
    # as a term now goes to a person (OUTLINE 3.3, "Still open"), so the same setup gives the opposite result.
    apply(imported, kb, 1, run_id, [name(kb, 1, "ユキ", "小雪")])
    word = term(kb, 2, "ユキ", "雪")

    outcomes = apply(imported, kb, 2, run_id, [word], accept_all([word]))

    assert results(outcomes) == [CONFLICT_ACROSS_KINDS] and entries(imported, kb) == []


# --- rule 3, across kinds: a name under the wrong kind goes to a person, and nothing is written for it ---


def insert_profile(conn, kb_id, name_src, name_tgt="小雪", aliases=(), status="approved"):
    """A profile in any state, as a person or an older Curator left it. Committed: `apply` wants no open transaction."""
    store.insert_profile(conn, kb_id=kb_id, name_src=name_src, name_tgt=name_tgt, aliases=list(aliases), notes="",
                         status=status, page_index=1, event={"event": "created"})
    conn.commit()


def insert_entry(conn, kb_id, source, target="封印", status="approved"):
    store.insert_entry(conn, kb_id=kb_id, source_term=source, target_term=target, category="", status=status,
                       provenance="human", page_index=1, event={"event": "created"})
    conn.commit()


def kinds_untouched(conn, kb_id, before):
    """Entries and profiles are as they were; only the proposal queue grew."""
    now = dump(conn)
    assert (now["glossary_entries"], now["character_profiles"]) == (before["glossary_entries"], before["character_profiles"])


@pytest.mark.parametrize("decision", [None, accept("worth keeping"), reject("too generic")], ids=["none", "accept", "reject"])
@pytest.mark.parametrize("status", ["approved", "pending_human"])
def test_a_term_whose_source_is_a_profiles_name_goes_to_a_person_and_no_entry_is_written(
        imported, kb, run_id, status, decision):
    insert_profile(imported, kb, "ユキ", aliases=["ゆき"], status=status)
    before = dump(imported)
    word = term(kb, 2, "ユキ", "雪")

    outcomes = apply(imported, kb, 2, run_id, [word], {} if decision is None else {word.proposal_id: decision})

    assert results(outcomes) == [CONFLICT_ACROSS_KINDS]   # whatever the Curator Agent said: the rules are code
    assert entries(imported, kb) == []
    kinds_untouched(imported, kb, before)
    row = proposal_row(imported, word)
    assert (row["status"], row["decided_at"] is not None) == ("pending_human", True)
    assert "across kinds" in row["reason"] and "ユキ" in row["reason"] and "profile" in row["reason"]
    assert load_snapshot(imported, kb).term("ユキ") is None   # still no term for the agents to find


def test_a_term_whose_source_is_a_profiles_alias_goes_to_a_person_and_no_entry_is_written(imported, kb, run_id):
    insert_profile(imported, kb, "ユキ", aliases=["ゆき", "雪ちゃん"])
    before = dump(imported)
    word = term(kb, 2, "雪ちゃん", "小雪酱")

    outcomes = apply(imported, kb, 2, run_id, [word], accept_all([word]))

    assert results(outcomes) == [CONFLICT_ACROSS_KINDS] and entries(imported, kb) == []
    kinds_untouched(imported, kb, before)
    row = proposal_row(imported, word)
    assert row["status"] == "pending_human" and "across kinds" in row["reason"] and "an alias" in row["reason"]


@pytest.mark.parametrize("decision", [None, accept("a real name"), reject("not a name")], ids=["none", "accept", "reject"])
@pytest.mark.parametrize("status", ["approved", "pending_human"])
def test_a_character_whose_name_is_an_entrys_source_goes_to_a_person_and_no_profile_is_written(
        imported, kb, run_id, status, decision):
    insert_entry(imported, kb, "封印", "封印术", status=status)
    before = dump(imported)
    person = name(kb, 2, "封印", "封印", aliases=["ふういん"])

    outcomes = apply(imported, kb, 2, run_id, [person], {} if decision is None else {person.proposal_id: decision})

    assert results(outcomes) == [CONFLICT_ACROSS_KINDS] and profiles(imported, kb) == []
    kinds_untouched(imported, kb, before)
    row = proposal_row(imported, person)
    assert (row["status"], row["decided_at"] is not None) == ("pending_human", True)
    assert "across kinds" in row["reason"] and "封印" in row["reason"] and "entry" in row["reason"]
    assert load_snapshot(imported, kb).character("封印") is None


def test_on_one_page_a_name_then_a_term_with_its_source_meet_as_a_conflict_across_kinds(imported, kb, run_id):
    person, word = name(kb, 1, "ユキ", "小雪"), term(kb, 1, "ユキ", "雪")

    outcomes = apply(imported, kb, 1, run_id, [person, word], accept_all([person, word]))

    assert results(outcomes) == [PENDING, CONFLICT_ACROSS_KINDS]   # the earlier proposal counts as existing for the later one
    assert [p["name_src"] for p in profiles(imported, kb)] == ["ユキ"] and entries(imported, kb) == []
    assert proposal_row(imported, word)["status"] == "pending_human"


def test_on_one_page_a_term_then_a_name_with_its_source_meet_as_a_conflict_across_kinds(imported, kb, run_id):
    word, person = term(kb, 1, "ユキ", "雪"), name(kb, 1, "ユキ", "小雪")

    outcomes = apply(imported, kb, 1, run_id, [word, person], accept_all([word]))

    assert results(outcomes) == [ACCEPTED, CONFLICT_ACROSS_KINDS]
    assert [e["source_term"] for e in entries(imported, kb)] == ["ユキ"] and profiles(imported, kb) == []
    row = proposal_row(imported, person)
    assert row["status"] == "pending_human" and "across kinds" in row["reason"]


def test_on_one_page_a_name_after_a_term_that_was_not_stored_is_a_new_name(imported, kb, run_id):
    word, person = term(kb, 1, "ユキ", "雪"), name(kb, 1, "ユキ", "小雪")   # the term is rejected: there is no entry

    outcomes = apply(imported, kb, 1, run_id, [word, person], {word.proposal_id: reject("a name, not a term")})

    assert results(outcomes) == [REJECTED, PENDING] and [p["name_src"] for p in profiles(imported, kb)] == ["ユキ"]


@pytest.mark.parametrize("status", ["proposed", "rejected"])
def test_a_proposed_or_rejected_profile_never_triggers_the_rule_for_a_term(imported, kb, run_id, status):
    insert_profile(imported, kb, "ユキ", aliases=["ゆき"], status=status)
    words = [term(kb, 2, "ユキ", "雪"), term(kb, 2, "ゆき", "雪儿")]

    outcomes = apply(imported, kb, 2, run_id, words, accept_all(words))

    assert results(outcomes) == [ACCEPTED, ACCEPTED]   # the profile is not readable, so it is not there to collide with


@pytest.mark.parametrize("status", ["proposed", "rejected"])
def test_a_proposed_or_rejected_entry_never_triggers_the_rule_for_a_character(imported, kb, run_id, status):
    insert_entry(imported, kb, "封印", status=status)
    person = name(kb, 2, "封印", "封印")

    outcomes = apply(imported, kb, 2, run_id, [person])

    assert results(outcomes) == [PENDING] and [p["name_src"] for p in profiles(imported, kb)] == ["封印"]


def test_a_same_kind_match_comes_first_so_a_confirmation_is_still_a_confirmation(imported, kb, run_id):
    insert_entry(imported, kb, "ユキ", "雪")   # a term, and a profile with the same string: a KB older than the rule
    insert_profile(imported, kb, "ユキ")
    word = term(kb, 2, "ユキ", "雪")

    outcomes = apply(imported, kb, 2, run_id, [word])

    assert results(outcomes) == [CONFIRMED]   # nothing new is created, so no name ends up under the wrong kind
    assert pages(entries(imported, kb)[0]) == [1, 2]


def test_a_term_and_a_name_that_share_nothing_are_not_touched_by_the_rule(imported, kb, run_id):
    insert_profile(imported, kb, "ユキ", aliases=["ゆき"])
    insert_entry(imported, kb, "封印")
    proposals = [term(kb, 2, "結界", "结界"), name(kb, 2, "ハル", "小春", aliases=["ユキ以外"])]

    outcomes = apply(imported, kb, 2, run_id, proposals, accept_all(proposals))

    assert results(outcomes) == [ACCEPTED, PENDING]


# --- rule 4: a name becomes a pending profile, whatever the decision says ---


@pytest.mark.parametrize("decision", [None, accept("a real name"), reject("not a name")], ids=["none", "accept", "reject"])
def test_a_new_name_becomes_a_pending_profile_whatever_the_decision(imported, kb, run_id, decision):
    proposal = name(kb, 4, "ユキ", "小雪", aliases=["ゆき"], notes="轻声细语")

    outcomes = apply(imported, kb, 4, run_id, [proposal], {} if decision is None else {proposal.proposal_id: decision})

    assert results(outcomes) == [PENDING]
    (profile,) = profiles(imported, kb)
    assert (profile["name_src"], profile["name_tgt"], profile["status"]) == ("ユキ", "小雪", "pending_human")
    assert (json.loads(profile["aliases"]), profile["notes"], pages(profile)) == (["ゆき"], "轻声细语", [4])
    (created,) = history(profile)
    assert (created["event"], created["page"], created["run_id"], created["proposal_id"]) == (
        "created", 4, run_id, proposal.proposal_id)
    row = proposal_row(imported, proposal)
    assert row["status"] == "pending_human" and row["decided_at"] is not None
    if decision is not None:
        assert decision.reason in row["reason"]  # the human gets to read what the Curator thought
    assert entries(imported, kb) == []
    snapshot_profile = load_snapshot(imported, kb).character("ゆき")  # readable at once, marked pending
    assert (snapshot_profile.name_tgt, snapshot_profile.status) == ("小雪", "pending_human")


# --- rule 5: a new term is accepted, rejected or left undecided ---


def test_an_accepted_term_becomes_an_approved_entry_made_by_the_agent(imported, kb, run_id):
    proposal = term(kb, 5, "封印", "封印", category="magic")

    outcomes = apply(imported, kb, 5, run_id, [proposal], {proposal.proposal_id: accept("a recurring spell")})

    assert results(outcomes) == [ACCEPTED]
    (entry,) = entries(imported, kb)
    assert (entry["source_term"], entry["target_term"], entry["category"]) == ("封印", "封印", "magic")
    assert (entry["status"], entry["provenance"], pages(entry)) == ("approved", "agent", [5])
    (created,) = history(entry)
    assert (created["event"], created["page"], created["run_id"], created["proposal_id"], created["reason"]) == (
        "created", 5, run_id, proposal.proposal_id, "a recurring spell")
    row = proposal_row(imported, proposal)
    assert (row["status"], row["reason"], row["decided_at"] is not None) == ("approved", "a recurring spell", True)
    assert (row["kind"], row["proposer"], row["page_index"], row["run_id"], row["kb_id"]) == (
        "term", "translator", 5, run_id, kb)
    assert json.loads(row["payload"]) == proposal.payload
    assert load_snapshot(imported, kb).term("封印").status == "approved"


def test_a_rejected_term_writes_no_entry_and_keeps_its_reason(imported, kb, run_id):
    proposal = term(kb, 5, "ありがとう", "谢谢")

    outcomes = apply(imported, kb, 5, run_id, [proposal], {proposal.proposal_id: reject("an everyday word")})

    assert results(outcomes) == [REJECTED]
    assert entries(imported, kb) == []
    row = proposal_row(imported, proposal)
    assert (row["status"], row["reason"], row["decided_at"] is not None) == ("rejected", "an everyday word", True)
    assert load_snapshot(imported, kb).term("ありがとう") is None
    later = term(kb, 6, "ありがとう", "谢谢")  # a rejection is not forever: the same pair on another page is new
    assert results(apply(imported, kb, 6, run_id, [later], accept_all([later]))) == [ACCEPTED]


def test_a_term_without_a_decision_stays_proposed(imported, kb, run_id):
    proposal = term(kb, 5, "封印", "封印")

    outcomes = apply(imported, kb, 5, run_id, [proposal])  # for example the Curator's call failed

    assert results(outcomes) == [UNDECIDED]
    row = proposal_row(imported, proposal)
    assert (row["status"], row["reason"], row["decided_at"]) == ("proposed", None, None)
    assert entries(imported, kb) == []
    assert load_snapshot(imported, kb).term("封印") is None and store.kb_proposals(imported, kb, "proposed")


def test_one_page_with_every_kind_of_outcome(imported, kb, run_id):
    apply(imported, kb, 1, run_id, [term(kb, 1, "魔法", "魔法")], {term(kb, 1, "魔法", "魔法").proposal_id: accept()})
    page = [
        term(kb, 2, "魔法", "魔法"),  # confirmation
        term(kb, 2, "魔法", "魔术"),  # conflict
        term(kb, 2, "封印", "封印"),  # accepted
        term(kb, 2, "王国", "王国"),  # rejected
        term(kb, 2, "城", "城"),  # undecided
        name(kb, 2, "ユキ", "小雪"),  # pending profile
        term(kb, 2, "封印", "封印"),  # the same proposal again
    ]
    decisions = {page[2].proposal_id: accept(), page[3].proposal_id: reject("generic")}

    outcomes = apply(imported, kb, 2, run_id, page, decisions)

    assert outcomes == [Outcome(p.proposal_id, r) for p, r in zip(page, [
        CONFIRMED, CONFLICT, ACCEPTED, REJECTED, UNDECIDED, PENDING, DUPLICATE])]
    queue = store.kb_proposals(imported, kb, "pending_human")  # the Human Review Queue
    assert [r["proposal_id"] for r in queue] == [page[1].proposal_id, page[5].proposal_id]
    snapshot = load_snapshot(imported, kb)
    assert sorted(e.source_term for e in snapshot.entries) == ["封印", "魔法"]
    assert [p.name_src for p in snapshot.profiles] == ["ユキ"]
    assert snapshot.term("王国") is None and snapshot.term("城") is None


# --- errors: nothing is written ---


def test_a_decision_for_an_unknown_id_is_an_error_and_writes_nothing(imported, kb, run_id):
    earlier = term(kb, 1, "魔法", "魔法")
    apply(imported, kb, 1, run_id, [earlier], accept_all([earlier]))
    before = dump(imported)
    page = [term(kb, 2, "封印", "封印"), name(kb, 2, "ユキ", "小雪")]
    stranger = term(kb, 2, "王国", "王国")  # a real proposal, but not among this call's

    for decisions in ({"no-such-id": accept()}, {**accept_all(page), stranger.proposal_id: accept()}):
        with pytest.raises(ValueError, match="not among this page's"):
            apply(imported, kb, 2, run_id, page, decisions)
        assert dump(imported) == before
        assert not imported.in_transaction


def test_a_proposal_that_is_not_this_pages_or_does_not_match_its_id_is_refused_and_writes_nothing(imported, kb, run_id):
    other_kb = new_kb(imported, "run-2")
    good = term(kb, 2, "封印", "封印")
    bad = {
        "another page": term(kb, 3, "魔法", "魔法"),
        "another kb": term(other_kb, 2, "魔法", "魔法"),
        "forged id": dataclasses.replace(term(kb, 2, "魔法", "魔法"), proposal_id="f" * 32),
        "changed payload": dataclasses.replace(
            term(kb, 2, "魔法", "魔法"), payload={"source_term": "魔法", "target_term": "魔术", "category": ""}),
    }
    before = dump(imported)

    for label, proposal in bad.items():
        with pytest.raises(ValueError, match="is not a proposal of page 2 of KB"):
            apply(imported, kb, 2, run_id, [good, proposal], accept_all([good]))
        assert dump(imported) == before, label


def test_an_unknown_kb_is_an_error(imported, run_id):
    with pytest.raises(KBNotFound):
        apply(imported, 99, 1, run_id, [], {})


def test_no_proposals_write_nothing(imported, kb, run_id):
    before = dump(imported)

    assert apply(imported, kb, 1, run_id, []) == [] and dump(imported) == before
    assert not imported.in_transaction


def test_apply_does_not_commit_what_the_caller_has_pending(imported, kb, run_id):
    imported.execute("INSERT INTO books (book_title) VALUES ('pending_book')")  # an open transaction of the caller's

    with pytest.raises(sqlite3.OperationalError, match="within a transaction"):
        apply(imported, kb, 1, run_id, [term(kb, 1, "封印", "封印")])

    imported.rollback()
    assert imported.execute("SELECT COUNT(*) FROM books WHERE book_title = 'pending_book'").fetchone()[0] == 0
    assert store.kb_proposals(imported, kb) == []


# --- all or nothing ---


@pytest.mark.parametrize("function, fails_on_call", [
    ("insert_proposal", 5), ("set_proposal_status", 4), ("insert_entry", 1), ("insert_entry", 2),
    ("insert_profile", 1), ("add_entry_evidence", 1), ("add_profile_evidence", 1),
])
def test_when_a_write_fails_partway_nothing_of_the_call_is_stored(imported, kb, run_id, monkeypatch, function, fails_on_call):
    earlier = [term(kb, 1, "魔法", "魔法"), name(kb, 1, "ハル", "小春")]
    apply(imported, kb, 1, run_id, earlier, accept_all(earlier))
    page = [
        term(kb, 2, "魔法", "魔法"),  # confirms an entry
        term(kb, 2, "封印", "封印"),  # a new entry
        name(kb, 2, "ハル", "小春"),  # confirms a profile
        term(kb, 2, "王国", "王国"),  # another new entry
        name(kb, 2, "ユキ", "小雪"),  # a new profile
    ]
    before = dump(imported)
    real = getattr(store, function)
    calls = []

    def failing(*args, **kwargs):
        calls.append(function)
        if len(calls) == fails_on_call:
            raise sqlite3.OperationalError("disk I/O error")
        return real(*args, **kwargs)

    monkeypatch.setattr(store, function, failing)

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        apply(imported, kb, 2, run_id, page, accept_all(page))

    assert len(calls) == fails_on_call  # the writes before the failure did happen...
    assert dump(imported) == before  # ...and are gone
    assert not imported.in_transaction
    monkeypatch.setattr(store, function, real)  # once the fault is gone the page goes through whole, not as duplicates
    assert results(apply(imported, kb, 2, run_id, page, accept_all(page))) == [
        CONFIRMED, ACCEPTED, CONFIRMED, ACCEPTED, PENDING]


# --- a conflict across kinds is a result of its own (task 016), and writes what it always wrote ---


def test_a_conflict_across_kinds_is_counted_apart_from_a_conflict_of_one_kind_and_writes_the_same_things(
        imported, kb, run_id):
    insert_entry(imported, kb, "封印", "封印术")
    insert_profile(imported, kb, "ユキ")
    same_kind, term_as_name, name_as_term = (
        term(kb, 2, "封印", "封缄"), term(kb, 2, "ユキ", "雪"), name(kb, 2, "封印", "封印"))
    before = dump(imported)

    outcomes = apply(imported, kb, 2, run_id, [same_kind, term_as_name, name_as_term],
                     accept_all([same_kind, term_as_name]))

    assert results(outcomes) == [CONFLICT, CONFLICT_ACROSS_KINDS, CONFLICT_ACROSS_KINDS]
    assert len(RESULTS) == 9 and set(results(outcomes)) <= set(RESULTS)
    kinds_untouched(imported, kb, before)   # in neither case is an entry or a profile written or changed
    for proposal in (same_kind, term_as_name, name_as_term):
        row = proposal_row(imported, proposal)
        assert (row["status"], row["decided_at"] is not None) == ("pending_human", True)
    assert proposal_row(imported, same_kind)["reason"].startswith("conflicts with entry")
    assert proposal_row(imported, term_as_name)["reason"].startswith("conflicts across kinds")
    assert proposal_row(imported, name_as_term)["reason"].startswith("conflicts across kinds")


# --- rule 6: one character, one profile (task 026) ---


def profile_named(conn, kb_id, name_src):
    (row,) = [r for r in profiles(conn, kb_id) if r["name_src"] == name_src]
    return row


def aliases(row):
    return json.loads(row["aliases"])


def profiles_untouched(conn, before):
    """No entry or profile was written or changed (only the proposal queue may have grown)."""
    kinds_untouched(conn, None, before)


@pytest.mark.parametrize("status", ["approved", "pending_human"])
def test_a_given_name_inside_a_profiles_full_name_with_an_agreeing_rendering_joins_that_profile(
        imported, kb, run_id, status):
    insert_profile(imported, kb, "桜野すずめ", "樱野雀", aliases=["桜野"], status=status)
    given = name(kb, 6, "すずめ", "雀", aliases=["すずめさん"])

    outcomes = apply(imported, kb, 6, run_id, [given], accept_all([given]))   # a decision changes nothing (rule 2)

    assert results(outcomes) == ["merged"]
    (profile,) = profiles(imported, kb)
    assert (profile["name_src"], profile["name_tgt"], profile["status"]) == ("桜野すずめ", "樱野雀", status)   # kept
    assert aliases(profile) == ["桜野", "すずめ", "すずめさん"]   # after the ones it had
    assert pages(profile) == [1, 6]
    merged = history(profile)[-1]
    assert (merged["event"], merged["page"], merged["run_id"], merged["proposal_id"], merged["aliases"]) == (
        "merged", 6, run_id, given.proposal_id, ["すずめ", "すずめさん"])
    row = proposal_row(imported, given)
    assert (row["status"], row["decided_at"] is not None) == ("approved", True)
    assert f"profile {profile['id']}" in row["reason"] and "桜野すずめ" in row["reason"]
    assert "すずめ is part of 桜野すずめ" in row["reason"] and "new aliases すずめ, すずめさん" in row["reason"]


def test_a_full_name_around_a_profiles_given_name_with_an_agreeing_rendering_joins_that_profile(imported, kb, run_id):
    insert_profile(imported, kb, "ツバメ", "燕", aliases=["ツバメさん"], status="pending_human")
    full = name(kb, 7, "桜野ツバメ", "樱野燕", aliases=["お姉さん", "ツバメさん"])

    outcomes = apply(imported, kb, 7, run_id, [full])

    assert results(outcomes) == ["merged"]
    (profile,) = profiles(imported, kb)
    assert (profile["name_src"], profile["name_tgt"]) == ("ツバメ", "燕")   # the profile keeps its name and rendering
    assert aliases(profile) == ["ツバメさん", "桜野ツバメ", "お姉さん"]   # no duplicate
    assert "ツバメ is part of 桜野ツバメ" in proposal_row(imported, full)["reason"]


@pytest.mark.parametrize("profile_aliases, proposed, rendering, proposed_aliases, shared", [
    (["雪ちゃん"], "雪ちゃん", "小雪", [], "雪ちゃん"),          # the proposal's name is the profile's alias
    ([], "白瀬", "小雪", ["ユキ"], "ユキ"),                      # the proposal's alias is the profile's name
    (["お嬢"], "ユッキー", "小雪", ["お嬢"], "お嬢"),             # an alias of each is the same
], ids=["name-is-their-alias", "alias-is-their-name", "alias-is-their-alias"])
def test_a_name_or_alias_equal_to_a_profiles_name_or_alias_joins_that_profile(
        imported, kb, run_id, profile_aliases, proposed, rendering, proposed_aliases, shared):
    insert_profile(imported, kb, "ユキ", "小雪", aliases=profile_aliases)
    proposal = name(kb, 2, proposed, rendering, aliases=proposed_aliases)

    outcomes = apply(imported, kb, 2, run_id, [proposal])

    assert results(outcomes) == ["merged"] and len(profiles(imported, kb)) == 1
    profile = profiles(imported, kb)[0]
    assert proposed in aliases(profile) or proposed == profile["name_src"]
    assert "ユキ" not in aliases(profile)   # never the profile's own name
    assert len(aliases(profile)) == len(set(aliases(profile)))
    assert f"both are called {shared}" in proposal_row(imported, proposal)["reason"]


@pytest.mark.parametrize("profile_rendering, rendering", [
    ("葬仪屋先生", "葬仪屋"),   # the proposal's rendering inside the profile's
    ("燕", "樱野燕"),           # the profile's inside the proposal's
    ("小雪", "小雪"),           # the same
])
def test_a_rendering_agrees_when_one_contains_the_other(imported, kb, run_id, profile_rendering, rendering):
    insert_profile(imported, kb, "葬儀屋さん", profile_rendering)
    proposal = name(kb, 2, "葬儀屋", rendering)

    assert results(apply(imported, kb, 2, run_id, [proposal])) == ["merged"]


def test_a_rendering_that_agrees_with_neither_is_a_conflict_and_writes_nothing(imported, kb, run_id):
    insert_profile(imported, kb, "桜野すずめ", "樱野雀", aliases=["桜野"], status="pending_human")
    before = dump(imported)
    rival = name(kb, 6, "すずめ", "铃", aliases=["すずめさん"])

    outcomes = apply(imported, kb, 6, run_id, [rival], accept_all([rival]))   # even an "accept" cannot override it

    assert results(outcomes) == [CONFLICT]
    profiles_untouched(imported, before)   # no second profile, the first one as it was
    row = proposal_row(imported, rival)
    assert (row["status"], row["decided_at"] is not None) == ("pending_human", True)
    profile_id = profiles(imported, kb)[0]["id"]
    assert row["reason"] == (f"conflicts with profile {profile_id} (桜野すずめ → 樱野雀): すずめ is part of 桜野すずめ, "
                             f"but 铃 does not agree with 樱野雀")
    snapshot = load_snapshot(imported, kb)
    assert snapshot.character("すずめ") is None and [p.name_tgt for p in snapshot.profiles] == ["樱野雀"]   # 铃 never spreads


def test_a_name_that_matches_more_than_one_profile_is_a_conflict_and_writes_nothing(imported, kb, run_id):
    insert_profile(imported, kb, "葬儀屋さん", "葬仪屋先生", aliases=["葬儀屋"], status="pending_human")
    insert_profile(imported, kb, "烏丸枢", "乌丸枢", aliases=["烏丸", "社長"], status="pending_human")
    before = dump(imported)
    ambiguous = name(kb, 16, "枢", "枢", aliases=["葬儀屋さん", "葬儀屋"])   # agrees with 乌丸枢, and still waits

    outcomes = apply(imported, kb, 16, run_id, [ambiguous])

    assert results(outcomes) == [CONFLICT]
    profiles_untouched(imported, before)
    row = proposal_row(imported, ambiguous)
    first, second = (p["id"] for p in profiles(imported, kb))
    assert row["status"] == "pending_human" and row["reason"].startswith("conflicts with more than one profile")
    assert f"profile {first} (葬儀屋さん → 葬仪屋先生): both are called 葬儀屋さん, both are called 葬儀屋" in row["reason"]
    assert f"profile {second} (烏丸枢 → 乌丸枢): 枢 is part of 烏丸枢" in row["reason"]


def test_two_characters_who_share_an_alias_are_taken_for_one_and_a_person_decides(imported, kb, run_id):
    # A known false positive (task 026): sisters both called by their family name as an alias. The proposal is not
    # made a profile of its own; it waits for a person, and nothing is written, which is the safe side.
    insert_profile(imported, kb, "桜野すずめ", "樱野雀", aliases=["桜野"], status="pending_human")
    before = dump(imported)
    sister = name(kb, 7, "桜野ツバメ", "樱野燕", aliases=["桜野"])

    outcomes = apply(imported, kb, 7, run_id, [sister])

    assert results(outcomes) == [CONFLICT]
    profiles_untouched(imported, before)
    assert "both are called 桜野" in proposal_row(imported, sister)["reason"]


@pytest.mark.parametrize("proposed_aliases", [[], ["桜野ツバメさん", "お姉さん"]], ids=["no-alias", "aliases-around-it"])
def test_a_family_name_alias_alone_does_not_tie_two_names(imported, kb, run_id, proposed_aliases):
    # 桜野 is the profile's alias, and it is part of the new name and of its aliases: aliases are compared for equality
    # only, so this is a new character.
    insert_profile(imported, kb, "桜野すずめ", "樱野雀", aliases=["桜野"], status="pending_human")
    sister = name(kb, 7, "桜野ツバメ", "樱野燕", aliases=proposed_aliases)

    outcomes = apply(imported, kb, 7, run_id, [sister])

    assert results(outcomes) == [PENDING]
    assert [(p["name_src"], aliases(p)) for p in profiles(imported, kb)] == [
        ("桜野すずめ", ["桜野"]), ("桜野ツバメ", proposed_aliases)]


@pytest.mark.parametrize("status", ["proposed", "rejected"])
def test_a_proposed_or_rejected_profile_is_not_matched(imported, kb, run_id, status):
    insert_profile(imported, kb, "桜野すずめ", "樱野雀", status=status)

    outcomes = apply(imported, kb, 6, run_id, [name(kb, 6, "すずめ", "铃")])

    assert results(outcomes) == [PENDING]   # not readable, so not there for the rule


def test_the_same_name_is_still_a_confirmation_or_a_conflict_before_the_rule_looks_further(imported, kb, run_id):
    insert_profile(imported, kb, "桜野すずめ", "樱野雀", status="pending_human")
    insert_profile(imported, kb, "すずめ", "小雀", status="pending_human")   # a KB older than the rule
    same, other = name(kb, 2, "すずめ", "小雀"), name(kb, 3, "すずめ", "雀")

    assert results(apply(imported, kb, 2, run_id, [same])) == [CONFIRMED]   # though it also matches the full name
    assert results(apply(imported, kb, 3, run_id, [other])) == [CONFLICT]
    same_name_id = profiles(imported, kb)[1]["id"]
    assert proposal_row(imported, other)["reason"].startswith(f"conflicts with profile {same_name_id},")   # rule 3's


def test_a_name_that_is_a_terms_source_is_still_a_conflict_across_kinds(imported, kb, run_id):
    insert_entry(imported, kb, "すずめ", "麻雀")
    insert_profile(imported, kb, "桜野すずめ", "樱野雀")
    before = dump(imported)
    proposal = name(kb, 2, "すずめ", "雀")

    assert results(apply(imported, kb, 2, run_id, [proposal])) == [CONFLICT_ACROSS_KINDS]
    kinds_untouched(imported, kb, before)


def test_on_one_page_a_later_name_joins_the_profile_an_earlier_one_made(imported, kb, run_id):
    full, given = name(kb, 4, "桜野すずめ", "樱野雀", aliases=["桜野"]), name(kb, 4, "すずめ", "雀")

    outcomes = apply(imported, kb, 4, run_id, [full, given])

    assert results(outcomes) == [PENDING, "merged"]
    (profile,) = profiles(imported, kb)
    assert (aliases(profile), pages(profile)) == (["桜野", "すずめ"], [4])   # the page once in the evidence...
    assert [h["event"] for h in history(profile)] == ["created", "merged"]   # ...and the merge in the history


def test_a_merge_that_brings_no_new_name_still_joins_the_evidence(imported, kb, run_id):
    insert_profile(imported, kb, "葬儀屋さん", "葬仪屋先生", aliases=["葬儀屋"], status="pending_human")
    title = name(kb, 11, "葬儀屋", "葬仪屋", aliases=["葬儀屋さん"])

    assert results(apply(imported, kb, 11, run_id, [title])) == ["merged"]

    (profile,) = profiles(imported, kb)
    assert (aliases(profile), pages(profile)) == (["葬儀屋"], [1, 11])
    assert history(profile)[-1]["aliases"] == [] and proposal_row(imported, title)["reason"].endswith("no new alias")


def test_a_merged_alias_is_found_by_the_snapshot_and_by_the_cross_kind_rule(imported, kb, run_id):
    insert_profile(imported, kb, "桜野ツバメ", "樱野燕", aliases=["お姉さん"], status="pending_human")
    apply(imported, kb, 10, run_id, [name(kb, 10, "ツバメ", "燕", aliases=["ツバメさん"])])

    snapshot = load_snapshot(imported, kb)

    for called in ("ツバメ", "ツバメさん", "お姉さん", "桜野ツバメ"):
        found = snapshot.character(called)
        assert (found.name_src, found.name_tgt, found.status) == ("桜野ツバメ", "樱野燕", "pending_human"), called
    assert len(snapshot.profiles) == 1
    word = term(kb, 12, "ツバメさん", "燕小姐")   # a later term with the new alias as its source meets the profile
    assert results(apply(imported, kb, 12, run_id, [word], accept_all([word]))) == [CONFLICT_ACROSS_KINDS]


def test_add_profile_aliases_keeps_the_order_and_never_repeats_a_name(imported, kb, run_id):
    insert_profile(imported, kb, "ユキ", "小雪", aliases=["ゆき"])
    profile_id = profiles(imported, kb)[0]["id"]

    with store.transaction(imported):
        added = store.add_profile_aliases(imported, profile_id, ["雪ちゃん", "ユキ", "ゆき", "雪ちゃん", "白瀬"], 1,
                                          {"event": "merged"})

    assert added == ["雪ちゃん", "白瀬"]
    profile = profiles(imported, kb)[0]
    assert (aliases(profile), pages(profile)) == (["ゆき", "雪ちゃん", "白瀬"], [1])
    assert [h["event"] for h in history(profile)] == ["created", "merged"]


def test_when_the_alias_write_fails_nothing_of_the_call_is_stored(imported, kb, run_id, monkeypatch):
    insert_profile(imported, kb, "桜野すずめ", "樱野雀", status="pending_human")
    page = [term(kb, 6, "封印", "封印"), name(kb, 6, "すずめ", "雀")]
    before = dump(imported)

    def failing(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "add_profile_aliases", failing)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        apply(imported, kb, 6, run_id, page, accept_all(page))

    assert dump(imported) == before and not imported.in_transaction


# --- the replay of the dev runs' character proposals (#44), names, renderings and aliases only ---

BOUREISOUGI = [   # (page, name_src, name_tgt, aliases, result), in page order
    (1, "葬儀屋さん", "葬仪屋先生", ["葬儀屋"], PENDING),
    (4, "烏丸枢", "乌丸枢", ["烏丸", "社長"], PENDING),
    (4, "桜野すずめ", "樱野雀", ["桜野"], PENDING),
    (6, "すずめ", "铃", ["すずめさん"], CONFLICT),                       # matches 桜野すずめ; 铃 does not agree with 樱野雀
    (7, "桜野ツバメ", "樱野燕", ["桜野ツバメさん", "お姉さん", "お姉ちゃん"], PENDING),
    (10, "ツバメ", "燕", ["ツバメさん", "桜野燕"], "merged"),             # into 桜野ツバメ
    (11, "葬儀屋", "葬仪屋", ["葬儀屋さん"], "merged"),                   # into 葬儀屋さん, no new alias
    (16, "枢", "枢", ["葬儀屋さん", "葬儀屋"], CONFLICT),                 # 葬儀屋さん by an alias, 烏丸枢 by containment
]
TOJIME_NO_SIORA = [
    (1, "シオラ", "希奥拉", ["綴じ眼のシオラ"], PENDING),
    (2, "レーネ", "蕾妮", [], PENDING),
    (7, "ビル・トルティヤー", "比尔·托尔蒂亚", ["ビル"], PENDING),
    (21, "レーネ", "蕾妮", [], CONFIRMED),
]


def replay(conn, kb_id, run_id, rows):
    """Apply `rows` page by page, as the eval applied them, and return each proposal's result."""
    got = []
    for page_index in dict.fromkeys(r[0] for r in rows):
        proposals = [name(kb_id, n, src, tgt, aliases=al) for n, src, tgt, al, _ in rows if n == page_index]
        got += results(apply(conn, kb_id, page_index, run_id, proposals))
    return got


def final_profiles(conn, kb_id):
    return [(p["name_src"], p["name_tgt"], aliases(p), p["status"], pages(p)) for p in profiles(conn, kb_id)]


def test_the_replay_of_boureisougi_ends_with_one_profile_for_each_of_four_characters(imported, kb, run_id):
    assert replay(imported, kb, run_id, BOUREISOUGI) == [r[-1] for r in BOUREISOUGI]

    assert final_profiles(imported, kb) == [
        ("葬儀屋さん", "葬仪屋先生", ["葬儀屋"], "pending_human", [1, 11]),
        ("烏丸枢", "乌丸枢", ["烏丸", "社長"], "pending_human", [4]),
        ("桜野すずめ", "樱野雀", ["桜野"], "pending_human", [4]),
        ("桜野ツバメ", "樱野燕", ["桜野ツバメさん", "お姉さん", "お姉ちゃん", "ツバメ", "ツバメさん", "桜野燕"],
         "pending_human", [7, 10]),
    ]   # 4 profiles, not 8, and the 铃 profile is never created
    assert [r["status"] for r in store.kb_proposals(imported, kb)] == [
        "pending_human", "pending_human", "pending_human", "pending_human", "pending_human", "approved", "approved",
        "pending_human"]
    assert load_snapshot(imported, kb).character("ツバメさん").name_tgt == "樱野燕"


def test_the_replay_of_tojime_no_siora_is_as_it_was(imported, kb, run_id):
    assert replay(imported, kb, run_id, TOJIME_NO_SIORA) == [r[-1] for r in TOJIME_NO_SIORA]

    assert final_profiles(imported, kb) == [
        ("シオラ", "希奥拉", ["綴じ眼のシオラ"], "pending_human", [1]),
        ("レーネ", "蕾妮", [], "pending_human", [2, 21]),
        ("ビル・トルティヤー", "比尔·托尔蒂亚", ["ビル"], "pending_human", [7]),
    ]
