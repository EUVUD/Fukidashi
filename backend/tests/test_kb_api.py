"""The Editor KB over HTTP (task 040; OUTLINE §8, "A person's decisions on KB items"): `GET /api/books/{book}/kb` and the
three decision routes, with their 404, 409 and 422. The decisions themselves are tested in `test_kb_decisions.py`."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kb_support import accept, dump, name, new_kb, new_run, reject, term

from fukidashi import store
from fukidashi.db import connect
from fukidashi.kb.curator import HUMAN_OUTCOMES, apply

GET = "/api/books/book_a/kb"

ROUTES = ("proposals", "entries", "profiles")


def url(route: str, row_id, book: str = "book_a") -> str:
    return f"/api/books/{book}/kb/{route}/{row_id}/decision"


def decide(client, route: str, row_id, decision: str = "approve", book: str = "book_a"):
    return client.post(url(route, row_id, book), json={"decision": decision})


def kb_rows(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM kbs").fetchone()[0]


@pytest.fixture
def editor(api, conn):
    """The Editor KB of book_a, and a Run to attribute its proposals to."""
    return SimpleNamespace(kb=store.editor_kb(conn, "book_a"), run_id=new_run(conn))


@pytest.fixture
def scene(conn, editor):
    """Something in every queue: an entry the agent made, a term nobody decided, a new name, a conflict, a rejected term."""
    kb, run_id = editor.kb, editor.run_id
    accepted, rejected = term(kb, 1, "封印", "封印"), term(kb, 1, "王国", "王国")
    undecided, new_name = term(kb, 1, "魔法", "魔法", category="magic"), name(kb, 1, "ユキ", "小雪", aliases=["ゆき"])
    apply(conn, kb, 1, run_id, [accepted, rejected, undecided, new_name],
          {accepted.proposal_id: accept("keep"), rejected.proposal_id: reject("generic")})
    rival = term(kb, 2, "封印", "封缄")
    apply(conn, kb, 2, run_id, [rival], {rival.proposal_id: accept()})
    return SimpleNamespace(
        kb=kb, run_id=run_id, accepted=accepted, rejected=rejected, undecided=undecided, new_name=new_name, rival=rival,
        entry=store.find_entry(conn, kb, "封印")["id"], profile=store.find_profile(conn, kb, "ユキ")["id"])


# --- GET ---


def test_a_book_with_no_editor_kb_answers_an_empty_kb_and_the_get_writes_nothing(api, conn):
    client, _ = api
    eval_kb = new_kb(conn, "run-1")  # an eval run's KB of the same book is not the Editor KB
    apply(conn, eval_kb, 1, new_run(conn), [term(eval_kb, 1, "封印", "封印")])
    before, kbs = dump(conn), kb_rows(conn)

    resp = client.get(GET)

    assert resp.status_code == 200
    assert resp.json() == {
        "kb_id": None, "entries": [], "profiles": [], "proposal_queue": [], "human_review_queue": [],
        "proposal_counts": {"proposed": 0, "approved": 0, "rejected": 0, "pending_human": 0}}
    assert kb_rows(conn) == kbs and dump(conn) == before  # not a `kbs` row, not a write of any kind
    assert not conn.in_transaction


def test_an_unknown_book_is_a_404_on_every_route(api):
    client, _ = api

    assert client.get("/api/books/no_such_book/kb").status_code == 404
    for route in ROUTES:
        resp = decide(client, route, 1, book="no_such_book")
        assert (resp.status_code, resp.json()["detail"]) == (404, "no book no_such_book")


def test_the_get_answers_the_kb_with_both_queues_and_the_counts(api, conn, scene):
    client, _ = api

    body = client.get(GET).json()

    assert body["kb_id"] == scene.kb
    assert [(e["source_term"], e["target_term"], e["status"], e["provenance"], e["evidence_pages"]) for e in body["entries"]] == [
        ("封印", "封印", "approved", "agent", [1])]
    (profile,) = body["profiles"]
    assert (profile["id"], profile["name_src"], profile["name_tgt"], profile["aliases"], profile["notes"]) == (
        scene.profile, "ユキ", "小雪", ["ゆき"], "")
    assert (profile["status"], profile["provenance"], profile["evidence_pages"]) == ("pending_human", "agent", [1])
    assert [h["event"] for h in profile["history"]] == ["created"] and profile["history"][0]["at"]
    (queued,) = body["proposal_queue"]  # the Proposal Queue: the term nobody decided
    assert queued == {
        "proposal_id": scene.undecided.proposal_id, "kind": "term", "proposer": "translator", "page_index": 1,
        "run_id": scene.run_id, "reason": None, "payload": {"source_term": "魔法", "target_term": "魔法", "category": "magic"}}
    waiting = body["human_review_queue"]  # the Human Review Queue: the new name, then the conflict, in the order queued
    assert [(w["proposal_id"], w["kind"], w["waits_for"], w["profile_id"]) for w in waiting] == [
        (scene.new_name.proposal_id, "character", "new_name", scene.profile),
        (scene.rival.proposal_id, "term", "conflict", None)]
    assert waiting[0]["payload"] == {"name_src": "ユキ", "name_tgt": "小雪", "aliases": ["ゆき"], "notes": ""}
    assert waiting[0]["reason"] == "a new name waits for a human" and waiting[1]["reason"].startswith("conflicts with entry")
    assert (waiting[1]["page_index"], waiting[1]["run_id"], waiting[1]["proposer"]) == (2, scene.run_id, "translator")
    assert body["proposal_counts"] == {"proposed": 1, "approved": 1, "rejected": 1, "pending_human": 2}


def test_the_get_lists_a_rejected_entry_so_that_a_person_can_approve_it_again(api, conn, scene):
    client, _ = api
    decide(client, "entries", scene.entry, "reject")

    body = client.get(GET).json()

    (entry,) = body["entries"]
    assert (entry["status"], entry["provenance"]) == ("rejected", "human")
    assert [h["event"] for h in entry["history"]] == ["created", "rejected"]


def test_the_get_shows_the_editor_kb_and_not_an_eval_kb_of_the_same_book(api, conn, scene):
    client, _ = api
    eval_kb = new_kb(conn, "run-1")
    run_id = new_run(conn)
    stray = [term(eval_kb, 1, "城", "城"), name(eval_kb, 1, "ハル", "小春")]
    apply(conn, eval_kb, 1, run_id, stray, {stray[0].proposal_id: accept()})

    body = client.get(GET).json()

    assert body["kb_id"] == scene.kb
    assert [e["source_term"] for e in body["entries"]] == ["封印"] and [p["name_src"] for p in body["profiles"]] == ["ユキ"]
    assert body["proposal_counts"] == {"proposed": 1, "approved": 1, "rejected": 1, "pending_human": 2}


def test_a_profiles_provenance_is_derived_from_its_history(api, conn, scene):
    client, _ = api
    assert client.get(GET).json()["profiles"][0]["provenance"] == "agent"

    body = decide(client, "profiles", scene.profile, "approve").json()

    (profile,) = body["profiles"]
    assert (profile["status"], profile["provenance"]) == ("approved", "human")
    assert profile["history"][-1]["by"] == "human"


def test_the_kb_response_has_the_fields_the_panel_mirrors(api, conn, scene):
    client, _ = api

    body = client.get(GET).json()

    assert set(body) == {"kb_id", "entries", "profiles", "proposal_queue", "human_review_queue", "proposal_counts"}
    assert set(body["entries"][0]) == {"id", "source_term", "target_term", "category", "status", "provenance",
                                       "evidence_pages", "history"}
    assert set(body["profiles"][0]) == {"id", "name_src", "name_tgt", "aliases", "notes", "status", "provenance",
                                        "evidence_pages", "history"}
    proposal = {"proposal_id", "kind", "payload", "proposer", "page_index", "run_id", "reason"}
    assert set(body["proposal_queue"][0]) == proposal
    assert set(body["human_review_queue"][0]) == proposal | {"waits_for", "profile_id"}
    assert set(body["proposal_counts"]) == {"proposed", "approved", "rejected", "pending_human"}


def test_the_view_is_read_in_one_snapshot_even_when_another_connection_writes_meanwhile(api, conn, editor, db_path):
    kb, run_id = editor.kb, editor.run_id
    reader, writer = connect(db_path), connect(db_path)
    try:
        with store.reading(reader):
            assert reader.in_transaction
            before = len(store.kb_proposals(reader, kb))
            apply(writer, kb, 1, run_id, [term(kb, 1, "封印", "封印")])  # committed by another connection, mid-read
            assert len(store.kb_proposals(reader, kb)) == before  # the view does not see it
        assert not reader.in_transaction  # it wrote nothing and holds nothing
        assert len(store.kb_proposals(reader, kb)) == before + 1  # the next view does
        with pytest.raises(RuntimeError, match="stop"), store.reading(reader):
            raise RuntimeError("stop")
        assert not reader.in_transaction  # whatever ends the block
    finally:
        reader.close()
        writer.close()


# --- POST: proposals ---


def test_approving_an_undecided_term_answers_the_outcome_and_the_kb_after_it(api, conn, scene):
    client, _ = api

    resp = decide(client, "proposals", scene.undecided.proposal_id, "approve")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"outcome", "kb"} and body["outcome"] == "entry_created"
    assert body["kb"] == client.get(GET).json()  # the KB as the GET gives it
    made = next(e for e in body["kb"]["entries"] if e["source_term"] == "魔法")
    assert (made["status"], made["provenance"], made["evidence_pages"]) == ("approved", "human", [1])
    assert made["history"][0]["by"] == "human" and made["history"][0]["proposal_id"] == scene.undecided.proposal_id
    assert body["kb"]["proposal_queue"] == []
    assert body["kb"]["proposal_counts"] == {"proposed": 0, "approved": 2, "rejected": 1, "pending_human": 2}


def test_rejecting_an_undecided_term_answers_the_outcome(api, conn, scene):
    client, _ = api

    body = decide(client, "proposals", scene.undecided.proposal_id, "reject").json()

    assert body["outcome"] == "proposal_rejected"
    assert body["kb"]["proposal_queue"] == [] and len(body["kb"]["entries"]) == 1
    assert body["kb"]["proposal_counts"] == {"proposed": 0, "approved": 1, "rejected": 2, "pending_human": 2}


def test_an_undecided_term_approved_after_the_kb_changed_answers_what_the_rules_made_of_it(api, conn, editor):
    client, _ = api
    kb, run_id = editor.kb, editor.run_id
    waiting = term(kb, 2, "封印", "封缄")
    apply(conn, kb, 2, run_id, [waiting])
    first = term(kb, 1, "封印", "封印")
    apply(conn, kb, 1, run_id, [first], {first.proposal_id: accept()})

    body = decide(client, "proposals", waiting.proposal_id, "approve").json()

    assert body["outcome"] == "held_as_conflict"
    assert [q["proposal_id"] for q in body["kb"]["proposal_queue"]] == []
    (held,) = body["kb"]["human_review_queue"]
    assert (held["proposal_id"], held["waits_for"], held["profile_id"]) == (waiting.proposal_id, "conflict", None)
    assert held["reason"].startswith("approved by a person; conflicts with entry")


def test_approving_a_new_name_approves_its_profile_and_leaves_the_queue(api, conn, scene):
    client, _ = api

    body = decide(client, "proposals", scene.new_name.proposal_id, "approve").json()

    assert body["outcome"] == "profile_approved"
    assert [w["proposal_id"] for w in body["kb"]["human_review_queue"]] == [scene.rival.proposal_id]
    (profile,) = body["kb"]["profiles"]
    assert (profile["status"], profile["provenance"]) == ("approved", "human")
    assert profile["history"][-1] == {**profile["history"][-1], "event": "approved", "by": "human",
                                      "from": "pending_human", "proposal_id": scene.new_name.proposal_id}


def test_rejecting_a_new_name_rejects_its_profile(api, conn, scene):
    client, _ = api

    body = decide(client, "proposals", scene.new_name.proposal_id, "reject").json()

    assert body["outcome"] == "profile_rejected"
    assert body["kb"]["profiles"][0]["status"] == "rejected"
    assert [w["proposal_id"] for w in body["kb"]["human_review_queue"]] == [scene.rival.proposal_id]


def test_rejecting_a_conflict_closes_it_and_the_kb_keeps_what_it_holds(api, conn, scene):
    client, _ = api
    entry_before = [dict(e) for e in store.kb_entries(conn, scene.kb, store.ALL_STATUSES)]

    body = decide(client, "proposals", scene.rival.proposal_id, "reject").json()

    assert body["outcome"] == "proposal_rejected"
    assert [w["proposal_id"] for w in body["kb"]["human_review_queue"]] == [scene.new_name.proposal_id]
    assert [dict(e) for e in store.kb_entries(conn, scene.kb, store.ALL_STATUSES)] == entry_before
    assert [e["target_term"] for e in body["kb"]["entries"]] == ["封印"]


def test_approving_a_conflict_is_a_409_that_says_why_and_writes_nothing(api, conn, scene):
    client, _ = api
    before = dump(conn)

    resp = decide(client, "proposals", scene.rival.proposal_id, "approve")

    assert resp.status_code == 409
    assert "a conflict cannot be approved" in resp.json()["detail"] and "rename" in resp.json()["detail"]
    assert dump(conn) == before


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_a_proposal_in_no_queue_is_a_409_and_writes_nothing(api, conn, scene, decision):
    client, _ = api
    before = dump(conn)

    for decided in (scene.accepted, scene.rejected):
        resp = decide(client, "proposals", decided.proposal_id, decision)
        assert resp.status_code == 409 and "already" in resp.json()["detail"]
    assert dump(conn) == before


def test_a_decided_proposal_is_a_409_when_decided_again(api, conn, scene):
    client, _ = api
    assert decide(client, "proposals", scene.undecided.proposal_id, "reject").status_code == 200

    resp = decide(client, "proposals", scene.undecided.proposal_id, "reject")

    assert (resp.status_code, resp.json()["detail"].startswith(f"proposal {scene.undecided.proposal_id} is already rejected")) == (409, True)


# --- POST: entries and profiles ---


def test_an_entry_rejected_and_then_approved_again_answers_the_kb_each_time(api, conn, scene):
    client, _ = api

    rejected = decide(client, "entries", scene.entry, "reject")
    approved = decide(client, "entries", scene.entry, "approve")

    assert rejected.status_code == approved.status_code == 200
    assert "outcome" not in rejected.json() and "kb" not in rejected.json()  # the KB itself, as the GET gives it
    assert [(e["status"], e["provenance"]) for e in rejected.json()["entries"]] == [("rejected", "human")]
    assert [(e["status"], e["provenance"]) for e in approved.json()["entries"]] == [("approved", "human")]
    assert approved.json() == client.get(GET).json()
    history = approved.json()["entries"][0]["history"]
    assert [(h["event"], h.get("by"), h.get("from")) for h in history] == [
        ("created", None, None), ("rejected", "human", "approved"), ("approved", "human", "rejected")]


def test_a_decision_that_changes_nothing_is_a_409(api, conn, scene):
    client, _ = api
    before = dump(conn)

    resp = decide(client, "entries", scene.entry, "approve")  # the entry is approved already

    assert (resp.status_code, resp.json()["detail"]) == (409, f"entry {scene.entry} is already approved")
    assert dump(conn) == before
    assert decide(client, "entries", scene.entry, "reject").status_code == 200
    assert decide(client, "profiles", scene.profile, "reject").status_code == 200
    again = decide(client, "entries", scene.entry, "reject")
    assert (again.status_code, again.json()["detail"]) == (409, f"entry {scene.entry} is already rejected")
    again = decide(client, "profiles", scene.profile, "reject")
    assert (again.status_code, again.json()["detail"]) == (409, f"profile {scene.profile} is already rejected")


def test_approving_a_second_readable_row_for_one_source_is_a_409(api, conn, scene):
    client, _ = api
    decide(client, "entries", scene.entry, "reject")  # the first entry is not readable now...
    second = term(scene.kb, 3, "封印", "封缄")
    apply(conn, scene.kb, 3, scene.run_id, [second], {second.proposal_id: accept()})  # ...so a new one is made for the source
    before = dump(conn)

    resp = decide(client, "entries", scene.entry, "approve")

    assert resp.status_code == 409
    assert "封印" in resp.json()["detail"] and "two readable entries" in resp.json()["detail"]
    assert dump(conn) == before


def test_approving_an_entry_whose_source_is_a_profiles_name_is_a_409(api, conn, scene):
    client, _ = api
    entry_id = store.insert_entry(conn, kb_id=scene.kb, source_term="ユキ", target_term="雪", category="", status="rejected",
                                  provenance="human", page_index=1, event={"event": "created"})
    conn.commit()
    before = dump(conn)

    resp = decide(client, "entries", entry_id, "approve")

    assert resp.status_code == 409 and "a character" in resp.json()["detail"]
    assert dump(conn) == before


def test_approving_a_profile_whose_name_is_an_entrys_source_is_a_409(api, conn, scene):
    client, _ = api
    profile_id = store.insert_profile(conn, kb_id=scene.kb, name_src="封印", name_tgt="印", aliases=[], notes="",
                                      status="rejected", page_index=1, event={"event": "created"})
    conn.commit()
    before = dump(conn)

    resp = decide(client, "profiles", profile_id, "approve")

    assert resp.status_code == 409 and "a term" in resp.json()["detail"]
    assert dump(conn) == before


def test_a_profile_decided_as_an_item_settles_its_proposal_in_the_answer(api, conn, scene):
    client, _ = api

    body = decide(client, "profiles", scene.profile, "reject").json()

    assert body["profiles"][0]["status"] == "rejected"
    assert [w["proposal_id"] for w in body["human_review_queue"]] == [scene.rival.proposal_id]  # the two queues agree
    assert body["proposal_counts"] == {"proposed": 1, "approved": 1, "rejected": 2, "pending_human": 1}


# --- the 404s: nothing is written ---


def test_a_decision_on_a_book_with_no_editor_kb_is_a_404_and_makes_no_kb(api, conn):
    client, _ = api
    kbs, before = kb_rows(conn), dump(conn)

    for route in ROUTES:
        resp = decide(client, route, 1)
        assert (resp.status_code, resp.json()["detail"]) == (404, "book book_a has no Editor KB yet, so there is nothing to decide")

    assert kb_rows(conn) == kbs and dump(conn) == before


@pytest.mark.parametrize("route, row_id", [("proposals", "no-such-proposal"), ("entries", 424242), ("profiles", 424242),
                                           ("entries", 0), ("profiles", -1), ("entries", 10**30), ("profiles", -(10**30))])
def test_an_id_the_editor_kb_does_not_have_is_a_404(api, conn, scene, route, row_id):
    client, _ = api
    before = dump(conn)

    resp = decide(client, route, row_id, "reject")

    assert resp.status_code == 404
    assert dump(conn) == before


def test_a_decision_aimed_at_an_eval_kbs_row_through_a_books_route_is_a_404_and_changes_nothing(api, conn, scene):
    client, _ = api
    eval_kb = new_kb(conn, "run-1")  # book_a's, like the Editor KB
    run_id = new_run(conn)
    proposals = [term(eval_kb, 1, "城", "城"), term(eval_kb, 1, "王", "王"), name(eval_kb, 1, "ハル", "小春"),
                 term(eval_kb, 2, "城", "砦")]
    apply(conn, eval_kb, 1, run_id, proposals[:3], {proposals[1].proposal_id: accept()})
    apply(conn, eval_kb, 2, run_id, proposals[3:], {proposals[3].proposal_id: accept()})
    undecided, entry, profile = proposals[0], store.find_entry(conn, eval_kb, "王"), store.find_profile(conn, eval_kb, "ハル")
    conflict = proposals[3]
    new_name_id = next(r["proposal_id"] for r in store.kb_proposals(conn, eval_kb, "pending_human") if r["kind"] == "character")
    before = dump(conn)

    for decision in ("approve", "reject"):
        for route, row_id in [("proposals", undecided.proposal_id), ("proposals", conflict.proposal_id),
                              ("proposals", new_name_id), ("entries", entry["id"]), ("profiles", profile["id"])]:
            resp = decide(client, route, row_id, decision)
            assert resp.status_code == 404, (route, decision)

    assert dump(conn) == before  # the eval KB's rows, and the Editor KB's, are as they were


def test_an_id_of_another_books_editor_kb_is_a_404_through_this_books_route(api, conn, scene):
    client, _ = api
    conn.execute("INSERT INTO books (book_title) VALUES ('book_b')")
    conn.commit()
    other = store.editor_kb(conn, "book_b")
    waiting = name(other, 1, "ハル", "小春")
    apply(conn, other, 1, scene.run_id, [waiting])
    other_profile = store.find_profile(conn, other, "ハル")["id"]
    before = dump(conn)

    assert decide(client, "proposals", waiting.proposal_id, "approve").status_code == 404
    assert decide(client, "profiles", other_profile, "approve").status_code == 404
    assert dump(conn) == before
    assert decide(client, "profiles", other_profile, "approve", book="book_b").status_code == 200  # its own route finds it


# --- the 422s: nothing is written ---


BAD_BODIES = [
    pytest.param({}, id="empty-object"),
    pytest.param({"decision": "maybe"}, id="another-word"),
    pytest.param({"decision": "Approve"}, id="wrong-case"),
    pytest.param({"decision": None}, id="null"),
    pytest.param({"decision": 1}, id="number"),
    pytest.param({"decision": ["approve"]}, id="list"),
    pytest.param({"decision": "approve", "reason": "x"}, id="extra-key"),
    pytest.param({"approve": True}, id="another-key"),
    pytest.param([], id="array"),
    pytest.param("approve", id="string"),
    pytest.param(None, id="null-body"),
]


@pytest.mark.parametrize("body", BAD_BODIES)
@pytest.mark.parametrize("route", ROUTES)
def test_a_body_that_is_not_exactly_a_decision_is_a_422_and_writes_nothing(api, conn, scene, route, body):
    client, _ = api
    row_id = {"proposals": scene.undecided.proposal_id, "entries": scene.entry, "profiles": scene.profile}[route]
    before = dump(conn)

    resp = client.post(url(route, row_id), json=body)

    assert resp.status_code == 422
    assert dump(conn) == before


@pytest.mark.parametrize("route", ROUTES)
def test_a_missing_or_unreadable_body_is_a_422(api, conn, scene, route):
    client, _ = api
    row_id = {"proposals": scene.undecided.proposal_id, "entries": scene.entry, "profiles": scene.profile}[route]
    before = dump(conn)

    assert client.post(url(route, row_id)).status_code == 422
    assert client.post(url(route, row_id), content=b"{not json", headers={"content-type": "application/json"}).status_code == 422
    assert dump(conn) == before


@pytest.mark.parametrize("route", ["entries", "profiles"])
def test_an_id_that_is_not_an_integer_is_a_422(api, conn, scene, route):
    client, _ = api

    for bad in ("abc", "1.5"):
        assert decide(client, route, bad).status_code == 422


# --- what a decision answers is what the outcome strings say ---


def test_every_outcome_a_route_answers_is_one_the_curator_names(api, conn, scene):
    client, _ = api
    seen = {
        decide(client, "proposals", scene.undecided.proposal_id, "approve").json()["outcome"],
        decide(client, "proposals", scene.new_name.proposal_id, "reject").json()["outcome"],
        decide(client, "proposals", scene.rival.proposal_id, "reject").json()["outcome"],
    }

    assert seen == {"entry_created", "profile_rejected", "proposal_rejected"} and seen <= set(HUMAN_OUTCOMES)
