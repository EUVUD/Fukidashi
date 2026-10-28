"""The `full` config (task 016) through `runs.translate_page` and the eval harness, on a scripted provider: no network.

The real LLMClient, the real openai SDK and LangChain's create_agent run; only the provider's replies are scripted. A
page is the Orchestrator with the KB, then the finalize: the Curator Agent's one call and `curator.apply`. Requests are
told apart by their system prompt: the Curator's, the Reviewer's, the Translator's, `baseline`'s.
"""

from __future__ import annotations

import json

import pytest
from conftest import mock_llm_client
from kb_support import dump, new_kb, term
from kb_support import name as name_proposal
from test_agent_loop import Provider, answer, ask, call, upstream_500
from test_curator_agent import decision, decisions_json
from test_evaluation import run_cli
from test_kb_tools import HEADER, NOTHING, lookup_term, propose
from test_reviewer import APPROVE, issue, verdict_json
from test_tracing import one_trace, traced  # noqa: F401 (fixture)
from test_translator import (  # noqa: F401 (fixtures)
    BOOK, CALL_COST, attempts_of, evaluate, flag, imported, previous_context, root, tool_result, translate, zh,
    zh_json, zh_of,
)

from fukidashi import config as app_config
from fukidashi import runs, store
from fukidashi.agents import curator, reviewer
from fukidashi.evaluation import harness
from fukidashi.kb import curator as curator_rules
from fukidashi.kb.entities import KEY_FIELDS
from fukidashi.kb.loader import load_snapshot
from fukidashi.llm.metering import Meter
from fukidashi.pipeline import baseline

CONFIG = "full"
REVISION_MARK = "sent it back for revision"  # in the text part that only a revision's user message has
ALL_STATUSES = ("proposed", "approved", "rejected", "pending_human")

SEEN_TERM = ("宇宙船", "飞船")   # an entry the KB has already
NAME = ("ユキ", "小雪")


def run_page(conn, root, provider, kb_id, page_index=3, **options):
    """`full` on one page; the Meter is returned so that a test can compare it with the requests sent."""
    meter = Meter()
    outcome = translate(conn, root, mock_llm_client(provider, meter=meter), page_index, config=CONFIG, kb_id=kb_id,
                        **options)
    return outcome, meter


def who(body):
    """Which agent a request belongs to. A Translator's follow-up and its Repair belong to its stage too."""
    system, user = body["messages"][0]["content"], body["messages"][1]["content"]
    if system == curator.SYSTEM_PROMPT:
        return "curate"
    if system in (reviewer.SYSTEM_PROMPT, reviewer.KB_SYSTEM_PROMPT):
        return "review"
    if system == baseline.SYSTEM_PROMPT:
        return "baseline"
    return "revise" if any(REVISION_MARK in part.get("text", "") for part in user) else "translate"


def sent(bodies):
    return json.dumps(bodies, ensure_ascii=False)


def seed_entry(conn, kb_id, source, target, *, status="approved", page_index=1):
    store.insert_entry(conn, kb_id=kb_id, source_term=source, target_term=target, category="", status=status,
                       provenance="human", page_index=page_index, event={"event": "created"})
    conn.commit()


def seed_profile(conn, kb_id, name, rendering, *, aliases=(), status="approved", page_index=1):
    store.insert_profile(conn, kb_id=kb_id, name_src=name, name_tgt=rendering, aliases=list(aliases), notes="",
                         status=status, page_index=page_index, event={"event": "created"})
    conn.commit()


def entries(conn, kb_id):
    return [(r["source_term"], r["target_term"], r["status"], r["provenance"], json.loads(r["evidence_pages"]))
            for r in store.kb_entries(conn, kb_id, ALL_STATUSES)]


def profiles(conn, kb_id):
    return [(r["name_src"], r["name_tgt"], r["status"], json.loads(r["evidence_pages"]))
            for r in store.kb_profiles(conn, kb_id, ALL_STATUSES)]


def queue(conn, kb_id):
    """The Proposal Queue as (kind, source, status, reason), in the order the proposals were queued."""
    return [(r["kind"], json.loads(r["payload"])[KEY_FIELDS[r["kind"]][0]], r["status"], r["reason"])
            for r in store.kb_proposals(conn, kb_id)]


def count_translations(conn, run_id):
    return conn.execute("SELECT COUNT(*) FROM translations WHERE run_id = ?", (run_id,)).fetchone()[0]


def proposals_reply(call_prefix="p"):
    """One reply from the Translator that proposes a new term, one the KB has already, and a name."""
    return ask(propose(f"{call_prefix}1", "term", source_term="封印", target_term="封印术", category="technique"),
               propose(f"{call_prefix}2", "term", source_term=SEEN_TERM[0], target_term=SEEN_TERM[1]),
               propose(f"{call_prefix}3", "character", name_src=NAME[0], name_tgt=NAME[1], aliases=["ゆき"]))


@pytest.fixture
def kb(imported):
    """An empty KB of the book, and the KB id; `SEEN_TERM` is in it already, as a person's entry from page 1."""
    kb_id = new_kb(imported, book=BOOK)
    seed_entry(imported, kb_id, *SEEN_TERM)
    return kb_id


def terms_of(kb_id, page_index=3):
    return term(kb_id, page_index, "封印", "封印术", category="technique"), term(kb_id, page_index, *SEEN_TERM)


# --- AC1: a full page ----------------------------------------------------------------------------------------------

def test_a_full_page_is_one_succeeded_run_that_finalizes_the_curators_call_and_the_kb_write(imported, root, kb):
    new, seen = terms_of(kb)
    provider = Provider(proposals_reply(), answer(zh_json(3)), answer(APPROVE),
                        answer(decisions_json(decision(new, True, "招式名，译法一致"), decision(seen, True, "同一译法"))))

    outcome, meter = run_page(imported, root, provider, kb)

    assert outcome.result.ok
    run = store.get_run(imported, outcome.run_id)
    assert (run["config"], run["status"], run["error"]) == ("full", "succeeded", None)
    assert zh_of(imported, outcome.run_id) == [zh(3, 0), zh(3, 1)]
    assert [who(b) for b in provider.bodies] == ["translate", "translate", "review", "curate"]
    assert run["calls"] == len(provider.bodies) == len(meter.records) == 4   # the Curator's call is the Run's
    assert run["cost_usd"] == pytest.approx(4 * CALL_COST) and run["unknown_usage_calls"] == 0

    assert entries(imported, kb) == [("宇宙船", "飞船", "approved", "human", [1, 3]),    # one more evidence page
                                     ("封印", "封印术", "approved", "agent", [3])]       # the new term, by the Curator
    assert profiles(imported, kb) == [("ユキ", "小雪", "pending_human", [3])]
    history = json.loads(imported.execute("SELECT history FROM glossary_entries WHERE source_term = '封印'")
                         .fetchone()[0])
    assert [(h["event"], h["run_id"], h["reason"]) for h in history] == [("created", outcome.run_id, "招式名，译法一致")]
    assert queue(imported, kb) == [("term", "封印", "approved", "招式名，译法一致"),
                                   ("term", "宇宙船", "approved", "confirms entry 1"),
                                   ("character", "ユキ", "pending_human", "a new name waits for a human")]
    assert {r["run_id"] for r in store.kb_proposals(imported, kb)} == {outcome.run_id}

    attempts = attempts_of(imported, outcome.run_id)
    assert set(attempts) == {"config", "stages", "send_backs", "final", "finalize"} and attempts["config"] == "full"
    finalize = attempts["finalize"]
    assert set(finalize) == {"kb_id", "curator", "outcomes"} and finalize["kb_id"] == kb
    assert set(finalize["curator"]) == {"status", "error", "asked", "decisions", "attempts"}
    assert (finalize["curator"]["status"], finalize["curator"]["error"]) == ("ok", None)
    assert finalize["curator"]["asked"] == [new.proposal_id, seen.proposal_id]
    assert finalize["curator"]["decisions"] == [
        {"id": new.proposal_id, "accept": True, "reason": "招式名，译法一致"},
        {"id": seen.proposal_id, "accept": True, "reason": "同一译法"}]
    assert set(finalize["curator"]["attempts"]) == {"agent", "repair"}
    assert finalize["outcomes"] == [
        {"proposal_id": new.proposal_id, "kind": "term", "source": "封印", "target": "封印术", "result": "accepted"},
        {"proposal_id": seen.proposal_id, "kind": "term", "source": "宇宙船", "target": "飞船", "result": "confirmed"},
        {"proposal_id": name_proposal(kb, 3, *NAME).proposal_id,
         "kind": "character", "source": "ユキ", "target": "小雪", "result": "pending_human"}]
    assert runs.outcome_counts(finalize["outcomes"]) == {
        "duplicate": 0, "confirmed": 1, "conflict": 0, "conflict_across_kinds": 0, "pending_human": 1, "accepted": 1,
        "rejected": 0, "undecided": 0}

    curator_body = provider.bodies[3]
    listed = [json.loads(line) for line in curator_body["messages"][1]["content"][2]["text"].splitlines()[1:]]
    assert [p["id"] for p in listed] == [new.proposal_id, seen.proposal_id]   # the terms only
    assert "ユキ" not in sent(curator_body) and "小雪" not in sent(curator_body)   # not the name
    assert "tools" not in curator_body


def test_the_run_records_every_budget_and_the_curators(imported, root, kb):
    provider = Provider(answer(zh_json(1)), answer(APPROVE))

    outcome, _ = run_page(imported, root, provider, kb, 1, page_budget=567, agent_budget=1234, review_budget=890,
                          curator_budget=345, max_steps=3, max_tokens=99)

    assert json.loads(store.get_run(imported, outcome.run_id)["config_json"]) == {
        "agent_budget": 1234, "review_budget": 890, "curator_budget": 345, "fallback_budget": 567, "max_steps": 3,
        "max_tokens": 99, "max_send_backs": 2}
    assert runs.run_params("full", page_budget=1, max_tokens=2)["curator_budget"] == curator.DEFAULT_CURATOR_BUDGET
    assert "curator_budget" not in runs.run_params("full_no_kb", page_budget=1, max_tokens=2)


def test_the_curator_is_given_the_budget_the_run_was_given(imported, root, kb):
    new, seen = terms_of(kb)
    provider = Provider(proposals_reply(), answer(zh_json(3)), answer(APPROVE), answer("not json"),
                        answer(decisions_json(decision(new), decision(seen))))

    outcome, _ = run_page(imported, root, provider, kb, curator_budget=120)   # the answer used all of it

    assert attempts_of(imported, outcome.run_id)["finalize"]["curator"]["status"] == "failed"
    assert len(provider.bodies) == 4


def test_a_full_page_whose_proposals_are_only_names_skips_the_curator_and_the_rules_still_apply(imported, root, kb):
    provider = Provider(ask(propose("n", "character", name_src=NAME[0], name_tgt=NAME[1])), answer(zh_json(3)),
                        answer(APPROVE), answer(decisions_json()))

    outcome, _ = run_page(imported, root, provider, kb)

    assert outcome.result.ok and [who(b) for b in provider.bodies] == ["translate", "translate", "review"]
    finalize = attempts_of(imported, outcome.run_id)["finalize"]
    assert (finalize["curator"]["status"], finalize["curator"]["asked"], finalize["curator"]["decisions"]) == (
        "skipped", [], [])
    assert finalize["curator"]["attempts"] == {"agent": None, "repair": None}
    assert [o["result"] for o in finalize["outcomes"]] == ["pending_human"]
    assert profiles(imported, kb) == [("ユキ", "小雪", "pending_human", [3])]
    assert store.get_run(imported, outcome.run_id)["calls"] == 3


def test_a_full_page_with_no_proposal_makes_no_curator_call_and_does_not_call_apply(imported, root, kb, monkeypatch):
    applied = []
    monkeypatch.setattr(curator_rules, "apply", lambda *args, **kwargs: applied.append(args) or [])   # a spy
    provider = Provider(answer(zh_json(3)), answer(APPROVE))
    before = dump(imported)

    outcome, _ = run_page(imported, root, provider, kb)

    assert outcome.result.ok and [who(b) for b in provider.bodies] == ["translate", "review"]
    assert applied == []   # not even an empty transaction
    finalize = attempts_of(imported, outcome.run_id)["finalize"]
    assert finalize["curator"]["status"] == "skipped" and finalize["outcomes"] == []
    assert dump(imported) == before


def test_decisions_for_proposals_the_rules_settle_are_recorded_and_change_nothing(imported, root, kb):
    seed_profile(imported, kb, "ユキ", "小雪", status="pending_human")
    confirm, conflict, across = term(kb, 3, *SEEN_TERM), term(kb, 3, "宇宙船", "太空船"), term(kb, 3, "ユキ", "雪")
    provider = Provider(
        ask(propose("a", "term", source_term="宇宙船", target_term="飞船"),
            propose("b", "term", source_term="宇宙船", target_term="太空船"),
            propose("c", "term", source_term="ユキ", target_term="雪")),
        answer(zh_json(3)), answer(APPROVE),
        answer(decisions_json(decision(confirm, False, "拒绝"), decision(conflict, True, "接受"),
                              decision(across, True, "接受"))))

    outcome, _ = run_page(imported, root, provider, kb)

    finalize = attempts_of(imported, outcome.run_id)["finalize"]
    assert [o["result"] for o in finalize["outcomes"]] == ["confirmed", "conflict", "conflict_across_kinds"]
    assert [(d["accept"], d["reason"]) for d in finalize["curator"]["decisions"]] == [
        (False, "拒绝"), (True, "接受"), (True, "接受")]   # recorded, as the model gave them
    assert entries(imported, kb) == [("宇宙船", "飞船", "approved", "human", [1, 3])]   # no new entry, no new target
    assert profiles(imported, kb) == [("ユキ", "小雪", "pending_human", [1])]
    assert [q[2] for q in queue(imported, kb)] == ["approved", "pending_human", "pending_human"]


# --- AC2: pages in order -------------------------------------------------------------------------------------------

def test_page_two_reads_what_page_one_wrote_and_page_one_never_sees_its_own_proposals(imported, root):
    kb_id = new_kb(imported, book=BOOK)
    language = term(kb_id, 1, "日本語", "日语", category="language")
    first = Provider(
        ask(lookup_term("l1", "日本語"),
            propose("p1", "term", source_term="日本語", target_term="日语", category="language")),
        answer(zh_json(1)), answer(APPROVE), answer(decisions_json(decision(language, True, "书名里的词"))))
    second = Provider(ask(lookup_term("l2", "日本語")), answer(zh_json(2)), answer(APPROVE))

    outcome_1, _ = run_page(imported, root, first, kb_id, 1)
    outcome_2, _ = run_page(imported, root, second, kb_id, 2)

    assert outcome_1.result.ok and outcome_2.result.ok
    assert json.loads(tool_result(first.bodies[1], "l1"))["found"] is False   # its own proposal is not visible to it
    assert first.bodies[0]["messages"][1]["content"][0]["text"] == NOTHING
    assert entries(imported, kb_id) == [("日本語", "日语", "approved", "agent", [1])]

    snippets = second.bodies[0]["messages"][1]["content"][0]["text"]
    assert snippets == HEADER + "term: 日本語 → 日语 (language)"   # the entry page 1 wrote
    assert second.bodies[2]["messages"][1]["content"][0]["text"] == snippets   # the Reviewer reads it too
    found = json.loads(tool_result(second.bodies[1], "l2"))
    assert found == {"found": True, "source_term": "日本語", "target_term": "日语", "category": "language",
                     "status": "approved"}
    finalize = attempts_of(imported, outcome_2.run_id)["finalize"]
    assert finalize["curator"]["status"] == "skipped" and len(second.bodies) == 3   # nothing to judge on page 2


def test_the_snapshot_is_the_kb_as_the_page_started_and_the_whole_page_reads_it(imported, root, kb):
    provider = Provider(
        ask(propose("p", "term", source_term="封印", target_term="封印术")), ask(lookup_term("a", "封印")),
        answer(zh_json(3)), ask(lookup_term("c", "封印")), answer(APPROVE),
        answer(decisions_json(decision(term(kb, 3, "封印", "封印术"), True, "好"))))

    outcome, _ = run_page(imported, root, provider, kb)

    assert outcome.result.ok and [who(b) for b in provider.bodies] == [
        "translate", "translate", "translate", "review", "review", "curate"]
    for body, call_id in ((provider.bodies[2], "a"), (provider.bodies[4], "c")):   # after the proposal, on its page
        assert json.loads(tool_result(body, call_id))["found"] is False   # the translator and the reviewer: not yet
    assert entries(imported, kb)[-1][:3] == ("封印", "封印术", "approved")   # written only at finalize


def test_the_earlier_pages_are_the_ones_of_the_same_config(imported, root, kb):
    other = new_kb(imported, label="other", book=BOOK)
    for page_index in (1, 2):
        assert run_page(imported, root, Provider(answer(zh_json(page_index)), answer(APPROVE)), kb,
                        page_index)[0].result.ok
    assert translate(imported, root, mock_llm_client(Provider(answer(zh_json(2, "别")))), 2).result.ok  # `translator`
    two_pages_back = call("call_3", {"pages_back": 2}, name="get_previous_context")
    provider = Provider(ask(two_pages_back), answer(zh_json(3)), answer(APPROVE))

    outcome, _ = run_page(imported, root, provider, other)   # another KB: the earlier pages are still the book's

    assert outcome.result.ok
    result = json.loads(tool_result(provider.bodies[1], "call_3"))
    assert [p["translated"] for p in result["pages"]] == [True, True]
    assert result["pages"][1]["regions"][0]["zh"] == zh(2, 0)   # not the `translator` Run's "译文2-0别"


# --- AC3: what the Curator says, and when it says nothing ----------------------------------------------------------

def test_a_curator_that_rejects_stores_no_term_entry_and_keeps_the_reason(imported, root, kb):
    new, seen = terms_of(kb)
    provider = Provider(proposals_reply(), answer(zh_json(3)), answer(APPROVE),
                        answer(decisions_json(decision(new, False, "普通的词，不需要固定译法"), decision(seen))))

    outcome, _ = run_page(imported, root, provider, kb)

    assert outcome.result.ok
    assert [e[0] for e in entries(imported, kb)] == ["宇宙船"]   # no entry for 封印
    assert queue(imported, kb)[0] == ("term", "封印", "rejected", "普通的词，不需要固定译法")
    assert [o["result"] for o in attempts_of(imported, outcome.run_id)["finalize"]["outcomes"]] == [
        "rejected", "confirmed", "pending_human"]
    assert load_snapshot(imported, kb).term("封印") is None   # a rejected term is not an entry: no lookup finds it


NO_ANSWER = [   # the Curator's replies, the Run's options, the requests sent in all, the Curator's status, its error
    ([answer("not json"), answer(decisions_json())], {}, 5, "unusable", "unparseable"),
    ([answer("", finish_reason="length")], {}, 4, "unusable", "incomplete"),
    ([upstream_500()], {}, 4, "failed", "API error 500"),
    ([answer("not json")], {"curator_budget": 120}, 4, "failed", "Stopped:"),   # spent before the repair
    ([answer("not json"), upstream_500()], {}, 5, "failed", "API error 500"),
]


@pytest.mark.parametrize("curator_replies, options, requests, status, why", NO_ANSWER,
                         ids=["unusable-after-the-repair", "cut-off", "provider-error", "spent-budget",
                              "provider-error-on-the-repair"])
def test_a_curator_without_an_answer_leaves_the_terms_undecided_and_the_run_succeeds(
        imported, root, kb, curator_replies, options, requests, status, why):
    new, seen = terms_of(kb)
    spare = answer(decisions_json(decision(new), decision(seen)))   # what a resend would have been answered with
    provider = Provider(proposals_reply(), answer(zh_json(3)), answer(APPROVE), *curator_replies, spare, spare)

    outcome, meter = run_page(imported, root, provider, kb, **options)

    assert outcome.result.ok   # the translation is complete and reviewed
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["error"]) == ("succeeded", None)
    assert zh_of(imported, outcome.run_id) == [zh(3, 0), zh(3, 1)]
    assert len(provider.bodies) == requests == len(meter.records) == run["calls"]   # nothing after the failure
    assert who(provider.bodies[-1]) == "curate"
    finalize = attempts_of(imported, outcome.run_id)["finalize"]
    assert finalize["curator"]["status"] == status and why in finalize["curator"]["error"]   # the record says why
    assert finalize["curator"]["decisions"] == [] and finalize["curator"]["asked"] == [new.proposal_id, seen.proposal_id]
    assert [o["result"] for o in finalize["outcomes"]] == ["undecided", "confirmed", "pending_human"]
    assert entries(imported, kb) == [("宇宙船", "飞船", "approved", "human", [1, 3])]   # the confirmation, by rule
    assert profiles(imported, kb) == [("ユキ", "小雪", "pending_human", [3])]           # the name, by rule
    assert queue(imported, kb)[0] == ("term", "封印", "proposed", None)   # every new term still waits


def test_a_provider_error_of_the_curator_makes_the_runs_cost_unknown_but_the_run_is_there(imported, root, kb):
    provider = Provider(proposals_reply(), answer(zh_json(3)), answer(APPROVE), upstream_500())

    outcome, _ = run_page(imported, root, provider, kb)

    run = store.get_run(imported, outcome.run_id)
    assert run["status"] == "succeeded" and run["calls"] == 4
    assert (run["unknown_usage_calls"], run["cost_usd"]) == (1, None)   # the failed request's usage is unknown, not 0


# --- AC4: failures that write nothing ------------------------------------------------------------------------------

def test_a_page_whose_orchestrator_fails_gets_no_curator_call_and_writes_nothing(imported, root, kb):
    new, seen = terms_of(kb)
    provider = Provider(proposals_reply(), answer(zh_json(3)), upstream_500(),
                        answer(decisions_json(decision(new), decision(seen))))
    before = dump(imported)

    outcome, meter = run_page(imported, root, provider, kb)

    assert not outcome.result.ok and "API error 500" in outcome.result.error
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"]) == ("failed", 3) and len(provider.bodies) == len(meter.records) == 3   # no Curator call
    assert [who(b) for b in provider.bodies] == ["translate", "translate", "review"]
    assert count_translations(imported, outcome.run_id) == 0
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["finalize"] is None and attempts["config"] == "full"
    proposed = [c for c in attempts["stages"][0]["attempts"]["agent"]["steps"][0]["tool_calls"]
                if c["name"] == "propose_entry" and c["ran"]]
    assert len(proposed) == 3   # the Translator did propose: the KB is untouched by choice, not by luck
    assert dump(imported) == before


def boom(*args, **kwargs):
    raise RuntimeError("disk on fire")


@pytest.mark.parametrize("target, name", [(curator_rules, "apply"), (store, "insert_entry")],
                         ids=["apply-raises", "a-write-inside-apply-raises"])
def test_when_the_write_fails_the_run_is_failed_through_run_crashed_and_the_kb_is_unchanged(
        imported, root, kb, monkeypatch, target, name):
    new, seen = terms_of(kb)
    provider = Provider(proposals_reply(), answer(zh_json(3)), answer(APPROVE),
                        answer(decisions_json(decision(new, True, "好"), decision(seen))))
    before = dump(imported)
    monkeypatch.setattr(target, name, boom)

    with pytest.raises(runs.RunCrashed) as crashed:
        run_page(imported, root, provider, kb)

    run = store.get_run(imported, crashed.value.run_id)
    assert (run["status"], run["error"]) == ("failed", "RuntimeError: disk on fire")
    assert crashed.value.error == run["error"] and isinstance(crashed.value.__cause__, RuntimeError)
    assert count_translations(imported, run["id"]) == 0   # no Translation
    assert run["calls"] == 4   # the Curator was asked before the write, and its call is the Run's
    assert dump(imported) == before   # not one row of the KB
    assert imported.in_transaction is False
    assert load_snapshot(imported, kb).term("封印") is None


def test_full_without_a_kb_or_with_the_wrong_one_is_a_value_error_before_any_client_or_run(imported, root):
    def no_client():
        raise AssertionError("make_client must not be called")

    imported.execute("INSERT INTO books (book_title) VALUES ('other_book')")
    imported.commit()
    elsewhere, mine = new_kb(imported, label="x", book="other_book"), new_kb(imported, book=BOOK)
    for options, message in (({}, "needs a `kb_id`"), ({"kb_id": elsewhere}, "is for the book 'other_book'"),
                             ({"kb_id": mine + 100}, "no knowledge base has id")):
        with pytest.raises(ValueError, match=message):
            runs.translate_page(imported, BOOK, 3, make_client=no_client, dataset_root=root, config="full", **options)

    with pytest.raises(ValueError, match="has no Knowledge Base"):   # a KB goes with `full` and nothing else
        runs.translate_page(imported, BOOK, 3, make_client=no_client, dataset_root=root, config="full_no_kb",
                            kb_id=mine)
    assert imported.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


# --- AC5: what the Curator sees ------------------------------------------------------------------------------------

def test_nothing_of_the_translators_or_the_reviewers_conversations_or_the_references_reaches_the_curator(imported, root, kb):
    for page_index in (1, 2):   # the earlier pages the Translator's tool returns: "译文2-…" and "日本語2-…"
        assert run_page(imported, root, Provider(answer(zh_json(page_index)), answer(APPROVE)), kb,
                        page_index)[0].result.ok
    new, seen = terms_of(kb)
    reviewed = verdict_json("revise", issue(0, "REVIEW-COMMENT 第一句译得生硬", "REVIEW-SUGGESTION 更自然的第一句"))
    provider = Provider(
        ask(previous_context(3), flag("f", 1, "FLAG-REASON"), propose("p1", "term", source_term="封印",
                                                                       target_term="封印术"),
            propose("p2", "term", source_term=SEEN_TERM[0], target_term=SEEN_TERM[1]),
            content="TRANSLATOR-TEXT", reasoning_content="TRANSLATOR-REASONING"),
        answer(zh_json(3, "初")),
        ask(lookup_term("rl", "宇宙船"), reasoning_content="REVIEWER-REASONING"),
        answer(reviewed),
        ask(previous_context(3), content="REVISER-TEXT", reasoning_content="REVISER-REASONING"),
        answer(zh_json(3, "改")),
        answer(APPROVE),
        answer(decisions_json(decision(new), decision(seen))))

    outcome, _ = run_page(imported, root, provider, kb)

    assert outcome.result.ok and [who(b) for b in provider.bodies] == [
        "translate", "translate", "review", "review", "revise", "revise", "review", "curate"]
    others = sent(provider.bodies[:-1])
    translator_only = ["TRANSLATOR-TEXT", "TRANSLATOR-REASONING", "FLAG-REASON", "REVISER-TEXT", "REVISER-REASONING",
                       "译文2-", "日本語2-", "get_previous_context", "flag_uncertain", "propose_entry", "tool_call"]
    reviewer_only = ["REVIEWER-REASONING", "REVIEW-COMMENT", "REVIEW-SUGGESTION"]
    assert all(s in others for s in translator_only + reviewer_only)   # the other agents' requests do carry them
    curator_sent = sent(provider.bodies[-1])
    for forbidden in (*translator_only, *reviewer_only, "译文3-0初", "REF-ZH", "REF-EN", "image_url"):
        assert forbidden not in curator_sent, forbidden
    text = provider.bodies[-1]["messages"][1]["content"][1]["text"]
    assert zh(3, 0, "改") in text and zh(3, 1, "改") in text   # the stored translation, the revision
    assert "REF-ZH" not in sent(provider.bodies) and "REF-EN" not in sent(provider.bodies)   # in no request at all
    assert [m["role"] for m in provider.bodies[-1]["messages"]] == ["system", "user"]


def test_the_curator_is_asked_for_decisions_in_simplified_chinese(imported, root, kb):
    new, seen = terms_of(kb)
    provider = Provider(proposals_reply(), answer(zh_json(3)), answer(APPROVE), answer("not json"),
                        answer(decisions_json(decision(new), decision(seen))))

    outcome, _ = run_page(imported, root, provider, kb)

    assert outcome.result.ok
    curator_bodies = [b for b in provider.bodies if who(b) == "curate"]
    assert len(curator_bodies) == 2   # the request and its repair
    assert "Write every reason in Simplified Chinese (zh-Hans)" in curator_bodies[0]["messages"][0]["content"]
    assert all(b["messages"][0]["content"] == curator.SYSTEM_PROMPT for b in curator_bodies)


# --- the root span -------------------------------------------------------------------------------------------------

def test_the_pages_root_span_says_the_curators_status_and_how_often_each_result_happened(imported, root, kb, traced):
    new, seen = terms_of(kb)
    provider = Provider(proposals_reply(), answer(zh_json(3)), answer(APPROVE),
                        answer(decisions_json(decision(new), decision(seen))))

    outcome, _ = run_page(imported, root, provider, kb)

    top = one_trace(traced.spans())
    assert top.attributes["fukidashi.config"] == "full" and top.attributes["fukidashi.curator_status"] == "ok"
    assert {k.removeprefix("fukidashi.proposals_"): v for k, v in top.attributes.items()
            if k.startswith("fukidashi.proposals_")} == {
        "duplicate": 0, "confirmed": 1, "conflict": 0, "conflict_across_kinds": 0, "pending_human": 1, "accepted": 1,
        "rejected": 0, "undecided": 0}
    assert top.attributes["fukidashi.send_backs"] == 0 and top.attributes["fukidashi.final_stop"] == "approved"
    assert outcome.trace_id is not None


def test_a_full_no_kb_pages_root_span_has_no_curator_attributes(imported, root, traced):
    outcome = translate(imported, root, mock_llm_client(Provider(answer(zh_json(3)), answer(APPROVE))), 3,
                        config="full_no_kb")

    top = one_trace(traced.spans())
    assert outcome.result.ok and not [k for k in top.attributes if "curator" in k or "proposals_" in k]


# --- AC6: the eval harness -----------------------------------------------------------------------------------------

def three_pages(kb_id=1):
    """Page 1 writes a term (that occurs in the text of every page) and a name; page 2 meets them again under other
    targets and kinds; on page 3 the Curator gives no usable answer. `kb_id` is the KB the eval will make."""
    return Provider(
        ask(propose("a", "term", source_term="日本語", target_term="日语", category="language"),
            propose("b", "character", name_src="ユキ", name_tgt="小雪")),
        answer(zh_json(1)), answer(APPROVE),
        answer(decisions_json(decision(term(kb_id, 1, "日本語", "日语"), True, "书名里的词"))),
        ask(propose("c", "term", source_term="日本語", target_term="日文"),
            propose("d", "term", source_term="ユキ", target_term="雪"),
            propose("e", "term", source_term="魔法", target_term="魔术")),
        answer(zh_json(2)), answer(APPROVE),
        answer(decisions_json(decision(term(kb_id, 2, "日本語", "日文")), decision(term(kb_id, 2, "ユキ", "雪")),
                              decision(term(kb_id, 2, "魔法", "魔术"), False, "普通的词"))),
        ask(propose("f", "term", source_term="日本語", target_term="日语"),
            propose("g", "term", source_term="竜", target_term="龙")),
        answer(zh_json(3)), answer(APPROVE), answer("not json"), answer("still not"))


OLD_SCORE_KEYS = {"config", "book", "pages_scored", "segments_scored", "fallback_pages", "chrf", "bleu", "cost",
                  "latency", "pages", "failed_pages", "skipped_pages"}
OLD_PAGE_KEYS = {"book", "page_index", "run_id", "trace_id", "segments", "calls", "unknown_usage_calls", "fell_back",
                 "fallback_reason", "cost_usd", "latency_s"}
LOOP_KEYS = {"send_backs", "final_stop", "final_reviewed", "review_statuses", "review_verdicts"}
RESULTS = ("duplicate", "confirmed", "conflict", "conflict_across_kinds", "pending_human", "accepted", "rejected",
           "undecided")


def outcomes(**counted):
    return {result: counted.get(result, 0) for result in RESULTS}


def kb_rows(conn):
    return [dict(r) for r in conn.execute("SELECT * FROM kbs ORDER BY id")]


def test_the_harness_makes_one_kb_for_a_full_eval_and_passes_it_to_every_page_in_order(imported, root, tmp_path):
    provider = three_pages()

    result = evaluate(imported, root, tmp_path, provider, [1, 2, 3], config=CONFIG)

    assert result.exit_code == 0 and len(provider.bodies) == 13
    (kb_row,) = kb_rows(imported)   # exactly one
    assert (kb_row["book_title"], kb_row["purpose"], kb_row["label"]) == (BOOK, "eval", result.manifest["eval_id"])
    assert result.manifest["kb"] == {"id": kb_row["id"], "label": result.manifest["eval_id"]}
    assert result.out_dir.name == result.manifest["eval_id"]
    assert {r["config"] for r in imported.execute("SELECT config FROM runs")} == {"full"}
    assert {r["kb_id"] for r in imported.execute("SELECT kb_id FROM proposals")} == {kb_row["id"]}
    assert provider.bodies[0]["messages"][1]["content"][0]["text"] == NOTHING
    for first_request in (4, 8):   # the first Translator request of pages 2 and 3 read what page 1 wrote
        assert provider.bodies[first_request]["messages"][1]["content"][0]["text"] == (
            HEADER + "term: 日本語 → 日语 (language)")
    assert [r["page_index"] for r in imported.execute(
        "SELECT p.page_index FROM runs r JOIN pages p ON p.id = r.page_id ORDER BY r.id")] == [1, 2, 3]

    scores = json.loads((result.out_dir / "scores.json").read_text(encoding="utf-8"))
    assert set(scores) == OLD_SCORE_KEYS | {"final_stops", "curator_statuses", "proposal_outcomes", "kb"}
    assert scores["config"] == CONFIG and (scores["pages_scored"], scores["segments_scored"]) == (3, 6)
    assert all(set(p) == OLD_PAGE_KEYS | LOOP_KEYS | {"curator_status", "proposal_outcomes"} for p in scores["pages"])
    assert [(p["page_index"], p["curator_status"]) for p in scores["pages"]] == [(1, "ok"), (2, "ok"), (3, "unusable")]
    assert [p["proposal_outcomes"] for p in scores["pages"]] == [
        outcomes(accepted=1, pending_human=1),
        outcomes(conflict=1, conflict_across_kinds=1, rejected=1),   # apart from each other
        outcomes(confirmed=1, undecided=1)]
    assert scores["proposal_outcomes"] == outcomes(accepted=1, pending_human=1, conflict=1, conflict_across_kinds=1,
                                                   rejected=1, confirmed=1, undecided=1)
    assert scores["curator_statuses"] == {"ok": 2, "unusable": 1, "failed": 0, "skipped": 0}
    assert scores["final_stops"]["approved"] == 3
    assert scores["kb"] == {
        "id": kb_row["id"], "label": result.manifest["eval_id"],
        "entries": {"proposed": 0, "approved": 1, "rejected": 0, "pending_human": 0},
        "profiles": {"proposed": 0, "approved": 0, "rejected": 0, "pending_human": 1}}
    assert scores["cost"]["total_usd"] == pytest.approx(13 * CALL_COST, abs=1e-6)
    assert [p["calls"] for p in scores["pages"]] == [4, 4, 5]   # the Curator's calls are the pages'
    assert result.manifest["config_params"] == {
        "agent_budget": 60_000, "review_budget": 30_000, "curator_budget": 20_000, "fallback_budget": 40_000,
        "max_steps": 6, "max_tokens": 16_000, "max_send_backs": 2}
    assert entries(imported, kb_row["id"]) == [("日本語", "日语", "approved", "agent", [1, 3])]   # page 3 confirms it

    summary = harness.format_summary(result)
    assert "curator  ok 2, unusable 1, failed 0, skipped 0" in summary
    assert "proposals  " in summary and "conflict_across_kinds 1" in summary and "undecided 1" in summary
    assert f"knowledge base {kb_row['id']} ({result.manifest['eval_id']})" in summary
    assert "entries approved 1" in summary and "profiles pending_human 1" in summary
    assert "CURATOR  book_t/3 (run" in summary and "unusable, proposals confirmed 1, undecided 1" in summary


def test_two_full_evals_of_the_same_pages_start_from_two_empty_kbs(imported, root, tmp_path):
    first = evaluate(imported, root, tmp_path, three_pages(), [1, 2, 3], config=CONFIG)
    first_kb = first.manifest["kb"]
    after_first = dump(imported)

    provider = three_pages(first_kb["id"] + 1)   # the second KB's id
    second = evaluate(imported, root, tmp_path, provider, [1, 2, 3], config=CONFIG)

    assert len(kb_rows(imported)) == 2 and first_kb != second.manifest["kb"]
    assert first_kb["label"] != second.manifest["kb"]["label"] and first.manifest["eval_id"] != second.manifest["eval_id"]
    assert provider.bodies[0]["messages"][1]["content"][0]["text"] == NOTHING   # the first page started from nothing
    for table in ("glossary_entries", "character_profiles", "proposals"):
        assert [r for r in dump(imported)[table] if r[1] == first_kb["id"]] == [r for r in after_first[table]]
    assert second.scores["kb"]["entries"] == first.scores["kb"]["entries"]   # the same script wrote the same
    assert {r["kb_id"] for r in imported.execute("SELECT kb_id FROM proposals")} == {first_kb["id"],
                                                                                    second.manifest["kb"]["id"]}


def test_a_page_that_fails_is_left_out_of_the_scores_and_the_kb_keeps_what_the_earlier_pages_wrote(imported, root, tmp_path):
    provider = Provider(
        ask(propose("a", "term", source_term="封印", target_term="封印术")), answer(zh_json(1)), answer(APPROVE),
        answer(decisions_json(decision(term(1, 1, "封印", "封印术")))),
        ask(propose("b", "term", source_term="魔法", target_term="魔术")), answer(zh_json(2)), upstream_500())

    result = evaluate(imported, root, tmp_path, provider, [1, 2], config=CONFIG)

    assert result.exit_code == 1 and len(provider.bodies) == 7
    (failed,) = result.scores["failed_pages"]
    assert failed["page_index"] == 2 and "API error 500" in failed["error"]
    assert [p["page_index"] for p in result.scores["pages"]] == [1]
    assert result.scores["proposal_outcomes"] == outcomes(accepted=1)   # page 2's proposal wrote nothing
    assert result.scores["kb"]["entries"]["approved"] == 1 and result.scores["curator_statuses"]["ok"] == 1
    assert "FAILED  book_t/2" in harness.format_summary(result)


def test_a_full_eval_that_stops_before_any_model_call_makes_no_kb(imported, root, tmp_path):
    with pytest.raises(harness.EvalAborted):
        evaluate(imported, root, tmp_path, Provider(), [1, 99], config=CONFIG)

    assert kb_rows(imported) == [] and not (tmp_path / "out").exists()


def test_the_kb_label_is_the_eval_id_and_a_taken_label_gets_a_suffix(imported, root, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_now", lambda: "2026-01-01T00:00:00+00:00")
    new_kb(imported, label="20260101T000000Z-full", book=BOOK)   # an earlier KB with the label of this eval's id

    result = evaluate(imported, root, tmp_path, Provider(answer(zh_json(1)), answer(APPROVE)), [1], config=CONFIG)

    assert result.manifest["eval_id"] == "20260101T000000Z-full-2" == result.manifest["kb"]["label"]
    assert result.out_dir.name == "20260101T000000Z-full-2"
    assert len(kb_rows(imported)) == 2


def test_the_other_configs_keep_every_key_they_have_and_make_no_kb(imported, root, tmp_path):
    reviewed = evaluate(imported, root, tmp_path, Provider(answer(zh_json(1)), answer(APPROVE)), [1], config="full_no_kb")
    translated = evaluate(imported, root, tmp_path, Provider(answer(zh_json(2))), [2], config="translator")

    assert set(reviewed.scores) == OLD_SCORE_KEYS | {"final_stops"}
    assert all(set(p) == OLD_PAGE_KEYS | LOOP_KEYS for p in reviewed.scores["pages"])
    assert set(translated.scores) == OLD_SCORE_KEYS
    assert "kb" not in reviewed.manifest and "kb" not in translated.manifest
    assert reviewed.manifest["config_params"] == {
        "agent_budget": 60_000, "review_budget": 30_000, "fallback_budget": 40_000, "max_steps": 6,
        "max_tokens": 16_000, "max_send_backs": 2}
    assert kb_rows(imported) == []
    for result in (reviewed, translated):
        assert "curator" not in harness.format_summary(result) and "knowledge base" not in harness.format_summary(result)


def test_the_summary_of_a_full_eval_that_wrote_nothing_says_the_kb_is_empty(imported, root, tmp_path):
    result = evaluate(imported, root, tmp_path, Provider(answer(zh_json(1)), answer(APPROVE)), [1], config=CONFIG)

    summary = harness.format_summary(result)

    assert result.scores["kb"]["entries"] == result.scores["kb"]["profiles"] == {
        "proposed": 0, "approved": 0, "rejected": 0, "pending_human": 0}
    assert f"knowledge base {result.manifest['kb']['id']} ({result.manifest['eval_id']})  empty" in summary
    assert "proposals  none" in summary and "entries ;" not in summary


def test_the_command_line_accepts_full_and_its_usage_names_it(tmp_path):
    stopped = run_cli(tmp_path, "--config", "full", "--book", "book_t", "--pages", "1")

    assert "invalid choice" not in stopped.stderr
    assert stopped.returncode == 2 and "Stopped before any model call" in stopped.stderr
    assert "--config full --book" in app_config.ROOT.joinpath("eval", "run.py").read_text(encoding="utf-8")
