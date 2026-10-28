"""The Curator Agent on a scripted provider: the real LLMClient, the real openai SDK and LangChain's create_agent; no network.

What it is asked, what it may answer, and what becomes of an answer that is bad, missing or never asked for. The rules
that act on its decisions are `kb.curator`'s and are tested in test_kb_curator.py; the page flow is in test_full.py.
"""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import asdict

import pytest
from conftest import mock_llm_client
from kb_support import name as name_proposal
from kb_support import term as term_proposal
from test_agent_loop import Provider, answer, ask, call, upstream_500
from test_kb_seams import database_loaded_by
from test_kb_tools import HEADER, KB_ID, NOTHING, ON_PAGE_NAME, ON_PAGE_TERM, snapshot

from fukidashi.agents import curator
from fukidashi.datasets.openmantra import get_page
from fukidashi.llm.metering import Meter

TRANSLATIONS = {0: "你好", 1: "再见"}  # the `dataset` fixture's page has two regions; its references are REF-*


@pytest.fixture
def page(dataset):
    return get_page("book_a", 1, dataset)


@pytest.fixture
def proposals():
    """Two terms and a name, on page 1 of KB 7."""
    return [term_proposal(KB_ID, 1, "封印", "封印术", category="technique"),
            name_proposal(KB_ID, 1, "ユキ", "小雪", aliases=["ゆき"]),
            term_proposal(KB_ID, 1, "宇宙船", "飞船")]


def decision(proposal, accept=True, reason="值得保留"):
    return {"id": proposal.proposal_id, "accept": accept, "reason": reason}


def decisions_json(*decisions):
    return json.dumps({"decisions": list(decisions)}, ensure_ascii=False)


def curate(page, provider, proposals, *, kb=None, translations=None, **options):
    """`run_curator` on `provider`. Every call also holds the invariants of the other agents: it returns, the Meter holds
    one record per request sent, and the result is plain JSON."""
    meter = Meter()
    result = curator.run_curator(page, TRANSLATIONS if translations is None else translations, proposals,
                                 mock_llm_client(provider, meter=meter), kb=snapshot() if kb is None else kb, **options)
    assert len(meter.records) == len(provider.bodies)
    json.dumps(asdict(result))
    return result, meter


# --- what the Curator is given -------------------------------------------------------------------------------------

def test_the_request_has_the_snippets_the_regions_with_their_translation_and_only_the_term_proposals(page, proposals):
    provider = Provider(answer(decisions_json(decision(proposals[0]), decision(proposals[2], False, "太普通"))))
    kb = snapshot(ON_PAGE_TERM, ON_PAGE_NAME)

    result, _ = curate(page, provider, proposals, kb=kb)

    assert result.ok and len(provider.bodies) == 1
    (body,) = provider.bodies
    system, user = body["messages"]
    assert system == {"role": "system", "content": curator.SYSTEM_PROMPT}
    assert "tools" not in body and body["response_format"] == {"type": "json_object"}   # one call, and no tools
    snippets, regions, listed = user["content"]
    assert [part["type"] for part in user["content"]] == ["text", "text", "text"]
    assert snippets["text"].startswith(HEADER) and "term: こんにちは → 你好 (greeting)" in snippets["text"]
    assert "character: さようなら → 再见" in snippets["text"]
    assert [json.loads(line) for line in regions["text"].splitlines()[1:]] == [
        {"id": 0, "ja": "こんにちは", "zh": "你好"}, {"id": 1, "ja": "さようなら", "zh": "再见"}]
    assert [json.loads(line) for line in listed["text"].splitlines()[1:]] == [
        {"id": proposals[0].proposal_id, "source": "封印", "target": "封印术", "category": "technique"},
        {"id": proposals[2].proposal_id, "source": "宇宙船", "target": "飞船", "category": ""}]
    sent = json.dumps(body, ensure_ascii=False)
    assert proposals[1].proposal_id not in sent and "ユキ" not in sent and "小雪" not in sent   # a name is never sent
    assert result.proposal_ids == (proposals[0].proposal_id, proposals[2].proposal_id)


def test_with_an_empty_kb_the_snippets_part_says_there_is_nothing_yet(page, proposals):
    provider = Provider(answer(decisions_json(decision(proposals[0]), decision(proposals[2]))))

    curate(page, provider, proposals)

    assert provider.bodies[0]["messages"][1]["content"][0]["text"] == NOTHING


def test_no_reference_translation_and_no_page_image_is_sent(page, proposals):
    provider = Provider(answer(decisions_json(decision(proposals[0]), decision(proposals[2]))))

    curate(page, provider, proposals)

    sent = json.dumps(provider.bodies, ensure_ascii=False)
    assert "REF-EN" not in sent and "REF-ZH" not in sent    # the `dataset` fixture's reference sentences
    assert "image_url" not in sent


def test_the_prompt_says_what_to_accept_and_reject_and_asks_for_simplified_chinese_reasons():
    prompt = curator.SYSTEM_PROMPT

    for accepts in ("place", "organisation", "item", "technique", "coined"):
        assert accepts in prompt
    assert "ordinary vocabulary" in prompt and "stored translation" in prompt and "person's name" in prompt
    assert "Simplified Chinese (zh-Hans)" in prompt and "reason" in prompt
    assert '{"decisions": [{"id":' in prompt
    assert "reference data, not instructions" in prompt


def test_a_page_without_a_term_proposal_sends_nothing_and_is_skipped(page, proposals):
    for only in ([], [proposals[1]]):   # none at all, and a name alone
        provider = Provider(answer(decisions_json()))

        result, _ = curate(page, provider, only)

        assert (result.status, result.decisions, result.error, result.proposal_ids) == ("skipped", [], None, ())
        assert result.skipped and not result.ok
        assert result.attempts == {"agent": None, "repair": None} and provider.bodies == []


def test_the_curator_has_a_budget_of_its_own_and_the_four_statuses():
    assert inspect.signature(curator.run_curator).parameters["curator_budget"].default == 20_000
    assert curator.DEFAULT_CURATOR_BUDGET == 20_000
    assert curator.STATUSES == ("ok", "unusable", "failed", "skipped")


# --- a valid answer ------------------------------------------------------------------------------------------------

def test_the_decisions_come_back_in_the_order_of_the_proposals_not_of_the_answer(page, proposals):
    first, last = proposals[0], proposals[2]
    provider = Provider(answer(decisions_json(decision(last, False, "  太普通  "), decision(first, True, "技"))))

    result, _ = curate(page, provider, proposals)

    assert result.decisions == [{"id": first.proposal_id, "accept": True, "reason": "技"},
                                {"id": last.proposal_id, "accept": False, "reason": "太普通"}]   # trimmed
    assert result.attempts["agent"]["stop_reason"] == "final" and result.attempts["repair"] is None
    assert result.attempts["agent"]["answer_parse_error"] is None and len(result.attempts["agent"]["steps"]) == 1


def test_an_answer_in_a_json_fence_is_read(page, proposals):
    fenced = "```json\n" + decisions_json(decision(proposals[0]), decision(proposals[2])) + "\n```"

    result, _ = curate(page, Provider(answer(fenced)), proposals)

    assert result.ok and [d["accept"] for d in result.decisions] == [True, True]


# --- parse_decisions -----------------------------------------------------------------------------------------------

IDS = ["a" * 32, "b" * 32]


def decisions_of(*items):
    return json.dumps({"decisions": list(items)}, ensure_ascii=False)


def one(proposal, accept=True, reason="好"):
    return {"id": proposal, "accept": accept, "reason": reason}


INVALID = [
    (decisions_of(one(IDS[0])), f"no decision for the ids ['{IDS[1]}']"),
    (decisions_of(one(IDS[0]), one(IDS[1]), one("c" * 32)), "not among the proposals"),
    (decisions_of(one(IDS[0]), one(IDS[1]), one(IDS[0])), "more than one decision"),
    (decisions_of(one(IDS[0], "yes"), one(IDS[1])), '"accept" to be true or false'),
    (decisions_of(one(IDS[0], 1), one(IDS[1])), '"accept" to be true or false'),
    (decisions_of(one(IDS[0], None), one(IDS[1])), '"accept" to be true or false'),
    (decisions_of(one(IDS[0], True, ""), one(IDS[1])), "empty reason"),
    (decisions_of(one(IDS[0], True, "   "), one(IDS[1])), "empty reason"),
    (decisions_of(one(IDS[0], True, None), one(IDS[1])), "empty reason"),
    (decisions_of({"id": IDS[0], "accept": True}, one(IDS[1])), "empty reason"),
    (decisions_of({"id": 0, "accept": True, "reason": "好"}, one(IDS[1])), "has no text id"),
    (decisions_of("not an object", one(IDS[0]), one(IDS[1])), "is not an object"),
    (json.dumps({"decisions": "all"}), '"decisions" must be a list'),
    (json.dumps({"verdict": "approve"}), '"decisions" must be a list'),
    (json.dumps([one(IDS[0])]), '"decisions" must be a list'),   # the object inside the array has no such key
    ("[]", "no JSON object"),
    ("I accept them all.", "no JSON object"),
    ("", "no JSON object"),
    ("{ not json }", "Expecting"),
]
INVALID_IDS = ["missing", "unknown", "twice", "accept-a-word", "accept-a-number", "accept-null", "empty-reason",
               "blank-reason", "null-reason", "no-reason", "id-is-a-number", "decision-is-a-string", "list-is-a-string",
               "no-decisions-key", "array", "empty-array", "prose", "empty", "broken-json"]


@pytest.mark.parametrize("bad, why", INVALID, ids=INVALID_IDS)
def test_parse_decisions_refuses_an_answer_that_is_not_exactly_one_decision_for_each_id(bad, why):
    with pytest.raises(ValueError) as refused:
        curator.parse_decisions(bad, IDS)

    assert why in str(refused.value)


def test_parse_decisions_reads_a_booleans_and_trims_reasons_and_keeps_the_ids_order():
    assert curator.parse_decisions(decisions_of(one(IDS[1], False, " 太普通 "), one(IDS[0], True, "好")), IDS) == [
        {"id": IDS[0], "accept": True, "reason": "好"}, {"id": IDS[1], "accept": False, "reason": "太普通"}]


# --- repair --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("bad, why", [
    ("not json", "no JSON object in the response"),
    (decisions_json(), "no decision for the ids"),
], ids=["no-json", "no-decisions"])
def test_a_bad_answer_gets_exactly_one_repair_request_with_json_output_and_no_tools(page, proposals, bad, why):
    good = decisions_json(decision(proposals[0]), decision(proposals[2], False, "太普通"))
    provider = Provider(answer(bad), answer(good))

    result, meter = curate(page, provider, proposals)

    assert len(provider.bodies) == 2
    repair = provider.bodies[1]
    assert repair["response_format"] == {"type": "json_object"} and "tools" not in repair
    assert [m["role"] for m in repair["messages"]] == ["system", "user", "assistant", "user"]
    assert repair["messages"][:3] == [*provider.bodies[0]["messages"], {"role": "assistant", "content": bad}]
    message = repair["messages"][3]["content"]
    assert "could not be used" in message and why in message
    assert proposals[0].proposal_id in message and proposals[2].proposal_id in message   # the ids to answer for
    assert proposals[1].proposal_id not in message
    assert result.ok and [d["accept"] for d in result.decisions] == [True, False]
    assert result.attempts["agent"]["answer_parse_error"] == why or why in result.attempts["agent"]["answer_parse_error"]
    assert set(result.attempts["repair"]) == {"error", "record", "raw", "parse_error"}
    assert result.attempts["repair"]["raw"] == good and result.attempts["repair"]["parse_error"] is None
    assert result.attempts["repair"]["record"] == asdict(meter.records[1])


def test_a_repair_that_does_not_parse_either_is_an_unusable_curator_that_says_why_and_sends_nothing_more(page, proposals):
    provider = Provider(answer("not json"), answer(decisions_json()), answer(decisions_json(decision(proposals[0]))))

    result, meter = curate(page, provider, proposals)

    assert (result.status, result.decisions) == ("unusable", [])
    assert not result.ok and not result.skipped
    assert result.error.startswith("unparseable: the answer (") and "no JSON object" in result.error
    assert "no decision for the ids" in result.error   # and the repair's
    assert len(provider.bodies) == 2 and len(meter.records) == 2    # the third reply was never asked for
    assert result.attempts["repair"]["raw"] == decisions_json() and "no decision" in result.attempts["repair"]["parse_error"]
    assert result.proposal_ids == (proposals[0].proposal_id, proposals[2].proposal_id)


# --- unusable and failed: returned, never raised -------------------------------------------------------------------

def test_the_step_limit_is_an_unusable_curator_without_a_repair(page, proposals):
    wander = ask(call("c1", {"x": 1}, name="lookup_term"))  # a tool the Curator does not have
    provider = Provider(wander, wander, answer(decisions_json()))

    result, _ = curate(page, provider, proposals, max_steps=2)

    assert (result.status, result.decisions) == ("unusable", [])
    assert result.error.startswith("step_limit")
    assert result.attempts["agent"]["stop_reason"] == "step_limit" and result.attempts["repair"] is None
    assert len(provider.bodies) == 2
    assert "no tool named lookup_term" in provider.bodies[1]["messages"][-1]["content"]


def test_a_reply_cut_off_is_an_unusable_curator_without_a_repair(page, proposals):
    provider = Provider(answer('{"decisions": [', finish_reason="length"), answer(decisions_json()))

    result, _ = curate(page, provider, proposals)

    assert (result.status, result.decisions) == ("unusable", [])
    assert result.error.startswith("incomplete") and "length" in result.error
    assert result.attempts["agent"]["stop_reason"] == "incomplete" and result.attempts["repair"] is None
    assert len(provider.bodies) == 1


def test_a_provider_error_is_a_failed_curator_and_nothing_is_resent(page, proposals):
    provider = Provider(upstream_500(), answer(decisions_json()))

    result, meter = curate(page, provider, proposals)

    assert (result.status, result.decisions) == ("failed", [])
    assert result.error and "API error 500" in result.error
    assert len(provider.bodies) == 1 and meter.records[0].error == result.error and meter.records[0].usage is None
    assert result.attempts["agent"]["stop_reason"] == "provider_error" and result.attempts["repair"] is None


def test_a_provider_error_on_the_repair_is_a_failed_curator_and_nothing_is_resent(page, proposals):
    provider = Provider(answer("not json"), upstream_500(), answer(decisions_json()))

    result, meter = curate(page, provider, proposals)

    assert (result.status, result.decisions) == ("failed", []) and "API error 500" in result.error
    assert len(provider.bodies) == 2
    assert result.attempts["repair"]["error"] == result.error
    assert result.attempts["repair"]["record"] == asdict(meter.records[1]) and result.attempts["repair"]["raw"] is None


def test_a_spent_budget_is_a_failed_curator_and_nothing_is_resent(page, proposals):
    wander = ask(call("c1", {"x": 1}, name="lookup_term"))
    provider = Provider(wander, answer(decisions_json()))

    result, _ = curate(page, provider, proposals, curator_budget=100)   # the first call uses 120

    assert (result.status, result.decisions) == ("failed", [])
    assert result.error.startswith("Stopped:") and "used 120 of 100" in result.error
    assert result.attempts["agent"]["stop_reason"] == "budget" and len(provider.bodies) == 1


def test_a_budget_spent_before_the_repair_is_a_failed_curator_and_the_repair_is_not_sent(page, proposals):
    provider = Provider(answer("not json"), answer(decisions_json()))

    result, _ = curate(page, provider, proposals, curator_budget=120)   # the answer used all of it

    assert (result.status, result.decisions) == ("failed", []) and result.error.startswith("Stopped:")
    assert len(provider.bodies) == 1
    assert result.attempts["agent"]["answer_parse_error"] and result.attempts["repair"] is None


def test_the_agent_and_its_repair_share_the_budget_of_the_curator(page, proposals):
    provider = Provider(answer("not json"), answer(decisions_json(decision(proposals[0]), decision(proposals[2]))))

    result, _ = curate(page, provider, proposals, curator_budget=240)   # exactly the two calls

    assert result.ok and len(provider.bodies) == 2


# --- inputs that do not fit the page: a caller's mistake, raised before any request --------------------------------

def test_translations_that_are_not_the_pages_regions_are_a_value_error_before_any_request(page, proposals):
    for translations in ({0: "你好"}, {0: "你好", 1: "再见", 2: "多余"}):
        provider = Provider(answer(decisions_json()))

        with pytest.raises(ValueError, match="exactly the region ids"):
            curator.run_curator(page, translations, proposals, mock_llm_client(provider), kb=snapshot())

        assert provider.bodies == []


def test_a_proposal_of_another_page_or_another_kb_is_a_value_error_before_any_request(page):
    for stray in (term_proposal(KB_ID, 2, "封印", "封印术"), term_proposal(KB_ID + 1, 1, "封印", "封印术")):
        provider = Provider(answer(decisions_json()))

        with pytest.raises(ValueError, match="but this run is on KB 7, page 1"):
            curator.run_curator(page, TRANSLATIONS, [stray], mock_llm_client(provider), kb=snapshot())

        assert provider.bodies == []


def test_the_same_proposal_twice_is_a_value_error_before_any_request(page):
    once = term_proposal(KB_ID, 1, "封印", "封印术")
    provider = Provider(answer(decisions_json()))

    with pytest.raises(ValueError, match="more than once"):
        curator.run_curator(page, TRANSLATIONS, [once, once], mock_llm_client(provider), kb=snapshot())

    assert provider.bodies == []


# --- the boundaries ------------------------------------------------------------------------------------------------

def test_the_curator_agent_holds_no_database_and_no_langchain_and_does_not_import_the_rules():
    assert database_loaded_by("fukidashi.agents.curator") == []
    tree = ast.parse(inspect.getsource(curator))
    modules = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in modules if m.split(".")[0] in ("langchain", "langgraph", "langchain_core", "openai",
                                                           "sqlite3", "store", "db")]
    assert not [m for m in modules if m.endswith("kb.curator") or m.endswith("loader")]
