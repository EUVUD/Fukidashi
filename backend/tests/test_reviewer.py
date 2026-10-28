"""The Reviewer Agent on a scripted provider: the real LLMClient, the real openai SDK and LangChain's create_agent; no network."""

from __future__ import annotations

import ast
import inspect
import json
from dataclasses import asdict

import pytest
from conftest import mock_llm_client
from test_agent_loop import Provider, answer, ask, call, upstream_500

from fukidashi.agents import reviewer, translator
from fukidashi.datasets.openmantra import get_page
from fukidashi.llm.metering import Meter
from fukidashi.pipeline import baseline

TRANSLATIONS = {0: "你好", 1: "再见"}  # the `dataset` fixture's page has two regions; its references are REF-*


@pytest.fixture
def page(dataset):
    return get_page("book_a", 1, dataset)


def verdict_json(verdict, *issues):
    return json.dumps({"verdict": verdict, "issues": list(issues)}, ensure_ascii=False)


def issue(region, comment="译得不对", suggestion="更好的译文"):
    return {"id": region, "comment": comment, "suggestion": suggestion}


APPROVE = verdict_json("approve")
REVISE = verdict_json("revise", issue(0))


def review(page, provider, *, flags=(), translations=None, **options):
    """`run_reviewer` on `provider`. Every review also holds the invariants of AC5: it returns, the Meter holds one
    record per request sent, and the result is plain JSON."""
    meter = Meter()
    result = reviewer.run_reviewer(page, TRANSLATIONS if translations is None else translations, list(flags),
                                   mock_llm_client(provider, meter=meter), **options)
    assert len(meter.records) == len(provider.bodies)
    json.dumps(asdict(result))
    return result, meter


# --- AC3: what the Reviewer is given ------------------------------------------------------------------------------

def test_the_first_request_is_the_system_prompt_and_one_user_message_with_the_image_the_regions_and_the_flags(page):
    provider = Provider(answer(APPROVE))

    result, _ = review(page, provider, flags=[{"id": 1, "reason": "unsure about the farewell"}])

    assert result.ok
    (body,) = provider.bodies
    system, user = body["messages"]
    assert system == {"role": "system", "content": reviewer.SYSTEM_PROMPT}
    assert user["role"] == "user" and [part["type"] for part in user["content"]] == ["image_url", "text"]
    assert user["content"][0] == baseline.build_messages(page)[1]["content"][0]  # the page image, as baseline sends it
    text = user["content"][1]["text"]
    assert '{"id": 0, "box": [10, 20, 30, 40], "ja": "こんにちは", "zh": "你好"}' in text
    assert '{"id": 1, "box": [50, 60, 70, 80], "ja": "さようなら", "zh": "再见"}' in text
    assert '{"id": 1, "reason": "unsure about the farewell"}' in text and "flagged no region" not in text
    assert "tools" not in body and body["response_format"] == {"type": "json_object"}
    assert body["messages"] == [system, user]  # a fresh conversation: nothing else in it


def test_a_page_without_flags_says_so_and_a_flag_carries_only_its_id_and_reason(page):
    without = Provider(answer(APPROVE))
    review(page, without)
    assert "The translator flagged no region." in without.bodies[0]["messages"][1]["content"][1]["text"]

    extra = Provider(answer(APPROVE))
    review(page, extra, flags=[{"id": 0, "reason": "a name", "trace": "SENTINEL-EXTRA-KEY"}])
    assert "SENTINEL-EXTRA-KEY" not in json.dumps(extra.bodies, ensure_ascii=False)


def test_nothing_of_the_translators_run_or_the_references_reaches_the_reviewer(page):
    previous = [translator.PreviousPage(
        0, (translator.PreviousRegion(0, "SENTINEL-PREVIOUS-JA", "SENTINEL-PREVIOUS-ZH"),), translated=True)]
    translator_provider = Provider(
        ask(call("c1", {"pages_back": 1}, name="get_previous_context"),
            call("c2", {"id": 1, "reason": "FLAG-REASON"}, name="flag_uncertain"),
            content="SENTINEL-TRANSLATOR-TEXT", reasoning_content="SENTINEL-REASONING-1"),
        answer(json.dumps({"translations": [{"id": i, "zh": zh} for i, zh in TRANSLATIONS.items()]}, ensure_ascii=False),
               reasoning_content="SENTINEL-REASONING-2"),
    )
    made = translator.run_translator(page, mock_llm_client(translator_provider), previous=previous, flagging=True)
    assert made.ok and made.translations == TRANSLATIONS and made.flags == [{"id": 1, "reason": "FLAG-REASON"}]
    translator_sent = json.dumps(translator_provider.bodies, ensure_ascii=False)
    sentinels = ["SENTINEL-PREVIOUS-JA", "SENTINEL-PREVIOUS-ZH", "SENTINEL-TRANSLATOR-TEXT", "SENTINEL-REASONING-1"]
    assert all(s in translator_sent for s in sentinels)  # the Translator's own requests do carry them

    reviewer_provider = Provider(answer(APPROVE))
    result, _ = review(page, reviewer_provider, flags=made.flags, translations=made.translations)

    assert result.ok
    sent = json.dumps(reviewer_provider.bodies, ensure_ascii=False)
    for forbidden in (*sentinels, "SENTINEL-REASONING-2", "get_previous_context", "flag_uncertain", "tool_call",
                      "REF-ZH", "REF-EN"):
        assert forbidden not in sent, forbidden
    assert "FLAG-REASON" in sent and "你好" in sent  # the translations and the flags are what it gets
    assert [m["role"] for m in reviewer_provider.bodies[0]["messages"]] == ["system", "user"]


# --- AC4: the verdict ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, verdict, issues", [
    (APPROVE, "approve", []),
    (verdict_json("approve", issue(1, "可以更口语", None)), "approve", [{"id": 1, "comment": "可以更口语", "suggestion": None}]),
    (verdict_json("revise", issue(0, "漏译", "你好呀"), issue(1)), "revise",
     [{"id": 0, "comment": "漏译", "suggestion": "你好呀"}, {"id": 1, "comment": "译得不对", "suggestion": "更好的译文"}]),
    (json.dumps({"verdict": "approve"}), "approve", []),  # no "issues" at all is an empty list
    ('```json\n' + APPROVE + '\n```', "approve", []),  # a fence, as parse_translations tolerates
    (verdict_json("revise", issue(0, " 译得不对 ", "  ")), "revise",
     [{"id": 0, "comment": "译得不对", "suggestion": None}]),  # an empty suggestion is no suggestion
], ids=["approve", "approve-with-notes", "revise", "approve-without-issues-key", "fenced", "stripped"])
def test_a_valid_verdict_is_read_with_one_request_and_no_repair(page, text, verdict, issues):
    provider = Provider(answer(text))

    result, _ = review(page, provider)

    assert (result.status, result.verdict, result.issues, result.error) == ("ok", verdict, issues, None)
    assert result.ok and not result.unusable and not result.failed
    assert len(provider.bodies) == 1 and result.attempts["repair"] is None
    assert result.attempts["agent"]["stop_reason"] == "final" and result.attempts["agent"]["answer_parse_error"] is None
    assert set(result.attempts) == {"agent", "repair"}
    assert set(result.attempts["agent"]) == {"stop_reason", "error", "steps", "answer_parse_error"}


INVALID = [
    (verdict_json("maybe"), '"verdict" must be "approve" or "revise", got \'maybe\''),
    (verdict_json("revise"), 'the verdict "revise" needs at least one issue'),
    (verdict_json("revise", issue(7)), "issues for region ids not on the page [7]"),
    (verdict_json("approve", issue(0, "  ")), "the issue for region 0 has an empty comment"),
    (verdict_json("revise", issue(0), issue(0, "また", None)), "more than one issue for region ids [0]"),
    ("not json", "no JSON object in the response"),
    ("", "no JSON object in the response"),
    ("[1, 2]", "no JSON object in the response"),
    (json.dumps({"issues": []}), '"verdict" must be "approve" or "revise", got None'),
    (json.dumps({"verdict": "approve", "issues": "none"}), '"issues" must be a list'),
    (json.dumps({"verdict": "revise", "issues": [{"id": "0", "comment": "x"}]}), "issue 0 has no integer region id"),
    (json.dumps({"verdict": "revise", "issues": [{"id": True, "comment": "x"}]}), "issue 0 has no integer region id"),
    (json.dumps({"verdict": "revise", "issues": ["x"]}), "issue 0 is not an object"),
    (json.dumps({"verdict": "revise", "issues": [{"id": 0, "comment": "x", "suggestion": 5}]}),
     "the suggestion for region 0 must be text or null"),
]
INVALID_IDS = ["unknown-verdict", "revise-without-issues", "region-not-on-page", "empty-comment", "region-twice",
               "no-json", "empty", "array", "no-verdict", "issues-not-a-list", "id-is-a-string", "id-is-a-bool",
               "issue-is-not-an-object", "suggestion-is-a-number"]


@pytest.mark.parametrize("bad, why", INVALID, ids=INVALID_IDS)
def test_an_invalid_verdict_gets_exactly_one_repair_request_with_json_output_and_no_tools(page, bad, why):
    provider = Provider(answer(bad), answer(verdict_json("revise", issue(1))))

    result, meter = review(page, provider)

    assert len(provider.bodies) == 2
    repair = provider.bodies[1]
    assert repair["response_format"] == {"type": "json_object"} and "tools" not in repair
    assert [m["role"] for m in repair["messages"]] == ["system", "user", "assistant", "user"]  # the history, then the error
    assert repair["messages"][:3] == [*provider.bodies[0]["messages"], {"role": "assistant", "content": bad}]
    assert "could not be used" in repair["messages"][3]["content"] and why in repair["messages"][3]["content"]
    assert "[0, 1]" in repair["messages"][3]["content"]  # the region ids of the page
    assert (result.status, result.verdict, result.issues) == ("ok", "revise", [
        {"id": 1, "comment": "译得不对", "suggestion": "更好的译文"}])
    assert result.attempts["agent"]["answer_parse_error"] == why
    assert set(result.attempts["repair"]) == {"error", "record", "raw", "parse_error"}
    assert result.attempts["repair"]["raw"] == verdict_json("revise", issue(1))
    assert (result.attempts["repair"]["error"], result.attempts["repair"]["parse_error"]) == (None, None)
    assert result.attempts["repair"]["record"] == asdict(meter.records[1])


def test_a_repair_that_does_not_parse_either_gives_an_unusable_review_that_says_why_and_sends_nothing_more(page):
    provider = Provider(answer(verdict_json("maybe")), answer(verdict_json("revise")), answer(APPROVE))

    result, meter = review(page, provider)

    assert (result.status, result.verdict, result.issues) == ("unusable", None, [])
    assert result.unusable and not result.ok and not result.failed
    assert result.error.startswith("unparseable: the answer (") and "got 'maybe'" in result.error
    assert "needs at least one issue" in result.error  # and the repair's
    assert len(provider.bodies) == 2 and len(meter.records) == 2  # the third reply was never asked for
    assert "needs at least one issue" in result.attempts["repair"]["parse_error"]
    assert result.attempts["repair"]["raw"] == verdict_json("revise")


# --- AC5: unusable and failed reviews are returned, never raised ---------------------------------------------------

def test_the_step_limit_is_an_unusable_review_without_a_repair(page):
    wander = ask(call("c1", {"x": 1}, name="lookup_term"))  # a tool the Reviewer does not have
    provider = Provider(wander, wander, answer(APPROVE))

    result, _ = review(page, provider, max_steps=2)

    assert (result.status, result.verdict, result.issues) == ("unusable", None, [])
    assert result.error.startswith("step_limit") and "2" in result.error
    assert result.attempts["agent"]["stop_reason"] == "step_limit" and result.attempts["repair"] is None
    assert len(provider.bodies) == 2
    assert "no tool named lookup_term" in provider.bodies[1]["messages"][-1]["content"]  # the loop's guardrail answered


def test_a_reply_cut_off_is_an_unusable_review_without_a_repair(page):
    provider = Provider(answer('{"verdict": "app', finish_reason="length"), answer(APPROVE))

    result, _ = review(page, provider)

    assert (result.status, result.verdict) == ("unusable", None)
    assert result.error.startswith("incomplete") and "length" in result.error
    assert result.attempts["agent"]["stop_reason"] == "incomplete" and result.attempts["repair"] is None
    assert len(provider.bodies) == 1


def test_a_provider_error_is_a_failed_review_and_nothing_is_resent(page):
    provider = Provider(upstream_500(), answer(APPROVE))

    result, meter = review(page, provider)

    assert (result.status, result.verdict, result.issues) == ("failed", None, [])
    assert result.failed and not result.ok and not result.unusable
    assert "API error 500" in result.error
    assert len(provider.bodies) == 1 and meter.records[0].error == result.error and meter.records[0].usage is None
    assert result.attempts["agent"]["stop_reason"] == "provider_error"
    assert result.attempts["agent"]["steps"][0]["error"] == result.error and result.attempts["repair"] is None


def test_a_provider_error_on_the_repair_is_a_failed_review_and_nothing_is_resent(page):
    provider = Provider(answer("not json"), upstream_500(), answer(APPROVE))

    result, meter = review(page, provider)

    assert (result.status, result.verdict) == ("failed", None) and "API error 500" in result.error
    assert len(provider.bodies) == 2
    assert result.attempts["repair"]["error"] == result.error
    assert result.attempts["repair"]["record"] == asdict(meter.records[1]) and result.attempts["repair"]["raw"] is None


def test_a_spent_budget_is_a_failed_review_and_nothing_is_resent(page):
    provider = Provider(ask(call("c1", {"x": 1}, name="lookup_term")), answer(APPROVE))

    result, _ = review(page, provider, review_budget=100)  # the first call uses 120

    assert (result.status, result.verdict) == ("failed", None)
    assert result.error.startswith("Stopped:") and "used 120 of 100" in result.error
    assert result.attempts["agent"]["stop_reason"] == "budget" and len(provider.bodies) == 1


def test_a_budget_spent_before_the_repair_is_a_failed_review_and_the_repair_is_not_sent(page):
    provider = Provider(answer("not json"), answer(APPROVE))

    result, _ = review(page, provider, review_budget=120)  # the answer used all of it

    assert (result.status, result.verdict) == ("failed", None) and result.error.startswith("Stopped:")
    assert len(provider.bodies) == 1
    assert result.attempts["agent"]["answer_parse_error"] and result.attempts["repair"] is None


# --- the Reviewer's own budget and its inputs ----------------------------------------------------------------------

def test_the_review_has_a_budget_of_its_own(page):
    assert inspect.signature(reviewer.run_reviewer).parameters["review_budget"].default == 30_000
    assert reviewer.DEFAULT_REVIEW_BUDGET == 30_000
    provider = Provider(answer("not json"), answer(APPROVE))

    result, _ = review(page, provider, review_budget=240)  # exactly the two calls

    assert result.ok and len(provider.bodies) == 2


@pytest.mark.parametrize("translations, flags", [
    ({0: "你好"}, []),
    ({0: "你好", 1: "再见", 2: "多余"}, []),
    (TRANSLATIONS, [{"id": 5, "reason": "not a region of this page"}]),
], ids=["missing-region", "extra-region", "flag-on-an-unknown-region"])
def test_inputs_that_do_not_fit_the_page_are_a_value_error_before_any_request(page, translations, flags):
    provider = Provider(answer(APPROVE))

    with pytest.raises(ValueError):
        reviewer.run_reviewer(page, translations, flags, mock_llm_client(provider))

    assert provider.bodies == []


def test_the_reviewer_module_imports_no_langchain():
    tree = ast.parse(inspect.getsource(reviewer))
    modules = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]

    assert not [m for m in modules if m.split(".")[0] in ("langchain", "langgraph", "langchain_core", "openai")]
