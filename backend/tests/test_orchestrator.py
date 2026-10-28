"""The Orchestrator (`full_no_kb`) through `runs.translate_page` and the eval harness, on a scripted provider: no network.

The real LLMClient, the real openai SDK and LangChain's create_agent run; only the provider's replies are scripted.
Requests are told apart by their system prompt: the Reviewer's, the flagging Translator's, `baseline`'s.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from conftest import FakeClient, mock_llm_client
from test_agent_loop import Provider, answer, ask, call, upstream_500
from test_evaluation import run_cli
from test_reviewer import APPROVE, REVISE, issue, verdict_json
from test_tracing import one_trace, traced  # noqa: F401 (fixture)
from test_translator import (  # noqa: F401 (fixtures)
    BOOK, CALL_COST, attempts_of, evaluate, flag, imported, page_for, previous_context, root, tool_result, translate,
    zh, zh_json, zh_of,
)

from fukidashi import config as app_config
from fukidashi import orchestrator, runs, store
from fukidashi.agents import reviewer, translator
from fukidashi.evaluation import harness
from fukidashi.llm.metering import Meter
from fukidashi.pipeline import baseline

CONFIG = "full_no_kb"
REVISION_MARK = "sent it back for revision"  # in the text part that only a revision's user message has

# Two Issues, then one other: the second round's Issues must be the ones its own review gave.
REVISE_TWO = verdict_json("revise", issue(0, "第一句译得生硬", "更自然的第一句"), issue(1, "漏译了语气词", None))
REVISE_AGAIN = verdict_json("revise", issue(1, "还是漏了语气词", "带语气词的第二句"))
TWO_PAGES_BACK = call("call_3", {"pages_back": 2}, name="get_previous_context")


def run_page(conn, root, provider, page_index=3, **options):
    """`full_no_kb` on one page; the Meter is returned so that a test can compare it with the requests sent."""
    meter = Meter()
    outcome = translate(conn, root, mock_llm_client(provider, meter=meter), page_index, config=CONFIG, **options)
    return outcome, meter


def who(body):
    """Which stage a request belongs to. A Translator's follow-up and its Repair belong to its stage too."""
    system, user = body["messages"][0]["content"], body["messages"][1]["content"]
    if system == reviewer.SYSTEM_PROMPT:
        return "review"
    if system == baseline.SYSTEM_PROMPT:
        return "baseline"
    return "revise" if any(REVISION_MARK in part.get("text", "") for part in user) else "translate"


def stages_of(conn, run_id):
    return attempts_of(conn, run_id)["stages"]


def sent(bodies):
    return json.dumps(bodies, ensure_ascii=False)


def count_translations(conn, run_id):
    return conn.execute("SELECT COUNT(*) FROM translations WHERE run_id = ?", (run_id,)).fetchone()[0]


# --- AC1: translate, review, approve -------------------------------------------------------------------------------

def test_an_approved_page_is_one_succeeded_full_no_kb_run_with_a_translate_and_a_review_stage(imported, root, page_for):
    provider = Provider(answer(zh_json(3)), answer(APPROVE))

    outcome, meter = run_page(imported, root, provider)

    assert outcome.result.ok and outcome.result.fallback_reason is None
    run = store.get_run(imported, outcome.run_id)
    assert (run["config"], run["status"], run["error"]) == ("full_no_kb", "succeeded", None)
    assert zh_of(imported, outcome.run_id) == [zh(3, 0), zh(3, 1)]  # every segment has a Translation
    assert run["calls"] == len(provider.bodies) == len(meter.records) == 2  # the Run's calls are the requests sent
    assert run["cost_usd"] == pytest.approx(2 * CALL_COST) and run["unknown_usage_calls"] == 0
    attempts = attempts_of(imported, outcome.run_id)
    assert set(attempts) == {"config", "stages", "send_backs", "final"} and attempts["config"] == "full_no_kb"
    assert [(s["kind"], s["status"]) for s in attempts["stages"]] == [("translate", "ok"), ("review", "ok")]
    assert attempts["send_backs"] == 0
    assert attempts["final"] == {"stage": 0, "kind": "translate", "reviewed": True, "stop": "approved"}
    assert (outcome.result.send_backs, outcome.result.final_stop) == (0, "approved")

    first = provider.bodies[0]
    assert first["messages"][0] == {"role": "system", "content": translator.FLAG_SYSTEM_PROMPT}
    assert [t["function"]["name"] for t in first["tools"]] == ["get_previous_context", "flag_uncertain"]
    assert first["messages"][1] == baseline.build_messages(page_for(3))[1]  # the user message baseline sends
    assert [who(b) for b in provider.bodies] == ["translate", "review"]


def test_each_stage_records_its_status_error_and_the_agents_own_attempts(imported, root):
    provider = Provider(ask(flag("f", 1, "初译不确定")), answer(zh_json(3)), answer(REVISE_TWO),
                        answer(zh_json(3, "改")), answer(APPROVE))

    outcome, _ = run_page(imported, root, provider)
    translate_stage, review_stage, *rest = stages_of(imported, outcome.run_id)

    assert outcome.result.ok
    assert set(translate_stage) == {"kind", "status", "error", "flags", "attempts"}
    assert translate_stage["flags"] == [{"id": 1, "reason": "初译不确定"}] and translate_stage["error"] is None
    assert set(translate_stage["attempts"]) == {"config", "agent", "repair", "fallback"}  # the Translator's own shape
    assert translate_stage["attempts"]["agent"]["steps"][0]["tool_calls"][0]["name"] == "flag_uncertain"
    assert set(review_stage) == {"kind", "status", "error", "verdict", "issues", "attempts"}
    assert review_stage["verdict"] == "revise" and review_stage["error"] is None
    assert review_stage["issues"] == [   # as the Reviewer returned them
        {"id": 0, "comment": "第一句译得生硬", "suggestion": "更自然的第一句"},
        {"id": 1, "comment": "漏译了语气词", "suggestion": None}]
    assert set(review_stage["attempts"]) == {"agent", "repair"}  # the Reviewer's own shape
    assert [s["kind"] for s in rest] == ["revise", "review"]


def test_the_run_records_every_budget_and_the_send_back_limit(imported, root):
    provider = Provider(answer(zh_json(1)), answer(APPROVE))

    outcome, _ = run_page(imported, root, provider, 1, page_budget=567, agent_budget=1234, review_budget=890,
                          max_steps=3, max_tokens=99)

    assert json.loads(store.get_run(imported, outcome.run_id)["config_json"]) == {
        "agent_budget": 1234, "review_budget": 890, "fallback_budget": 567, "max_steps": 3, "max_tokens": 99,
        "max_send_backs": 2}
    assert provider.bodies[0]["max_tokens"] == provider.bodies[1]["max_tokens"] == 99


def test_the_knowledge_base_is_off_and_cannot_be_switched_on_yet(page_for):
    provider = Provider(answer(zh_json(3)))

    with pytest.raises(ValueError, match="Knowledge Base"):
        orchestrator.run_orchestrator(page_for(3), mock_llm_client(provider), use_kb=True)

    assert provider.bodies == []


def test_the_earlier_pages_are_the_ones_of_the_same_config(imported, root):
    for page_index in (1, 2):
        assert run_page(imported, root, Provider(answer(zh_json(page_index)), answer(APPROVE)), page_index)[0].result.ok
    assert translate(imported, root, mock_llm_client(Provider(answer(zh_json(2, "别")))), 2).result.ok  # `translator`
    provider = Provider(ask(TWO_PAGES_BACK), answer(zh_json(3)), answer(APPROVE))

    outcome, _ = run_page(imported, root, provider)

    assert outcome.result.ok
    result = json.loads(tool_result(provider.bodies[1], "call_3"))
    assert [p["translated"] for p in result["pages"]] == [True, True]
    assert result["pages"][1]["regions"][0]["zh"] == zh(2, 0)  # not the later `translator` Run's "译文2-0别"


# --- AC2: send-backs ------------------------------------------------------------------------------------------------

def test_a_revise_verdict_sends_the_page_back_once_and_the_revision_is_what_is_stored(imported, root, page_for):
    provider = Provider(
        ask(flag("f", 1, "初译不确定")), answer(zh_json(3)), answer(REVISE_TWO),
        ask(flag("r", 0, "改后仍不确定")), answer(zh_json(3, "改")), answer(APPROVE))

    outcome, meter = run_page(imported, root, provider)

    assert outcome.result.ok
    assert [who(b) for b in provider.bodies] == ["translate", "translate", "review", "revise", "revise", "review"]
    run = store.get_run(imported, outcome.run_id)
    assert zh_of(imported, outcome.run_id) == [zh(3, 0, "改"), zh(3, 1, "改")]
    assert run["calls"] == len(provider.bodies) == len(meter.records) == 6
    attempts = attempts_of(imported, outcome.run_id)
    assert [s["kind"] for s in attempts["stages"]] == ["translate", "review", "revise", "review"]
    assert attempts["send_backs"] == 1
    assert attempts["final"] == {"stage": 2, "kind": "revise", "reviewed": True, "stop": "approved"}
    assert attempts["stages"][2]["flags"] == [{"id": 0, "reason": "改后仍不确定"}]

    revision = provider.bodies[3]  # the revision's first request
    assert revision["messages"][0] == {"role": "system", "content": translator.FLAG_SYSTEM_PROMPT}
    assert [t["function"]["name"] for t in revision["tools"]] == ["get_previous_context", "flag_uncertain"]
    user = revision["messages"][1]["content"]
    assert user[:2] == baseline.build_messages(page_for(3))[1]["content"]  # the page, as baseline sends it
    assert [part["type"] for part in user] == ["image_url", "text", "text"]
    text = user[2]["text"]
    assert f'{{"id": 0, "zh": "{zh(3, 0)}"}}' in text and f'{{"id": 1, "zh": "{zh(3, 1)}"}}' in text  # current translation
    assert '{"id": 0, "comment": "第一句译得生硬", "suggestion": "更自然的第一句"}' in text  # every Issue
    assert '{"id": 1, "comment": "漏译了语气词", "suggestion": null}' in text
    assert "Simplified Chinese" in text

    # the second review reads the revision, with the flags of the run that made it
    second_review = provider.bodies[5]["messages"][1]["content"][1]["text"]
    assert f'"zh": "{zh(3, 0, "改")}"' in second_review and f'"zh": "{zh(3, 0)}"' not in second_review
    assert "改后仍不确定" in second_review and "初译不确定" not in second_review


def test_a_page_sent_back_twice_is_not_reviewed_a_third_time_and_keeps_the_second_revision(imported, root):
    provider = Provider(
        answer(zh_json(3)), answer(REVISE_TWO), answer(zh_json(3, "改")), answer(REVISE_AGAIN),
        answer(zh_json(3, "改改")), answer(APPROVE))  # the last reply must stay unused

    outcome, meter = run_page(imported, root, provider)

    assert outcome.result.ok and len(provider.bodies) == 5  # no third review request
    assert [who(b) for b in provider.bodies] == ["translate", "review", "revise", "review", "revise"]
    run = store.get_run(imported, outcome.run_id)
    assert zh_of(imported, outcome.run_id) == [zh(3, 0, "改改"), zh(3, 1, "改改")]
    assert run["status"] == "succeeded" and run["calls"] == len(meter.records) == 5
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["send_backs"] == 2
    assert [s["kind"] for s in attempts["stages"]] == ["translate", "review", "revise", "review", "revise"]
    assert attempts["final"] == {"stage": 4, "kind": "revise", "reviewed": False, "stop": "send_back_limit"}
    assert (outcome.result.send_backs, outcome.result.final_stop) == (2, "send_back_limit")
    second_revision = provider.bodies[4]["messages"][1]["content"][2]["text"]
    assert zh(3, 0, "改") in second_revision and "还是漏了语气词" in second_revision  # round two's translation and Issues
    assert "第一句译得生硬" not in second_revision


# --- AC3: an unusable review or revision ends the loop with the translation it has -----------------------------------

@pytest.mark.parametrize("replies", [
    (answer("not json"), answer("still not json")),
    (answer("", finish_reason="length"),),
], ids=["unparseable-after-the-repair", "reply-cut-off"])
def test_an_unusable_review_stores_the_current_translation_and_sends_nothing_more(imported, root, replies):
    provider = Provider(answer(zh_json(3)), *replies, answer(zh_json(3, "改")))

    outcome, meter = run_page(imported, root, provider)

    assert outcome.result.ok and len(provider.bodies) == 1 + len(replies) == len(meter.records)  # no revision, no review
    assert zh_of(imported, outcome.run_id) == [zh(3, 0), zh(3, 1)]
    attempts = attempts_of(imported, outcome.run_id)
    assert [(s["kind"], s["status"]) for s in attempts["stages"]] == [("translate", "ok"), ("review", "unusable")]
    assert attempts["stages"][1]["verdict"] is None and attempts["stages"][1]["issues"] == []
    assert attempts["stages"][1]["error"]
    assert attempts["send_backs"] == 0
    assert attempts["final"] == {"stage": 0, "kind": "translate", "reviewed": False, "stop": "review_unusable"}
    assert store.get_run(imported, outcome.run_id)["status"] == "succeeded"


def test_an_unusable_second_review_keeps_the_revision(imported, root):
    provider = Provider(answer(zh_json(3)), answer(REVISE_TWO), answer(zh_json(3, "改")),
                        answer("", finish_reason="length"), answer(zh_json(3, "改改")))

    outcome, _ = run_page(imported, root, provider)

    assert outcome.result.ok and len(provider.bodies) == 4
    assert zh_of(imported, outcome.run_id) == [zh(3, 0, "改"), zh(3, 1, "改")]
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["send_backs"] == 1
    assert attempts["final"] == {"stage": 2, "kind": "revise", "reviewed": False, "stop": "review_unusable"}


UNUSABLE_REVISIONS = [   # the revision's replies, the options of the run, and the requests the revision sends
    ((ask(flag("a", 0)), ask(flag("b", 1))), {"max_steps": 2}, "step_limit"),
    ((answer("", finish_reason="length"),), {}, "incomplete"),
    ((answer("not json"), answer("still not json")), {}, "unparseable"),
]


@pytest.mark.parametrize("replies, options, why", UNUSABLE_REVISIONS, ids=[u[2] for u in UNUSABLE_REVISIONS])
def test_an_unusable_revision_keeps_the_previous_translation_without_a_fallback_or_another_review(
        imported, root, replies, options, why):
    provider = Provider(answer(zh_json(3)), answer(REVISE_TWO), *replies, answer(zh_json(3, "改")), answer(APPROVE))

    outcome, meter = run_page(imported, root, provider, **options)

    sent_by = [who(b) for b in provider.bodies]
    assert outcome.result.ok and len(provider.bodies) == 2 + len(replies) == len(meter.records)
    assert "baseline" not in sent_by and sent_by[2:] == ["revise"] * len(replies)  # no fallback; no review after it
    assert zh_of(imported, outcome.run_id) == [zh(3, 0), zh(3, 1)]  # the translation the Reviewer sent back
    attempts = attempts_of(imported, outcome.run_id)
    assert [(s["kind"], s["status"]) for s in attempts["stages"]] == [
        ("translate", "ok"), ("review", "ok"), ("revise", "unusable")]
    revise_stage = attempts["stages"][2]
    assert revise_stage["error"].startswith(why) and revise_stage["flags"] == []
    assert revise_stage["attempts"]["fallback"] is None
    assert attempts["send_backs"] == 1
    assert attempts["final"] == {"stage": 0, "kind": "translate", "reviewed": True, "stop": "revision_unusable"}


def test_an_unusable_second_revision_keeps_the_first_revision(imported, root):
    provider = Provider(answer(zh_json(3)), answer(REVISE_TWO), answer(zh_json(3, "改")), answer(REVISE_AGAIN),
                        answer("", finish_reason="length"), answer(zh_json(3, "改改")))

    outcome, _ = run_page(imported, root, provider)

    assert outcome.result.ok and len(provider.bodies) == 5
    assert zh_of(imported, outcome.run_id) == [zh(3, 0, "改"), zh(3, 1, "改")]
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["send_backs"] == 2
    assert attempts["final"] == {"stage": 2, "kind": "revise", "reviewed": True, "stop": "revision_unusable"}


def test_a_revision_repaired_by_its_one_repair_call_is_usable(imported, root):
    provider = Provider(answer(zh_json(3)), answer(REVISE_TWO), answer("not json"), answer(zh_json(3, "改")),
                        answer(APPROVE))

    outcome, _ = run_page(imported, root, provider)

    assert [who(b) for b in provider.bodies] == ["translate", "review", "revise", "revise", "review"]
    assert "tools" not in provider.bodies[3] and provider.bodies[3]["response_format"] == {"type": "json_object"}
    assert zh_of(imported, outcome.run_id) == [zh(3, 0, "改"), zh(3, 1, "改")]
    stages = stages_of(imported, outcome.run_id)
    assert stages[2]["status"] == "ok" and stages[2]["attempts"]["repair"]["parse_error"] is None


# --- AC4: a provider error or a spent budget in any stage fails the Run ---------------------------------------------

FAILURES = [   # replies, options, requests sent, the failing stage, why
    ([upstream_500()], {}, 1, "translate", "provider"),
    ([answer(zh_json(3)), upstream_500()], {}, 2, "review", "provider"),
    ([answer(zh_json(3)), answer(REVISE_TWO), upstream_500()], {}, 3, "revise", "provider"),
    ([answer("not json"), answer(zh_json(3))], {"agent_budget": 120}, 1, "translate", "budget"),
    ([answer(zh_json(3)), answer("not json"), answer(APPROVE)], {"review_budget": 120}, 2, "review", "budget"),
    ([answer(zh_json(3)), answer(REVISE_TWO), answer("not json"), answer(zh_json(3, "改"))],
     {"agent_budget": 120}, 3, "revise", "budget"),   # each Translator run has a budget of its own: only this one is spent
]


@pytest.mark.parametrize("replies, options, requests, stage, why", FAILURES,
                         ids=[f"{f[3]}-{f[4]}" for f in FAILURES])
def test_a_provider_error_or_a_spent_budget_fails_the_run_with_that_stages_error_and_sends_nothing_more(
        imported, root, replies, options, requests, stage, why):
    provider = Provider(*replies)

    outcome, meter = run_page(imported, root, provider, **options)

    assert not outcome.result.ok and outcome.result.translations is None
    assert len(provider.bodies) == len(meter.records) == requests  # nothing followed, and no fallback
    assert "baseline" not in [who(b) for b in provider.bodies]
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"]) == ("failed", requests)
    assert run["error"] == outcome.result.error == stages_of(imported, outcome.run_id)[-1]["error"]
    assert ("API error 500" if why == "provider" else "Stopped:") in run["error"]
    if why == "provider":  # the failed request is a call whose usage is unknown, never 0
        assert run["unknown_usage_calls"] == 1 and run["cost_usd"] is None
    else:
        assert run["unknown_usage_calls"] == 0 and run["cost_usd"] == pytest.approx(requests * CALL_COST)
    assert count_translations(imported, outcome.run_id) == 0  # the earlier stages' translations are not stored
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["stages"][-1]["kind"] == stage and attempts["stages"][-1]["status"] == "failed"
    assert attempts["final"] == {"stage": None, "kind": None, "reviewed": False, "stop": "failed"}
    assert outcome.result.final_stop == "failed"


def test_a_page_that_failed_does_not_become_the_page_shown_or_the_next_pages_context(imported, root):
    assert not run_page(imported, root, Provider(answer(zh_json(2)), upstream_500()), 2)[0].result.ok
    provider = Provider(ask(TWO_PAGES_BACK), answer(zh_json(3)), answer(APPROVE))

    run_page(imported, root, provider)

    result = json.loads(tool_result(provider.bodies[1], "call_3"))
    assert [p["translated"] for p in result["pages"]] == [False, False]  # page 2's failed Run is not context


# --- AC5: the first pass may fall back, and the fallback is reviewed ------------------------------------------------

def test_the_first_pass_falls_back_to_baseline_and_the_fallbacks_translation_is_reviewed(imported, root, tmp_path):
    provider = Provider(ask(flag("f", 1, "SENTINEL-FLAG")), answer("", finish_reason="length"),
                        answer(zh_json(1, "底")), answer(APPROVE))

    result = evaluate(imported, root, tmp_path, provider, [1], config=CONFIG)

    assert [who(b) for b in provider.bodies] == ["translate", "translate", "baseline", "review"]
    review_text = provider.bodies[3]["messages"][1]["content"][1]["text"]
    assert zh(1, 0, "底") in review_text and zh(1, 1, "底") in review_text   # the fallback's translation is what it reads
    assert "SENTINEL-FLAG" not in review_text and "flagged no region" in review_text  # flags go with the agent's answer
    (row,) = result.scores["pages"]
    assert row["fell_back"] is True and row["fallback_reason"].startswith("incomplete")
    assert (row["final_stop"], row["send_backs"], row["calls"]) == ("approved", 0, 4)
    assert result.scores["fallback_pages"] == 1
    stages = stages_of(imported, row["run_id"])
    assert stages[0]["attempts"]["fallback"]["reason"].startswith("incomplete") and stages[0]["flags"] == []
    assert zh_of(imported, row["run_id"]) == [zh(1, 0, "底"), zh(1, 1, "底")]
    assert "FELL BACK  book_t/1" in harness.format_summary(result)


def test_a_first_pass_that_fell_back_and_failed_too_is_a_failed_page_marked_as_fallen_back(imported, root, tmp_path):
    provider = Provider(answer("", finish_reason="length"), answer("x"), answer("y"))

    result = evaluate(imported, root, tmp_path, provider, [1], config=CONFIG)

    assert len(provider.bodies) == 3 and result.exit_code == 1  # no review was sent
    (failed,) = result.scores["failed_pages"]
    assert failed["fell_back"] is True and failed["fallback_reason"].startswith("incomplete")
    assert result.scores["pages"] == [] and result.scores["fallback_pages"] == 0


# --- AC6: independence, over two rounds -----------------------------------------------------------------------------

def test_nothing_of_the_translators_runs_or_the_references_reaches_a_review_in_either_round(imported, root):
    for page_index in (1, 2):   # the earlier pages the Translator's tool returns: "译文2-…" and "日本語2-…"
        assert run_page(imported, root, Provider(answer(zh_json(page_index)), answer(APPROVE)), page_index)[0].result.ok
    provider = Provider(
        ask(previous_context(3), flag("f", 1, "FLAG-ONE"), content="SENTINEL-TEXT-1", reasoning_content="SENTINEL-REASON-1"),
        answer(zh_json(3)),
        answer(REVISE_TWO),
        ask(previous_context(3), flag("g", 0, "FLAG-TWO"), content="SENTINEL-TEXT-2", reasoning_content="SENTINEL-REASON-2"),
        answer(zh_json(3, "改")),
        answer(REVISE_AGAIN),
        answer(zh_json(3, "改改")),
    )

    outcome, _ = run_page(imported, root, provider)

    assert outcome.result.ok and outcome.result.final_stop == "send_back_limit"
    assert [who(b) for b in provider.bodies] == [
        "translate", "translate", "review", "revise", "revise", "review", "revise"]
    translator_sent = sent([b for b in provider.bodies if who(b) != "review"])
    reviewer_bodies = [b for b in provider.bodies if who(b) == "review"]
    assert len(reviewer_bodies) == 2
    translator_only = ["SENTINEL-TEXT-1", "SENTINEL-REASON-1", "SENTINEL-TEXT-2", "SENTINEL-REASON-2", "译文2-", "日本語2-"]
    assert all(s in translator_sent for s in translator_only)  # the Translator's own requests do carry them
    reviewer_sent = sent(reviewer_bodies)
    for forbidden in (*translator_only, "get_previous_context", "flag_uncertain", "tool_call", "REF-ZH", "REF-EN"):
        assert forbidden not in reviewer_sent, forbidden
    for body, flag_reason, other in zip(reviewer_bodies, ("FLAG-ONE", "FLAG-TWO"), ("FLAG-TWO", "FLAG-ONE")):
        text = body["messages"][1]["content"][1]["text"]   # each review gets the flags of the run it reads
        assert flag_reason in text and other not in text
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "REF-ZH" not in sent(provider.bodies) and "REF-EN" not in sent(provider.bodies)  # not in any request
    assert "第一句译得生硬" in translator_sent   # the Issues do reach the Translator's revision: that is their purpose


def test_the_reviewers_comments_are_asked_for_in_simplified_chinese():
    assert "every comment and every suggestion in Simplified Chinese" in reviewer.SYSTEM_PROMPT


# --- AC7: a revision that does not fit the page ----------------------------------------------------------------------

@pytest.mark.parametrize("revision", [
    translator.Revision({0: "只有一句"}, []),
    translator.Revision({0: "甲", 1: "乙", 2: "丙"}, []),
    translator.Revision({0: "甲", 1: "乙"}, [{"id": 5, "comment": "没有这个区域", "suggestion": None}]),
], ids=["a-region-missing", "a-region-too-many", "an-issue-for-an-unknown-region"])
def test_a_revision_that_does_not_fit_the_page_is_refused_before_any_request(page_for, revision):
    provider = Provider(answer(zh_json(3)))

    with pytest.raises(ValueError):
        translator.run_translator(page_for(3), mock_llm_client(provider), revision=revision)

    assert provider.bodies == []


def test_a_first_pass_is_never_unusable_and_a_revision_result_says_which_it_is(page_for):
    revision = translator.Revision({0: "甲", 1: "乙"}, [{"id": 0, "comment": "不对", "suggestion": None}])
    cut_off = Provider(answer("", finish_reason="length"), answer(zh_json(3)))

    first_pass = translator.run_translator(page_for(3), mock_llm_client(cut_off))
    revised = translator.run_translator(page_for(3), mock_llm_client(Provider(answer("", finish_reason="length"))),
                                        revision=revision)
    failed = translator.run_translator(page_for(3), mock_llm_client(Provider(upstream_500())), revision=revision)

    assert first_pass.ok and first_pass.fallback_reason.startswith("incomplete") and first_pass.unusable is False
    assert (revised.ok, revised.unusable, revised.translations) == (False, True, None)
    assert revised.error.startswith("incomplete") and revised.fallback_reason is None
    assert (failed.ok, failed.unusable) == (False, False)  # a provider error is a failure, not an unusable revision


# --- AC8: the eval harness --------------------------------------------------------------------------------------------

OLD_PAGE_KEYS = {"book", "page_index", "run_id", "trace_id", "segments", "calls", "unknown_usage_calls", "fell_back",
                 "fallback_reason", "cost_usd", "latency_s"}
OLD_SCORE_KEYS = {"config", "book", "pages_scored", "segments_scored", "fallback_pages", "chrf", "bleu", "cost",
                  "latency", "pages", "failed_pages", "skipped_pages"}
LOOP_KEYS = {"send_backs", "final_stop", "final_reviewed", "review_statuses", "review_verdicts"}


def three_pages_provider():
    return Provider(
        answer(zh_json(1)), answer(APPROVE),                                                          # page 1: approved
        answer(zh_json(2)), answer(REVISE_TWO), answer(zh_json(2, "改")), answer(APPROVE),             # page 2: once
        answer(zh_json(3)), answer(REVISE_TWO), answer(zh_json(3, "改")), answer(REVISE_AGAIN),        # page 3: the limit
        answer(zh_json(3, "改改")),
    )


def test_the_harness_reports_the_review_loop_per_page_and_counts_the_pages_by_how_it_ended(imported, root, tmp_path):
    provider = three_pages_provider()

    result = evaluate(imported, root, tmp_path, provider, [1, 2, 3], config=CONFIG)

    assert result.exit_code == 0 and len(provider.bodies) == 11
    scores = json.loads((result.out_dir / "scores.json").read_text(encoding="utf-8"))
    assert set(scores) == OLD_SCORE_KEYS | {"final_stops"}
    assert scores["config"] == CONFIG and (scores["pages_scored"], scores["segments_scored"]) == (3, 6)
    assert scores["final_stops"] == {"approved": 2, "review_unusable": 0, "revision_unusable": 0, "send_back_limit": 1}
    pages = scores["pages"]
    assert all(set(p) == OLD_PAGE_KEYS | LOOP_KEYS for p in pages)
    assert [(p["page_index"], p["send_backs"], p["final_stop"], p["final_reviewed"]) for p in pages] == [
        (1, 0, "approved", True), (2, 1, "approved", True), (3, 2, "send_back_limit", False)]
    assert [p["review_statuses"] for p in pages] == [["ok"], ["ok", "ok"], ["ok", "ok"]]
    assert [p["review_verdicts"] for p in pages] == [["approve"], ["revise", "approve"], ["revise", "revise"]]
    assert [p["calls"] for p in pages] == [2, 4, 5] and all(p["fell_back"] is False for p in pages)
    assert scores["cost"]["total_usd"] == pytest.approx(11 * CALL_COST, abs=1e-6)
    assert [r["config"] for r in imported.execute("SELECT config FROM runs ORDER BY id")] == [CONFIG] * 3
    manifest = json.loads((result.out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_name"] == CONFIG
    assert manifest["config_params"] == {"agent_budget": 60_000, "review_budget": 30_000, "fallback_budget": 40_000,
                                         "max_steps": 6, "max_tokens": 16_000, "max_send_backs": 2}
    lines = (result.out_dir / "segments.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["hyp"] for line in lines][-2:] == [zh(3, 0, "改改"), zh(3, 1, "改改")]
    summary = harness.format_summary(result)
    assert "review loop  approved 2, review_unusable 0, revision_unusable 0, send_back_limit 1; send-backs 3" in summary
    assert "REVIEW LOOP  book_t/2 (run" in summary and "approved, 1 send-back(s), reviews revise/approve" in summary


def test_an_unusable_review_is_counted_and_a_failed_page_is_left_out_of_the_scores(imported, root, tmp_path):
    provider = Provider(
        answer(zh_json(1)), answer("", finish_reason="length"),    # page 1: the review is unusable
        answer(zh_json(2)), upstream_500(),                        # page 2: the review fails the page
    )

    result = evaluate(imported, root, tmp_path, provider, [1, 2], config=CONFIG)

    assert result.exit_code == 1 and len(provider.bodies) == 4
    (row,) = result.scores["pages"]
    assert (row["page_index"], row["final_stop"], row["final_reviewed"], row["review_statuses"]) == (
        1, "review_unusable", False, ["unusable"])
    assert result.scores["final_stops"]["review_unusable"] == 1 and sum(result.scores["final_stops"].values()) == 1
    (failed,) = result.scores["failed_pages"]
    assert failed["page_index"] == 2 and "API error 500" in failed["error"] and failed["fell_back"] is False
    assert failed["cost_usd"] is None   # the failed request's usage is unknown
    assert "FAILED  book_t/2" in harness.format_summary(result)


def test_the_other_configs_keep_every_key_they_have_today(imported, root, tmp_path):
    translated = evaluate(imported, root, tmp_path, Provider(answer(zh_json(1))), [1], config="translator")
    based = harness.run_eval(imported, config="baseline", book=BOOK, page_indexes=[2],
                             make_client=lambda: FakeClient([zh_json(2)]), dataset_root=root,
                             out_root=tmp_path / "out", repo_root=app_config.ROOT)

    for result in (translated, based):
        assert set(result.scores) == OLD_SCORE_KEYS
        assert all(set(p) == OLD_PAGE_KEYS for p in result.scores["pages"])
        assert "review loop" not in harness.format_summary(result)
    assert translated.scores["pages"][0]["fell_back"] is False and translated.scores["fallback_pages"] == 0
    assert based.scores["fallback_pages"] == 0


def test_the_harness_and_its_command_line_know_the_new_config(tmp_path):
    assert harness.CONFIGS == runs.CONFIGS == ("baseline", "translator", "full_no_kb")

    stopped = run_cli(tmp_path, "--config", "full_no_kb", "--book", "book_t", "--pages", "1")
    refused = run_cli(tmp_path, "--config", "full", "--book", "book_t", "--pages", "1")

    assert "invalid choice" not in stopped.stderr
    assert stopped.returncode == 2 and "Stopped before any model call" in stopped.stderr  # no such page: nothing sent
    assert refused.returncode == 2 and "invalid choice" in refused.stderr   # `full` waits for the KB


# --- the page's trace -------------------------------------------------------------------------------------------------

SCRIPTS = [   # the replies, the send-backs, and why the loop stopped
    ((answer(zh_json(2)), answer(APPROVE)), 0, "approved"),
    ((answer(zh_json(2)), answer(REVISE_TWO), answer(zh_json(2, "改")), answer(APPROVE)), 1, "approved"),
    ((answer(zh_json(2)), answer(REVISE_TWO), answer(zh_json(2, "改")), answer(REVISE_AGAIN), answer(zh_json(2, "改改"))),
     2, "send_back_limit"),
    ((answer(zh_json(2)), answer("", finish_reason="length")), 0, "review_unusable"),
    ((answer(zh_json(2)), upstream_500()), 0, "failed"),
]


@pytest.mark.parametrize("replies, send_backs, stop", SCRIPTS, ids=[s[2] + str(s[1]) for s in SCRIPTS])
def test_the_pages_root_span_says_how_often_it_was_sent_back_and_why_the_loop_stopped(
        imported, root, traced, replies, send_backs, stop):
    outcome, _ = run_page(imported, root, Provider(*replies), 2)

    top = one_trace(traced.spans())
    assert top.attributes["fukidashi.config"] == CONFIG
    assert top.attributes["fukidashi.send_backs"] == send_backs and top.attributes["fukidashi.final_stop"] == stop
    assert outcome.trace_id is not None and "fukidashi.fallback_reason" not in top.attributes


# --- AC10: the boundaries ---------------------------------------------------------------------------------------------

def test_the_orchestrator_imports_neither_langchain_nor_the_database():
    imported_names = set()
    for path in Path(orchestrator.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported_names |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):   # `from ..store import x` and `from .. import store`
                imported_names |= {(node.module or "").split(".")[0], *(a.name for a in node.names)}

    assert {"reviewer", "translator", "baseline"} <= imported_names   # the scan does see the imports
    assert not imported_names & {"langchain", "langchain_core", "langgraph", "openai", "sqlite3", "store", "db"}
