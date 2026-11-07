"""The live trace (task 035): the events a Run reports to an optional sink, on a scripted provider; no network.

The real LLMClient, the real openai SDK and LangChain's create_agent run; only the provider's replies are scripted. Every
sink here is a `Recorder` (live_trace_support.py), which holds the contract on every event it gets. The checks, in order:
the agent loop's own events, the kinds of each config, what the events say against what the Run stored, that they are
live, what they leave out, the thread they come in, that the hook only observes, and what happens when a sink raises.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import request_scenarios
from conftest import CrashAfterCallClient, mock_llm_client
from kb_support import new_kb
from live_trace_support import VIOLATIONS, Recorder, assert_well_formed, check_event, checked, without_latency
from test_agent_loop import Args, Provider, answer, ask, call, make_tool, run, upstream_500
from test_curator_agent import curate, decision, decisions_json
from test_curator_agent import page as curator_page  # noqa: F401 (fixture)
from test_curator_agent import proposals as curator_proposals  # noqa: F401 (fixture)
from test_full import NAME, SEEN_TERM, kb, proposals_reply, run_page, seed_entry, terms_of  # noqa: F401 (fixture: kb)
from test_kb_seams import DIGESTS, database_loaded_by, digest
from test_kb_tools import propose
from test_reviewer import APPROVE, REVISE, issue
from test_translator import (  # noqa: F401 (fixtures)
    BOOK, attempts_of, imported, previous_context, root, translate, zh_json, zh_of,
)

from fukidashi import live_trace, runs, store
from fukidashi.agents.loop import Tool
from fukidashi.datasets.importer import import_openmantra
from fukidashi.datasets.openmantra import get_page
from fukidashi.db import connect, init_db
from fukidashi.llm.client import LLMNotConfigured


@pytest.fixture(autouse=True)
def every_event_holds_the_contract():
    """AC1: every event any test of this module sees was held to the contract (live_trace_support.check_event)."""
    VIOLATIONS.clear()
    yield
    assert VIOLATIONS == []


def model_call_of(step: dict, n: int) -> dict:
    """The `model_call` event of the n-th step of an agent's record: the step, with its tool calls as they were asked for."""
    asked = [{k: c[k] for k in ("id", "name", "arguments")} for c in step["tool_calls"]]
    return {"step": n, **step, "tool_calls": asked}


def tool_results_of(steps: list[dict]) -> list[dict]:
    """The `tool_result` events of an agent's record, in the order of the calls: one for each call that got a reply."""
    return [{"tool_call_id": c["id"], "name": c["name"], **{k: c[k] for k in ("result", "error", "duplicate", "ran")}}
            for step in steps for c in step["tool_calls"] if c["result"] is not None or c["error"] is not None]


# --- the contract ---------------------------------------------------------------------------------------------------

def test_every_kind_of_the_contract_is_named_once_and_described_in_the_modules_docstring():
    kinds = [live_trace.RUN_START, live_trace.STAGE_START, live_trace.MODEL_CALL, live_trace.TOOL_RESULT,
             live_trace.REPAIR, live_trace.FALLBACK, live_trace.STAGE_END, live_trace.RUN_END]
    assert list(live_trace.KINDS) == kinds and len(set(kinds)) == 8
    assert all(f"`{kind}`" in live_trace.__doc__ for kind in kinds)


def test_the_check_every_sink_runs_catches_an_unknown_kind_a_bad_id_and_data_that_is_not_json():
    check_event("not_a_kind", {})
    check_event("run_start", {}, run_id=0, seq=1)
    check_event("run_start", {}, run_id=1, seq=0)
    check_event("run_end", {"cost_usd": float("nan")})
    check_event("run_end", {"x": object()})
    check_event("run_end", ["not", "a", "dict"])
    check_event("run_end", {"cost_usd": 0.1}, run_id=1, seq=1)  # this one is fine

    assert len(VIOLATIONS) == 6
    VIOLATIONS.clear()


def test_a_run_event_is_immutable():
    event = live_trace.RunEvent(1, 1, "run_start", {})
    with pytest.raises(dataclasses.FrozenInstanceError):
        event.seq = 2


def test_the_contract_imports_only_the_standard_library_so_it_loads_no_database():
    assert database_loaded_by("fukidashi.live_trace") == []


# --- run_agent: a model call, a tool result ----------------------------------------------------------------------------

def test_each_model_call_and_tool_result_is_reported_and_the_users_message_and_the_reasoning_never_are():
    sink, handled = Recorder(), []
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1}), content="Let me look.",
                            reasoning_content="SECRET-REASONING"), answer("done"))

    result, _ = run(provider, [make_tool(handled, returns={"zh": "阴泽"})], emit=sink.emit)

    assert [kind for kind, _ in sink.emitted] == ["model_call", "tool_result", "model_call"]
    first, tool, last = (data for _, data in sink.emitted)
    assert first == model_call_of(result.steps[0], 1) and last == model_call_of(result.steps[1], 2)
    assert first["reasoning_chars"] == len("SECRET-REASONING") and first["text"] == "Let me look."
    assert first["tool_calls"] == [{"id": "call_1", "name": "lookup", "arguments": json.dumps({"term": "a", "n": 1})}]
    assert tool == {"tool_call_id": "call_1", "name": "lookup", "result": '{"zh": "阴泽"}', "error": None,
                    "duplicate": False, "ran": True}
    serialized = json.dumps(sink.emitted, ensure_ascii=False)
    assert "SECRET-REASONING" not in serialized and "USER" not in serialized  # the user message of `run` is "USER"


def test_the_events_are_the_steps_of_the_record_including_a_duplicate_a_mistake_and_a_tool_that_raised():
    sink, handled = Recorder(), []
    same = call("c1", {"term": "a", "n": 1})
    provider = Provider(
        ask(same, call("c2", {"term": "b", "n": 2}), call("c3", {"term": "a"}), call("c4", {"x": 1}, name="nope")),
        ask(call("c5", {"term": "a", "n": 1})),   # c1 again: a duplicate, not run again
        answer("done"))

    result, _ = run(provider, [make_tool(handled)], emit=sink.emit)

    steps = result.steps
    assert [d for k, d in sink.emitted if k == "model_call"] == [model_call_of(s, n) for n, s in enumerate(steps, 1)]
    assert [d for k, d in sink.emitted if k == "tool_result"] == tool_results_of(steps)
    outcomes = {d["tool_call_id"]: (d["ran"], d["duplicate"], d["error"] is not None)
                for k, d in sink.emitted if k == "tool_result"}
    assert outcomes == {"c1": (True, False, False), "c2": (True, False, False), "c3": (False, False, True),
                        "c4": (False, False, True), "c5": (False, True, False)}


def test_a_failed_call_is_a_model_call_with_its_error_and_the_next_step_number():
    sink = Recorder()
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})), upstream_500())

    result, meter = run(provider, [make_tool([])], emit=sink.emit)

    assert result.stop_reason == "provider_error"
    calls = [d for k, d in sink.emitted if k == "model_call"]
    assert [d["step"] for d in calls] == [1, 2] and len(calls) == len(result.steps) == len(meter.records)
    failed = calls[1]
    assert failed == model_call_of(result.steps[1], 2)
    assert "upstream exploded" in failed["error"] and failed["finish_reason"] is None and failed["tool_calls"] == []
    assert failed["record"]["error"] == failed["error"]  # the call as the Meter has it


@pytest.mark.parametrize("options, stop", [({"max_steps": 1}, "step_limit"), ({"limit": 100}, "budget")])
def test_a_step_limit_or_a_spent_budget_sends_no_request_so_it_reports_no_model_call(options, stop):
    sink = Recorder()
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})))

    result, _ = run(provider, [make_tool([])], emit=sink.emit, **options)

    assert result.stop_reason == stop
    assert [kind for kind, _ in sink.emitted] == ["model_call", "tool_result"]
    assert len(result.steps) == len(provider.bodies) == 1


def test_the_tools_of_one_reply_run_in_worker_threads_and_every_event_comes_from_the_calling_thread():
    both_running, handler_threads = threading.Barrier(2, timeout=10), []

    def lookup(term: str, n: int):
        handler_threads.append(threading.get_ident())
        both_running.wait()  # the two calls must be running at once, or this raises and the tool's result is an error
        return f"found {term}"

    sink = Recorder()
    provider = Provider(ask(call("c1", {"term": "a", "n": 1}), call("c2", {"term": "b", "n": 2})), answer())

    run(provider, [Tool("lookup", "Look a term up.", Args, lookup)], emit=sink.emit)

    me = threading.get_ident()
    assert len(set(handler_threads)) == 2 and me not in handler_threads
    assert set(sink.threads) == {me} and len(sink.emitted) == 4
    assert [d["error"] for k, d in sink.emitted if k == "tool_result"] == [None, None]


def test_the_tool_results_of_one_reply_are_reported_in_the_order_of_the_calls_not_of_the_threads():
    second_done = threading.Event()

    def lookup(term: str, n: int):
        if term == "slow":
            assert second_done.wait(10)  # finishes after the second call
        else:
            second_done.set()
        return f"{term} done"

    sink = Recorder()
    provider = Provider(ask(call("c1", {"term": "slow", "n": 1}), call("c2", {"term": "quick", "n": 2})), answer())

    run(provider, [Tool("lookup", "Look a term up.", Args, lookup)], emit=sink.emit)

    assert [d["tool_call_id"] for k, d in sink.emitted if k == "tool_result"] == ["c1", "c2"]


# --- AC2: the kinds, in order, of each config ----------------------------------------------------------------------------

def test_a_baseline_run_emits_its_start_and_its_end_and_no_model_call_of_its_own(imported, root):
    sink = Recorder()

    outcome = translate(imported, root, mock_llm_client(Provider(answer(zh_json(3)))), 3, config="baseline",
                        on_event=sink)

    assert sink.kinds == ["run_start", "run_end"]
    assert_well_formed(sink.events, outcome.run_id)
    assert sink.events[0].data == {"run_id": outcome.run_id, "config": "baseline", "model": "mock-model"}
    assert sink.events[1].data["status"] == "succeeded" and sink.events[1].data["calls"] == 1


def test_a_translator_run_reports_its_calls_and_its_tool_result_and_has_no_stages(imported, root):
    sink = Recorder()

    outcome = translate(imported, root, mock_llm_client(Provider(ask(previous_context(3)), answer(zh_json(3)))), 3,
                        on_event=sink)

    assert sink.kinds == ["run_start", "model_call", "tool_result", "model_call", "run_end"]
    assert_well_formed(sink.events, outcome.run_id)
    assert sink.events[0].data["config"] == "translator"
    assert sink.events[2].data["name"] == "get_previous_context" and sink.events[2].data["ran"] is True


def test_a_bad_answer_and_its_repair_are_a_repair_event_before_the_end(imported, root):
    sink = Recorder()
    provider = Provider(ask(previous_context(3)), answer("not json"), answer(zh_json(3)))

    outcome = translate(imported, root, mock_llm_client(provider), 3, on_event=sink)

    assert sink.kinds == ["run_start", "model_call", "tool_result", "model_call", "repair", "run_end"]
    assert_well_formed(sink.events, outcome.run_id)
    repair = sink.of("repair")[0].data
    assert repair["error"] is None and repair["record"]["requested_model"] == "mock-model"
    assert repair["parse_error"] == attempts_of(imported, outcome.run_id)["agent"]["answer_parse_error"] != ""


def test_a_repair_that_the_provider_refuses_is_a_repair_event_with_its_error(imported, root):
    sink = Recorder()
    provider = Provider(answer("not json"), upstream_500())

    outcome = translate(imported, root, mock_llm_client(provider), 3, on_event=sink)

    assert not outcome.result.ok
    assert sink.kinds == ["run_start", "model_call", "repair", "run_end"]
    repair = sink.of("repair")[0].data
    assert "upstream exploded" in repair["error"] and repair["record"]["error"] == repair["error"]
    assert sink.events[-1].data["status"] == "failed"


def test_the_translators_fallback_is_an_event_before_baseline_runs(imported, root):
    sink = Recorder()
    provider = Provider(answer("", finish_reason="length"), answer(zh_json(3)))

    outcome = translate(imported, root, mock_llm_client(provider), 3, on_event=sink)

    assert outcome.result.fallback_reason is not None
    assert sink.kinds == ["run_start", "model_call", "fallback", "run_end"]  # baseline's own call is no model call
    assert sink.of("model_call")[0].data["finish_reason"] == "length"
    assert sink.of("fallback")[0].data == {"reason": outcome.result.fallback_reason}
    assert sink.events[-1].data["calls"] == 2


def test_the_fallback_of_the_first_pass_ends_the_translate_stage(imported, root):
    sink = Recorder()
    provider = Provider(answer("", finish_reason="length"), answer(zh_json(3)), answer(APPROVE))

    translate(imported, root, mock_llm_client(provider), 3, config="full_no_kb", on_event=sink)

    assert sink.trail == ["run_start", "stage_start:translate", "model_call", "fallback", "stage_end:translate",
                          "stage_start:review", "model_call", "stage_end:review", "run_end"]


def one_send_back(*, first=None):
    """A page that is translated, sent back once with an Issue on region 0, revised, and approved."""
    return [*(first or [answer(zh_json(3))]), answer(REVISE), answer(zh_json(3, "改")), answer(APPROVE)]


STAGES_OF_ONE_SEND_BACK = ["stage_start:translate", "model_call", "stage_end:translate",
                           "stage_start:review", "model_call", "stage_end:review",
                           "stage_start:revise", "model_call", "stage_end:revise",
                           "stage_start:review", "model_call", "stage_end:review"]


def test_full_no_kb_with_one_send_back_reports_each_stage_and_the_end_of_the_loop(imported, root):
    sink = Recorder()

    outcome = translate(imported, root, mock_llm_client(Provider(*one_send_back())), 3, config="full_no_kb",
                        on_event=sink)

    assert sink.trail == ["run_start", *STAGES_OF_ONE_SEND_BACK, "run_end"]
    assert_well_formed(sink.events, outcome.run_id)
    assert [(e.data["stage"], e.data["index"]) for e in sink.of("stage_start")] == [
        ("translate", 0), ("review", 1), ("revise", 2), ("review", 3)]
    first_review, second_review = (e.data for e in sink.of("stage_end") if e.data["stage"] == "review")
    assert (first_review["verdict"], first_review["issues"]) == ("revise", [issue(0)])
    assert (second_review["verdict"], second_review["issues"]) == ("approve", [])
    assert sink.events[-1].data["send_backs"] == 1 and sink.events[-1].data["final_stop"] == "approved"
    assert sink.events[-1].data["final_reviewed"] is True


def test_full_with_a_term_proposal_adds_finalize_and_the_curators_call(imported, root, kb):
    sink = Recorder()
    new, seen = terms_of(kb)
    provider = Provider(*one_send_back(first=[proposals_reply(), answer(zh_json(3))]),
                        answer(decisions_json(decision(new, True, "招式名"), decision(seen, True, "同一译法"))))

    outcome, _ = run_page(imported, root, provider, kb, on_event=sink)

    assert sink.trail == ["run_start", "stage_start:translate", "model_call", "tool_result", "tool_result",
                          "tool_result", "model_call", "stage_end:translate", *STAGES_OF_ONE_SEND_BACK[3:],
                          "stage_start:finalize", "model_call", "stage_end:finalize", "run_end"]
    assert_well_formed(sink.events, outcome.run_id)
    start, end = (e.data for e in sink.events if e.data.get("stage") == "finalize")
    assert start == {"stage": "finalize", "index": None}
    assert (end["status"], end["error"], end["index"]) == ("ok", None, None)
    assert sum(end["results"].values()) == 3 and end["results"]["accepted"] == 1   # the new term; two others by rule
    assert sink.events[-1].data["final_stop"] == "approved" and sink.events[-1].data["send_backs"] == 1


def test_a_provider_error_in_the_review_is_that_calls_model_call_a_failed_stage_and_a_failed_run(imported, root):
    sink = Recorder()

    outcome = translate(imported, root, mock_llm_client(Provider(answer(zh_json(3)), upstream_500())), 3,
                        config="full_no_kb", on_event=sink)

    assert not outcome.result.ok
    assert sink.trail == ["run_start", "stage_start:translate", "model_call", "stage_end:translate",
                          "stage_start:review", "model_call", "stage_end:review", "run_end"]
    review_call, review_end, end = sink.events[5].data, sink.events[6].data, sink.events[7].data
    assert "upstream exploded" in review_call["error"] and review_call["step"] == 1
    assert (review_end["status"], review_end["error"]) == ("failed", review_call["error"])
    assert (end["status"], end["error"], end["final_stop"], end["final_reviewed"]) == (
        "failed", review_call["error"], "failed", False)


def test_a_crash_after_the_run_row_exists_ends_with_a_failed_run_end(imported, root):
    sink = Recorder()

    with pytest.raises(runs.RunCrashed) as crashed:
        translate(imported, root, CrashAfterCallClient([zh_json(3)]), 3, config="baseline", on_event=sink)

    assert sink.kinds == ["run_start", "run_end"]
    assert_well_formed(sink.events, crashed.value.run_id)
    end = sink.events[-1].data
    assert (end["status"], end["error"], end["calls"]) == ("failed", "RuntimeError: crashed after the call", 1)
    assert "send_backs" not in end


def test_a_crash_in_finalize_has_no_stage_end_for_it_and_ends_with_a_failed_run_end(imported, root, kb, monkeypatch):
    sink = Recorder()
    new, seen = terms_of(kb)
    provider = Provider(proposals_reply(), answer(zh_json(3)), answer(APPROVE),
                        answer(decisions_json(decision(new, True, "好"), decision(seen))))

    def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(runs.curator_rules, "apply", boom)

    with pytest.raises(runs.RunCrashed) as crashed:
        run_page(imported, root, provider, kb, on_event=sink)

    assert sink.trail[-3:] == ["stage_start:finalize", "model_call", "run_end"]
    assert_well_formed(sink.events, crashed.value.run_id)
    assert sink.events[-1].data["error"] == "RuntimeError: disk on fire" and sink.events[-1].data["status"] == "failed"


def test_when_marking_the_run_failed_fails_too_there_is_no_run_end(imported, root, monkeypatch):
    sink = Recorder()

    def broken(*args, **kwargs):
        raise RuntimeError("the disk is full")

    monkeypatch.setattr(store, "finish_run", broken)

    with pytest.raises(runs.RunCrashed) as crashed:
        translate(imported, root, mock_llm_client(Provider(answer(zh_json(3)))), 3, config="baseline", on_event=sink)

    assert sink.kinds == ["run_start"]
    assert store.get_run(imported, crashed.value.run_id)["status"] == "running"


# --- AC3: seq, and the pages refused before a Run exists -------------------------------------------------------------------

REFUSALS = ["unknown-page", "no-text-regions", "image-missing", "client-not-configured", "full-without-a-kb"]


@pytest.mark.parametrize("case", REFUSALS)
def test_a_page_refused_before_its_run_exists_emits_nothing(imported, root, dataset, case):
    import_openmantra(imported, dataset)  # book_a: page 1 has text, page 2 has none
    sink, book, page, options = Recorder(), BOOK, 3, {}
    make_client = lambda: mock_llm_client(Provider(answer(zh_json(3))))  # noqa: E731
    expected = runs.TranslateError
    if case == "unknown-page":
        page = 99
    elif case == "no-text-regions":
        book, page, root = "book_a", 2, dataset
    elif case == "image-missing":
        (root / "images" / BOOK / "ja" / "002.jpg").unlink()
    elif case == "client-not-configured":
        expected = LLMNotConfigured

        def make_client():
            raise LLMNotConfigured("no key")
    else:
        expected, options = ValueError, {"config": "full"}

    with pytest.raises(expected):
        runs.translate_page(imported, book, page, make_client=make_client, dataset_root=root, on_event=sink, **options)

    assert sink.events == []
    assert imported.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_every_run_has_its_own_seq_from_one(imported, root):
    first, second = Recorder(), Recorder()

    one = translate(imported, root, mock_llm_client(Provider(answer(zh_json(3)))), 3, config="baseline", on_event=first)
    two = translate(imported, root, mock_llm_client(Provider(answer(zh_json(3)))), 3, config="baseline", on_event=second)

    assert one.run_id != two.run_id
    assert_well_formed(first.events, one.run_id)
    assert_well_formed(second.events, two.run_id)


# --- AC4: the events say what the Run stored -----------------------------------------------------------------------------

def rich_replies(kb_id):
    """A `full` page that uses every kind of event: five tool calls in one reply (three proposals, a flag, and a lookup with
    arguments that are wrong), a bad answer that is repaired, one send-back, and the Curator's decisions."""
    new, seen = terms_of(kb_id)
    return [
        ask(propose("p1", "term", source_term="封印", target_term="封印术", category="technique"),
            propose("p2", "term", source_term=SEEN_TERM[0], target_term=SEEN_TERM[1]),
            propose("p3", "character", name_src=NAME[0], name_tgt=NAME[1], aliases=["ゆき"]),
            call("f1", {"id": 1, "reason": "迷い"}, name="flag_uncertain"),
            call("bad", {"wrong": 1}, name="lookup_term"),
            reasoning_content="SECRET-REASONING-A"),
        answer("not json", reasoning_content="SECRET-REASONING-B"),
        answer(zh_json(3)),                       # the repair
        answer(REVISE),
        answer(zh_json(3, "改")),
        answer(APPROVE),
        answer(decisions_json(decision(new, True, "招式名"), decision(seen, True, "同一译法"))),
    ]


def inside_stages(events):
    """The events between each stage's start and end: {(stage, index): [events]}."""
    inside, current = {}, None
    for e in events:
        if e.kind == "stage_end":
            current = None
        elif current is not None:
            inside[current].append(e)
        if e.kind == "stage_start":
            current = (e.data["stage"], e.data["index"])
            inside[current] = []
    return inside


def test_the_events_equal_what_the_run_stored(imported, root, kb):
    sink = Recorder()

    outcome, _ = run_page(imported, root, Provider(*rich_replies(kb)), kb, on_event=sink)

    assert outcome.result.ok
    assert_well_formed(sink.events, outcome.run_id)
    attempts, row = attempts_of(imported, outcome.run_id), store.get_run(imported, outcome.run_id)
    inside = inside_stages(sink.events)
    ends = {e.data["index"]: e.data for e in sink.of("stage_end") if e.data["index"] is not None}
    assert len(attempts["stages"]) == 4 and sorted(ends) == [0, 1, 2, 3]

    for index, stage in enumerate(attempts["stages"]):
        agent, events = stage["attempts"]["agent"], inside[(stage["kind"], index)]
        assert [e.data for e in events if e.kind == "model_call"] == [
            model_call_of(s, n) for n, s in enumerate(agent["steps"], 1)]
        assert [e.data for e in events if e.kind == "tool_result"] == tool_results_of(agent["steps"])
        repairs = [e.data for e in events if e.kind == "repair"]
        repair = stage["attempts"]["repair"]
        assert repairs == ([] if repair is None else [
            {"record": repair["record"], "error": repair["error"], "parse_error": agent["answer_parse_error"]}])
        carried = {field: stage[field] for field in ("status", "error", "flags", "verdict", "issues") if field in stage}
        assert ends[index] == {"stage": stage["kind"], "index": index, **carried}
    first = attempts["stages"][0]
    assert first["flags"] == [{"id": 1, "reason": "迷い"}] and first["attempts"]["repair"] is not None
    assert [e.kind for e in inside[("translate", 0)]][-1] == "repair"  # the repair comes after the agent's calls
    errors = [d["error"] for d in tool_results_of(first["attempts"]["agent"]["steps"])]
    assert sum(error is not None for error in errors) == 1  # the lookup with wrong arguments

    finalize = attempts["finalize"]
    agent = finalize["curator"]["attempts"]["agent"]
    assert [e.data for e in inside[("finalize", None)] if e.kind == "model_call"] == [
        model_call_of(s, n) for n, s in enumerate(agent["steps"], 1)]
    (finalize_end,) = [e.data for e in sink.of("stage_end") if e.data["stage"] == "finalize"]
    assert finalize_end == {"stage": "finalize", "index": None, "status": finalize["curator"]["status"],
                            "error": finalize["curator"]["error"], "results": runs.outcome_counts(finalize["outcomes"])}

    assert sink.events[-1].data == {
        "status": row["status"], "error": row["error"], "calls": row["calls"],
        "unknown_usage_calls": row["unknown_usage_calls"], "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"], "cost_usd": row["cost_usd"], "send_backs": attempts["send_backs"],
        "final_stop": attempts["final"]["stop"], "final_reviewed": attempts["final"]["reviewed"]}


def test_a_failed_runs_end_has_the_stored_error_and_a_translator_runs_has_no_loop_fields(imported, root):
    sink = Recorder()

    outcome = translate(imported, root, mock_llm_client(Provider(upstream_500())), 3, on_event=sink)

    row = store.get_run(imported, outcome.run_id)
    assert row["status"] == "failed" and row["input_tokens"] is None and row["cost_usd"] is None  # unknown, not 0
    assert sink.events[-1].data == {
        "status": "failed", "error": row["error"], "calls": 1, "unknown_usage_calls": 1, "input_tokens": None,
        "output_tokens": None, "cost_usd": None}


# --- AC5: the events are live ------------------------------------------------------------------------------------------

def test_each_event_reaches_the_sink_while_the_run_runs_and_a_second_connection_reads_it(imported, root, kb, db_path):
    other = connect(db_path)  # another connection to the same file
    sink, stored, seen = Recorder(), runs.store_events(imported), []

    def watch(event):
        sink(event)  # a Recorder
        stored(event)
        seen.append((event.kind, store.get_run(other, event.run_id)["status"], store.run_events(other, event.run_id)))

    outcome, _ = run_page(imported, root, Provider(*rich_replies(kb)), kb, on_event=watch)

    assert outcome.result.ok and len(seen) == len(sink.events) > 10
    assert {status for kind, status, _ in seen if kind == "model_call"} == {"running"}
    assert [status for _, status, _ in seen[:-1]] == ["running"] * (len(seen) - 1)  # all but the last: still running
    assert (seen[-1][0], seen[-1][1]) == ("run_end", "succeeded")  # the end comes after the Run was finished
    for n, (_, _, readable) in enumerate(seen, 1):
        assert readable == sink.events[:n]  # what had been emitted so far was already readable, and nothing more
    assert store.run_events(other, outcome.run_id) == sink.events
    other.close()


def test_without_a_sink_no_events_are_stored_and_the_run_is_the_same(imported, root):
    translate(imported, root, mock_llm_client(Provider(*one_send_back())), 3, config="full_no_kb")

    assert imported.execute("SELECT COUNT(*) FROM run_events").fetchone()[0] == 0


# --- AC6: what no event carries ---------------------------------------------------------------------------------------

def test_no_event_carries_the_image_the_user_message_the_reasoning_or_a_reference_translation(imported, root, kb):
    sink = Recorder()
    provider = Provider(*rich_replies(kb))

    run_page(imported, root, provider, kb, on_event=sink)

    serialized = json.dumps([dataclasses.asdict(e) for e in sink.events], ensure_ascii=False)
    sent = json.dumps(provider.bodies, ensure_ascii=False)
    assert "data:image" in sent and "SECRET-REASONING" in sent  # the check below would be empty otherwise
    assert "data:image" not in serialized and "SECRET-REASONING" not in serialized
    assert "REF-EN" not in serialized and "REF-ZH" not in serialized  # the page's text_en and text_zh_hant
    parts = [part for body in provider.bodies for part in body["messages"][1]["content"]]
    assert len(parts) >= 8  # the user messages of the Translator, its revision, the Reviewer and the Curator
    for part in parts:
        text = part["text"] if part["type"] == "text" else part["image_url"]["url"]
        assert text not in serialized
    assert "日本語3-0" not in serialized and "日本語3-1" not in serialized  # the page's text only reaches the user messages
    reasoning = [e.data["reasoning_chars"] for e in sink.of("model_call")][:2]
    assert reasoning == [len("SECRET-REASONING-A"), len("SECRET-REASONING-B")]


# --- AC7: the thread -----------------------------------------------------------------------------------------------------

def test_every_sink_call_is_in_the_thread_that_called_translate_page_though_the_tools_run_in_others(imported, root, kb):
    sink = Recorder()

    def in_a_worker():
        run_page(imported, root, Provider(*rich_replies(kb)), kb, on_event=sink)  # a reply with five parallel tool calls
        return threading.get_ident()

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(in_a_worker).result()

    assert worker != threading.get_ident()
    assert len(sink.threads) == len(sink.events) > 10
    assert set(sink.threads) == {worker}


# --- AC8: the hook only observes ------------------------------------------------------------------------------------------

def fresh_database(tmp_path, root, name, *, with_kb):
    conn = connect(tmp_path / f"{name}.db")
    init_db(conn)
    import_openmantra(conn, root)
    kb_id = None
    if with_kb:
        kb_id = new_kb(conn, book=BOOK)
        seed_entry(conn, kb_id, *SEEN_TERM)
    return conn, kb_id


def stored_run(conn, run_id):
    """Everything a Run stored that is not a clock: its row, its attempts without latency, and its Translations."""
    row = dict(store.get_run(conn, run_id))
    for clock in ("id", "started_at", "finished_at"):
        del row[clock]
    row["attempts_json"] = without_latency(json.loads(row["attempts_json"]))
    return {**row, "translations": zh_of(conn, run_id)}


SCENARIOS = {
    "baseline": lambda kb_id: [answer(zh_json(3))],
    "translator": lambda kb_id: [ask(previous_context(3)), answer("not json"), answer(zh_json(3))],
    "full_no_kb": lambda kb_id: [answer("", finish_reason="length"), *one_send_back(first=[answer(zh_json(3))])],
    "full": rich_replies,
}


def wipe(event):
    """A sink that damages what it is given, at every depth: whatever it got must not be what the Run keeps."""
    def damage(value):
        if isinstance(value, (dict, list)):
            for inner in list(value.values() if isinstance(value, dict) else value):
                damage(inner)
            value.clear()

    damage(event.data)


@pytest.mark.parametrize("config", list(SCENARIOS))
def test_a_run_with_a_sink_stores_and_sends_what_a_run_without_one_does(tmp_path, root, config):
    results, requests = [], []
    for name, on_event in (("plain", None), ("recording", Recorder()), ("wiping", checked(wipe))):
        conn, kb_id = fresh_database(tmp_path, root, name, with_kb=config == "full")
        provider = Provider(*SCENARIOS[config](kb_id))
        options = {} if on_event is None else {"on_event": on_event}
        if config == "full":
            outcome, _ = run_page(conn, root, provider, kb_id, **options)
        else:
            outcome = translate(conn, root, mock_llm_client(provider), 3, config=config, **options)
        assert outcome.result.ok
        results.append(stored_run(conn, outcome.run_id))
        requests.append(provider.bodies)
        if name == "recording":
            assert on_event.events and on_event.kinds[0] == "run_start"  # the sink was live in this scenario
    assert results[0] == results[1] == results[2]
    assert requests[0] == requests[1] == requests[2]


def test_the_sixteen_scenarios_send_the_pinned_requests_with_an_emit_too(dataset):
    sink = Recorder()

    collected = request_scenarios.collect(get_page("book_a", 1, dataset), emit=sink.emit)

    assert {name: digest(bodies) for name, bodies in collected.items()} == DIGESTS
    assert len(sink.emitted) > 40  # the hook was live in them


def test_the_curators_requests_are_the_same_with_and_without_an_emit(curator_page, curator_proposals):
    good = decisions_json(decision(curator_proposals[0]), decision(curator_proposals[2]))
    for replies, kinds in (([answer(good)], ["model_call"]), ([answer("nope"), answer(good)], ["model_call", "repair"])):
        plain, hooked, sink = Provider(*replies), Provider(*replies), Recorder()

        without, _ = curate(curator_page, plain, curator_proposals)
        with_emit, _ = curate(curator_page, hooked, curator_proposals, emit=sink.emit)

        assert plain.bodies == hooked.bodies
        assert (with_emit.status, with_emit.decisions, without_latency(with_emit.attempts)) == (
            without.status, without.decisions, without_latency(without.attempts))
        assert [kind for kind, _ in sink.emitted] == kinds


# --- AC9: a sink that raises ------------------------------------------------------------------------------------------

class FailsOnThird:
    """Raises once, on the third event it is given. Counts its calls."""

    def __init__(self):
        self.calls, self.raised, self.events = 0, False, []

    def __call__(self, event):
        self.calls += 1
        self.events.append(event)
        if event.seq == 3 and not self.raised:
            self.raised = True
            raise RuntimeError("the sink is broken")


@pytest.mark.parametrize("config", ["full_no_kb", "full"])
def test_a_sink_that_raises_changes_nothing_is_logged_once_and_is_not_called_again(tmp_path, root, caplog, config):
    with_kb = config == "full"
    plain_conn, plain_kb = fresh_database(tmp_path, root, "plain", with_kb=with_kb)
    conn, kb_id = fresh_database(tmp_path, root, "flaky", with_kb=with_kb)
    replies = SCENARIOS[config]
    sink = FailsOnThird()

    def go(conn, kb_id, **options):
        provider = Provider(*replies(kb_id))
        if with_kb:
            return run_page(conn, root, provider, kb_id, **options)[0]
        return translate(conn, root, mock_llm_client(provider), 3, config=config, **options)

    plain = go(plain_conn, plain_kb)
    with caplog.at_level(logging.WARNING, logger="fukidashi.runs"):
        outcome = go(conn, kb_id, on_event=checked(sink))

    assert outcome.result.ok and stored_run(conn, outcome.run_id) == stored_run(plain_conn, plain.run_id)
    assert sink.calls == 3  # it raised on the third event and was not called again for this Run
    warnings = [r for r in caplog.records if r.name == "fukidashi.runs" and r.levelno == logging.WARNING]
    assert len(warnings) == 1 and f"Run {outcome.run_id}" in warnings[0].getMessage()
    assert "the sink is broken" in warnings[0].getMessage()

    with caplog.at_level(logging.WARNING, logger="fukidashi.runs"):
        again = go(conn, kb_id, on_event=checked(sink))  # the next Run's sink gets every event of it
    assert again.result.ok
    assert_well_formed([e for e in sink.events if e.run_id == again.run_id], again.run_id)
    assert len([r for r in caplog.records if r.name == "fukidashi.runs"]) == 1  # and nothing more was logged


def test_a_sink_that_raises_on_the_first_event_still_lets_the_run_succeed(imported, root, caplog):
    def broken(event):
        raise RuntimeError("no")

    with caplog.at_level(logging.WARNING, logger="fukidashi.runs"):
        outcome = translate(imported, root, mock_llm_client(Provider(answer(zh_json(3)))), 3, config="baseline",
                            on_event=checked(broken))

    assert outcome.result.ok and store.get_run(imported, outcome.run_id)["status"] == "succeeded"
    assert len([r for r in caplog.records if r.name == "fukidashi.runs"]) == 1


# --- AC10: a full Run's finalize events are stored ------------------------------------------------------------------------

def test_a_full_run_with_proposals_stores_its_finalize_events_on_the_same_connection(imported, root, kb, caplog):
    sink, stored = Recorder(), runs.store_events(imported)

    def both(event):
        sink(event)
        stored(event)

    with caplog.at_level(logging.WARNING, logger="fukidashi.runs"):
        outcome, _ = run_page(imported, root, Provider(*rich_replies(kb)), kb, on_event=both)

    assert [r for r in caplog.records if r.name == "fukidashi.runs"] == []  # none refused for coming inside a transaction
    events = store.run_events(imported, outcome.run_id)
    assert events == sink.events and len(events) > 10
    finalize = [e.kind for e in events if e.data.get("stage") == "finalize"]
    assert finalize == ["stage_start", "stage_end"] and events[-1].kind == "run_end"
    assert imported.in_transaction is False
