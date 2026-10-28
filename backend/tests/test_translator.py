"""The Translator Agent through `runs.translate_page` and the eval harness, on a scripted provider: no network.

The real LLMClient, the real openai SDK and LangChain's create_agent run; only the provider's replies are scripted.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, replace

import pytest
from conftest import FakeClient, mock_llm_client
from test_agent_loop import Provider, answer, ask, call, upstream_500
from test_evaluation import run_cli

from fukidashi import config as app_config
from fukidashi import runs, store
from fukidashi.agents import translator
from fukidashi.datasets.importer import import_openmantra
from fukidashi.datasets.openmantra import get_page
from fukidashi.evaluation import harness
from fukidashi.llm.metering import Meter
from fukidashi.pipeline import baseline

BOOK = "book_t"
PAGES = 3
CALL_COST = 0.00014  # what one scripted reply costs: 100 input and 20 output tokens at $1 / $2 per million


def ja(page_index, i):
    return f"日本語{page_index}-{i}"


def zh(page_index, i, tag=""):
    return f"译文{page_index}-{i}{tag}"


def zh_json(page_index, tag=""):
    return json.dumps({"translations": [{"id": i, "zh": zh(page_index, i, tag)} for i in range(2)]}, ensure_ascii=False)


def previous_context(page_index):
    """The tool call that asks for the page before."""
    return call(f"call_{page_index}", {"pages_back": 1}, name="get_previous_context")


@pytest.fixture
def root(tmp_path):
    """A book of three pages with two regions each. Every reference text is distinctive, so a test can look for it."""
    root = tmp_path / "open-mantra-dataset"
    (root / "images" / BOOK / "ja").mkdir(parents=True)
    pages = []
    for page_index in range(1, PAGES + 1):
        image = f"images/{BOOK}/ja/{page_index - 1:03d}.jpg"
        (root / image).write_bytes(b"\xff\xd8fake-jpeg")
        text = [{"x": 1, "y": 2, "w": 3, "h": 4, "text_ja": ja(page_index, i), "text_en": f"REF-EN-{page_index}-{i}",
                 "text_zh": f"REF-ZH-{page_index}-{i}"} for i in range(2)]
        pages.append({"page_index": page_index, "image_paths": {"ja": image}, "frame": [], "text": text})
    (root / "annotation.json").write_text(json.dumps([{"book_title": BOOK, "pages": pages}]), encoding="utf-8")
    return root


@pytest.fixture
def imported(conn, root):
    import_openmantra(conn, root)
    return conn


def translate(conn, root, client, page_index=3, *, config="translator", **options):
    return runs.translate_page(conn, BOOK, page_index, make_client=lambda: client, dataset_root=root, config=config,
                               **options)


def translate_earlier_pages(conn, root, *page_indexes, config="translator"):
    """A succeeded Run of `config` for each page, as an earlier eval would have left them."""
    for page_index in page_indexes:
        provider = Provider(answer(zh_json(page_index)))
        assert translate(conn, root, mock_llm_client(provider), page_index, config=config).result.ok


def attempts_of(conn, run_id):
    return json.loads(store.get_run(conn, run_id)["attempts_json"])


def zh_of(conn, run_id):
    return [r["text_zh_hans"] for r in conn.execute(
        "SELECT t.text_zh_hans FROM translations t JOIN segments s ON s.id = t.segment_id WHERE t.run_id = ?"
        " ORDER BY s.idx", (run_id,))]


def tool_result(body, call_id):
    (message,) = [m for m in body["messages"] if m["role"] == "tool" and m["tool_call_id"] == call_id]
    return message["content"]


@pytest.fixture
def page_for(root):
    return lambda page_index: get_page(BOOK, page_index, root)


# --- AC1: a translator Run ----------------------------------------------------------------------------------------

def test_a_tool_call_then_a_valid_answer_is_a_succeeded_translator_run(imported, root, page_for):
    translate_earlier_pages(imported, root, 1, 2)
    provider, meter = Provider(ask(previous_context(3)), answer(zh_json(3))), Meter()

    outcome = translate(imported, root, mock_llm_client(provider, meter=meter), 3)

    assert outcome.result.ok and outcome.result.fallback_reason is None
    run = store.get_run(imported, outcome.run_id)
    assert (run["config"], run["status"], run["error"]) == ("translator", "succeeded", None)
    assert zh_of(imported, outcome.run_id) == [zh(3, 0), zh(3, 1)]  # every segment has a Translation
    assert run["calls"] == len(provider.bodies) == len(meter.records) == 2  # the Run's calls are the requests sent
    assert run["cost_usd"] == pytest.approx(2 * CALL_COST) and run["unknown_usage_calls"] == 0
    attempts = attempts_of(imported, outcome.run_id)
    assert set(attempts) == {"config", "agent", "repair", "fallback"} and attempts["config"] == "translator"
    assert attempts["agent"]["stop_reason"] == "final" and attempts["agent"]["error"] is None
    assert attempts["agent"]["answer_parse_error"] is None
    assert len(attempts["agent"]["steps"]) == 2 and attempts["agent"]["steps"][0]["tool_calls"][0]["ran"] is True
    assert attempts["repair"] is None and attempts["fallback"] is None

    first = provider.bodies[0]
    system, user = first["messages"]
    assert system == {"role": "system", "content": translator.SYSTEM_PROMPT}
    assert translator.SYSTEM_PROMPT != baseline.SYSTEM_PROMPT and "get_previous_context" in translator.SYSTEM_PROMPT
    assert user == baseline.build_messages(page_for(3))[1]  # the same user message baseline sends
    assert [t["function"]["name"] for t in first["tools"]] == ["get_previous_context"]
    assert first["response_format"] == {"type": "json_object"}


def test_the_run_records_its_parameters_and_the_two_budgets(imported, root):
    provider = Provider(answer(zh_json(1)))

    outcome = translate(imported, root, mock_llm_client(provider), 1, page_budget=567, agent_budget=1234, max_steps=3,
                        max_tokens=99)

    assert json.loads(store.get_run(imported, outcome.run_id)["config_json"]) == {
        "agent_budget": 1234, "fallback_budget": 567, "max_steps": 3, "max_tokens": 99}
    assert provider.bodies[0]["max_tokens"] == 99


def test_the_default_parameters_are_the_documented_ones(imported, root):
    outcome = translate(imported, root, mock_llm_client(Provider(answer(zh_json(1)))), 1)

    assert json.loads(store.get_run(imported, outcome.run_id)["config_json"]) == {
        "agent_budget": 60_000, "fallback_budget": 40_000, "max_steps": 6, "max_tokens": 16_000}


def test_baseline_is_still_the_default_config_and_records_the_parameters_it_always_did(imported, root):
    client = FakeClient([zh_json(1)])

    outcome = runs.translate_page(imported, BOOK, 1, make_client=lambda: client, dataset_root=root)

    run = store.get_run(imported, outcome.run_id)
    assert (run["config"], json.loads(run["config_json"])) == ("baseline", {"page_budget": 40_000, "max_tokens": 16_000})
    assert isinstance(json.loads(run["attempts_json"]), list)


def test_an_unknown_config_is_a_value_error_before_a_client_is_built_or_a_run_exists(imported, root):
    def no_client():
        raise AssertionError("make_client must not be called")

    with pytest.raises(ValueError, match="unknown config 'nonexistent'"):
        runs.translate_page(imported, BOOK, 1, make_client=no_client, dataset_root=root, config="nonexistent")

    assert imported.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_a_crash_inside_the_agent_marks_the_translator_run_failed(imported, root):
    class Exploding(FakeClient):
        def chat(self, messages, **kwargs):
            raise RuntimeError("provider exploded")

    with pytest.raises(runs.RunCrashed) as crashed:
        translate(imported, root, Exploding([]), 1)

    run = store.get_run(imported, crashed.value.run_id)
    assert (run["config"], run["status"], run["error"]) == ("translator", "failed", "RuntimeError: provider exploded")


def test_tracing_switched_on_fails_the_run_without_a_request(imported, root, monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    provider = Provider(answer(zh_json(1)))

    with pytest.raises(runs.RunCrashed, match="LANGSMITH_TRACING") as crashed:
        translate(imported, root, mock_llm_client(provider), 1)

    assert provider.bodies == []
    assert store.get_run(imported, crashed.value.run_id)["status"] == "failed"


# --- AC2: no reference translation reaches a request --------------------------------------------------------------

def test_no_request_and_no_tool_result_holds_a_reference_translation(imported, root):
    bodies = []
    for page_index in (1, 2, 3):
        replies = [answer(zh_json(page_index))] if page_index == 1 else [ask(previous_context(page_index)),
                                                                         answer(zh_json(page_index))]
        provider = Provider(*replies)
        outcome = translate(imported, root, mock_llm_client(provider), page_index)
        assert outcome.result.ok
        bodies += provider.bodies

    sent = json.dumps(bodies, ensure_ascii=False)
    assert ja(2, 0) in sent and zh(2, 0) in sent  # page 3 really was given page 2's text and translation
    assert "REF-ZH" not in sent and "REF-EN" not in sent


def test_no_reference_reaches_the_repair_or_the_fallback_either(imported, root):
    translate_earlier_pages(imported, root, 1)
    provider = Provider(ask(previous_context(2)), answer("not json"), answer("not json either"), answer(zh_json(2)))

    outcome = translate(imported, root, mock_llm_client(provider), 2)

    assert outcome.result.fallback_reason and len(provider.bodies) == 4  # agent (2), repair, baseline
    sent = json.dumps(provider.bodies, ensure_ascii=False)
    assert "REF-ZH" not in sent and "REF-EN" not in sent


# --- AC3: get_previous_context ------------------------------------------------------------------------------------

def context_of(imported, root, page_index, arguments):
    """The result `get_previous_context` gave when the model asked with `arguments` while translating the page."""
    provider = Provider(ask(call("ctx", arguments, name="get_previous_context")), answer(zh_json(page_index)))
    outcome = translate(imported, root, mock_llm_client(provider), page_index)
    assert outcome.result.ok
    return json.loads(tool_result(provider.bodies[1], "ctx"))


def region(page_index, i, *, translated=True, tag=""):
    return {"id": i, "ja": ja(page_index, i), "zh": zh(page_index, i, tag) if translated else None}


def test_pages_back_1_returns_the_previous_page_with_its_translation(imported, root):
    translate_earlier_pages(imported, root, 1, 2)

    assert context_of(imported, root, 3, {"pages_back": 1}) == {"pages": [
        {"page_index": 2, "pages_back": 1, "translated": True, "regions": [region(2, 0), region(2, 1)]}]}


def test_pages_back_2_adds_the_page_before_that_oldest_first(imported, root):
    translate_earlier_pages(imported, root, 1, 2)

    context = context_of(imported, root, 3, {"pages_back": 2})

    assert [(p["page_index"], p["pages_back"]) for p in context["pages"]] == [(1, 2), (2, 1)]
    assert context["pages"][0]["regions"] == [region(1, 0), region(1, 1)]


def test_without_arguments_it_goes_one_page_back(imported, root):
    translate_earlier_pages(imported, root, 1, 2)

    assert [p["page_index"] for p in context_of(imported, root, 3, {})["pages"]] == [2]


def test_a_page_with_no_translator_run_is_given_in_japanese_only_and_says_so(imported, root):
    runs.translate_page(imported, BOOK, 1, make_client=lambda: FakeClient([zh_json(1)]), dataset_root=root)  # baseline
    translate_earlier_pages(imported, root, 2)

    pages = context_of(imported, root, 3, {"pages_back": 2})["pages"]

    assert pages[0] == {"page_index": 1, "pages_back": 2, "translated": False,
                        "regions": [region(1, 0, translated=False), region(1, 1, translated=False)]}
    assert pages[1]["translated"] is True  # page 1 has only a baseline Run: that is not the translator's own output


def test_the_latest_succeeded_translator_run_of_a_page_is_the_one_used(imported, root):
    translate_earlier_pages(imported, root, 2)
    assert translate(imported, root, mock_llm_client(Provider(answer(zh_json(2, "-v2")))), 2).result.ok
    assert not translate(imported, root, mock_llm_client(Provider(upstream_500())), 2).result.ok  # a failed Run is skipped

    (page,) = context_of(imported, root, 3, {})["pages"]

    assert page["regions"] == [region(2, 0, tag="-v2"), region(2, 1, tag="-v2")]


def test_on_the_first_page_it_says_there_is_no_earlier_context(imported, root):
    assert context_of(imported, root, 1, {"pages_back": 2}) == {
        "pages": [], "note": "There is no earlier page in this book."}


def test_pages_back_3_goes_back_to_the_model_as_a_validation_error_and_the_run_goes_on(imported, root):
    provider = Provider(ask(call("ctx", {"pages_back": 3}, name="get_previous_context")), answer(zh_json(3)))

    outcome = translate(imported, root, mock_llm_client(provider), 3)

    assert outcome.result.ok
    assert "pages_back: Input should be less than or equal to 2" in tool_result(provider.bodies[1], "ctx")
    (tool_call,) = attempts_of(imported, outcome.run_id)["agent"]["steps"][0]["tool_calls"]
    assert tool_call["ran"] is False and "pages_back" in tool_call["error"]


def test_the_tool_definition_says_what_pages_back_may_be(imported, root):
    provider = Provider(answer(zh_json(1)))
    translate(imported, root, mock_llm_client(provider), 1)

    (tool,) = provider.bodies[0]["tools"]
    pages_back = tool["function"]["parameters"]["properties"]["pages_back"]
    assert (pages_back["minimum"], pages_back["maximum"], pages_back["default"]) == (1, 2, 1)


def test_the_handler_runs_in_a_worker_thread_with_the_database_closed(imported, root, page_for, monkeypatch):
    translate_earlier_pages(imported, root, 1, 2)
    previous = runs.previous_pages(imported, BOOK, 3, "translator")
    imported.close()  # the handler gets its data from `previous` and must not need the connection
    threads = []
    real = translator.previous_context_tool

    def spying(*args):
        tool = real(*args)
        return replace(tool, handler=lambda **kw: (threads.append(threading.current_thread()), tool.handler(**kw))[1])

    monkeypatch.setattr(translator, "previous_context_tool", spying)
    provider = Provider(ask(call("ctx", {"pages_back": 2}, name="get_previous_context")), answer(zh_json(3)))

    result = translator.run_translator(page_for(3), mock_llm_client(provider), previous=previous)

    assert result.ok and result.translations == {0: zh(3, 0), 1: zh(3, 1)}
    (handler_thread,) = threads
    assert handler_thread is not threading.main_thread()  # a worker thread, where a plain connection would fail
    context = json.loads(tool_result(provider.bodies[1], "ctx"))
    assert [p["page_index"] for p in context["pages"]] == [1, 2]
    assert context["pages"][1]["regions"] == [region(2, 0), region(2, 1)]


def test_previous_pages_reads_only_what_the_tool_may_show(imported, root):
    translate_earlier_pages(imported, root, 1, 2)

    pages = runs.previous_pages(imported, BOOK, 3, "translator")

    assert [(p.page_index, p.translated) for p in pages] == [(1, True), (2, True)]
    assert set(vars(pages[0].regions[0])) == {"idx", "ja", "zh"}
    assert runs.previous_pages(imported, BOOK, 1, "translator") == []
    assert [p.page_index for p in runs.previous_pages(imported, BOOK, 2, "translator")] == [1]


# --- AC4: a bad answer is repaired once ---------------------------------------------------------------------------

def test_an_answer_that_does_not_parse_gets_one_repair_request_with_json_output_and_no_tools(imported, root):
    provider, meter = Provider(ask(previous_context(3)), answer("not json"), answer(zh_json(3))), Meter()

    outcome = translate(imported, root, mock_llm_client(provider, meter=meter), 3)

    assert outcome.result.ok and len(provider.bodies) == 3
    repair = provider.bodies[2]
    assert repair["response_format"] == {"type": "json_object"} and "tools" not in repair
    roles = [m["role"] for m in repair["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]  # the history, then the parse error
    assert repair["messages"][4]["content"] == "not json"
    assert "could not be used" in repair["messages"][5]["content"] and "no JSON object" in repair["messages"][5]["content"]
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["error"], run["calls"]) == ("succeeded", None, 3)
    assert zh_of(imported, outcome.run_id) == [zh(3, 0), zh(3, 1)]
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["fallback"] is None
    assert attempts["agent"]["answer_parse_error"] == "no JSON object in the response"
    assert set(attempts["repair"]) == {"error", "record", "raw", "parse_error"}
    assert (attempts["repair"]["raw"], attempts["repair"]["error"], attempts["repair"]["parse_error"]) == (
        zh_json(3), None, None)
    assert attempts["repair"]["record"] == asdict(meter.records[2])


@pytest.mark.parametrize("bad", [
    json.dumps({"translations": [{"id": 0, "zh": "字"}]}),
    json.dumps({"translations": [{"id": i, "zh": "字"} for i in (0, 1, 7)]}),
    json.dumps({"translations": [{"id": i, "zh": "字"} for i in (0, 1, 1)]}),
], ids=["missing-id", "unknown-id", "duplicate-id"])
def test_an_answer_with_the_wrong_region_ids_is_repaired_too(imported, root, bad):
    provider = Provider(answer(bad), answer(zh_json(1)))

    outcome = translate(imported, root, mock_llm_client(provider), 1)

    assert outcome.result.ok and len(provider.bodies) == 2
    assert attempts_of(imported, outcome.run_id)["agent"]["answer_parse_error"]


def test_a_repair_that_does_not_parse_either_falls_back_to_baseline(imported, root, page_for):
    provider = Provider(answer("not json"), answer("still not json"), answer(zh_json(3)))

    outcome = translate(imported, root, mock_llm_client(provider), 3)

    assert outcome.result.ok and len(provider.bodies) == 3  # the agent's answer, the one repair, baseline's call
    assert provider.bodies[2]["messages"] == baseline.build_messages(page_for(3))  # exactly baseline's request
    assert "tools" not in provider.bodies[2]
    run = store.get_run(imported, outcome.run_id)
    assert (run["config"], run["status"], run["error"], run["calls"]) == ("translator", "succeeded", None, 3)
    assert run["cost_usd"] == pytest.approx(3 * CALL_COST)
    assert zh_of(imported, outcome.run_id) == [zh(3, 0), zh(3, 1)]
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["repair"]["raw"] == "still not json" and "no JSON object" in attempts["repair"]["parse_error"]
    assert attempts["fallback"]["reason"].startswith("unparseable")
    assert [a["raw"] for a in attempts["fallback"]["attempts"]] == [zh_json(3)]
    assert outcome.result.fallback_reason == attempts["fallback"]["reason"]


def test_an_empty_final_answer_is_repaired_like_any_other_unusable_one(imported, root):
    provider = Provider(answer(""), answer(zh_json(1)))

    outcome = translate(imported, root, mock_llm_client(provider), 1)

    assert outcome.result.ok and len(provider.bodies) == 2


# --- AC5: fallback, and where it must not happen ------------------------------------------------------------------

def test_the_step_limit_falls_back_to_baseline_and_names_the_cause(imported, root, page_for):
    asks = [ask(call("c1", {"pages_back": 1}, name="get_previous_context")),
            ask(call("c2", {"pages_back": 2}, name="get_previous_context"))]
    provider = Provider(*asks, answer(zh_json(3)))

    outcome = translate(imported, root, mock_llm_client(provider), 3, max_steps=2)

    assert outcome.result.ok and len(provider.bodies) == 3  # two steps, then baseline's one call
    assert provider.bodies[2]["messages"] == baseline.build_messages(page_for(3)) and "tools" not in provider.bodies[2]
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["error"], run["calls"]) == ("succeeded", None, 3)
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["agent"]["stop_reason"] == "step_limit" and len(attempts["agent"]["steps"]) == 2
    assert attempts["repair"] is None
    assert attempts["fallback"]["reason"].startswith("step_limit") and "2" in attempts["fallback"]["reason"]


def test_a_reply_cut_off_falls_back_to_baseline_and_names_the_cause(imported, root):
    provider = Provider(answer("{\"translations\": [{\"id\"", finish_reason="length"), answer(zh_json(2)))

    outcome = translate(imported, root, mock_llm_client(provider), 2)

    assert outcome.result.ok and len(provider.bodies) == 2  # no repair: the reply was never an answer
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["agent"]["stop_reason"] == "incomplete" and attempts["repair"] is None
    assert attempts["fallback"]["reason"].startswith("incomplete") and "length" in attempts["fallback"]["reason"]


def test_the_fallback_gets_its_own_budget(imported, root, monkeypatch):
    budgets = []
    real = baseline.run_baseline

    def spy(page, client, budget, **kwargs):
        budgets.append((budget.limit_tokens, budget.used_tokens))
        return real(page, client, budget, **kwargs)

    monkeypatch.setattr(baseline, "run_baseline", spy)
    asks = [ask(call("c1", {"pages_back": 1}, name="get_previous_context")),
            ask(call("c2", {"pages_back": 2}, name="get_previous_context"))]
    provider = Provider(*asks, answer(zh_json(3)))

    outcome = translate(imported, root, mock_llm_client(provider), 3, max_steps=2, agent_budget=10_000, page_budget=5_000)

    assert outcome.result.ok
    assert budgets == [(5_000, 0)]  # the agent's 240 tokens are not counted against it


def test_a_fallback_that_fails_is_a_failed_run_with_the_cost_of_every_request(imported, root):
    provider = Provider(answer("a"), answer("b"), answer("c"), answer("d"))

    outcome = translate(imported, root, mock_llm_client(provider), 3)

    assert not outcome.result.ok and len(provider.bodies) == 4  # agent, repair, and baseline's two attempts
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"], run["cost_usd"]) == ("failed", 4, pytest.approx(4 * CALL_COST))
    assert run["error"].startswith("Could not parse the model response after 2 attempt(s)")
    assert imported.execute("SELECT COUNT(*) FROM translations").fetchone()[0] == 0
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["fallback"]["reason"].startswith("unparseable") and len(attempts["fallback"]["attempts"]) == 2


def test_a_provider_error_in_the_agent_fails_the_run_without_another_request(imported, root):
    provider = Provider(ask(previous_context(3)), upstream_500())

    outcome = translate(imported, root, mock_llm_client(provider), 3)

    assert not outcome.result.ok and len(provider.bodies) == 2  # nothing was resent, and baseline was not tried
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"], run["unknown_usage_calls"], run["cost_usd"]) == ("failed", 2, 1, None)
    assert "API error 500" in run["error"] and run["error"] == outcome.result.error
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["agent"]["stop_reason"] == "provider_error" and attempts["agent"]["error"] == run["error"]
    assert attempts["agent"]["steps"][1]["error"] == run["error"]
    assert attempts["repair"] is None and attempts["fallback"] is None
    assert imported.execute("SELECT COUNT(*) FROM translations").fetchone()[0] == 0


def test_a_provider_error_on_the_first_call_is_the_only_request(imported, root):
    provider = Provider(upstream_500())

    outcome = translate(imported, root, mock_llm_client(provider), 1)

    assert not outcome.result.ok and len(provider.bodies) == 1
    assert store.get_run(imported, outcome.run_id)["calls"] == 1


def test_a_spent_budget_fails_the_run_without_another_request(imported, root):
    provider = Provider(ask(previous_context(3)), answer(zh_json(3)))

    outcome = translate(imported, root, mock_llm_client(provider), 3, agent_budget=100)  # the first call uses 120

    assert not outcome.result.ok and len(provider.bodies) == 1
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"]) == ("failed", 1)
    assert run["error"].startswith("Stopped:") and "used 120 of 100" in run["error"]
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["agent"]["stop_reason"] == "budget" and attempts["fallback"] is None


def test_a_provider_error_on_the_repair_fails_the_run_without_a_fallback(imported, root):
    provider = Provider(answer("not json"), upstream_500())

    outcome = translate(imported, root, mock_llm_client(provider), 1)

    assert not outcome.result.ok and len(provider.bodies) == 2
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"], run["unknown_usage_calls"]) == ("failed", 2, 1)
    assert "API error 500" in run["error"]
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["repair"]["error"] == run["error"] and attempts["repair"]["record"]["usage"] is None
    assert attempts["repair"]["raw"] is None and attempts["fallback"] is None


def test_a_budget_spent_before_the_repair_fails_the_run_without_a_fallback(imported, root):
    provider = Provider(answer("not json"), answer(zh_json(1)))

    outcome = translate(imported, root, mock_llm_client(provider), 1, agent_budget=120)  # the answer used all of it

    assert not outcome.result.ok and len(provider.bodies) == 1
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"]) == ("failed", 1) and run["error"].startswith("Stopped:")
    attempts = attempts_of(imported, outcome.run_id)
    assert attempts["agent"]["answer_parse_error"] and attempts["repair"] is None and attempts["fallback"] is None


# --- AC6: the eval harness ----------------------------------------------------------------------------------------

def evaluate(conn, root, tmp_path, provider, pages, *, config="translator"):
    return harness.run_eval(
        conn, config=config, book=BOOK, page_indexes=pages, make_client=lambda: mock_llm_client(provider),
        dataset_root=root, out_root=tmp_path / "out", repo_root=app_config.ROOT,
    )


def test_the_harness_flags_the_pages_that_fell_back_per_page_and_in_total(imported, root, tmp_path):
    provider = Provider(
        answer(zh_json(1)),                                          # page 1: the agent answers
        answer("", finish_reason="length"), answer(zh_json(2)),      # page 2: cut off, baseline answers
        ask(previous_context(3)), answer(zh_json(3)),                # page 3: a tool call, then the answer
    )

    result = evaluate(imported, root, tmp_path, provider, [1, 2, 3])

    assert result.exit_code == 0 and len(provider.bodies) == 5
    scores = json.loads((result.out_dir / "scores.json").read_text(encoding="utf-8"))
    assert scores["config"] == "translator" and scores["fallback_pages"] == 1
    assert [(p["page_index"], p["fell_back"]) for p in scores["pages"]] == [(1, False), (2, True), (3, False)]
    assert scores["pages"][1]["fallback_reason"].startswith("incomplete") and scores["pages"][0]["fallback_reason"] is None
    assert [p["calls"] for p in scores["pages"]] == [1, 2, 2]
    assert (scores["pages_scored"], scores["segments_scored"]) == (3, 6)
    assert scores["cost"]["total_usd"] == pytest.approx(5 * CALL_COST, abs=1e-6)
    manifest = json.loads((result.out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_name"] == "translator"
    assert manifest["config_params"] == {"agent_budget": 60_000, "fallback_budget": 40_000, "max_steps": 6,
                                         "max_tokens": 16_000}
    assert [r["config"] for r in imported.execute("SELECT config FROM runs ORDER BY id")] == ["translator"] * 3
    # the eval goes page by page: page 3 was given page 2's output, which came from the fallback
    assert zh(2, 0) in tool_result(provider.bodies[4], "call_3")
    summary = harness.format_summary(result)
    assert "fallback to baseline  1 of 3 scored pages" in summary
    assert "FELL BACK  book_t/2" in summary and "incomplete" in summary


def test_a_page_that_fell_back_and_failed_too_is_listed_as_failed_and_marked(imported, root, tmp_path):
    provider = Provider(
        answer("", finish_reason="length"), answer("x"), answer("y"),   # page 1: cut off, baseline fails twice
        answer(zh_json(2)),                                             # page 2: clean
    )

    result = evaluate(imported, root, tmp_path, provider, [1, 2])

    assert result.exit_code == 1
    (failed,) = result.scores["failed_pages"]
    assert (failed["page_index"], failed["fell_back"]) == (1, True) and failed["fallback_reason"].startswith("incomplete")
    assert result.scores["fallback_pages"] == 0  # counted over the scored pages
    assert [p["fell_back"] for p in result.scores["pages"]] == [False]
    assert "(after falling back: incomplete" in harness.format_summary(result)


def test_a_translator_eval_that_never_falls_back_counts_zero(imported, root, tmp_path):
    result = evaluate(imported, root, tmp_path, Provider(answer(zh_json(1))), [1])

    assert result.scores["fallback_pages"] == 0 and result.scores["pages"][0]["fell_back"] is False
    assert "FELL BACK" not in harness.format_summary(result)


def test_a_baseline_eval_reports_zero_fallback_pages(imported, root, tmp_path):
    result = harness.run_eval(imported, config="baseline", book=BOOK, page_indexes=[1],
                              make_client=lambda: FakeClient([zh_json(1)]), dataset_root=root,
                              out_root=tmp_path / "out", repo_root=app_config.ROOT)

    assert result.scores["fallback_pages"] == 0 and result.scores["pages"][0]["fell_back"] is False
    assert "fallback to baseline" not in harness.format_summary(result)  # nothing to fall back to for baseline


def test_the_harness_knows_both_configs_and_the_cli_accepts_translator(tmp_path):
    assert harness.CONFIGS == ("baseline", "translator", "full_no_kb", "full")  # `full_no_kb` joined in task 012, `full` in 016

    stopped = run_cli(tmp_path, "--config", "translator", "--book", "book_t", "--pages", "1")

    assert "invalid choice" not in stopped.stderr
    assert stopped.returncode == 2 and "Stopped before any model call" in stopped.stderr  # no such page: nothing sent


# --- flagging: the `flag_uncertain` tool (task 011) ---------------------------------------------------------------

PREVIOUS_CONTEXT_TOOL = {   # the tool definition of M2, as the provider got it
    "type": "function",
    "function": {
        "name": "get_previous_context",
        "description": "Get the Japanese text, and the Chinese translation when there is one, of the pages just before "
                       "this one in the book.",
        "parameters": {
            "properties": {"pages_back": {
                "default": 1, "maximum": 2, "minimum": 1, "type": "integer",
                "description": "How many pages back to look: 1 is the previous page, 2 also adds the page before that."}},
            "type": "object"},
    },
}


def flag(call_id, region, reason="言い回しが曖昧"):
    return call(call_id, {"id": region, "reason": reason}, name="flag_uncertain")


def run_flagging(page_for, provider, *, page_index=3, **options):
    return translator.run_translator(page_for(page_index), mock_llm_client(provider), flagging=True, **options)


@pytest.mark.parametrize("options", [{}, {"flagging": False}], ids=["default", "explicitly-off"])
def test_with_the_switch_off_the_requests_are_what_they_were_in_m2(page_for, options):
    provider = Provider(ask(previous_context(3)), answer("not json"), answer(zh_json(3)))

    result = translator.run_translator(page_for(3), mock_llm_client(provider), **options)

    assert result.ok and len(provider.bodies) == 3 and result.flags == []
    first = provider.bodies[0]
    assert list(first) == ["model", "messages", "max_tokens", "response_format", "tools"]
    assert first["messages"][0] == {"role": "system", "content": translator.SYSTEM_PROMPT}
    assert first["messages"][1] == baseline.build_messages(page_for(3))[1]
    assert first["tools"] == [PREVIOUS_CONTEXT_TOOL]
    assert "flag_uncertain" not in json.dumps(provider.bodies) and "flag_uncertain" not in translator.SYSTEM_PROMPT
    assert provider.bodies[1]["tools"] == [PREVIOUS_CONTEXT_TOOL] and "tools" not in provider.bodies[2]  # and the repair


def test_two_flags_in_one_reply_are_both_kept_and_the_first_request_has_both_tools(page_for):
    provider = Provider(
        ask(flag("f1", 1, "名前かもしれない"), flag("f0", 0, "言い回しが曖昧")), answer(zh_json(3)))

    result = run_flagging(page_for, provider)

    assert result.ok and result.translations == {0: zh(3, 0), 1: zh(3, 1)}
    assert result.flags == [{"id": 0, "reason": "言い回しが曖昧"}, {"id": 1, "reason": "名前かもしれない"}]  # region order
    first = provider.bodies[0]
    assert [t["function"]["name"] for t in first["tools"]] == ["get_previous_context", "flag_uncertain"]
    assert first["tools"][0] == PREVIOUS_CONTEXT_TOOL
    assert first["messages"][0] == {"role": "system", "content": translator.FLAG_SYSTEM_PROMPT}
    assert translator.FLAG_SYSTEM_PROMPT != translator.SYSTEM_PROMPT
    assert "get_previous_context" in translator.FLAG_SYSTEM_PROMPT and "flag_uncertain" in translator.FLAG_SYSTEM_PROMPT
    assert first["messages"][1] == baseline.build_messages(page_for(3))[1]  # the user message is baseline's, as before
    parameters = first["tools"][1]["function"]["parameters"]
    assert parameters["required"] == ["id", "reason"]
    assert (parameters["properties"]["reason"]["minLength"], parameters["properties"]["reason"]["maxLength"]) == (1, 200)
    assert parameters["properties"]["id"]["type"] == "integer"
    assert tool_result(provider.bodies[1], "f0") == '{"flagged": 0}'
    steps = result.attempts["agent"]["steps"]
    assert [c["ran"] for c in steps[0]["tool_calls"]] == [True, True]
    assert set(result.attempts) == {"config", "agent", "repair", "fallback"}  # the attempts record keeps its shape


def test_a_run_that_flags_nothing_has_no_flags(page_for):
    result = run_flagging(page_for, Provider(answer(zh_json(3))))

    assert result.ok and result.flags == []


def test_an_unknown_region_goes_back_to_the_model_and_is_not_recorded(page_for):
    provider = Provider(ask(flag("bad", 7)), ask(flag("good", 1, "固有名詞")), answer(zh_json(3)))

    result = run_flagging(page_for, provider)

    assert result.ok and result.flags == [{"id": 1, "reason": "固有名詞"}]
    message = tool_result(provider.bodies[1], "bad")
    assert message.startswith("Error: flag_uncertain failed with ValueError:") and "region 7 is not on this page" in message
    assert "[0, 1]" in message  # the model is told which ids there are
    (bad_call,) = result.attempts["agent"]["steps"][0]["tool_calls"]
    assert bad_call["ran"] is True and bad_call["error"] == message


@pytest.mark.parametrize("arguments, complaint", [
    ({"id": 0, "reason": ""}, "reason: String should have at least 1 character"),
    ({"id": 0, "reason": "   "}, "reason: String should have at least 1 character"),
    ({"id": 0, "reason": "x" * 201}, "reason: String should have at most 200 characters"),
    ({"id": 0}, "reason: Field required"),
    ({"id": "zero", "reason": "x"}, "id: Input should be a valid integer"),
], ids=["empty", "blank", "too-long", "missing", "id-not-a-number"])
def test_an_argument_that_does_not_validate_goes_back_to_the_model_and_is_not_recorded(page_for, arguments, complaint):
    provider = Provider(ask(call("bad", arguments, name="flag_uncertain")), answer(zh_json(3)))

    result = run_flagging(page_for, provider)

    assert result.ok and result.flags == []
    assert complaint in tool_result(provider.bodies[1], "bad")
    (bad_call,) = result.attempts["agent"]["steps"][0]["tool_calls"]
    assert bad_call["ran"] is False and complaint in bad_call["error"]


def test_the_longest_reason_is_kept_and_a_padded_one_is_trimmed(page_for):
    provider = Provider(ask(flag("a", 0, "x" * 200), flag("b", 1, "  迷い  ")), answer(zh_json(3)))

    result = run_flagging(page_for, provider)

    assert result.flags == [{"id": 0, "reason": "x" * 200}, {"id": 1, "reason": "迷い"}]


def test_the_same_flag_twice_is_not_run_again_and_not_recorded_twice(page_for):
    provider = Provider(ask(flag("a", 1, "同じ")), ask(flag("b", 1, "同じ")), answer(zh_json(3)))

    result = run_flagging(page_for, provider)

    assert result.flags == [{"id": 1, "reason": "同じ"}]
    assert tool_result(provider.bodies[2], "b").startswith("Duplicate call: flag_uncertain")
    assert [c["ran"] for s in result.attempts["agent"]["steps"] for c in s["tool_calls"]] == [True, False]


def test_a_region_flagged_with_two_reasons_keeps_both_in_the_order_they_came(page_for):
    provider = Provider(ask(flag("a", 1, "一つ目")), ask(flag("b", 1, "二つ目")), answer(zh_json(3)))

    result = run_flagging(page_for, provider)

    assert [f["reason"] for f in result.flags] == ["一つ目", "二つ目"]


def test_both_tools_can_be_called_in_one_reply(imported, root, page_for):
    translate_earlier_pages(imported, root, 1, 2)
    previous = runs.previous_pages(imported, BOOK, 3, "translator")
    provider = Provider(ask(previous_context(3), flag("f", 0)), answer(zh_json(3)))

    result = run_flagging(page_for, provider, previous=previous)

    assert result.ok and result.flags == [{"id": 0, "reason": "言い回しが曖昧"}]
    assert zh(2, 0) in tool_result(provider.bodies[1], "call_3")


def test_flags_are_kept_when_the_repair_gives_the_answer(page_for):
    provider = Provider(ask(flag("f", 1)), answer("not json"), answer(zh_json(3)))

    result = run_flagging(page_for, provider)

    assert result.ok and result.fallback_reason is None and len(provider.bodies) == 3
    assert result.attempts["repair"]["parse_error"] is None
    assert result.flags == [{"id": 1, "reason": "言い回しが曖昧"}]


@pytest.mark.parametrize("replies, how", [
    ((answer("not json"), answer("still not json"), answer(zh_json(3))), "unparseable"),
    ((answer("", finish_reason="length"), answer(zh_json(3))), "incomplete"),
], ids=["repair-failed", "reply-cut-off"])
def test_after_a_fallback_there_are_no_flags_but_the_steps_still_show_the_calls(page_for, replies, how):
    provider = Provider(ask(flag("f", 1)), *replies)

    result = run_flagging(page_for, provider)

    assert result.ok and result.fallback_reason.startswith(how)
    assert result.flags == []
    assert result.attempts["agent"]["steps"][0]["tool_calls"][0]["name"] == "flag_uncertain"
    assert result.attempts["agent"]["steps"][0]["tool_calls"][0]["ran"] is True


def test_after_the_step_limit_there_are_no_flags(page_for):
    provider = Provider(ask(flag("a", 0)), ask(flag("b", 1)), answer(zh_json(3)))

    result = run_flagging(page_for, provider, max_steps=2)

    assert result.ok and result.fallback_reason.startswith("step_limit")
    assert result.flags == [] and len(provider.bodies) == 3  # two steps, then baseline's call


@pytest.mark.parametrize("replies", [(upstream_500(),), (answer("not json"), upstream_500())],
                         ids=["agent", "repair"])
def test_a_failed_run_has_no_flags(page_for, replies):
    provider = Provider(ask(flag("f", 0)), *replies)

    result = run_flagging(page_for, provider)

    assert not result.ok and result.translations is None and result.flags == []


def test_the_flag_handler_is_safe_when_many_calls_run_side_by_side():
    ids = list(range(200))
    collector = translator.FlagCollector(ids)
    handler = collector.tool().handler
    barrier = threading.Barrier(len(ids))

    def flag_one(region):
        barrier.wait()  # all the threads start together
        handler(id=region, reason=f"reason {region}")

    threads = [threading.Thread(target=flag_one, args=(i,)) for i in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert [f["id"] for f in collector.flags] == ids
    assert all(f["reason"] == f"reason {f['id']}" for f in collector.flags)
    with pytest.raises(ValueError, match="region 200 is not on this page"):
        handler(id=200, reason="x")
    assert len(collector.flags) == 200


def test_the_flag_handler_runs_in_a_worker_thread(page_for, monkeypatch):
    threads = []
    real = translator.FlagCollector.tool

    def spying(self):
        tool = real(self)
        return replace(tool, handler=lambda **kw: (threads.append(threading.current_thread()), tool.handler(**kw))[1])

    monkeypatch.setattr(translator.FlagCollector, "tool", spying)

    result = run_flagging(page_for, Provider(ask(flag("a", 0), flag("b", 1)), answer(zh_json(3))))

    assert [f["id"] for f in result.flags] == [0, 1] and len(threads) == 2
    assert all(t is not threading.main_thread() for t in threads)


def test_no_reference_translation_reaches_a_flagging_request(page_for):
    provider = Provider(ask(flag("f", 0)), answer(zh_json(3)))

    run_flagging(page_for, provider)

    sent = json.dumps(provider.bodies, ensure_ascii=False)
    assert "REF-ZH" not in sent and "REF-EN" not in sent


def test_the_translator_config_does_not_flag(imported, root):
    provider = Provider(answer(zh_json(1)))

    outcome = translate(imported, root, mock_llm_client(provider), 1)

    assert outcome.result.flags == []
    assert [t["function"]["name"] for t in provider.bodies[0]["tools"]] == ["get_previous_context"]
    assert provider.bodies[0]["messages"][0]["content"] == translator.SYSTEM_PROMPT


# --- folded in from task 011's review (task 012) ------------------------------------------------------------------

class AppendsUnderTheLock(list):
    """The collector's list, made to say whether the collector's lock was held at each append.

    Under CPython's GIL a bare `list.append` is atomic, so contention cannot show a missing lock; this can.
    """

    def __init__(self, collector):
        super().__init__()
        self.collector = collector
        self.held = []

    def append(self, item):
        self.held.append(self.collector._lock.locked())
        super().append(item)


def test_the_flag_handler_appends_only_while_it_holds_the_lock():
    collector = translator.FlagCollector([0, 1])
    collector._flags = recorder = AppendsUnderTheLock(collector)
    handler = collector.tool().handler

    handler(id=0, reason="一つ目")
    handler(id=1, reason="二つ目")

    assert recorder.held == [True, True]  # false for an append outside `with self._lock`
    assert [f["id"] for f in collector.flags] == [0, 1]


@pytest.mark.parametrize("region", [True, False])
def test_a_boolean_region_id_goes_back_to_the_model_and_is_not_recorded(page_for, region):
    provider = Provider(ask(call("bad", f'{{"id": {str(region).lower()}, "reason": "x"}}', name="flag_uncertain")),
                        answer(zh_json(3)))

    result = run_flagging(page_for, provider)

    assert result.ok and result.flags == []
    assert "Error: the arguments of flag_uncertain are not valid" in tool_result(provider.bodies[1], "bad")
    (bad_call,) = result.attempts["agent"]["steps"][0]["tool_calls"]
    assert bad_call["ran"] is False


@pytest.mark.parametrize("region", ["1", 1.0])
def test_a_number_written_as_text_or_a_float_is_still_a_region_id(page_for, region):
    provider = Provider(ask(call("f", {"id": region, "reason": "x"}, name="flag_uncertain")), answer(zh_json(3)))

    result = run_flagging(page_for, provider)

    assert result.ok and result.flags == [{"id": 1, "reason": "x"}]  # only the booleans are refused
