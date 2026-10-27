"""The agent loop on a scripted provider: the real LLMClient, the real openai SDK, LangChain's create_agent, no network."""

import json
from dataclasses import asdict

import httpx
import pytest
from conftest import mock_llm_client
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from fukidashi.agents import loop
from fukidashi.agents.loop import Tool, run_agent
from fukidashi.agents.model import to_openai_message
from fukidashi.llm.metering import Meter, PageBudget


class Args(BaseModel):
    term: str
    n: int


def make_tool(calls, *, returns=None, raises=None):
    """A `lookup(term, n)` tool; `calls` collects what its handler was called with."""

    def lookup(term: str, n: int):
        calls.append((term, n))
        if raises is not None:
            raise raises
        return returns if returns is not None else f"found {term} {n}"

    return Tool(name="lookup", description="Look a term up.", args=Args, handler=lookup)


# --- a scripted provider -----------------------------------------------------------------------------------------

def call(call_id, args, name="lookup"):
    """A tool call as the provider sends it; `args` is a dict (sent as compact JSON) or the raw arguments string."""
    arguments = args if isinstance(args, str) else json.dumps(args)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def reply(message, finish_reason):
    body = {"id": "cmpl-1", "object": "chat.completion", "created": 0, "model": "mock-model-1",
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", **message}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
    return httpx.Response(200, json=body)


def ask(*calls, content=None, finish_reason="tool_calls", **extra):
    return reply({"content": content, "tool_calls": list(calls), **extra}, finish_reason)


def answer(text="done", finish_reason="stop", **extra):
    return reply({"content": text, **extra}, finish_reason)


def upstream_500():
    return httpx.Response(500, json={"error": {"message": "upstream exploded"}})


class Provider:
    """Replies in order (a callable reply gets the request's index); records each request body."""

    def __init__(self, *replies):
        self.replies = replies
        self.bodies = []

    def __call__(self, request):
        self.bodies.append(json.loads(request.content))
        r = self.replies[min(len(self.bodies), len(self.replies)) - 1]
        return r(len(self.bodies)) if callable(r) else r


def run(provider, tools=(), *, limit=10_000, **options):
    """`run_agent` on `provider`. Every run also holds the invariants of AC5: one step per call, plain JSON."""
    meter = Meter()
    result = run_agent(mock_llm_client(provider, meter=meter), PageBudget("book/001", limit),
                       system_prompt="SYS", user_message="USER", tools=list(tools), **options)
    assert len(result.steps) == len(meter.records) == len(provider.bodies)
    json.dumps(asdict(result))  # no LangChain objects anywhere in the result
    return result, meter


# --- one tool call, then the answer -------------------------------------------------------------------------------

def test_a_tool_call_then_an_answer_is_a_final_result_with_two_steps_and_two_calls():
    handled = []
    odd_spacing = call("call_1", '{ "term" : "カゲミザワ",  "n": 1 }')
    provider = Provider(ask(odd_spacing, content="Let me look.", reasoning_content="I should look it up."),
                        answer("阴泽前辈"))

    result, meter = run(provider, [make_tool(handled, returns={"zh": "阴泽"})])

    assert (result.stop_reason, result.text, result.error) == ("final", "阴泽前辈", None)
    assert len(result.steps) == 2 and meter.summary()["calls"] == 2
    assert handled == [("カゲミザワ", 1)]
    assistant, tool_message = provider.bodies[1]["messages"][2:]
    assert assistant == {"role": "assistant", "content": "Let me look.", "tool_calls": [odd_spacing],
                         "reasoning_content": "I should look it up."}  # as received, down to the argument spacing
    assert tool_message == {"role": "tool", "tool_call_id": "call_1", "content": '{"zh": "阴泽"}'}
    assert provider.bodies[1]["messages"][:2] == [{"role": "system", "content": "SYS"},
                                                  {"role": "user", "content": "USER"}]


def test_a_langchain_tool_is_accepted_as_it_is():
    handled = []
    structured = StructuredTool.from_function(lambda term, n: handled.append((term, n)) or "ok", name="lookup",
                                              description="Look a term up.", args_schema=Args)
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})), answer())

    result, _ = run(provider, [structured])

    assert (result.stop_reason, handled) == ("final", [("a", 1)])


def test_the_tool_definition_sent_has_the_fields_and_required_list_of_its_args_schema():
    provider = Provider(answer())

    run(provider, [make_tool([])])

    (tool,) = provider.bodies[0]["tools"]
    assert tool["type"] == "function" and tool["function"]["name"] == "lookup"
    assert tool["function"]["description"] == "Look a term up."
    parameters = tool["function"]["parameters"]
    assert set(parameters["properties"]) == set(Args.model_fields) and parameters["required"] == ["term", "n"]
    assert parameters["properties"]["n"]["type"] == "integer"


def test_the_steps_say_what_each_model_call_and_tool_call_did():
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1}), reasoning_content="think"), answer("ok"))

    result, meter = run(provider, [make_tool([])])

    first, second = result.steps
    assert first["record"] == asdict(meter.records[0]) and first["finish_reason"] == "tool_calls"
    assert first["reasoning_chars"] == 5 and first["error"] is None
    assert first["tool_calls"] == [{"id": "call_1", "name": "lookup", "arguments": '{"term": "a", "n": 1}',
                                    "result": "found a 1", "error": None, "duplicate": False, "ran": True}]
    assert (second["finish_reason"], second["text"], second["tool_calls"]) == ("stop", "ok", [])
    assert second["reasoning_chars"] is None  # the response had none


def test_the_history_is_plain_dicts_ending_with_the_answer():
    result, _ = run(Provider(ask(call("call_1", {"term": "a", "n": 1})), answer("ok")), [make_tool([])])

    assert [m["role"] for m in result.messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert result.messages[-1] == {"role": "assistant", "content": "ok"}


@pytest.mark.parametrize("reasoning", ["", "thinking"])
def test_reasoning_content_goes_back_unchanged_even_when_empty(reasoning):
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1}), reasoning_content=reasoning), answer())

    run(provider, [make_tool([])])

    assert provider.bodies[1]["messages"][2]["reasoning_content"] == reasoning


def test_a_reply_without_reasoning_content_sends_none_back():
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})), answer())

    run(provider, [make_tool([])])

    assert "reasoning_content" not in provider.bodies[1]["messages"][2]


def test_two_tool_calls_in_one_reply_both_run_and_both_results_go_back_in_order():
    handled = []
    provider = Provider(ask(call("call_a", {"term": "a", "n": 1}), call("call_b", {"term": "b", "n": 2})), answer())

    result, _ = run(provider, [make_tool(handled)])

    assert sorted(handled) == [("a", 1), ("b", 2)]
    results = provider.bodies[1]["messages"][3:]
    assert [(m["role"], m["tool_call_id"], m["content"]) for m in results] == [
        ("tool", "call_a", "found a 1"), ("tool", "call_b", "found b 2")]
    assert len(result.steps) == 2 and len(result.steps[0]["tool_calls"]) == 2


def test_json_output_is_sent_next_to_the_tools_only_when_asked():
    plain, json_mode = Provider(answer()), Provider(answer())

    run(plain, [make_tool([])])
    run(json_mode, [make_tool([])], json_output=True)

    assert "response_format" not in plain.bodies[0]
    assert json_mode.bodies[0]["response_format"] == {"type": "json_object"} and json_mode.bodies[0]["tools"]


def test_an_agent_without_tools_sends_no_tools_key():
    provider = Provider(answer("ok"))

    result, _ = run(provider)

    assert result.stop_reason == "final" and "tools" not in provider.bodies[0]


def test_max_tokens_reaches_the_request():
    provider = Provider(answer())

    run(provider, max_tokens=123)

    assert provider.bodies[0]["max_tokens"] == 123


# --- the guardrails -----------------------------------------------------------------------------------------------

def test_a_provider_that_always_asks_for_a_new_tool_call_stops_at_the_step_limit():
    handled = []
    provider = Provider(lambda i: ask(call(f"call_{i}", {"term": f"t{i}", "n": i})))

    result, meter = run(provider, [make_tool(handled)], max_steps=3)

    assert len(provider.bodies) == 3 and meter.summary()["calls"] == 3  # exactly max_steps requests
    assert (result.stop_reason, result.text) == ("step_limit", None)
    assert "limit" in result.error and "3" in result.error
    assert len(result.steps) == 3 and len(result.messages) == 2 + 3 * 2  # system, user, then call + result per step


def test_the_default_step_limit_is_six():
    provider = Provider(lambda i: ask(call(f"call_{i}", {"term": f"t{i}", "n": i})))

    result, _ = run(provider, [make_tool([])])

    assert (len(provider.bodies), result.stop_reason) == (6, "step_limit")


def test_a_budget_too_small_for_the_next_call_stops_the_run_without_another_request():
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})), answer())

    result, meter = run(provider, [make_tool([])], limit=100)  # the first call uses 120 tokens

    assert len(provider.bodies) == 1  # nothing was sent for the second call
    assert (result.stop_reason, result.text) == ("budget", None)
    assert "used 120 of 100" in result.error
    assert len(result.steps) == 1 and result.steps[0]["tool_calls"][0]["result"] == "found a 1"


def test_a_budget_spent_before_the_first_call_sends_nothing():
    provider = Provider(answer())

    result, meter = run(provider, limit=0)

    assert (result.stop_reason, result.steps, len(provider.bodies), meter.records) == ("budget", [], 0, [])


def test_a_provider_error_at_step_two_stops_the_run_after_exactly_two_requests():
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})), upstream_500())

    result, meter = run(provider, [make_tool([])])

    assert len(provider.bodies) == 2  # no retry
    assert (result.stop_reason, result.text) == ("provider_error", None)
    assert "API error 500" in result.error and "upstream exploded" in result.error
    first, second = result.steps
    assert first["tool_calls"][0]["result"] == "found a 1"
    assert second["error"] == result.error and second["record"] == asdict(meter.records[1])
    assert second["record"]["usage"] is None and second["tool_calls"] == [] and second["finish_reason"] is None
    assert meter.summary()["unknown_usage_calls"] == 1 and meter.summary()["cost_usd"] is None


def test_a_provider_error_at_step_one_is_the_only_step():
    result, _ = run(Provider(upstream_500()))

    assert (result.stop_reason, len(result.steps)) == ("provider_error", 1)
    assert [m["role"] for m in result.messages] == ["system", "user"]


@pytest.mark.parametrize("finish_reason", ["length", "content_filter", "insufficient_system_resource", None])
def test_a_reply_cut_off_for_any_reason_is_never_the_answer(finish_reason):
    provider = Provider(answer("", finish_reason=finish_reason))

    result, _ = run(provider)

    assert (result.stop_reason, result.text, len(provider.bodies)) == ("incomplete", None, 1)
    assert repr(finish_reason) in result.error


def test_a_cut_off_reply_with_text_is_not_the_answer_either():
    result, _ = run(Provider(answer("半截", finish_reason="length")))

    assert (result.stop_reason, result.text) == ("incomplete", None)
    assert result.steps[0]["text"] == "半截"  # kept in the step, for whoever debugs it


def test_a_cut_off_reply_with_tool_calls_does_not_run_them():
    handled = []
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1}), finish_reason="length"), answer())

    result, _ = run(provider, [make_tool(handled)])

    assert (result.stop_reason, len(provider.bodies), handled) == ("incomplete", 1, [])
    (tool_call,) = result.steps[0]["tool_calls"]  # recorded in the step, not run
    assert (tool_call["name"], tool_call["arguments"]) == ("lookup", '{"term": "a", "n": 1}')
    assert (tool_call["result"], tool_call["error"], tool_call["ran"]) == (None, None, False)


def test_an_empty_answer_with_finish_reason_stop_is_final():
    provider = Provider(answer(""))

    result, _ = run(provider, [make_tool([])])

    assert (result.stop_reason, result.text, result.error, len(provider.bodies)) == ("final", "", None, 1)


def test_a_failure_that_is_not_the_providers_is_not_swallowed(monkeypatch):
    client = mock_llm_client(Provider(answer()))
    monkeypatch.setattr(client, "chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("programming error")))

    with pytest.raises(RuntimeError, match="programming error"):
        run_agent(client, PageBudget("book/001", 10_000), system_prompt="SYS", user_message="USER")


def test_max_steps_must_be_at_least_one():
    with pytest.raises(ValueError, match="max_steps"):
        run_agent(mock_llm_client(Provider(answer())), PageBudget("book/001", 10_000),
                  system_prompt="SYS", user_message="USER", max_steps=0)


# --- a model's mistakes go back to it as the tool's result ---------------------------------------------------------

MISTAKES = [
    pytest.param(call("call_1", {"term": "a", "n": 1}, name="nope"), "there is no tool named nope", 0, False,
                 id="unknown-tool"),
    pytest.param(call("call_1", '{"term": '), "not a JSON object", 0, False, id="not-json"),
    pytest.param(call("call_1", "[1, 2]"), "not a JSON object", 0, False, id="not-an-object"),
    pytest.param(call("call_1", {"term": "a"}), "n: Field required", 0, False, id="missing-field"),
    pytest.param(call("call_1", {"term": "a", "n": "many"}), "n: Input should be a valid integer", 0, False,
                 id="wrong-type"),
    pytest.param(call("call_1", {"term": ["a"], "n": 1}), "term: Input should be a valid string", 0, False,
                 id="wrong-type-in-the-first-field"),
]


@pytest.mark.parametrize("mistake, phrase, handler_calls, ran", MISTAKES)
def test_a_mistake_of_the_model_is_the_tool_result_and_the_run_still_answers(mistake, phrase, handler_calls, ran):
    handled = []
    provider = Provider(ask(mistake), answer("fixed"))

    result, _ = run(provider, [make_tool(handled)])

    assert (result.stop_reason, result.text) == ("final", "fixed")
    assert len(handled) == handler_calls  # the handler never saw arguments that failed validation
    sent_back = provider.bodies[1]["messages"][-1]
    assert sent_back["role"] == "tool" and sent_back["tool_call_id"] == "call_1" and phrase in sent_back["content"]
    (tool_call,) = result.steps[0]["tool_calls"]
    assert phrase in tool_call["error"] and tool_call["result"] is None and tool_call["ran"] is ran
    assert tool_call["duplicate"] is False


def test_the_arguments_in_the_step_are_the_raw_string_even_when_it_is_not_json():
    result, _ = run(Provider(ask(call("call_1", '{"term": ')), answer()), [make_tool([])])

    assert result.steps[0]["tool_calls"][0]["arguments"] == '{"term": '
    assert provider_message_arguments(result) == '{"term": '


def provider_message_arguments(result):
    return result.messages[2]["tool_calls"][0]["function"]["arguments"]


def test_a_handler_that_raises_is_a_tool_error_and_the_run_goes_on():
    handled = []
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})), answer("recovered"))

    result, _ = run(provider, [make_tool(handled, raises=ValueError("the glossary is down"))])

    assert (result.stop_reason, result.text, handled) == ("final", "recovered", [("a", 1)])
    assert "failed with ValueError: the glossary is down" in provider.bodies[1]["messages"][-1]["content"]
    (tool_call,) = result.steps[0]["tool_calls"]
    assert "the glossary is down" in tool_call["error"] and tool_call["ran"] is True


def test_a_result_that_is_not_json_serializable_is_a_tool_error_not_a_crash():
    handled = []
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})), answer("recovered"))

    result, _ = run(provider, [make_tool(handled, returns={1, 2})])  # a set

    assert (result.stop_reason, result.text, handled) == ("final", "recovered", [("a", 1)])
    assert "returned a set, which is not JSON-serializable" in provider.bodies[1]["messages"][-1]["content"]
    (tool_call,) = result.steps[0]["tool_calls"]
    assert tool_call["result"] is None and tool_call["ran"] is True and "not JSON-serializable" in tool_call["error"]


def test_a_tool_result_that_is_structured_goes_back_as_json_text():
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})), answer())

    run(provider, [make_tool([], returns={"term": "カゲミザワ", "zh": "阴泽", "tags": [1, None]})])

    assert provider.bodies[1]["messages"][-1]["content"] == '{"term": "カゲミザワ", "zh": "阴泽", "tags": [1, null]}'


# --- duplicates ----------------------------------------------------------------------------------------------------

def test_the_same_call_with_its_keys_in_another_order_is_a_duplicate_and_is_not_run_again():
    handled = []
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})), ask(call("call_2", '{"n": 1, "term": "a"}')),
                        answer())

    result, _ = run(provider, [make_tool(handled)])

    assert handled == [("a", 1)] and result.stop_reason == "final"
    sent_back = provider.bodies[2]["messages"]
    assert [m["tool_call_id"] for m in sent_back if m["role"] == "tool"] == ["call_1", "call_2"]
    assert "Duplicate call" in sent_back[-1]["content"] and "found a 1" == sent_back[-3]["content"]
    first, second, _ = result.steps
    assert (first["tool_calls"][0]["duplicate"], first["tool_calls"][0]["ran"]) == (False, True)
    duplicate = second["tool_calls"][0]
    assert (duplicate["duplicate"], duplicate["ran"], duplicate["error"]) == (True, False, None)
    assert "Duplicate call" in duplicate["result"]


def test_arguments_are_compared_after_validation():
    handled = []
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})),
                        ask(call("call_2", {"term": "a", "n": "1", "extra": True})), answer())  # "1" reads as 1

    result, _ = run(provider, [make_tool(handled)])

    assert handled == [("a", 1)] and result.steps[1]["tool_calls"][0]["duplicate"] is True


def test_different_arguments_are_not_duplicates():
    handled = []
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})), ask(call("call_2", {"term": "a", "n": 2})),
                        answer())

    result, _ = run(provider, [make_tool(handled)])

    assert handled == [("a", 1), ("a", 2)] and not any(c["duplicate"] for s in result.steps for c in s["tool_calls"])


@pytest.mark.parametrize("attempt", range(10))
def test_two_identical_calls_in_one_reply_run_once_and_the_later_one_is_the_duplicate(attempt):
    handled = []
    provider = Provider(ask(call("call_a", {"term": "a", "n": 1}), call("call_b", {"n": 1, "term": "a"})), answer())

    result, _ = run(provider, [make_tool(handled)])

    first, second = result.steps[0]["tool_calls"]
    assert handled == [("a", 1)]
    assert (first["id"], first["duplicate"], first["ran"]) == ("call_a", False, True)
    assert (second["id"], second["duplicate"], second["ran"]) == ("call_b", True, False)


def test_repeating_a_call_whose_handler_raised_is_a_duplicate():
    handled = []
    provider = Provider(ask(call("call_1", {"term": "a", "n": 1})), ask(call("call_2", {"term": "a", "n": 1})),
                        answer())

    result, _ = run(provider, [make_tool(handled, raises=RuntimeError("down"))])

    assert handled == [("a", 1)]  # the second time it was not tried
    assert result.steps[1]["tool_calls"][0]["duplicate"] is True


def test_a_call_that_failed_validation_never_ran_so_repeating_it_is_not_a_duplicate():
    handled = []
    invalid = {"term": "a"}  # no n
    provider = Provider(ask(call("call_1", invalid)), ask(call("call_2", invalid)),
                        ask(call("call_3", {"term": "a", "n": 1})), answer())

    result, _ = run(provider, [make_tool(handled)])

    first, second, third = (s["tool_calls"][0] for s in result.steps[:3])
    assert "n: Field required" in first["error"] and "n: Field required" in second["error"]  # not "Duplicate call"
    assert (first["duplicate"], second["duplicate"], third["duplicate"]) == (False, False, False)
    assert handled == [("a", 1)] and third["ran"] is True


def test_an_unknown_tool_or_malformed_arguments_are_not_remembered_as_calls():
    provider = Provider(ask(call("call_1", '{"term": ')), ask(call("call_2", '{"term": ')), answer())

    result, _ = run(provider, [make_tool([])])

    assert all("not a JSON object" in s["tool_calls"][0]["error"] for s in result.steps[:2])


# --- the adapter ----------------------------------------------------------------------------------------------------

def test_an_ai_message_not_made_by_the_adapter_is_sent_with_its_parsed_arguments():
    message = AIMessage(content="", tool_calls=[{"id": "c1", "name": "lookup", "args": {"term": "日本", "n": 1},
                                                 "type": "tool_call"}])

    assert to_openai_message(message) == {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": '{"term": "日本", "n": 1}'}}]}


def test_a_message_the_provider_has_no_role_for_is_refused():
    from langchain_core.messages import ChatMessage

    with pytest.raises(TypeError, match="ChatMessage"):
        to_openai_message(ChatMessage(role="critic", content="hm"))


def test_the_model_adapter_refuses_a_tool_choice_it_cannot_send():
    from fukidashi.agents.model import LLMClientChatModel

    model = LLMClientChatModel(client=None, budget=None, max_tokens=10)

    with pytest.raises(ValueError, match="tool_choice"):
        model.bind_tools([make_tool([])], tool_choice="any")


def test_the_loop_module_names_its_stop_reasons():
    assert (loop.FINAL, loop.STEP_LIMIT, loop.INCOMPLETE, loop.BUDGET, loop.PROVIDER_ERROR) == (
        "final", "step_limit", "incomplete", "budget", "provider_error")
