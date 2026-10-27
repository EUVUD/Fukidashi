"""LLMClient against a mock provider (httpx.MockTransport): the real SDK, no network, no real model."""

import json

import httpx
import openai
import pytest
from conftest import completion, mock_llm_client

from fukidashi.llm.client import LLMClient, LLMConfig
from fukidashi.llm.errors import ProviderError, describe_api_error
from fukidashi.llm.metering import Meter, PageBudget, Usage

MESSAGES = [{"role": "user", "content": "こんにちは"}]
KEY = "sk-test-0123456789"


def refuse(request):
    raise httpx.ConnectError("connection refused", request=request)


def time_out(request):
    raise httpx.ReadTimeout("read timed out", request=request)


def upstream_500(request):
    return httpx.Response(500, json={"error": {"message": "upstream exploded"}})


def wrong_key(request):
    return httpx.Response(401, json={"error": {"message": "Incorrect API key provided"}})


def rate_limited(request):
    return httpx.Response(429, json={"error": {"message": "slow down"}})


# handler, the status code ProviderError carries, a phrase the readable description must contain
FAILURES = [
    pytest.param(refuse, None, "Cannot reach http://provider.test/v1", id="connection-error"),
    pytest.param(time_out, None, "timed out", id="timeout"),
    pytest.param(upstream_500, 500, "API error 500", id="http-500"),
    pytest.param(wrong_key, 401, "Authentication failed (401)", id="http-401"),
    pytest.param(rate_limited, 429, "Rate limited", id="http-429"),
]


def ledger_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("handler, status_code, readable", FAILURES)
def test_a_failed_request_is_recorded_as_one_unknown_call(handler, status_code, readable, tmp_path):
    ledger = tmp_path / "usage.jsonl"
    meter = Meter(ledger_path=ledger)
    client = mock_llm_client(handler, meter=meter, api_key=KEY)
    budget = PageBudget("book/001", 10_000)

    with pytest.raises(ProviderError) as raised:
        client.chat(MESSAGES, budget=budget, json_output=True)

    error = raised.value
    assert error.status_code == status_code
    assert readable in error.description and str(error) == error.description
    (rec,) = meter.records
    assert (rec.page_id, rec.requested_model, rec.response_model) == ("book/001", "mock-model", None)
    assert (rec.usage, rec.cost_usd, rec.error) == (None, None, error.description)
    assert rec.latency_s >= 0
    assert (budget.unknown_calls, budget.used_tokens, budget.verified) == (1, 0, False)
    assert meter.summary() == {"calls": 1, "unknown_usage_calls": 1, "input_tokens": 0, "output_tokens": 0,
                               "cost_usd": None}
    (line,) = ledger_lines(ledger)
    assert (line["usage"], line["cost_usd"], line["error"]) == (None, None, error.description)


def test_a_successful_call_is_recorded_with_its_usage_and_no_error(tmp_path):
    ledger = tmp_path / "usage.jsonl"
    meter = Meter(ledger_path=ledger)
    client = mock_llm_client(lambda request: completion("你好"), meter=meter)
    budget = PageBudget("book/001", 10_000)

    reply = client.chat(MESSAGES, budget=budget, json_output=True)

    assert (reply.text, reply.finish_reason) == ("你好", "stop")
    (rec,) = meter.records
    assert reply.record == rec
    assert (rec.requested_model, rec.response_model, rec.usage) == ("mock-model", "mock-model-1", Usage(100, 20, None))
    assert rec.cost_usd == pytest.approx(0.00014) and rec.error is None
    assert (budget.used_tokens, budget.unknown_calls) == (120, 0)
    (line,) = ledger_lines(ledger)
    assert line["error"] is None and line["usage"]["input_tokens"] == 100


def test_a_response_without_choices_is_a_provider_error_but_keeps_its_usage(tmp_path):
    meter = Meter(ledger_path=tmp_path / "usage.jsonl")
    client = mock_llm_client(lambda request: completion(choices=False), meter=meter)
    budget = PageBudget("book/001", 10_000)

    with pytest.raises(ProviderError, match="mock-model returned no choices") as raised:
        client.chat(MESSAGES, budget=budget)

    assert raised.value.status_code is None
    (rec,) = meter.records
    assert rec.usage == Usage(100, 20, None) and rec.cost_usd == pytest.approx(0.00014)  # it was billed, and is known
    assert (budget.used_tokens, budget.unknown_calls) == (120, 0)


def test_an_exception_that_is_not_a_provider_error_is_recorded_and_raised_unchanged(monkeypatch):
    meter = Meter()
    client = mock_llm_client(lambda request: completion(), meter=meter)
    budget = PageBudget("book/001", 10_000)
    boom = ValueError("bad arguments")

    def create(**kwargs):
        raise boom

    monkeypatch.setattr(client._client.chat.completions, "create", create)

    with pytest.raises(ValueError) as raised:
        client.chat(MESSAGES, budget=budget)

    assert raised.value is boom  # not wrapped in ProviderError
    (rec,) = meter.records
    assert (rec.usage, rec.cost_usd, rec.error) == (None, None, "ValueError: bad arguments")
    assert budget.unknown_calls == 1


def test_the_api_key_never_reaches_the_error_or_the_ledger(tmp_path):
    ledger = tmp_path / "usage.jsonl"

    def echoes_the_key(request):  # a provider may repeat the key it was sent in the error body
        return httpx.Response(404, json={"error": {"message": f"No such model for key {KEY}"}})

    client = mock_llm_client(echoes_the_key, meter=Meter(ledger_path=ledger), api_key=KEY)

    with pytest.raises(ProviderError) as raised:
        client.chat(MESSAGES, budget=PageBudget("book/001", 10_000))

    assert "Not found (404)" in raised.value.description and "***" in raised.value.description
    assert KEY not in raised.value.description and KEY not in ledger.read_text(encoding="utf-8")


def test_the_sdk_client_does_not_retry():
    assert mock_llm_client(lambda request: completion())._client.max_retries == 0


@pytest.mark.parametrize("handler, status_code, readable", FAILURES)
def test_a_failed_request_reaches_the_provider_exactly_once(handler, status_code, readable):
    sent = []

    def counting(request):
        sent.append(request)
        return handler(request)

    with pytest.raises(ProviderError):
        mock_llm_client(counting).chat(MESSAGES, budget=PageBudget("book/001", 10_000))

    assert len(sent) == 1  # the SDK's default would resend it up to twice, unseen by the Meter


@pytest.mark.parametrize("handler, status_code, readable", FAILURES)
def test_the_provider_error_carries_the_call_the_meter_recorded(handler, status_code, readable):
    meter = Meter()

    with pytest.raises(ProviderError) as raised:
        mock_llm_client(handler, meter=meter).chat(MESSAGES, budget=PageBudget("book/001", 10_000))

    (rec,) = meter.records
    assert raised.value.record is rec


# --- responses the client cannot read: still one call, still a ProviderError, never AttributeError ---------

def html_page(request):  # the SDK returns this body as a str: it is not JSON
    return httpx.Response(200, text="<html>gateway</html>", headers={"content-type": "text/html"})


def not_json(request):  # claims to be JSON, is not: the SDK raises JSONDecodeError
    return httpx.Response(200, content=b"{nope", headers={"content-type": "application/json"})


def body(**overrides):
    """A well-formed chat.completion body with `overrides` applied: 100 input and 20 output tokens."""
    b = json.loads(completion("ok").content)
    b.update(overrides)
    return b


def first_choice(message):
    return [{"index": 0, "finish_reason": "stop", "message": message}]


@pytest.mark.parametrize(
    "handler, readable",
    [pytest.param(html_page, "is not a chat completion", id="html-body"),
     pytest.param(not_json, "not valid JSON", id="invalid-json-body")],
)
def test_a_body_that_is_not_a_completion_is_one_unknown_call(handler, readable, tmp_path):
    ledger = tmp_path / "usage.jsonl"
    meter = Meter(ledger_path=ledger)
    budget = PageBudget("book/001", 10_000)

    with pytest.raises(ProviderError) as raised:
        mock_llm_client(handler, meter=meter, api_key=KEY).chat(MESSAGES, budget=budget, json_output=True)

    error = raised.value
    assert readable in error.description and "LLM_BASE_URL" in error.description
    assert error.status_code is None  # the SDK does not say what the status was
    (rec,) = meter.records
    assert error.record is rec
    assert (rec.response_model, rec.usage, rec.cost_usd, rec.error) == (None, None, None, error.description)
    assert (budget.unknown_calls, budget.used_tokens) == (1, 0)
    assert meter.summary()["cost_usd"] is None and meter.summary()["unknown_usage_calls"] == 1
    (line,) = ledger_lines(ledger)
    assert (line["usage"], line["error"]) == (None, error.description)


def test_an_exception_that_is_a_value_error_but_not_bad_json_still_arrives_unchanged(monkeypatch):
    client = mock_llm_client(lambda request: completion())
    boom = ValueError("bad arguments")  # JSONDecodeError is a ValueError; this one is not

    def create(**kwargs):
        raise boom

    monkeypatch.setattr(client._client.chat.completions, "create", create)

    with pytest.raises(ValueError) as raised:
        client.chat(MESSAGES, budget=PageBudget("book/001", 10_000))

    assert raised.value is boom


@pytest.mark.parametrize(
    "overrides",
    [{"choices": first_choice(None)}, {"choices": [None]}, {"choices": ["text"]}, {"choices": "text"},
     {"choices": first_choice({"role": "assistant", "content": ["not", "text"]})}],
    ids=["null-message", "null-choice", "string-choice", "string-choices", "content-is-a-list"],
)
def test_a_first_choice_that_cannot_be_read_is_one_call_that_keeps_its_usage(overrides, tmp_path):
    ledger = tmp_path / "usage.jsonl"
    meter = Meter(ledger_path=ledger)
    budget = PageBudget("book/001", 10_000)
    client = mock_llm_client(lambda request: httpx.Response(200, json=body(**overrides)), meter=meter)

    with pytest.raises(ProviderError, match="mock-model returned a first choice that cannot be read") as raised:
        client.chat(MESSAGES, budget=budget)

    (rec,) = meter.records
    assert raised.value.record is rec and raised.value.status_code is None
    assert rec.usage == Usage(100, 20, None) and rec.cost_usd == pytest.approx(0.00014)  # billed, and known
    assert rec.response_model == "mock-model-1" and rec.error == raised.value.description
    assert (budget.used_tokens, budget.unknown_calls) == (120, 0)
    (line,) = ledger_lines(ledger)
    assert line["usage"]["input_tokens"] == 100 and line["error"] == raised.value.description


def test_a_response_without_choices_is_marked_as_an_error_in_its_record():
    meter = Meter()

    with pytest.raises(ProviderError) as raised:
        mock_llm_client(lambda request: completion(choices=False), meter=meter).chat(
            MESSAGES, budget=PageBudget("book/001", 10_000))

    (rec,) = meter.records
    assert raised.value.record is rec and rec.error == "mock-model returned no choices"


def test_an_unreadable_message_with_no_text_is_still_a_reply():
    reply = mock_llm_client(lambda request: httpx.Response(200, json=body(choices=first_choice({})))).chat(
        MESSAGES, budget=PageBudget("book/001", 10_000))

    assert (reply.text, reply.finish_reason, reply.record.error) == ("", "stop", None)  # empty content is a parse matter


@pytest.mark.parametrize("usage", ["lots", {"prompt_tokens": 100}, None], ids=["not-an-object", "no-completion-tokens", "null"])
def test_usage_the_provider_sends_unreadably_is_unknown_not_a_crash(usage):
    meter = Meter()
    budget = PageBudget("book/001", 10_000)
    client = mock_llm_client(lambda request: httpx.Response(200, json=body(usage=usage)), meter=meter)

    reply = client.chat(MESSAGES, budget=budget)

    assert reply.text == "ok"
    (rec,) = meter.records
    assert (rec.usage, rec.cost_usd, rec.error) == (None, None, None)
    assert (budget.unknown_calls, budget.used_tokens) == (1, 0)


# --- describe_api_error for an openai.APIResponseValidationError ---------------------------------------------

def validation_error(status_code=200):
    request = httpx.Request("POST", "http://provider.test/v1/chat/completions")
    return openai.APIResponseValidationError(httpx.Response(status_code, request=request), body=None)


@pytest.mark.parametrize("status_code", [200, 502])
def test_a_response_validation_error_is_described_with_its_status_and_the_base_url_setting(status_code):
    description = describe_api_error(validation_error(status_code), "http://provider.test/v1")

    assert f"HTTP {status_code}" in description and "LLM_BASE_URL" in description
    assert not description.startswith("LLM call failed")


def test_a_response_validation_error_out_of_the_sdk_is_a_provider_error_with_its_status(monkeypatch):
    meter = Meter()
    client = mock_llm_client(lambda request: completion(), meter=meter)

    def create(**kwargs):
        raise validation_error()

    monkeypatch.setattr(client._client.chat.completions, "create", create)

    with pytest.raises(ProviderError) as raised:
        client.chat(MESSAGES, budget=PageBudget("book/001", 10_000))

    assert raised.value.status_code == 200 and "LLM_BASE_URL" in raised.value.description
    (rec,) = meter.records
    assert raised.value.record is rec and rec.usage is None


# --- tools, tool calls and reasoning_content (task 006) ---------------------------------------------------------

WEATHER = {"type": "function", "function": {"name": "weather", "description": "Weather in a city.", "parameters": {
    "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def tool_call(call_id="call_1", name="weather", arguments='{"city": "Kyoto"}'):
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool_reply(*calls, content=None, finish_reason="tool_calls", **extra):
    """A completion whose first choice asks for `calls`; `extra` are more message fields, like reasoning_content."""
    message = {"role": "assistant", "content": content, "tool_calls": list(calls), **extra}
    return httpx.Response(200, json=body(choices=[{"index": 0, "finish_reason": finish_reason, "message": message}]))


def sent_bodies(handler):
    """(handler that records the JSON body of each request, the list it records into)."""
    bodies = []

    def recording(request):
        bodies.append(json.loads(request.content))
        return handler(request)

    return recording, bodies


def test_tools_reach_the_provider_as_given():
    recording, bodies = sent_bodies(lambda request: completion("ok"))

    mock_llm_client(recording).chat(MESSAGES, budget=PageBudget("book/001", 10_000), tools=[WEATHER])

    assert bodies[0]["tools"] == [WEATHER]


@pytest.mark.parametrize("tools", [None, []], ids=["none", "empty-list"])
def test_without_tools_no_tools_key_is_sent(tools):
    recording, bodies = sent_bodies(lambda request: completion("ok"))

    mock_llm_client(recording).chat(MESSAGES, budget=PageBudget("book/001", 10_000), tools=tools)  # [] would be a 400

    assert "tools" not in bodies[0]


def test_two_tool_calls_and_reasoning_content_come_back_in_order():
    first = tool_call("call_a", "weather", '{"city": "Kyoto"}')
    second = tool_call("call_b", "weather", '{ "city" : "Osaka" ')  # not valid JSON: handed over as the model sent it
    client = mock_llm_client(lambda request: tool_reply(first, second, reasoning_content="needs the weather"))

    reply = client.chat(MESSAGES, budget=PageBudget("book/001", 10_000), tools=[WEATHER])

    assert [(c.id, c.name, c.arguments) for c in reply.tool_calls] == [
        ("call_a", "weather", '{"city": "Kyoto"}'), ("call_b", "weather", '{ "city" : "Osaka" ')]
    assert (reply.reasoning_content, reply.text, reply.finish_reason) == ("needs the weather", "", "tool_calls")


def test_a_reply_without_tool_calls_or_reasoning_has_empty_ones():
    reply = mock_llm_client(lambda request: completion("你好")).chat(MESSAGES, budget=PageBudget("book/001", 10_000))

    assert (reply.tool_calls, reply.reasoning_content) == ([], None)


# --- responses the client reads wrongly: one call, the right error or a reply (task 005 Review, finding 1) ----------

def test_choices_that_are_an_object_are_one_call_and_a_provider_error():
    meter = Meter()
    client = mock_llm_client(lambda request: httpx.Response(200, json=body(choices={"a": 1})), meter=meter)

    with pytest.raises(ProviderError, match="first choice that cannot be read") as raised:
        client.chat(MESSAGES, budget=PageBudget("book/001", 10_000))  # was a KeyError, with no call recorded

    (rec,) = meter.records
    assert raised.value.record is rec and rec.usage == Usage(100, 20, None)


@pytest.mark.parametrize("usage", [{"prompt_tokens": "100", "completion_tokens": "20"},
                                   {"prompt_tokens": [100], "completion_tokens": {"n": 20}},
                                   {"prompt_tokens": True, "completion_tokens": 20},
                                   {"prompt_tokens": 100.5, "completion_tokens": 20},
                                   {"prompt_tokens": -1, "completion_tokens": 20}],
                         ids=["strings", "containers", "bool", "float", "negative"])
def test_token_counts_that_are_not_integers_are_unknown_and_the_reply_is_used(usage):
    meter = Meter()
    budget = PageBudget("book/001", 10_000)
    client = mock_llm_client(lambda request: httpx.Response(200, json=body(usage=usage)), meter=meter)

    reply = client.chat(MESSAGES, budget=budget)  # was a TypeError in the cost, with no call recorded

    assert reply.text == "ok"
    (rec,) = meter.records
    assert (rec.usage, rec.cost_usd, rec.error) == (None, None, None)
    assert (budget.unknown_calls, budget.used_tokens) == (1, 0)


@pytest.mark.parametrize("details, cached", [("lots", None), ([1], None), ({"cached_tokens": [256]}, None),
                                             ({"cached_tokens": 256}, 256), (None, None)],
                         ids=["string", "list", "list-count", "integer-count", "null"])
def test_cached_tokens_that_cannot_be_read_are_none_and_the_usage_is_kept(details, cached):
    usage = {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": details}
    meter = Meter()
    client = mock_llm_client(lambda request: httpx.Response(200, json=body(usage=usage)), meter=meter)

    reply = client.chat(MESSAGES, budget=PageBudget("book/001", 10_000))  # was an AttributeError, with no call recorded

    assert reply.text == "ok"
    (rec,) = meter.records
    assert rec.usage == Usage(100, 20, cached) and rec.cost_usd == pytest.approx(0.00014) and rec.error is None


@pytest.mark.parametrize(
    "bad_call",
    [{"id": "call_1", "type": "function", "function": {"arguments": "{}"}},
     {"id": "call_1", "type": "function", "function": {"name": "", "arguments": "{}"}},
     {"id": "call_1", "type": "function"},
     {"type": "function", "function": {"name": "weather", "arguments": "{}"}},
     {"id": "call_1", "type": "function", "function": {"name": "weather", "arguments": {"city": "Kyoto"}}},
     "weather"],
    ids=["no-function-name", "empty-function-name", "no-function", "no-id", "arguments-not-a-string", "not-an-object"],
)
def test_a_tool_call_that_cannot_be_read_is_one_call_that_keeps_its_usage(bad_call):
    meter = Meter()
    budget = PageBudget("book/001", 10_000)
    client = mock_llm_client(lambda request: tool_reply(tool_call(), bad_call), meter=meter)

    with pytest.raises(ProviderError, match="mock-model returned a tool call that cannot be read") as raised:
        client.chat(MESSAGES, budget=budget, tools=[WEATHER])

    (rec,) = meter.records
    assert raised.value.record is rec and rec.error == raised.value.description
    assert rec.usage == Usage(100, 20, None)  # billed, and known
    assert (budget.used_tokens, budget.unknown_calls) == (120, 0)


def test_tool_calls_that_are_not_a_list_are_one_call_and_a_provider_error():
    meter = Meter()
    reply = httpx.Response(200, json=body(choices=first_choice({"role": "assistant", "content": None, "tool_calls": "x"})))

    with pytest.raises(ProviderError, match="tool call that cannot be read"):
        mock_llm_client(lambda request: reply, meter=meter).chat(MESSAGES, budget=PageBudget("book/001", 10_000))

    assert len(meter.records) == 1


# --- the call's error and the exception's description are one redacted string (task 005 Review, finding 4) ---------

def key_in_the_model_name(handler, meter):
    """A client whose model name contains the API key, so that the message of an unreadable response does too."""
    config = LLMConfig("http://provider.test/v1", KEY, f"model-{KEY}")
    return LLMClient(config, meter, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def reply_with(**overrides):
    return lambda request: httpx.Response(200, json=body(**overrides))


FAILURE_PATHS = [
    pytest.param(refuse, id="connection-error"),
    pytest.param(upstream_500, id="http-500"),
    pytest.param(html_page, id="html-body"),
    pytest.param(not_json, id="invalid-json-body"),
    pytest.param(lambda request: completion(choices=False), id="no-choices"),
    pytest.param(reply_with(choices=first_choice(None)), id="unreadable-choice"),
    pytest.param(reply_with(choices="text"), id="unreadable-choices"),
    pytest.param(lambda request: tool_reply({"id": "c"}), id="unreadable-tool-call"),
]


@pytest.mark.parametrize("handler", FAILURE_PATHS)
def test_the_call_error_is_the_provider_error_description_in_every_failure_path(handler):
    meter = Meter()

    with pytest.raises(ProviderError) as raised:
        key_in_the_model_name(handler, meter).chat(MESSAGES, budget=PageBudget("book/001", 10_000))

    (rec,) = meter.records
    assert rec.error == raised.value.description and raised.value.record is rec
    assert KEY not in rec.error  # the same string, and redacted
