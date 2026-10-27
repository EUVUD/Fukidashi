"""LLMClient against a mock provider (httpx.MockTransport): the real SDK, no network, no real model."""

import json

import httpx
import openai
import pytest
from conftest import completion, mock_llm_client

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
