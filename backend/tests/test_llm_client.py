"""LLMClient against a mock provider (httpx.MockTransport): the real SDK, no network, no real model."""

import json

import httpx
import pytest
from conftest import completion, mock_llm_client

from fukidashi.llm.errors import ProviderError
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
