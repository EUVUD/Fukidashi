"""The judge's configuration (OUTLINE 8): `LLMConfig.from_env(prefix)`, the optional temperature and reasoning effort,
the prefixed messages, and a ledger of its own. The real LLMClient and openai SDK on an httpx.MockTransport; no network.
"""

from __future__ import annotations

import json
import os

import httpx
import openai
import pytest

from conftest import completion
from fukidashi.llm.client import LLMClient, LLMConfig, LLMNotConfigured
from fukidashi.llm.errors import ProviderError, describe_api_error
from fukidashi.llm.metering import Meter, PageBudget

SECRET = "sk-judge-secret-value"
JUDGE = {"JUDGE_BASE_URL": "http://judge.test/v1", "JUDGE_API_KEY": SECRET, "JUDGE_MODEL": "gpt-4o-mini-2024-07-18",
         "JUDGE_JSON_MODE": "none", "JUDGE_TOKEN_PARAM": "max_completion_tokens", "JUDGE_PRICE_INPUT_PER_M": "0.15",
         "JUDGE_PRICE_OUTPUT_PER_M": "0.60", "JUDGE_TEMPERATURE": "0"}
JUDGE2 = {"JUDGE2_BASE_URL": "http://judge2.test/v1", "JUDGE2_API_KEY": "sk-judge2-secret",
          "JUDGE2_MODEL": "gpt-5.1-2025-11-13", "JUDGE2_TOKEN_PARAM": "max_completion_tokens",
          "JUDGE2_REASONING_EFFORT": "none", "JUDGE2_PRICE_INPUT_PER_M": "1.25", "JUDGE2_PRICE_OUTPUT_PER_M": "10.00"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No LLM_* or JUDGE*_* variable from the shell reaches these tests."""
    for name in list(os.environ):
        if name.startswith(("LLM_", "JUDGE")):
            monkeypatch.delenv(name)


def set_env(monkeypatch, values):
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def client_on(handler, config: LLMConfig, meter: Meter | None = None) -> LLMClient:
    return LLMClient(config, meter or Meter(), http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def recording(bodies: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return completion("ok")
    return handler


def budget() -> PageBudget:
    return PageBudget("p", 100_000)


# --- from_env ------------------------------------------------------------------------------------------------------

def test_the_judge_prefix_reads_every_judge_variable(monkeypatch):
    set_env(monkeypatch, JUDGE)

    config = LLMConfig.from_env(prefix="JUDGE")

    assert config == LLMConfig(
        base_url="http://judge.test/v1", api_key=SECRET, model="gpt-4o-mini-2024-07-18", json_mode="none",
        token_param="max_completion_tokens", price_input_per_m=0.15, price_output_per_m=0.60, temperature=0.0,
        reasoning_effort=None, env_prefix="JUDGE")


def test_the_fallback_prefix_reads_the_fallbacks_variables_and_its_reasoning_effort(monkeypatch):
    set_env(monkeypatch, {**JUDGE, **JUDGE2})

    config = LLMConfig.from_env(prefix="JUDGE2")

    assert (config.base_url, config.api_key, config.model) == ("http://judge2.test/v1", "sk-judge2-secret",
                                                                "gpt-5.1-2025-11-13")
    assert (config.token_param, config.reasoning_effort, config.temperature) == ("max_completion_tokens", "none", None)
    assert (config.price_input_per_m, config.price_output_per_m, config.env_prefix) == (1.25, 10.0, "JUDGE2")
    assert config.json_mode == "json_object"  # the default, as for LLM_*


@pytest.mark.parametrize("missing", ["JUDGE_BASE_URL", "JUDGE_API_KEY", "JUDGE_MODEL"])
def test_a_missing_judge_variable_is_named_and_no_value_is(monkeypatch, missing):
    set_env(monkeypatch, {k: v for k, v in JUDGE.items() if k != missing})

    with pytest.raises(LLMNotConfigured) as raised:
        LLMConfig.from_env(prefix="JUDGE")

    message = str(raised.value)
    assert missing in message and "LLM_" not in message
    assert not any(value in message for value in JUDGE.values() if value)


def test_without_a_prefix_it_reads_the_llm_variables_as_before(monkeypatch):
    set_env(monkeypatch, {**JUDGE, "LLM_BASE_URL": "http://llm.test/v1", "LLM_API_KEY": "sk-llm", "LLM_MODEL": "m",
                          "LLM_PRICE_INPUT_PER_M": " 1.5 ", "LLM_PRICE_OUTPUT_PER_M": ""})

    config = LLMConfig.from_env()

    assert config == LLMConfig("http://llm.test/v1", "sk-llm", "m", price_input_per_m=1.5)
    assert (config.temperature, config.reasoning_effort, config.env_prefix) == (None, None, "LLM")


def test_without_a_prefix_a_missing_variable_gives_the_same_message_as_before():
    with pytest.raises(LLMNotConfigured) as raised:
        LLMConfig.from_env()

    assert str(raised.value) == "Set LLM_BASE_URL, LLM_API_KEY, LLM_MODEL in .env (see .env.example)"


def test_an_empty_temperature_or_reasoning_effort_is_unset(monkeypatch):
    set_env(monkeypatch, {**JUDGE, "JUDGE_TEMPERATURE": " ", "JUDGE_REASONING_EFFORT": ""})

    config = LLMConfig.from_env("JUDGE")

    assert (config.temperature, config.reasoning_effort) == (None, None)


# --- what chat sends -------------------------------------------------------------------------------------------------

def test_chat_sends_neither_temperature_nor_reasoning_effort_when_they_are_unset():
    bodies: list[dict] = []
    client_on(recording(bodies), LLMConfig("http://provider.test/v1", "k", "m")).chat(
        [{"role": "user", "content": "hi"}], budget=budget())

    (body,) = bodies
    assert "temperature" not in body and "reasoning_effort" not in body
    assert set(body) == {"model", "messages", "max_tokens"}


@pytest.mark.parametrize("field, value", [("temperature", 0.0), ("temperature", 0.7), ("reasoning_effort", "none"),
                                          ("reasoning_effort", "low")])
def test_chat_sends_each_one_when_it_is_set(field, value):
    bodies: list[dict] = []
    other = "reasoning_effort" if field == "temperature" else "temperature"
    client_on(recording(bodies), LLMConfig("http://provider.test/v1", "k", "m", **{field: value})).chat(
        [{"role": "user", "content": "hi"}], budget=budget())

    (body,) = bodies
    assert body[field] == value and other not in body


def test_a_judge_config_read_from_the_environment_reaches_the_provider_as_configured(monkeypatch):
    set_env(monkeypatch, JUDGE)
    bodies: list[dict] = []

    client_on(recording(bodies), LLMConfig.from_env("JUDGE")).chat([{"role": "user", "content": "hi"}],
                                                                     budget=budget(), max_tokens=500)

    (body,) = bodies
    assert body["model"] == "gpt-4o-mini-2024-07-18" and body["temperature"] == 0
    assert body["max_completion_tokens"] == 500 and "max_tokens" not in body


# --- messages name the configuration's own variables -----------------------------------------------------------------

def unauthorised(request: httpx.Request) -> httpx.Response:
    return httpx.Response(401, json={"error": {"message": "bad key"}})


def html_page(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, text="<html>gateway</html>", headers={"content-type": "text/html"})


@pytest.mark.parametrize("handler, variable", [(unauthorised, "API_KEY"), (html_page, "BASE_URL")])
def test_the_clients_messages_name_the_judges_variables(handler, variable):
    config = LLMConfig("http://judge.test/v1", SECRET, "judge-model", env_prefix="JUDGE")

    with pytest.raises(ProviderError) as raised:
        client_on(handler, config).chat([{"role": "user", "content": "hi"}], budget=budget())

    assert f"JUDGE_{variable}" in raised.value.description and "LLM_" not in raised.value.description
    assert SECRET not in raised.value.description


@pytest.mark.parametrize("handler, variable", [(unauthorised, "API_KEY"), (html_page, "BASE_URL")])
def test_the_translation_clients_messages_still_name_the_llm_variables(handler, variable):
    with pytest.raises(ProviderError) as raised:
        client_on(handler, LLMConfig("http://provider.test/v1", "k", "m")).chat(
            [{"role": "user", "content": "hi"}], budget=budget())

    assert f"LLM_{variable}" in raised.value.description


def test_describe_api_error_names_the_prefix_it_is_given_and_llm_by_default():
    request = httpx.Request("POST", "http://judge.test/v1/chat/completions")
    error = openai.NotFoundError("no such model", response=httpx.Response(404, request=request), body=None)

    assert "Check JUDGE_BASE_URL and JUDGE_MODEL" in describe_api_error(error, "http://judge.test/v1", "JUDGE")
    assert "Check LLM_BASE_URL and LLM_MODEL" in describe_api_error(error, "http://judge.test/v1")


# --- two ledgers -----------------------------------------------------------------------------------------------------

def test_each_client_writes_only_to_its_own_ledger(tmp_path):
    usage, judge_usage = tmp_path / "usage.jsonl", tmp_path / "judge_usage.jsonl"
    translation = client_on(recording([]), LLMConfig("http://provider.test/v1", "k", "translation-model"),
                            Meter(ledger_path=usage))
    judge = client_on(recording([]), LLMConfig("http://judge.test/v1", "k", "judge-model", env_prefix="JUDGE"),
                      Meter(ledger_path=judge_usage))

    judge.chat([{"role": "user", "content": "a"}], budget=budget())
    judge.chat([{"role": "user", "content": "b"}], budget=budget())
    translation.chat([{"role": "user", "content": "c"}], budget=budget())

    judged = [json.loads(line) for line in judge_usage.read_text(encoding="utf-8").splitlines()]
    translated = [json.loads(line) for line in usage.read_text(encoding="utf-8").splitlines()]
    assert [r["requested_model"] for r in judged] == ["judge-model", "judge-model"]
    assert [r["requested_model"] for r in translated] == ["translation-model"]
