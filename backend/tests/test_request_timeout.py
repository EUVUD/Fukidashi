"""The request timeout (task 036): `LLMClient(timeout=...)`, the 180 s of the API's factory, and the eval's clients that have
none. A request that times out is a failed call like any other: recorded once, with `error`, raised as `ProviderError`."""

from __future__ import annotations

import ast
import socket
import threading
import time
from pathlib import Path

import httpx
import openai
import pytest
from conftest import mock_llm_client
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from fukidashi import config
from fukidashi.api import deps
from fukidashi.llm import client as llm_client
from fukidashi.llm.client import LLMClient, LLMConfig, LLMNotConfigured
from fukidashi.llm.errors import ProviderError
from fukidashi.llm.metering import Meter, PageBudget

CONFIG = LLMConfig("http://provider.test/v1", "test-key", "mock-model")


def chat(client: LLMClient):
    return client.chat([{"role": "user", "content": "hi"}], budget=PageBudget("book/001", 10_000))


@pytest.fixture
def sdk_calls(monkeypatch):
    """The keyword arguments `openai.OpenAI` is built with."""
    seen: list[dict] = []
    real = openai.OpenAI

    def spy(**kwargs):
        seen.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(llm_client.openai, "OpenAI", spy)
    return seen


def test_a_client_without_a_timeout_is_built_exactly_as_before(sdk_calls):
    http_client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500)))

    client = LLMClient(CONFIG, Meter(), http_client=http_client)

    assert sdk_calls == [{"base_url": CONFIG.base_url, "api_key": "test-key", "max_retries": 0, "http_client": http_client}]
    assert client._client.timeout == openai.DEFAULT_TIMEOUT  # the SDK's own default, untouched


def test_a_client_with_a_timeout_gives_it_to_the_sdk(sdk_calls):
    client = LLMClient(CONFIG, Meter(), timeout=180.0)

    assert sdk_calls[0]["timeout"] == 180.0 and sdk_calls[0]["max_retries"] == 0
    assert client._client.timeout == 180.0


def test_a_timeout_raised_by_the_transport_is_one_failed_call_and_a_provider_error():
    def handler(request):
        raise httpx.ReadTimeout("timed out", request=request)

    meter = Meter()
    client = mock_llm_client(handler, meter=meter)

    with pytest.raises(ProviderError) as raised:
        chat(client)

    (record,) = meter.records  # recorded once: the SDK does not retry
    assert record.error == raised.value.description and "timed out" in record.error.lower()
    assert record.usage is None and record.cost_usd is None and raised.value.record is record  # unknown, not 0
    assert meter.summary()["calls"] == 1 and meter.summary()["unknown_usage_calls"] == 1


def test_a_provider_that_never_answers_times_out_after_the_clients_timeout():
    """The timeout really reaches the socket: a local server that accepts and stays silent, 0.3 s."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    held = []
    threading.Thread(target=lambda: held.append(server.accept()), daemon=True).start()
    meter = Meter()
    client = LLMClient(LLMConfig(f"http://127.0.0.1:{server.getsockname()[1]}/v1", "k", "m"), meter, timeout=0.3)
    try:
        started = time.monotonic()
        with pytest.raises(ProviderError) as raised:
            chat(client)
        elapsed = time.monotonic() - started
    finally:
        for connection, _ in held:
            connection.close()
        server.close()

    assert 0.25 < elapsed < 5 and "timed out" in raised.value.description.lower()
    assert len(meter.records) == 1 and meter.records[0].error == raised.value.description


def test_the_apis_factory_builds_clients_with_180_seconds(monkeypatch, tmp_path):
    assert deps.API_REQUEST_TIMEOUT_S == 180.0
    for name, value in (("LLM_BASE_URL", "http://provider.test/v1"), ("LLM_API_KEY", "k"), ("LLM_MODEL", "m")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(config, "ROOT", tmp_path)  # no .env to read
    app = FastAPI()

    @app.get("/timeout")
    def timeout(factory=Depends(deps.get_client_factory)):  # FastAPI resolves it as it does for the real routes
        return {"timeout": factory()._client.timeout}

    assert TestClient(app).get("/timeout").json() == {"timeout": 180.0}


def test_the_factory_called_directly_builds_clients_without_one_which_is_how_the_eval_harness_calls_it(monkeypatch, tmp_path):
    for name, value in (("LLM_BASE_URL", "http://provider.test/v1"), ("LLM_API_KEY", "k"), ("LLM_MODEL", "m")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(config, "ROOT", tmp_path)

    client = deps.get_client_factory()()

    assert client._client.timeout == openai.DEFAULT_TIMEOUT  # no timeout of its own


def test_the_eval_harness_and_the_scripts_set_no_timeout():
    root = Path(config.ROOT)
    for source in (root / "eval" / "run.py", *sorted((root / "scripts").glob("*.py"))):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.keyword) and n.arg == "timeout"], source
    calls = [n for n in ast.walk(ast.parse((root / "eval" / "run.py").read_text(encoding="utf-8")))
             if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "get_client_factory"]
    assert len(calls) == 1 and not calls[0].args and not calls[0].keywords  # `get_client_factory()`: the default, none


def test_a_post_translate_builds_its_client_with_180_seconds(api, background, monkeypatch, tmp_path):
    from conftest import post_translate

    from fukidashi.main import app
    client, _ = api
    del app.dependency_overrides[deps.get_client_factory]  # the real factory
    built = []

    def fake_llm_client(config, meter, *, timeout=None, **kwargs):
        built.append(timeout)
        raise LLMNotConfigured("stop here")

    for name, value in (("LLM_BASE_URL", "http://provider.test/v1"), ("LLM_API_KEY", "k"), ("LLM_MODEL", "m")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(deps, "LLMClient", fake_llm_client)

    assert post_translate(client, background).status_code == 503

    assert built == [180.0]
