"""Fixtures: a tiny OpenMantra-shaped dataset, a temp SQLite file, and an app wired to a fake LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx
import pytest
from fastapi.testclient import TestClient

from fukidashi import tracing
from fukidashi.agents.loop import TRACING_VARIABLES
from fukidashi.api.deps import get_client_factory, get_dataset_root, get_db
from fukidashi.datasets.importer import import_openmantra
from fukidashi.db import connect, init_db
from fukidashi.llm.client import LLMClient, LLMConfig, LLMResult
from fukidashi.llm.metering import CallRecord, Meter, Usage
from fukidashi.main import app

# Reference strings are distinctive so a test can check they never reach the prompt.
SEGMENTS = [
    {"x": 10, "y": 20, "w": 30, "h": 40, "text_ja": "こんにちは", "text_en": "REF-EN-0", "text_zh": "REF-ZH-0"},
    {"x": 50, "y": 60, "w": 70, "h": 80, "text_ja": "さようなら", "text_en": "REF-EN-1", "text_zh": "REF-ZH-1"},
]


@pytest.fixture(autouse=True)
def no_tracing(monkeypatch):
    """The suite does not depend on what the shell exports. The agent loop refuses to run with LangSmith tracing on,
    `FUKIDASHI_TRACING` / `PHOENIX_*` would switch our own tracing on, and `LLM_JSON_MODE` changes what the
    provider is sent."""
    for name in (*TRACING_VARIABLES, "FUKIDASHI_TRACING", "PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_PROJECT_NAME",
                 "LLM_JSON_MODE"):
        monkeypatch.delenv(name, raising=False)
    yield
    tracing.shutdown()  # whatever a test switched on is switched off again, even when the test failed


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "open-mantra-dataset"
    (root / "images" / "book_a" / "ja").mkdir(parents=True)
    (root / "images" / "book_a" / "ja" / "000.jpg").write_bytes(b"\xff\xd8fake-jpeg")
    (root / "images" / "book_a" / "ja" / "001.jpg").write_bytes(b"\xff\xd8fake-jpeg")
    pages = [
        {"page_index": 1, "image_paths": {"ja": "images/book_a/ja/000.jpg"}, "frame": [], "text": SEGMENTS},
        {"page_index": 2, "image_paths": {"ja": "images/book_a/ja/001.jpg"}, "frame": [], "text": []},
    ]
    (root / "annotation.json").write_text(json.dumps([{"book_title": "book_a", "pages": pages}]))
    return root


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def conn(db_path):
    c = connect(db_path)
    init_db(c)
    yield c
    c.close()


@dataclass
class FakeClient:
    """Stands in for LLMClient: replays canned replies through a real Meter, like the real one does."""

    replies: list[str]
    config: LLMConfig = field(default_factory=lambda: LLMConfig("http://fake", "key", "fake-model"))
    meter: Meter = field(default_factory=Meter)
    requests: list[list[dict]] = field(default_factory=list)

    def chat(self, messages, *, budget, max_tokens=4096, json_output=False) -> LLMResult:
        self.requests.append(messages)
        usage = Usage(input_tokens=100, output_tokens=20)
        rec = CallRecord(budget.page_id, "fake-model", "fake-model-1", usage, 0.01, 0.0001)
        self.meter.record(rec)
        budget.charge(usage)
        return LLMResult(text=self.replies.pop(0), finish_reason="stop", record=rec)


@dataclass
class UsageClient(FakeClient):
    """Like FakeClient, but each call reports the usage listed in `usages` (None: the provider sent none)."""

    usages: list[Usage | None] = field(default_factory=list)

    def chat(self, messages, *, budget, max_tokens=4096, json_output=False):
        usage = self.usages.pop(0)
        rec = CallRecord(budget.page_id, "fake-model", "fake-model-1", usage, 0.01,
                         None if usage is None else 0.0001)
        self.meter.record(rec)
        budget.charge(usage)
        return LLMResult(text=self.replies.pop(0), finish_reason="stop", record=rec)


class CrashAfterCallClient(FakeClient):
    """Makes one priced call, then raises: a crash that happens after money was spent."""

    def chat(self, messages, **kwargs):
        super().chat(messages, **kwargs)
        raise RuntimeError("crashed after the call")


def completion(text: str = "ok", *, choices: bool = True, usage: bool = True) -> httpx.Response:
    """A chat.completion reply as an OpenAI-compatible provider sends it: 100 input and 20 output tokens."""
    body: dict = {
        "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": "mock-model-1",
        "choices": ([{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]
                    if choices else []),
    }
    if usage:
        body["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    return httpx.Response(200, json=body)


def mock_llm_client(handler, *, meter: Meter | None = None, api_key: str = "test-key") -> LLMClient:
    """The real LLMClient (and the real openai SDK) talking to `handler` through httpx.MockTransport: no network.

    Priced at $1 / $2 per million tokens, so `completion()` costs $0.00014.
    """
    config = LLMConfig("http://provider.test/v1", api_key, "mock-model",
                       price_input_per_m=1.0, price_output_per_m=2.0)
    return LLMClient(config, meter if meter is not None else Meter(),
                     http_client=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.fixture
def api(dataset, db_path, conn):
    import_openmantra(conn, dataset)
    clients: list[FakeClient] = []

    def use_replies(*replies: str) -> FakeClient:
        client = FakeClient(list(replies))
        clients.append(client)
        return client

    def override_db():
        c = connect(db_path)
        try:
            yield c
        finally:
            c.close()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_dataset_root] = lambda: dataset
    app.dependency_overrides[get_client_factory] = lambda: lambda: clients[-1]
    yield TestClient(app), use_replies
    app.dependency_overrides.clear()
