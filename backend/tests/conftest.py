"""Fixtures: a tiny OpenMantra-shaped dataset, a temp SQLite file, and an app wired to a fake LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from fukidashi.api.deps import get_client_factory, get_dataset_root, get_db
from fukidashi.datasets.importer import import_openmantra
from fukidashi.db import connect, init_db
from fukidashi.llm.client import LLMConfig, LLMResult
from fukidashi.llm.metering import CallRecord, Meter, Usage
from fukidashi.main import app

# Reference strings are distinctive so a test can check they never reach the prompt.
SEGMENTS = [
    {"x": 10, "y": 20, "w": 30, "h": 40, "text_ja": "こんにちは", "text_en": "REF-EN-0", "text_zh": "REF-ZH-0"},
    {"x": 50, "y": 60, "w": 70, "h": 80, "text_ja": "さようなら", "text_en": "REF-EN-1", "text_zh": "REF-ZH-1"},
]


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
