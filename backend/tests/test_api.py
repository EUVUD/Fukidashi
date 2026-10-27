import json

import httpx
import pytest
from conftest import mock_llm_client

from fukidashi import config
from fukidashi.api import deps
from fukidashi.api.deps import get_client_factory
from fukidashi.main import app

GOOD = json.dumps({"translations": [{"id": 0, "zh": "你好"}, {"id": 1, "zh": "再见"}]}, ensure_ascii=False)


def test_list_and_detail_before_any_translation(api):
    client, _ = api
    pages = client.get("/api/pages").json()
    assert pages == [
        {"book_title": "book_a", "page_index": 1, "segment_count": 2, "translated": False},
        {"book_title": "book_a", "page_index": 2, "segment_count": 0, "translated": False},
    ]
    detail = client.get("/api/pages/book_a/1").json()
    assert detail["run"] is None
    assert [s["text_zh_hans"] for s in detail["segments"]] == [None, None]
    assert detail["segments"][0] | {"text_zh_hans": None} == {
        "index": 0, "x": 10, "y": 20, "w": 30, "h": 40, "text_ja": "こんにちは",
        "text_zh_hant": "REF-ZH-0", "text_en": "REF-EN-0", "text_zh_hans": None,
    }
    assert client.get("/api/pages/book_a/1/image").content == b"\xff\xd8fake-jpeg"
    assert client.get("/api/pages/book_a/99").status_code == 404


def test_translate_stores_run_and_aligns_translations_with_segments(api):
    client, use_replies = api
    fake = use_replies(GOOD)

    resp = client.post("/api/pages/book_a/1/translate")

    assert resp.status_code == 200
    body = resp.json()
    assert [(s["index"], s["text_zh_hans"]) for s in body["segments"]] == [(0, "你好"), (1, "再见")]
    assert body["run"]["status"] == "succeeded"
    assert body["run"]["config"] == "baseline"
    assert (body["run"]["input_tokens"], body["run"]["output_tokens"], body["run"]["calls"]) == (100, 20, 1)
    # persisted: a fresh GET sees the same translations
    assert client.get("/api/pages/book_a/1").json()["segments"] == body["segments"]
    assert client.get("/api/pages").json()[0]["translated"] is True
    # the prompt carries Japanese text and boxes, never the reference translations
    prompt = json.dumps(fake.requests, ensure_ascii=False)
    assert "こんにちは" in prompt and "REF-ZH" not in prompt and "REF-EN" not in prompt


def test_empty_content_is_retried_once(api):
    client, use_replies = api
    fake = use_replies("", GOOD)

    resp = client.post("/api/pages/book_a/1/translate")

    assert resp.status_code == 200
    assert len(fake.requests) == 2
    assert resp.json()["run"]["calls"] == 2


def test_unparseable_output_fails_visibly_and_keeps_the_previous_translation(api):
    client, use_replies = api
    use_replies(GOOD)
    client.post("/api/pages/book_a/1/translate")

    use_replies("not json", '{"translations": [{"id": 0, "zh": "只有一条"}]}')
    resp = client.post("/api/pages/book_a/1/translate")

    assert resp.status_code == 502
    assert "Could not parse" in resp.json()["detail"]
    detail = client.get("/api/pages/book_a/1").json()
    assert [s["text_zh_hans"] for s in detail["segments"]] == ["你好", "再见"]  # still the first run
    assert detail["run"]["id"] == 1


def test_page_without_segments_is_not_sent_to_the_model(api):
    client, use_replies = api
    fake = use_replies(GOOD)

    resp = client.post("/api/pages/book_a/2/translate")

    assert resp.status_code == 422
    assert fake.requests == []


def test_a_region_id_that_is_not_on_the_page_is_retried_then_answers_502(api):
    client, use_replies = api
    bad = json.dumps({"translations": [{"id": i, "zh": "字"} for i in (0, 1, 7)]})
    fake = use_replies(bad, bad)

    resp = client.post("/api/pages/book_a/1/translate")

    assert resp.status_code == 502
    assert "Could not parse" in resp.json()["detail"] and "7" in resp.json()["detail"]
    assert len(fake.requests) == 2
    assert client.get("/api/pages/book_a/1").json()["run"] is None  # nothing succeeded, nothing stored


def test_a_duplicate_region_id_is_retried_and_a_good_retry_succeeds(api):
    client, use_replies = api
    duplicate = json.dumps({"translations": [{"id": i, "zh": "字"} for i in (0, 1, 1)]})
    fake = use_replies(duplicate, GOOD)

    resp = client.post("/api/pages/book_a/1/translate")

    assert resp.status_code == 200
    assert len(fake.requests) == 2
    assert [s["text_zh_hans"] for s in resp.json()["segments"]] == ["你好", "再见"]


def test_a_provider_error_answers_502_and_the_run_has_unknown_cost(api, conn):
    client, _ = api
    provider = mock_llm_client(lambda request: httpx.Response(500, json={"error": {"message": "upstream exploded"}}))
    app.dependency_overrides[get_client_factory] = lambda: lambda: provider  # the real LLMClient on a mock provider

    resp = client.post("/api/pages/book_a/1/translate")

    assert resp.status_code == 502
    assert "API error 500" in resp.json()["detail"] and "upstream exploded" in resp.json()["detail"]
    run = conn.execute("SELECT status, calls, unknown_usage_calls, cost_usd, error FROM runs").fetchone()
    assert tuple(run)[:4] == ("failed", 1, 1, None)  # unknown, not 0.0
    assert run["error"] == resp.json()["detail"]
    assert client.get("/api/pages/book_a/1").json()["run"] is None  # nothing succeeded, nothing stored


@pytest.mark.parametrize(
    "env, missing",
    [({}, ["LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"]),
     ({"LLM_API_KEY": "sk-secret-value"}, ["LLM_BASE_URL", "LLM_MODEL"])],
    ids=["nothing-set", "only-the-key-set"],
)
def test_missing_llm_settings_answer_503_naming_only_the_variables(api, conn, tmp_path, monkeypatch, env, missing):
    client, _ = api
    del app.dependency_overrides[get_client_factory]  # the real factory, as the app runs it
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    # The factory loads ROOT/.env on every request: point it at an empty directory so the real .env is never read.
    monkeypatch.setattr(config, "ROOT", tmp_path)
    loaded = []
    real_load_dotenv = deps.load_dotenv

    def spy(path, *args, **kwargs):
        loaded.append(path)
        return real_load_dotenv(path, *args, **kwargs)

    monkeypatch.setattr(deps, "load_dotenv", spy)

    resp = client.post("/api/pages/book_a/1/translate")

    assert loaded == [tmp_path / ".env"] and not loaded[0].exists()
    assert resp.status_code == 503
    assert resp.json()["detail"] == f"Set {', '.join(missing)} in .env (see .env.example)"
    assert "sk-secret-value" not in resp.text  # names, never values
    assert conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
