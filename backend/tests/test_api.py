import json

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
