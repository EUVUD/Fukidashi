"""A person's edit of one Translation (OUTLINE §8, "How a human edit is stored"): a Run of config `human`, no model call."""

import json
import sqlite3
import threading
import time

import pytest
from conftest import FakeClient, post_translate

from fukidashi import runs, store
from fukidashi.datasets.importer import import_openmantra
from fukidashi.db import connect
from fukidashi.llm.metering import Meter

GOOD = json.dumps({"translations": [{"id": 0, "zh": "你好"}, {"id": 1, "zh": "再见"}]}, ensure_ascii=False)
OTHER = json.dumps({"translations": [{"id": 0, "zh": "您好"}, {"id": 1, "zh": "拜拜"}]}, ensure_ascii=False)


@pytest.fixture
def imported(conn, dataset):
    import_openmantra(conn, dataset)
    return conn


def translate(conn, dataset, *replies, page_index=1) -> int:
    """A Run of `baseline` on book_a's page, through FakeClient. Returns the Run's id; the Run succeeded."""
    outcome = runs.translate_page(
        conn, "book_a", page_index, make_client=lambda: FakeClient(list(replies)), dataset_root=dataset)
    assert outcome.result.ok
    return outcome.run_id


def finished_run(conn, config: str, texts: dict[int, str] | None) -> int:
    """A Run of `config` on book_a/1 that finishes with `texts` (segment idx -> zh-Hans); None makes it a failed Run."""
    page_id = store.find_page(conn, "book_a", 1)["id"]
    seg_ids = {s["idx"]: s["id"] for s in store.page_segments(conn, page_id)}
    run_id = store.start_run(conn, page_id=page_id, config=config, params={}, model="some-model")
    store.finish_run(
        conn, run_id, error=None if texts is not None else "boom", response_models=[], usage=Meter().summary(),
        attempts={}, translations=None if texts is None else {seg_ids[i]: t for i, t in texts.items()},
    )
    return run_id


def texts_of(conn, run_id: int) -> list[str]:
    """The Run's Translations in segment order."""
    return [r["text_zh_hans"] for r in conn.execute(
        "SELECT t.text_zh_hans FROM translations t JOIN segments s ON s.id = t.segment_id WHERE t.run_id = ?"
        " ORDER BY s.idx", (run_id,))]


def run_ids(conn) -> list[int]:
    return [r["id"] for r in conn.execute("SELECT id FROM runs ORDER BY id")]


def attempts_of(conn, run_id: int) -> dict:
    return json.loads(store.get_run(conn, run_id)["attempts_json"])


# --- the service layer ---


def test_an_edit_records_a_human_run_with_the_base_translations_and_one_changed(imported, dataset):
    base = translate(imported, dataset, GOOD)

    run_id = runs.edit_translation(imported, "book_a", 1, 1, "再会")

    run = store.get_run(imported, run_id)
    assert (run["config"], run["model"], run["status"], run["config_json"]) == ("human", "human", "succeeded", "{}")
    assert (run["calls"], run["unknown_usage_calls"]) == (0, 0)
    assert (run["input_tokens"], run["output_tokens"], run["response_model"], run["error"]) == (None, None, None, None)
    assert run["cost_usd"] == 0.0  # no call, so nothing was spent: 0.0, not unknown
    assert run["started_at"] and run["finished_at"] == run["started_at"]
    assert json.loads(run["attempts_json"]) == {
        "config": "human", "base_run_id": base, "edits": [{"idx": 1, "before": "再见", "after": "再会"}]}
    assert texts_of(imported, run_id) == ["你好", "再会"]
    assert texts_of(imported, base) == ["你好", "再见"]  # the base Run keeps its own Translations
    page = store.find_page(imported, "book_a", 1)
    assert store.latest_succeeded_run(imported, page["id"])["id"] == run_id  # so the page shows the edit


def test_a_second_edit_is_based_on_the_first_human_run_so_both_show(imported, dataset):
    base = translate(imported, dataset, GOOD)

    first = runs.edit_translation(imported, "book_a", 1, 0, "喂")
    second = runs.edit_translation(imported, "book_a", 1, 1, "再会")

    assert attempts_of(imported, first)["base_run_id"] == base
    assert attempts_of(imported, second) == {
        "config": "human", "base_run_id": first, "edits": [{"idx": 1, "before": "再见", "after": "再会"}]}
    assert texts_of(imported, second) == ["喂", "再会"]


def test_editing_the_same_segment_again_takes_the_earlier_edit_as_before(imported, dataset):
    translate(imported, dataset, GOOD)
    runs.edit_translation(imported, "book_a", 1, 1, "再会")

    second = runs.edit_translation(imported, "book_a", 1, 1, "后会有期")

    assert attempts_of(imported, second)["edits"] == [{"idx": 1, "before": "再会", "after": "后会有期"}]
    assert texts_of(imported, second) == ["你好", "后会有期"]


def test_the_base_is_the_latest_succeeded_run_of_any_config(imported, dataset):
    translate(imported, dataset, GOOD)
    agent = finished_run(imported, "full", {0: "甲", 1: "乙"})  # newer, and not a baseline Run
    finished_run(imported, "baseline", None)  # newer still, but failed
    store.start_run(imported, page_id=store.find_page(imported, "book_a", 1)["id"], config="baseline", params={},
                    model="m")  # and one that is still running

    run_id = runs.edit_translation(imported, "book_a", 1, 0, "丙")

    assert attempts_of(imported, run_id) == {
        "config": "human", "base_run_id": agent, "edits": [{"idx": 0, "before": "甲", "after": "丙"}]}
    assert texts_of(imported, run_id) == ["丙", "乙"]


def test_a_text_equal_to_the_current_translation_records_no_run(imported, dataset):
    base = translate(imported, dataset, GOOD)

    assert runs.edit_translation(imported, "book_a", 1, 1, "再见") is None  # the model's text
    assert run_ids(imported) == [base]
    human = runs.edit_translation(imported, "book_a", 1, 1, "再会")
    assert runs.edit_translation(imported, "book_a", 1, 1, "再会") is None  # the person's text
    assert run_ids(imported) == [base, human]
    assert not imported.in_transaction


def test_the_text_is_stored_exactly_as_sent(imported, dataset):
    translate(imported, dataset, GOOD)
    text = "  再见，\n　再会 "  # spaces around it, a newline and an ideographic space inside

    run_id = runs.edit_translation(imported, "book_a", 1, 1, text)

    assert texts_of(imported, run_id)[1] == text
    assert attempts_of(imported, run_id)["edits"][0]["after"] == text
    # a text that differs from the current one only by whitespace is an edit, not "equal to the current translation"
    assert runs.edit_translation(imported, "book_a", 1, 1, text.strip()) is not None


def test_a_segment_the_base_run_has_no_translation_for_is_added(imported, dataset):
    """Every succeeded Run has a Translation for every Segment, so this cannot happen to a Run the system made: the edit
    adds the Translation (`before` is null) rather than failing."""
    base = translate(imported, dataset, GOOD)
    imported.execute("DELETE FROM translations WHERE run_id = ? AND segment_id = (SELECT id FROM segments WHERE idx = 1)",
                     (base,))
    imported.commit()

    run_id = runs.edit_translation(imported, "book_a", 1, 1, "再会")

    assert attempts_of(imported, run_id)["edits"] == [{"idx": 1, "before": None, "after": "再会"}]
    assert texts_of(imported, run_id) == ["你好", "再会"]


@pytest.mark.parametrize("text", ["", " ", "\n\t ", "　"], ids=["empty", "space", "whitespace", "ideographic-space"])
def test_an_empty_or_whitespace_only_text_is_refused_and_records_no_run(imported, dataset, text):
    base = translate(imported, dataset, GOOD)

    with pytest.raises(runs.EmptyTranslation):
        runs.edit_translation(imported, "book_a", 1, 1, text)

    assert run_ids(imported) == [base]
    assert not imported.in_transaction


@pytest.mark.parametrize(
    "book, page_index, idx, error",
    [("book_a", 99, 0, runs.PageNotFound), ("no_book", 1, 0, runs.PageNotFound),
     ("book_a", 1, 2, runs.SegmentNotFound), ("book_a", 1, -1, runs.SegmentNotFound),
     ("book_a", 2, 0, runs.SegmentNotFound)],  # page 2 has no segments
    ids=["unknown-page", "unknown-book", "unknown-idx", "negative-idx", "page-without-segments"],
)
def test_an_unknown_page_or_segment_is_refused_and_records_no_run(imported, dataset, book, page_index, idx, error):
    base = translate(imported, dataset, GOOD)

    with pytest.raises(error):
        runs.edit_translation(imported, book, page_index, idx, "你好")

    assert run_ids(imported) == [base]
    assert not imported.in_transaction


def test_a_page_without_a_succeeded_run_has_nothing_to_edit(imported, dataset):
    with pytest.raises(runs.NothingToEdit):
        runs.edit_translation(imported, "book_a", 1, 0, "你好")
    assert run_ids(imported) == []

    failed = runs.translate_page(  # a failed Run is not something to edit either
        imported, "book_a", 1, make_client=lambda: FakeClient(["not json", "not json"]), dataset_root=dataset)
    assert not failed.result.ok
    with pytest.raises(runs.NothingToEdit):
        runs.edit_translation(imported, "book_a", 1, 0, "你好")
    assert run_ids(imported) == [failed.run_id]
    assert not imported.in_transaction


def test_a_failure_while_the_run_is_written_leaves_no_run_and_no_translations(imported, dataset):
    base = translate(imported, dataset, GOOD)
    imported.execute(  # the Run row is inserted first; this makes the Translations insert that follows it raise
        "CREATE TRIGGER fail_translations BEFORE INSERT ON translations BEGIN SELECT RAISE(ABORT, 'disk on fire'); END")

    with pytest.raises(sqlite3.IntegrityError, match="disk on fire"):
        runs.edit_translation(imported, "book_a", 1, 1, "再会")

    assert run_ids(imported) == [base]  # no new Run row, and none stuck on `running`
    assert imported.execute("SELECT COUNT(*) FROM translations").fetchone()[0] == 2  # the base Run's, and no more
    assert not imported.in_transaction
    imported.execute("DROP TRIGGER fail_translations")  # the connection is still usable: the same edit goes through now
    run_id = runs.edit_translation(imported, "book_a", 1, 1, "再会")
    assert run_ids(imported) == [base, run_id] and texts_of(imported, run_id) == ["你好", "再会"]


def test_a_later_run_supersedes_the_edit_which_stays_stored(imported, dataset):
    translate(imported, dataset, GOOD)
    human = runs.edit_translation(imported, "book_a", 1, 1, "再会")

    newer = translate(imported, dataset, OTHER)

    page = store.find_page(imported, "book_a", 1)
    shown = store.latest_succeeded_run(imported, page["id"])
    assert (shown["id"], shown["config"]) == (newer, "baseline")
    assert [r["text_zh_hans"] for r in store.segments_with_translations(imported, page["id"], shown["id"])] == [
        "您好", "拜拜"]
    assert store.get_run(imported, human)["status"] == "succeeded"
    assert texts_of(imported, human) == ["你好", "再会"]


def test_two_edits_at_once_are_both_kept(imported, dataset, db_path, monkeypatch):
    """Each edit reads its base and writes its Run in one transaction that takes the write lock first, so whichever goes
    second is based on the first: no edit is lost. A pause between reading the base and writing the Run makes the two
    overlap: without the lock they would read the same base, and one edit would be lost or fail."""
    base = translate(imported, dataset, GOOD)
    read_translations = store.segments_with_translations

    def slow_read(*args):
        rows = read_translations(*args)
        time.sleep(0.2)
        return rows

    monkeypatch.setattr(store, "segments_with_translations", slow_read)
    barrier = threading.Barrier(2)
    errors = []

    def edit(idx, text):
        c = connect(db_path)
        try:
            barrier.wait(timeout=10)
            runs.edit_translation(c, "book_a", 1, idx, text)
        except Exception as e:  # reported by the main thread
            errors.append(e)
        finally:
            c.close()

    threads = [threading.Thread(target=edit, args=args) for args in [(0, "喂"), (1, "再会")]]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    first, second = run_ids(imported)[1:]
    assert attempts_of(imported, first)["base_run_id"] == base
    assert attempts_of(imported, second)["base_run_id"] == first
    assert texts_of(imported, second) == ["喂", "再会"]  # whichever order they ran in, the last Run has both edits


def test_translate_page_does_not_make_human_runs(imported, dataset):
    assert runs.HUMAN == "human" and runs.HUMAN not in runs.CONFIGS

    with pytest.raises(ValueError, match="unknown config"):
        runs.translate_page(imported, "book_a", 1, make_client=lambda: FakeClient([GOOD]), dataset_root=dataset,
                            config="human")
    assert run_ids(imported) == []


# --- the API ---


def edit(client, idx, text, *, page=1):
    return client.put(f"/api/pages/book_a/{page}/segments/{idx}/translation", json={"text_zh_hans": text})


def run_count(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]


def translated(api, background):
    """The `api` fixture with book_a/1 translated by `baseline`, its Run over."""
    client, use_replies = api
    use_replies(GOOD)
    assert post_translate(client, background).status_code == 202
    return client, use_replies


def test_put_edits_one_translation_and_it_survives_a_reload(api, conn, background):
    client, _ = translated(api, background)

    resp = edit(client, 1, "再会")

    assert resp.status_code == 200
    body = resp.json()
    assert [(s["index"], s["text_zh_hans"]) for s in body["segments"]] == [(0, "你好"), (1, "再会")]
    run = body["run"]
    assert (run["config"], run["model"], run["status"], run["calls"], run["unknown_usage_calls"]) == (
        "human", "human", "succeeded", 0, 0)
    assert (run["input_tokens"], run["output_tokens"], run["response_model"], run["cost_usd"]) == (None, None, None, 0.0)
    assert client.get("/api/pages/book_a/1").json() == body  # a reload shows the same page
    assert run_count(conn) == 2


def test_two_edits_through_the_api_both_show(api, conn, background):
    client, _ = translated(api, background)

    first = edit(client, 0, "喂").json()
    second = edit(client, 1, "再会")

    assert [s["text_zh_hans"] for s in second.json()["segments"]] == ["喂", "再会"]
    assert attempts_of(conn, second.json()["run"]["id"])["base_run_id"] == first["run"]["id"]


def test_put_keeps_the_text_exactly_as_sent(api, background):
    client, _ = translated(api, background)
    text = " 再见，\n再会  "

    body = edit(client, 1, text).json()

    assert body["segments"][1]["text_zh_hans"] == text
    assert client.get("/api/pages/book_a/1").json()["segments"][1]["text_zh_hans"] == text


def test_put_with_the_current_text_answers_200_with_the_page_unchanged(api, conn, background):
    client, _ = translated(api, background)
    before = client.get("/api/pages/book_a/1").json()

    resp = edit(client, 1, "再见")

    assert resp.status_code == 200 and resp.json() == before
    assert run_count(conn) == 1


@pytest.mark.parametrize(
    "path, body, status",
    [("/api/pages/book_a/99/segments/0/translation", {"text_zh_hans": "你好"}, 404),
     ("/api/pages/no_book/1/segments/0/translation", {"text_zh_hans": "你好"}, 404),
     ("/api/pages/book_a/1/segments/2/translation", {"text_zh_hans": "你好"}, 404),
     ("/api/pages/book_a/2/segments/0/translation", {"text_zh_hans": "你好"}, 404),
     ("/api/pages/book_a/1/segments/0/translation", {"text_zh_hans": ""}, 422),
     ("/api/pages/book_a/1/segments/0/translation", {"text_zh_hans": " \n　"}, 422),
     ("/api/pages/book_a/1/segments/0/translation", {}, 422),
     ("/api/pages/book_a/1/segments/0/translation", {"zh": "你好"}, 422),
     ("/api/pages/book_a/1/segments/0/translation", {"text_zh_hans": None}, 422),
     ("/api/pages/book_a/1/segments/0/translation", {"text_zh_hans": 7}, 422),
     ("/api/pages/book_a/1/segments/0/translation", ["你好"], 422)],
    ids=["unknown-page", "unknown-book", "unknown-idx", "page-without-segments", "empty", "whitespace",
         "no-field", "other-field", "null", "not-a-string", "not-an-object"],
)
def test_a_refused_edit_records_no_run(api, conn, path, body, status, background):
    client, _ = translated(api, background)
    before = client.get("/api/pages/book_a/1").json()

    resp = client.put(path, json=body)

    assert resp.status_code == status
    assert run_count(conn) == 1  # only the translation
    assert client.get("/api/pages/book_a/1").json() == before


@pytest.mark.parametrize("kwargs", [{}, {"content": b"{nope", "headers": {"content-type": "application/json"}}],
                         ids=["no-body", "not-json"])
def test_a_body_that_is_missing_or_not_json_answers_422_and_records_no_run(api, conn, kwargs, background):
    client, _ = translated(api, background)

    resp = client.put("/api/pages/book_a/1/segments/0/translation", **kwargs)

    assert resp.status_code == 422
    assert run_count(conn) == 1


def test_a_page_without_a_succeeded_run_answers_409_and_records_no_run(api, conn):
    client, _ = api

    resp = edit(client, 0, "你好")

    assert resp.status_code == 409 and "no translation to edit" in resp.json()["detail"]
    assert run_count(conn) == 0


def test_a_later_translate_supersedes_the_edit_and_the_human_run_stays(api, conn, background):
    client, use_replies = translated(api, background)
    edited = edit(client, 1, "再会").json()

    use_replies(OTHER)
    assert post_translate(client, background).status_code == 202
    page = client.get("/api/pages/book_a/1").json()

    assert page["run"]["config"] == "baseline"
    assert [s["text_zh_hans"] for s in page["segments"]] == ["您好", "拜拜"]
    human = conn.execute("SELECT id, status FROM runs WHERE config = 'human'").fetchall()
    assert [(r["id"], r["status"]) for r in human] == [(edited["run"]["id"], "succeeded")]
    assert texts_of(conn, edited["run"]["id"]) == ["你好", "再会"]
