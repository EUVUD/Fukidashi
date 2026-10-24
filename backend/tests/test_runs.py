"""The service layer, called the way the eval harness will call it: no HTTP."""

import json

import pytest
from conftest import FakeClient

from fukidashi import runs
from fukidashi.datasets.importer import import_openmantra

GOOD = json.dumps({"translations": [{"id": 0, "zh": "你好"}, {"id": 1, "zh": "再见"}]}, ensure_ascii=False)


class ExplodingClient(FakeClient):
    def chat(self, messages, **kwargs):
        raise RuntimeError("provider exploded")


@pytest.fixture
def imported(conn, dataset):
    import_openmantra(conn, dataset)
    return conn


def translate(conn, dataset, client, page_index=1):
    return runs.translate_page(conn, "book_a", page_index, make_client=lambda: client, dataset_root=dataset)


def test_translate_page_records_a_run_without_http(imported, dataset):
    outcome = translate(imported, dataset, FakeClient([GOOD]))

    assert outcome.result.ok
    run = imported.execute("SELECT * FROM runs WHERE id = ?", (outcome.run_id,)).fetchone()
    assert (run["status"], run["calls"], run["input_tokens"], run["response_model"]) == ("succeeded", 1, 100, "fake-model-1")
    texts = [r["text_zh_hans"] for r in imported.execute(
        "SELECT t.text_zh_hans FROM translations t JOIN segments s ON s.id = t.segment_id"
        " WHERE t.run_id = ? ORDER BY s.idx", (outcome.run_id,))]
    assert texts == ["你好", "再见"]


def test_unexpected_error_marks_the_run_failed_and_still_raises(imported, dataset):
    with pytest.raises(RuntimeError, match="provider exploded"):
        translate(imported, dataset, ExplodingClient([]))

    run = imported.execute("SELECT status, error FROM runs").fetchone()
    assert run["status"] == "failed"  # not stuck on 'running'
    assert "provider exploded" in run["error"]
    assert imported.execute("SELECT COUNT(*) FROM translations").fetchone()[0] == 0


def test_invalid_pages_are_refused_before_a_client_is_built_or_a_run_exists(imported, dataset):
    def no_client():
        raise AssertionError("make_client must not be called")

    for page_index, error in [(99, runs.PageNotFound), (2, runs.NothingToTranslate)]:
        with pytest.raises(error):
            runs.translate_page(imported, "book_a", page_index, make_client=no_client, dataset_root=dataset)
    assert imported.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
