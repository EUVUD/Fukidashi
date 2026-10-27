"""The service layer, called the way the eval harness will call it: no HTTP."""

import json

import httpx
import pytest
from conftest import CrashAfterCallClient, FakeClient, UsageClient, completion, mock_llm_client

from fukidashi import runs, store
from fukidashi.datasets.importer import import_openmantra
from fukidashi.llm.metering import Usage
from fukidashi.pipeline import baseline

GOOD = json.dumps({"translations": [{"id": 0, "zh": "你好"}, {"id": 1, "zh": "再见"}]}, ensure_ascii=False)
UNKNOWN_ID = json.dumps({"translations": [{"id": i, "zh": "字"} for i in (0, 1, 7)]})
DUPLICATE_ID = json.dumps({"translations": [{"id": i, "zh": "字"} for i in (0, 1, 1)]})


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


def run_rows(conn):
    return conn.execute("SELECT * FROM runs ORDER BY id").fetchall()


def test_a_run_with_an_unknown_usage_call_and_a_priced_call_has_null_cost(imported, dataset):
    client = UsageClient(["not json", GOOD], usages=[None, Usage(100, 20)])

    outcome = translate(imported, dataset, client)

    assert outcome.result.ok
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"], run["unknown_usage_calls"]) == ("succeeded", 2, 1)
    assert run["cost_usd"] is None  # not the partial 0.0001
    assert (run["input_tokens"], run["output_tokens"]) == (100, 20)


@pytest.mark.parametrize("bad, offender", [(UNKNOWN_ID, "7"), (DUPLICATE_ID, "1")], ids=["unknown-id", "duplicate-id"])
def test_a_bad_region_id_is_retried_once_and_a_good_retry_succeeds(imported, dataset, bad, offender):
    client = FakeClient([bad, GOOD])

    outcome = translate(imported, dataset, client)

    assert outcome.result.ok and len(client.requests) == 2
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"]) == ("succeeded", 2)
    assert offender in outcome.result.attempts[0]["parse_error"]
    assert "parse_error" not in outcome.result.attempts[1]


@pytest.mark.parametrize("bad, offender", [(UNKNOWN_ID, "7"), (DUPLICATE_ID, "1")], ids=["unknown-id", "duplicate-id"])
def test_a_bad_region_id_twice_is_a_failed_run_that_names_the_id(imported, dataset, bad, offender):
    client = FakeClient([bad, bad])

    outcome = translate(imported, dataset, client)

    assert not outcome.result.ok and len(client.requests) == 2
    run = store.get_run(imported, outcome.run_id)
    assert run["status"] == "failed"
    assert "Could not parse" in run["error"] and offender in run["error"]
    assert imported.execute("SELECT COUNT(*) FROM translations").fetchone()[0] == 0
    assert imported.execute("SELECT COUNT(*) FROM runs WHERE status = 'running'").fetchone()[0] == 0


@pytest.mark.parametrize("bad", ["not json", UNKNOWN_ID, DUPLICATE_ID], ids=["not-json", "unknown-id", "duplicate-id"])
def test_a_run_that_succeeds_on_its_retry_stores_no_error(imported, dataset, bad):
    outcome = translate(imported, dataset, FakeClient([bad, GOOD]))

    assert outcome.result.ok and outcome.result.error is None
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["error"], run["calls"]) == ("succeeded", None, 2)
    first, second = json.loads(run["attempts_json"])
    assert first["parse_error"]  # the first attempt's parse error stays in the attempts
    assert "parse_error" not in second


def test_two_bad_replies_are_still_a_failed_run_with_the_error_of_the_last_attempt(imported, dataset):
    outcome = translate(imported, dataset, FakeClient(["not json", UNKNOWN_ID]))

    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"]) == ("failed", 2)
    assert run["error"].startswith("Could not parse the model response after 2 attempt(s)")
    assert "region ids not on the page [7]" in run["error"]
    first, second = json.loads(run["attempts_json"])
    assert "no JSON object" in first["parse_error"] and "[7]" in second["parse_error"]


def test_an_unexpected_exception_raises_run_crashed_with_the_run_id(imported, dataset):
    with pytest.raises(runs.RunCrashed) as crashed:
        translate(imported, dataset, ExplodingClient([]))

    (run,) = run_rows(imported)
    assert crashed.value.run_id == run["id"]
    assert (run["status"], run["error"]) == ("failed", "RuntimeError: provider exploded")
    assert str(crashed.value) == run["error"]
    assert isinstance(crashed.value.__cause__, RuntimeError)
    assert not isinstance(crashed.value, runs.TranslateError)  # a Run was recorded
    assert imported.execute("SELECT COUNT(*) FROM runs WHERE status = 'running'").fetchone()[0] == 0


def test_a_crash_after_a_priced_call_keeps_the_spend_on_the_failed_run(imported, dataset):
    with pytest.raises(runs.RunCrashed) as crashed:
        translate(imported, dataset, CrashAfterCallClient([GOOD]))

    run = store.get_run(imported, crashed.value.run_id)
    assert (run["status"], run["calls"], run["cost_usd"]) == ("failed", 1, 0.0001)


def upstream_500(request):
    return httpx.Response(500, json={"error": {"message": "upstream exploded"}})


def test_a_provider_error_is_a_failed_run_with_unknown_cost(imported, dataset):
    sent = []

    def handler(request):
        sent.append(request)
        return upstream_500(request)

    outcome = translate(imported, dataset, mock_llm_client(handler))  # the real LLMClient on a mock provider

    assert not outcome.result.ok
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"], run["unknown_usage_calls"], run["cost_usd"]) == ("failed", 1, 1, None)
    assert (run["input_tokens"], run["output_tokens"]) == (None, None)  # unknown, not 0
    assert "API error 500" in run["error"] and "upstream exploded" in run["error"]
    assert run["error"] == outcome.result.error
    assert len(sent) == 1  # the SDK did not resend it
    assert imported.execute("SELECT COUNT(*) FROM translations").fetchone()[0] == 0


def test_a_failed_request_after_a_priced_call_makes_the_whole_cost_unknown(imported, dataset):
    replies = iter([completion(""), upstream_500(None)])  # empty content is retried; the retry fails at the provider

    outcome = translate(imported, dataset, mock_llm_client(lambda request: next(replies)))

    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"], run["unknown_usage_calls"]) == ("failed", 2, 1)
    assert run["cost_usd"] is None  # not the partial 0.00014
    assert (run["input_tokens"], run["output_tokens"]) == (100, 20)  # the known call's tokens stay
    assert "API error 500" in run["error"]


def test_a_response_without_choices_is_a_failed_run_not_a_crash(imported, dataset):
    outcome = translate(imported, dataset, mock_llm_client(lambda request: completion(choices=False)))

    assert not outcome.result.ok  # translate_page returned: no RunCrashed
    run = store.get_run(imported, outcome.run_id)
    assert (run["status"], run["calls"], run["unknown_usage_calls"]) == ("failed", 1, 0)
    assert (run["input_tokens"], run["output_tokens"]) == (100, 20)  # the call was recorded with its real usage
    assert run["cost_usd"] == pytest.approx(0.00014)
    assert "mock-model returned no choices" in run["error"]


def test_a_region_id_the_parser_let_through_cannot_leave_the_run_running(imported, dataset, monkeypatch):
    # The parser now rejects unknown ids; if one ever got past it, the mapping to segment ids must not
    # leave the Run on 'running'.
    monkeypatch.setattr(baseline, "parse_translations", lambda text, ids: {0: "a", 1: "b", 7: "c"})

    with pytest.raises(runs.RunCrashed) as crashed:
        translate(imported, dataset, FakeClient([GOOD]))

    run = store.get_run(imported, crashed.value.run_id)
    assert run["status"] == "failed" and "KeyError" in run["error"]
    assert imported.execute("SELECT COUNT(*) FROM translations").fetchone()[0] == 0


def test_a_failure_while_finishing_the_run_still_marks_it_failed(imported, dataset, monkeypatch):
    real_finish = store.finish_run
    calls = []

    def flaky_finish(*args, **kwargs):
        calls.append(kwargs["error"])
        if len(calls) == 1:
            raise RuntimeError("database is locked")
        return real_finish(*args, **kwargs)

    monkeypatch.setattr(store, "finish_run", flaky_finish)

    with pytest.raises(runs.RunCrashed, match="database is locked"):
        translate(imported, dataset, FakeClient([GOOD]))

    (run,) = run_rows(imported)
    assert (run["status"], run["error"]) == ("failed", "RuntimeError: database is locked")
    assert imported.execute("SELECT COUNT(*) FROM translations").fetchone()[0] == 0


@pytest.mark.parametrize(
    "client, original",
    [(ExplodingClient([]), "provider exploded"), (FakeClient([GOOD]), "write 1 failed")],
    ids=["client-raises", "saving-the-result-raises"],
)
def test_run_crashed_still_carries_the_run_id_when_the_recovery_write_fails_too(imported, dataset, monkeypatch, client, original):
    writes = []

    def broken_finish_run(*args, **kwargs):
        writes.append(kwargs["error"])
        raise RuntimeError(f"write {len(writes)} failed")

    monkeypatch.setattr(store, "finish_run", broken_finish_run)

    with pytest.raises(runs.RunCrashed) as crashed:
        translate(imported, dataset, client)

    (run,) = run_rows(imported)
    assert crashed.value.run_id == run["id"]
    assert run["status"] == "running"  # nothing could be written: the Run is left as it was
    assert isinstance(crashed.value.__cause__, RuntimeError) and str(crashed.value.__cause__) == original
    message = str(crashed.value)
    assert original in message and "marking the Run failed also failed" in message
    assert f"write {len(writes)} failed" in message  # the recovery write's own error
    assert crashed.value.error == message
    assert not isinstance(crashed.value, runs.TranslateError)
