"""Tracing: one trace per page. The backend is an in-memory exporter that goes in through the same seam Phoenix uses
(`tracing.BACKENDS`), so nothing here starts Phoenix, reads .env or sends a request: the provider is `mock_llm_client`."""

from __future__ import annotations

import base64
import json
import logging
import time
import types

import httpx
import pytest
from conftest import FakeClient, completion, mock_llm_client
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from test_agent_loop import Provider, answer, ask, call
from test_evaluation import run_cli
from test_translator import BOOK, imported, previous_context, root, translate, zh_json  # noqa: F401 (fixtures)

from fukidashi import config as app_config
from fukidashi import main, runs, store, tracing
from fukidashi.agents import loop
from fukidashi.evaluation import harness
from fukidashi.tracing import redact

OPENAI_SCOPE = "openinference.instrumentation.openai"
API_KEY = "sk-test-SECRET-0123456789"
IMAGE_B64 = base64.b64encode(b"\xff\xd8fake-jpeg").decode()  # what every test page's image becomes in the prompt


def backend_of(exporter, *, info=None):
    return lambda: tracing.Backend(exporter, {"openinference.project.name": "test"},
                                   info or {"backend": "memory", "project": "test", "endpoint": "memory"})


class Traced:
    def __init__(self, exporter):
        self.exporter = exporter

    def spans(self):
        tracing.flush()
        return self.exporter.get_finished_spans()


@pytest.fixture
def traced(monkeypatch):
    """Tracing switched on with an in-memory exporter, and switched off again afterwards."""
    exporter = InMemorySpanExporter()
    monkeypatch.setitem(tracing.BACKENDS, "memory", backend_of(exporter))
    monkeypatch.setenv("FUKIDASHI_TRACING", "memory")
    assert tracing.setup() is True
    yield Traced(exporter)
    tracing.shutdown()


def one_trace(spans):
    """The root span of `spans`, after checking that they are exactly one trace under that single root."""
    assert len({s.context.trace_id for s in spans}) == 1
    (root_span,) = [s for s in spans if s.parent is None]
    by_id = {s.context.span_id: s for s in spans}
    for s in spans:
        while s.parent is not None:  # every span reaches the root
            s = by_id[s.parent.span_id]
        assert s is root_span
    return root_span


def provider_calls(spans):
    """The spans of requests sent to the provider: the OpenAI instrumentor's, one per call of `LLMClient`."""
    return [s for s in spans if s.instrumentation_scope.name == OPENAI_SCOPE]


def trace_id_of(span):
    return format(span.context.trace_id, "032x")


def attribute_values(spans):
    """Every string a span exports: its attributes and the attributes of its events."""
    for s in spans:
        for value in [*(s.attributes or {}).values(), *(v for e in s.events for v in (e.attributes or {}).values())]:
            yield from (value if isinstance(value, (list, tuple)) else [value])


def assert_clean(spans, provider=None):
    """AC4: no key, no image and no reference translation in anything that is exported. When the provider's requests
    are given, it also checks that the image really was sent, so that the scan has something to find."""
    if provider is not None:
        sent = json.dumps(provider.bodies)
        assert IMAGE_B64 in sent and API_KEY not in sent
    assert not [k for s in spans for k in s.attributes if "message_content.image" in k]  # hidden, not just redacted
    for value in attribute_values(spans):
        text = str(value)
        assert API_KEY not in text
        assert "data:image" not in text and IMAGE_B64 not in text
        assert "REF-ZH" not in text and "REF-EN" not in text


# --- AC1: off by default --------------------------------------------------------------------------------------

def test_with_the_variable_unset_nothing_is_installed(imported, root):
    import openai.resources.chat.completions as completions
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor
    from opentelemetry import trace

    assert tracing.setup() is False
    assert tracing.describe() == {"enabled": False}
    assert type(trace.get_tracer_provider()).__name__ == "ProxyTracerProvider"  # no SDK provider anywhere
    assert not OpenAIInstrumentor().is_instrumented_by_opentelemetry
    assert not LangChainInstrumentor().is_instrumented_by_opentelemetry
    assert type(completions.Completions.create) is types.FunctionType  # not wrapped by an instrumentor
    with tracing.page_span(book="b", page_index=1, config="baseline") as span:
        assert span is None
    assert tracing.flush() is True

    outcome = translate(imported, root, mock_llm_client(Provider(completion(zh_json(1)))), 1, config="baseline")

    assert outcome.result.ok and outcome.trace_id is None


@pytest.mark.parametrize("value", ["", "  "])
def test_an_empty_variable_is_off_too(monkeypatch, value):
    monkeypatch.setenv("FUKIDASHI_TRACING", value)

    assert tracing.setup() is False and tracing.describe() == {"enabled": False}


# --- AC2: a baseline page is one trace -------------------------------------------------------------------------

def test_a_baseline_page_is_one_trace_with_its_run_on_the_root_and_a_span_per_provider_call(imported, root, traced):
    provider = Provider(completion("not json"), completion(zh_json(1)))  # empty or bad output is retried once

    outcome = translate(imported, root, mock_llm_client(provider, api_key=API_KEY), 1, config="baseline")

    spans = traced.spans()
    top = one_trace(spans)
    run = store.get_run(imported, outcome.run_id)
    assert outcome.result.ok and run["calls"] == len(provider.bodies) == 2
    assert top.name == f"translate_page {BOOK}/1"
    assert trace_id_of(top) == outcome.trace_id and len(outcome.trace_id) == 32
    assert dict(top.attributes) == {
        "openinference.span.kind": "CHAIN",
        "fukidashi.book": BOOK, "fukidashi.page_index": 1, "fukidashi.config": "baseline",
        "fukidashi.run_id": outcome.run_id, "fukidashi.run_status": "succeeded", "fukidashi.calls": 2,
    }
    assert len(provider_calls(spans)) == run["calls"]
    assert all(s.attributes["openinference.span.kind"] == "LLM" and s.parent.span_id == top.context.span_id
               for s in provider_calls(spans))
    assert len(spans) == 3  # baseline has no agent: the root and its two calls
    assert_clean(spans, provider)


def test_the_root_of_a_failed_run_says_so_and_a_crashed_run_still_gives_its_trace(imported, root, traced):
    failed = translate(imported, root, mock_llm_client(Provider(completion("x"), completion("y"))), 1,
                       config="baseline")

    assert not failed.result.ok
    (top,) = [s for s in traced.spans() if s.parent is None]
    assert top.attributes["fukidashi.run_status"] == "failed" and top.status.status_code.name == "ERROR"
    assert trace_id_of(top) == failed.trace_id

    with pytest.raises(runs.RunCrashed) as crashed:
        translate(imported, root, FakeCrash(), 1, config="baseline")

    crash_root = [s for s in traced.spans() if s.parent is None and s.attributes["fukidashi.run_id"] ==
                  crashed.value.run_id][0]
    assert trace_id_of(crash_root) == crashed.value.trace_id
    assert crash_root.attributes["fukidashi.run_status"] == "failed"
    assert any(e.name == "exception" for e in crash_root.events)


class FakeCrash(FakeClient):
    def __init__(self):
        super().__init__([])

    def chat(self, messages, **kwargs):
        raise RuntimeError("crashed")


def test_a_page_that_cannot_be_translated_still_leaves_its_trace_with_the_error(imported, root, traced):
    with pytest.raises(runs.PageNotFound):
        translate(imported, root, FakeClient([]), 99, config="baseline")

    (top,) = traced.spans()
    assert top.attributes["fukidashi.page_index"] == 99 and top.status.status_code.name == "ERROR"
    assert "fukidashi.run_id" not in top.attributes  # no Run existed


# --- AC3: a translator page is one trace -----------------------------------------------------------------------

def test_a_translator_page_holds_the_agent_the_tool_and_every_provider_call(imported, root, traced):
    provider = Provider(ask(previous_context(2)), answer(zh_json(2)))

    outcome = translate(imported, root, mock_llm_client(provider, api_key=API_KEY), 2)

    spans = traced.spans()
    top = one_trace(spans)
    run = store.get_run(imported, outcome.run_id)
    assert outcome.result.ok and run["calls"] == 2
    assert trace_id_of(top) == outcome.trace_id
    assert top.attributes["fukidashi.config"] == "translator" and top.attributes["fukidashi.stop_reason"] == "final"
    assert "fukidashi.fallback_reason" not in top.attributes
    assert len(provider_calls(spans)) == run["calls"]
    (tool,) = [s for s in spans if s.attributes.get("openinference.span.kind") == "TOOL"]
    assert tool.name == "get_previous_context"
    assert any(s.name == "LangGraph" for s in spans)  # the agent graph
    assert_clean(spans, provider)


def test_a_repaired_page_has_a_provider_span_for_the_repair_too(imported, root, traced):
    provider = Provider(ask(previous_context(2)), answer("not json"), answer(zh_json(2)))

    outcome = translate(imported, root, mock_llm_client(provider, api_key=API_KEY), 2)

    spans = traced.spans()
    top = one_trace(spans)
    assert outcome.result.ok and store.get_run(imported, outcome.run_id)["calls"] == 3
    assert len(provider_calls(spans)) == 3 and len(provider.bodies) == 3
    assert "tools" not in provider.bodies[2]  # the third span is the repair request
    assert top.attributes["fukidashi.stop_reason"] == "final" and "fukidashi.fallback_reason" not in top.attributes
    assert_clean(spans, provider)


@pytest.mark.parametrize("replies, stop_reason, fallback", [
    ((answer("not json"), answer("still not json"), answer(zh_json(2))), "final", "unparseable"),
    ((answer("", finish_reason="length"), answer(zh_json(2))), "incomplete", "incomplete"),
    ((ask(call("c1", {"pages_back": 1}, name="get_previous_context")),
      ask(call("c2", {"pages_back": 2}, name="get_previous_context")), answer(zh_json(2))), "step_limit", "step_limit"),
], ids=["repair-fails", "cut-off", "step-limit"])
def test_a_page_that_fell_back_to_baseline_is_still_one_trace_with_the_fallback_call(
        imported, root, traced, replies, stop_reason, fallback):
    provider = Provider(*replies)

    outcome = translate(imported, root, mock_llm_client(provider, api_key=API_KEY), 2, max_steps=2)

    spans = traced.spans()
    top = one_trace(spans)
    run = store.get_run(imported, outcome.run_id)
    assert outcome.result.ok and run["config"] == "translator" and run["calls"] == len(replies)
    assert len(provider_calls(spans)) == run["calls"]  # the repair and the fallback are in the trace
    assert top.attributes["fukidashi.stop_reason"] == stop_reason
    assert top.attributes["fukidashi.fallback_reason"].startswith(fallback)
    assert "tools" not in provider.bodies[-1]  # the last request is baseline's
    assert_clean(spans, provider)


def test_a_provider_error_is_a_failed_run_whose_trace_shows_the_failed_call(imported, root, traced):
    provider = Provider(ask(previous_context(2)), httpx.Response(500, json={"error": {"message": "upstream exploded"}}))

    outcome = translate(imported, root, mock_llm_client(provider, api_key=API_KEY), 2)

    spans = traced.spans()
    top = one_trace(spans)
    assert not outcome.result.ok and store.get_run(imported, outcome.run_id)["calls"] == 2
    assert len(provider_calls(spans)) == 2
    assert provider_calls(spans)[-1].status.status_code.name == "ERROR"
    assert top.attributes["fukidashi.stop_reason"] == "provider_error"


# --- AC4: nothing secret or bulky is exported ------------------------------------------------------------------

def test_the_key_a_provider_echoes_back_in_an_error_is_not_exported(imported, root, traced, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", API_KEY)
    echo = httpx.Response(401, json={"error": {"message": f"Incorrect API key provided: {API_KEY}"}})

    outcome = translate(imported, root, mock_llm_client(Provider(echo), api_key=API_KEY), 1, config="baseline")

    spans = traced.spans()
    assert not outcome.result.ok
    (call_span,) = provider_calls(spans)
    assert call_span.status.status_code.name == "ERROR" and any(e.name == "exception" for e in call_span.events)
    assert any("Incorrect API key provided" in str(v) for v in attribute_values(spans))  # the error is there ...
    assert_clean(spans)  # ... without the key


def test_the_redactor_removes_data_url_images_and_secrets_from_strings_and_sequences():
    url = "data:image/jpeg;base64," + "QUJD" * 50

    assert redact.redact(f'{{"url": "{url}", "text": "kept"}}') == '{"url": "[image omitted]", "text": "kept"}'
    assert redact.redact("data:image/png;base64,iVBOR+/=_-") == "[image omitted]"
    assert redact.redact(["a", f"x {url}"]) == ["a", "x [image omitted]"]
    assert redact.redact(("a", url)) == ("a", "[image omitted]")
    assert redact.redact("key sk-1 and sk-1", ["sk-1"]) == "key *** and ***"
    assert redact.redact(3) == 3 and redact.redact(True) is True and redact.redact(0.5) == 0.5
    assert redact.redact("日本語のテキスト") == "日本語のテキスト"  # text stays


def test_the_redactor_rewrites_a_span_without_changing_anything_else(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    inner = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(redact.RedactingExporter(inner, lambda: [API_KEY, ""])))
    with provider.get_tracer("t").start_as_current_span("s") as span:
        span.set_attribute("input.value", f"hello data:image/jpeg;base64,AAAA {API_KEY}")
        span.set_attribute("count", 3)
        span.add_event("e", {"message": f"bad {API_KEY}"})

    (done,) = inner.get_finished_spans()
    assert done.attributes["input.value"] == "hello [image omitted] ***" and done.attributes["count"] == 3
    assert done.events[0].attributes["message"] == "bad ***" and done.name == "s"


# --- AC5: tracing never fails a Run ----------------------------------------------------------------------------

class RaisingExporter(SpanExporter):
    def __init__(self):
        self.exports = 0

    def export(self, spans):
        self.exports += 1
        raise RuntimeError("the exporter exploded")


class FailingExporter(RaisingExporter):
    def export(self, spans):
        self.exports += 1
        return SpanExportResult.FAILURE


def run_facts(conn, run_id):
    run = store.get_run(conn, run_id)
    translations = [r["text_zh_hans"] for r in conn.execute(
        "SELECT t.text_zh_hans FROM translations t JOIN segments s ON s.id = t.segment_id WHERE t.run_id = ?"
        " ORDER BY s.idx", (run_id,))]
    return run["status"], run["error"], run["calls"], run["cost_usd"], run["unknown_usage_calls"], translations


@pytest.mark.parametrize("exporter_class", [RaisingExporter, FailingExporter])
def test_an_exporter_that_fails_leaves_the_run_as_it_is(imported, root, monkeypatch, exporter_class, caplog):
    untraced = translate(imported, root, mock_llm_client(Provider(ask(previous_context(2)), answer(zh_json(2)))), 2)
    exporter = exporter_class()
    monkeypatch.setitem(tracing.BACKENDS, "broken", backend_of(exporter))
    monkeypatch.setenv("FUKIDASHI_TRACING", "broken")
    assert tracing.setup()
    try:
        with caplog.at_level(logging.ERROR):
            outcome = translate(imported, root, mock_llm_client(Provider(ask(previous_context(2)),
                                                                         answer(zh_json(2)))), 2)
            tracing.flush()
    finally:
        tracing.shutdown()

    assert exporter.exports >= 1  # the exporter really was asked to export
    assert outcome.result.ok and outcome.trace_id is not None
    assert run_facts(imported, outcome.run_id) == run_facts(imported, untraced.run_id)
    assert run_facts(imported, outcome.run_id)[:3] == ("succeeded", None, 2)


def test_an_endpoint_where_nothing_listens_leaves_the_run_as_it_is(imported, root, monkeypatch):
    def refused(*args, **kwargs):  # what the operating system answers for a port with no listener
        raise ConnectionRefusedError(61, "Connection refused")

    monkeypatch.setattr("urllib3.util.connection.create_connection", refused)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "2")
    monkeypatch.setenv("FUKIDASHI_TRACING", "phoenix")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:6006/v1/traces")
    untraced = translate(imported, root, mock_llm_client(Provider(answer(zh_json(1)))), 1, config="baseline")
    assert tracing.setup() and tracing.describe()["endpoint"] == "http://127.0.0.1:6006/v1/traces"
    try:
        started = time.monotonic()
        outcome = translate(imported, root, mock_llm_client(Provider(answer(zh_json(1)))), 1, config="baseline")
        assert time.monotonic() - started < 1.5  # exporting happens in the background: the Run does not wait for it
        assert tracing.flush(4000) in (True, False)  # it returns, whether or not the export worked
    finally:
        started = time.monotonic()
        tracing.shutdown()
        assert time.monotonic() - started < 5

    assert outcome.result.ok and outcome.trace_id is not None
    assert run_facts(imported, outcome.run_id) == run_facts(imported, untraced.run_id)


# --- AC6: the backend and the endpoint are checked at start ---------------------------------------------------

def test_an_unknown_backend_raises_and_lists_the_supported_ones(monkeypatch):
    monkeypatch.setenv("FUKIDASHI_TRACING", "langfuse")

    with pytest.raises(tracing.TracingError) as refused:
        tracing.setup()

    assert "langfuse" in str(refused.value) and "Supported: phoenix" in str(refused.value)
    assert tracing.describe() == {"enabled": False}


@pytest.mark.parametrize("endpoint", [
    "https://phoenix.example.com/v1/traces",
    "http://10.0.0.5:6006/v1/traces",
    "http://0.0.0.0:6006/v1/traces",
    "http://localhost.example.com:6006/v1/traces",
    "http://127.0.0.1@example.com/v1/traces",
    "http://[::2]:6006/v1/traces",
    "localhost:6006/v1/traces",  # no scheme: there is no host to check
    "/v1/traces",
])
def test_a_phoenix_endpoint_that_is_not_on_this_machine_is_refused(monkeypatch, endpoint):
    monkeypatch.setenv("FUKIDASHI_TRACING", "phoenix")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", endpoint)

    with pytest.raises(tracing.TracingError) as refused:
        tracing.setup()

    assert "PHOENIX_COLLECTOR_ENDPOINT" in str(refused.value) and "not on this machine" in str(refused.value)
    assert tracing.describe() == {"enabled": False}  # nothing was installed


@pytest.mark.parametrize("endpoint", [
    None, "http://localhost:6006/v1/traces", "http://127.0.0.1:4318/v1/traces", "http://[::1]:6006/v1/traces",
    "http://LOCALHOST:6006/v1/traces",
])
def test_a_local_phoenix_endpoint_is_accepted_and_recorded_with_its_project(monkeypatch, endpoint):
    monkeypatch.setenv("FUKIDASHI_TRACING", "phoenix")
    if endpoint:
        monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", endpoint)
    try:
        assert tracing.setup() is True
        assert tracing.describe() == {"enabled": True, "backend": "phoenix", "project": "fukidashi",
                                      "endpoint": endpoint or "http://localhost:6006/v1/traces"}
    finally:
        tracing.shutdown()


def test_phoenix_gets_the_project_name_as_the_resource_attribute_it_reads(monkeypatch):
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "my-project")

    backend = tracing.BACKENDS["phoenix"]()

    assert backend.resource == {"openinference.project.name": "my-project"}
    assert backend.info["project"] == "my-project"
    backend.exporter.shutdown()


def test_setup_is_idempotent_and_shutdown_removes_the_instrumentation(monkeypatch, traced):
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from openinference.instrumentation.openai import OpenAIInstrumentor

    assert tracing.setup() is True and tracing.setup() is True  # a second call changes nothing
    assert OpenAIInstrumentor().is_instrumented_by_opentelemetry
    assert LangChainInstrumentor().is_instrumented_by_opentelemetry

    tracing.shutdown()

    assert tracing.describe() == {"enabled": False}
    assert not OpenAIInstrumentor().is_instrumented_by_opentelemetry
    assert not LangChainInstrumentor().is_instrumented_by_opentelemetry
    tracing.shutdown()  # and again: nothing to do


def test_the_cli_exits_2_before_any_run_for_an_unknown_backend_and_a_remote_endpoint(tmp_path):
    unknown = run_cli(tmp_path, "--config", "baseline", "--book", "b", "--pages", "1", FUKIDASHI_TRACING="langfuse")
    remote = run_cli(tmp_path, "--config", "baseline", "--book", "b", "--pages", "1", FUKIDASHI_TRACING="phoenix",
                     PHOENIX_COLLECTOR_ENDPOINT="https://phoenix.example.com/v1/traces")

    assert unknown.returncode == 2 and "Supported: phoenix" in unknown.stderr
    assert remote.returncode == 2 and "not on this machine" in remote.stderr
    assert "Stopped before any model call" in unknown.stderr + remote.stderr
    assert not (tmp_path / "cli.db").exists()  # no database, so no Run


def test_the_api_refuses_to_start_when_tracing_cannot_be_honoured(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "load_dotenv", lambda path: None)
    monkeypatch.setattr(app_config, "DB_PATH", tmp_path / "api.db")
    monkeypatch.setenv("FUKIDASHI_TRACING", "phoenix")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://example.com:6006/v1/traces")

    with pytest.raises(tracing.TracingError):
        with TestClient(main.app):
            pass

    assert not (tmp_path / "api.db").exists()


def test_the_api_switches_tracing_on_at_start_and_off_at_shutdown(monkeypatch, tmp_path):
    exporter = InMemorySpanExporter()
    monkeypatch.setitem(tracing.BACKENDS, "memory", backend_of(exporter))
    monkeypatch.setattr(main, "load_dotenv", lambda path: None)
    monkeypatch.setattr(app_config, "DB_PATH", tmp_path / "api.db")
    monkeypatch.setenv("FUKIDASHI_TRACING", "memory")

    with TestClient(main.app):
        assert tracing.describe()["enabled"] is True

    assert tracing.describe() == {"enabled": False}


def test_the_api_without_the_variable_starts_with_tracing_off(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "load_dotenv", lambda path: None)
    monkeypatch.setattr(app_config, "DB_PATH", tmp_path / "api.db")

    with TestClient(main.app):
        assert tracing.describe() == {"enabled": False}


def test_an_api_request_leaves_one_trace(api, traced):
    client, use_replies = api
    use_replies(json.dumps({"translations": [{"id": 0, "zh": "你好"}, {"id": 1, "zh": "再见"}]}))

    response = client.post("/api/pages/book_a/1/translate")

    assert response.status_code == 200
    (top,) = traced.spans()
    assert top.name == "translate_page book_a/1" and top.attributes["fukidashi.config"] == "baseline"
    assert top.attributes["fukidashi.run_status"] == "succeeded"


# --- AC7: the eval harness -------------------------------------------------------------------------------------

def eval_with(conn, root, tmp_path, provider, pages, *, config):
    return harness.run_eval(
        conn, config=config, book=BOOK, page_indexes=pages, make_client=lambda: mock_llm_client(provider),
        dataset_root=root, out_root=tmp_path / "out", repo_root=app_config.ROOT,
    )


def test_eval_rows_carry_the_trace_ids_and_the_manifest_says_where_the_traces_went(imported, root, tmp_path, traced):
    provider = Provider(answer(zh_json(1)), ask(previous_context(2)), answer(zh_json(2)), answer("", "length"),
                        answer("x"), answer("y"))  # page 3: cut off, then baseline fails twice

    result = eval_with(imported, root, tmp_path, provider, [1, 2, 3], config="translator")

    roots = {s.attributes["fukidashi.page_index"]: trace_id_of(s) for s in traced.spans() if s.parent is None}
    assert result.exit_code == 1 and set(roots) == {1, 2, 3}
    assert [p["trace_id"] for p in result.scores["pages"]] == [roots[1], roots[2]]
    assert [p["trace_id"] for p in result.scores["failed_pages"]] == [roots[3]]  # a failed page has its trace too
    assert {p["page_index"]: p["trace_id"] for p in result.manifest["pages"]} == roots
    assert result.manifest["tracing"] == {"enabled": True, "backend": "memory", "project": "test",
                                          "endpoint": "memory"}
    on_disk = json.loads((result.out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["tracing"]["enabled"] is True and on_disk["pages"][0]["trace_id"] == roots[1]


def test_a_crashed_page_has_its_trace_id_in_the_eval_rows_too(imported, root, tmp_path, traced):
    clients = iter([mock_llm_client(Provider(answer(zh_json(1)))), FakeCrash()])

    result = harness.run_eval(
        imported, config="baseline", book=BOOK, page_indexes=[1, 2], make_client=lambda: next(clients),
        dataset_root=root, out_root=tmp_path / "out", repo_root=app_config.ROOT,
    )

    roots = {s.attributes["fukidashi.page_index"]: trace_id_of(s) for s in traced.spans() if s.parent is None}
    (crashed,) = result.scores["failed_pages"]
    assert crashed["error"] == "RuntimeError: crashed" and crashed["trace_id"] == roots[2]
    assert [p["trace_id"] for p in result.manifest["pages"]] == [roots[1], roots[2]]


def test_eval_rows_have_a_null_trace_id_and_the_manifest_says_off_when_tracing_is_off(imported, root, tmp_path):
    result = eval_with(imported, root, tmp_path, Provider(answer(zh_json(1)), answer(zh_json(2))), [1, 2],
                       config="baseline")

    assert result.manifest["tracing"] == {"enabled": False}
    assert [(p["page_index"], p["trace_id"]) for p in result.scores["pages"]] == [(1, None), (2, None)]
    assert [(p["page_index"], p["trace_id"]) for p in result.manifest["pages"]] == [(1, None), (2, None)]


@pytest.mark.parametrize("config", ["baseline", "translator"])
@pytest.mark.parametrize("variable", ["LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"])
def test_a_langsmith_variable_stops_the_eval_before_any_run_or_request(imported, root, tmp_path, monkeypatch,
                                                                         variable, config):
    monkeypatch.setenv(variable, "true")
    provider = Provider(answer(zh_json(1)))

    def make_client():
        raise AssertionError("a client was asked for")

    with pytest.raises(loop.TracingEnabled) as refused:
        harness.run_eval(imported, config=config, book=BOOK, page_indexes=[1, 2, 3], make_client=make_client,
                         dataset_root=root, out_root=tmp_path / "out", repo_root=app_config.ROOT)

    assert variable in str(refused.value) and "true" not in str(refused.value)
    assert imported.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0 and provider.bodies == []
    assert not (tmp_path / "out").exists()


def test_the_cli_exits_2_with_no_run_when_a_langsmith_variable_is_on(tmp_path):
    stopped = run_cli(tmp_path, "--config", "translator", "--book", BOOK, "--pages", "1", LANGSMITH_TRACING="true")

    assert stopped.returncode == 2
    assert "Stopped before any model call" in stopped.stderr and "LANGSMITH_TRACING" in stopped.stderr
    import sqlite3
    with sqlite3.connect(tmp_path / "cli.db") as db:  # the database exists (the CLI opened it) and holds no Run
        assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_the_cli_prints_nothing_about_tracing_when_it_is_off(tmp_path):
    stopped = run_cli(tmp_path, "--config", "translator", "--book", BOOK, "--pages", "1")

    assert "Tracing" not in stopped.stdout + stopped.stderr


# --- the repair message is one function ------------------------------------------------------------------------

def test_the_translator_sends_exactly_what_repair_messages_builds(imported, root):
    from fukidashi.agents import translator

    provider = Provider(ask(previous_context(2)), answer("not json"), answer(zh_json(2)))

    translate(imported, root, mock_llm_client(provider), 2)

    history = provider.bodies[1]["messages"] + [{"role": "assistant", "content": "not json"}]
    assert provider.bodies[2]["messages"] == translator.repair_messages(
        history, "no JSON object in the response", [0, 1])
    assert translator.repair_messages([{"role": "system", "content": "S"}], "E", [3])[-1]["role"] == "user"
