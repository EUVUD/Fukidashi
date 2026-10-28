"""The CLI scripts that carry a little logic of their own. Nothing here sends a request."""

import importlib.util
import json
import sys

import httpx
import pytest

from fukidashi import config
from fukidashi.datasets.openmantra import get_page
from fukidashi.llm.client import LLMClient, LLMConfig

spec = importlib.util.spec_from_file_location("m0_first_call", config.ROOT / "scripts" / "m0_first_call.py")
m0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m0)

PRICED = LLMConfig("http://provider.test/v1", "key", "mock-model", price_input_per_m=1.0, price_output_per_m=2.0)
UNPRICED = LLMConfig("http://provider.test/v1", "key", "mock-model")
HINT = "set LLM_PRICE_INPUT_PER_M / LLM_PRICE_OUTPUT_PER_M in .env"


def summary(*, calls=1, unknown=0, cost=None):
    return {"calls": calls, "unknown_usage_calls": unknown, "input_tokens": 0, "output_tokens": 0, "cost_usd": cost}


def test_a_known_cost_is_shown_as_dollars():
    assert m0.cost_line(summary(cost=0.00014), PRICED) == "Cost: $0.0001"


@pytest.mark.parametrize("unknown, wording", [(1, "1 call returned no usage"), (3, "3 calls returned no usage")])
def test_an_unknown_cost_with_prices_set_says_unknown_and_how_many_calls_had_no_usage(unknown, wording):
    line = m0.cost_line(summary(calls=3, unknown=unknown), PRICED)

    assert line == f"Cost: unknown ({wording})"
    assert HINT not in line  # the prices are fine: pointing at .env would mislead


def test_a_missing_price_still_shows_the_price_hint():
    assert m0.cost_line(summary(), UNPRICED) == f"Cost: not computed ({HINT})"
    half = LLMConfig("http://provider.test/v1", "key", "mock-model", price_input_per_m=1.0)
    assert HINT in m0.cost_line(summary(), half)


def test_both_reasons_are_shown_when_both_apply():
    line = m0.cost_line(summary(calls=2, unknown=1), UNPRICED)

    assert "unknown (1 call returned no usage)" in line and HINT in line


# --- the script end to end, on a mock provider ------------------------------------------------------------

def run_main(monkeypatch, tmp_path, dataset, handler, capsys, *, prices=True):
    """`main()` with the real LLMClient on `handler`; nothing is read from .env and nothing leaves the process."""
    monkeypatch.setattr(sys, "argv", ["m0_first_call.py", "--book", "book_a", "--page", "1"])
    monkeypatch.setattr(m0, "load_dotenv", lambda path: None)
    monkeypatch.setattr(m0, "ROOT", tmp_path)  # the ledger and the output file go here, not into runs/
    monkeypatch.setattr(m0, "get_page", lambda book, index: get_page(book, index, dataset))
    monkeypatch.setattr(m0, "LLMClient", lambda config, meter: LLMClient(
        config, meter, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    env = {"LLM_BASE_URL": "http://provider.test/v1", "LLM_API_KEY": "sk-test", "LLM_MODEL": "mock-model",
           "LLM_PRICE_INPUT_PER_M": "1.0" if prices else "", "LLM_PRICE_OUTPUT_PER_M": "2.0" if prices else ""}
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as exited:
        m0.main()
    return exited.value.code, capsys.readouterr().out


def html_page(request):
    return httpx.Response(200, text="<html>gateway</html>", headers={"content-type": "text/html"})


def test_main_says_the_cost_is_unknown_when_a_response_could_not_be_read(monkeypatch, tmp_path, dataset, capsys):
    code, out = run_main(monkeypatch, tmp_path, dataset, html_page, capsys)

    assert code == 1  # the Run failed
    assert "Failed: The response from http://provider.test/v1 is not a chat completion" in out
    assert "unknown-usage calls: 1" in out
    assert "Cost: unknown (1 call returned no usage)" in out and "LLM_PRICE" not in out
    (saved,) = (tmp_path / "runs" / "m0").glob("*.json")
    attempts = json.loads(saved.read_text(encoding="utf-8"))["attempts"]
    assert len(attempts) == 1 and attempts[0]["error"] and attempts[0]["record"]["usage"] is None


def test_main_keeps_the_price_hint_when_the_price_is_missing(monkeypatch, tmp_path, dataset, capsys):
    _, out = run_main(monkeypatch, tmp_path, dataset, html_page, capsys, prices=False)

    assert "Cost: unknown (1 call returned no usage); not computed (set LLM_PRICE_INPUT_PER_M" in out


# --- scripts/m2_tool_smoke.py ---------------------------------------------------------------------------------

spec = importlib.util.spec_from_file_location("m2_tool_smoke", config.ROOT / "scripts" / "m2_tool_smoke.py")
m2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m2)


def test_the_smoke_dry_run_prints_the_first_request_with_the_tool_and_sends_nothing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["m2_tool_smoke.py", "--dry-run"])
    monkeypatch.setattr(m2, "LLMClient", None)  # a dry run must not even build the real client
    monkeypatch.setattr(m2, "load_dotenv", None)  # nor read .env

    m2.main()

    out = capsys.readouterr().out
    request = json.loads(out[: out.rindex("}") + 1])
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert request["messages"][1]["content"] == m2.LINE
    (tool,) = request["tools"]
    assert tool["function"]["name"] == "lookup_term" and tool["function"]["parameters"]["required"] == ["term"]
    assert request["json_output"] is False and "budget" not in request
    assert out.rstrip().endswith("Dry run: nothing was sent.")


def test_the_smoke_dry_run_with_json_output_says_so(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["m2_tool_smoke.py", "--dry-run", "--json-output"])

    m2.main()

    out = capsys.readouterr().out
    request = json.loads(out[: out.rindex("}") + 1])
    assert request["json_output"] is True and "JSON" in request["messages"][0]["content"]


def test_the_glossary_tool_renders_the_made_up_name_only_it_knows():
    assert m2.lookup_term("カゲミザワ先輩") == {"term": "カゲミザワ", "zh": "阴泽"}
    assert m2.lookup_term("屋上")["zh"] is None
    assert m2.uses_glossary("阴泽前辈，今天也在屋顶上吗？") and not m2.uses_glossary("影见泽前辈")


def test_the_smoke_script_end_to_end_on_a_mock_provider(monkeypatch, tmp_path, capsys):
    replies = iter([
        httpx.Response(200, json={"id": "c", "object": "chat.completion", "created": 0, "model": "mock-model-1",
                                  "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                                      "role": "assistant", "content": None, "reasoning_content": "look it up",
                                      "tool_calls": [{"id": "call_1", "type": "function", "function": {
                                          "name": "lookup_term", "arguments": '{"term": "カゲミザワ"}'}}]}}],
                                  "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}),
        httpx.Response(200, json={"id": "c", "object": "chat.completion", "created": 0, "model": "mock-model-1",
                                  "choices": [{"index": 0, "finish_reason": "stop", "message": {
                                      "role": "assistant", "content": "阴泽前辈，今天也在屋顶上吗？"}}],
                                  "usage": {"prompt_tokens": 150, "completion_tokens": 10, "total_tokens": 160}}),
    ])
    monkeypatch.setattr(sys, "argv", ["m2_tool_smoke.py"])
    monkeypatch.setattr(m2, "load_dotenv", lambda path: None)
    monkeypatch.setattr(m2, "ROOT", tmp_path)  # the ledger and the output file go here, not into runs/
    monkeypatch.setattr(m2, "LLMClient", lambda config, meter: LLMClient(
        config, meter, http_client=httpx.Client(transport=httpx.MockTransport(lambda request: next(replies)))))
    for name, value in {"LLM_BASE_URL": "http://provider.test/v1", "LLM_API_KEY": "sk-test",
                        "LLM_MODEL": "mock-model", "LLM_PRICE_INPUT_PER_M": "1.0",
                        "LLM_PRICE_OUTPUT_PER_M": "2.0"}.items():
        monkeypatch.setenv(name, value)

    m2.main()

    out = capsys.readouterr().out
    assert "Model (as returned by the API): mock-model-1" in out
    assert "reasoning_content came back (10 characters)" in out and "tool call lookup_term" in out
    assert "Final answer: 阴泽前辈，今天也在屋顶上吗？" in out and "Uses the glossary rendering 阴泽: yes" in out
    assert "Calls: 2  input tokens: 250  output tokens: 30" in out
    (saved,) = (tmp_path / "runs" / "m2").glob("*-tool-smoke.json")
    data = json.loads(saved.read_text(encoding="utf-8"))
    assert data["uses_glossary"] is True and data["result"]["stop_reason"] == "final"
    assert data["meter"]["calls"] == 2 and "sk-test" not in saved.read_text(encoding="utf-8")


def test_the_smoke_script_with_json_output_reports_whether_the_provider_accepted_it(monkeypatch, tmp_path, capsys):
    def rejects(request):
        return httpx.Response(400, json={"error": {"message": "response_format is not supported with tools"}})

    monkeypatch.setattr(sys, "argv", ["m2_tool_smoke.py", "--json-output"])
    monkeypatch.setattr(m2, "load_dotenv", lambda path: None)
    monkeypatch.setattr(m2, "ROOT", tmp_path)
    monkeypatch.setattr(m2, "LLMClient", lambda config, meter: LLMClient(
        config, meter, http_client=httpx.Client(transport=httpx.MockTransport(rejects))))
    for name, value in {"LLM_BASE_URL": "http://provider.test/v1", "LLM_API_KEY": "sk-test",
                        "LLM_MODEL": "mock-model"}.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(SystemExit) as exited:
        m2.main()

    out = capsys.readouterr().out
    assert exited.value.code == 1 and "Stop reason: provider_error" in out
    assert "JSON output together with tools: not accepted (Bad request:" in out
    assert "response_format is not supported with tools" in out


def smoke_run(monkeypatch, tmp_path, capsys, *args, json_mode=None):
    """The smoke script on a mock provider that calls the tool and then answers. Returns (output, saved file, request bodies)."""
    bodies = []
    replies = iter([
        httpx.Response(200, json={"id": "c", "object": "chat.completion", "created": 0, "model": "mock-model-1",
                                  "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                                      "role": "assistant", "content": None,
                                      "tool_calls": [{"id": "call_1", "type": "function", "function": {
                                          "name": "lookup_term", "arguments": '{"term": "カゲミザワ"}'}}]}}],
                                  "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}),
        httpx.Response(200, json={"id": "c", "object": "chat.completion", "created": 0, "model": "mock-model-1",
                                  "choices": [{"index": 0, "finish_reason": "stop", "message": {
                                      "role": "assistant", "content": '{"zh": "阴泽前辈"}'}}],
                                  "usage": {"prompt_tokens": 150, "completion_tokens": 10, "total_tokens": 160}}),
    ])

    def handler(request):
        bodies.append(json.loads(request.content))
        return next(replies)

    monkeypatch.setattr(sys, "argv", ["m2_tool_smoke.py", *args])
    monkeypatch.setattr(m2, "load_dotenv", lambda path: None)
    monkeypatch.setattr(m2, "ROOT", tmp_path)
    monkeypatch.setattr(m2, "LLMClient", lambda config, meter: LLMClient(
        config, meter, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    env = {"LLM_BASE_URL": "http://provider.test/v1", "LLM_API_KEY": "sk-test", "LLM_MODEL": "mock-model"}
    if json_mode is not None:
        env["LLM_JSON_MODE"] = json_mode
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    m2.main()
    (saved,) = (tmp_path / "runs" / "m2").glob("*.json")
    return capsys.readouterr().out, json.loads(saved.read_text(encoding="utf-8")), bodies


def test_the_smoke_script_says_response_format_was_sent_and_accepted(monkeypatch, tmp_path, capsys):
    out, saved, bodies = smoke_run(monkeypatch, tmp_path, capsys, "--json-output")

    assert all(b["response_format"] == {"type": "json_object"} and b["tools"] for b in bodies)
    assert "JSON output together with tools: accepted by the provider; response_format was sent" in out
    assert (saved["json_mode"], saved["response_format_sent"], saved["json_output_with_tools"]) == (
        "json_object", True, "accepted")


def test_the_smoke_script_does_not_call_it_accepted_when_response_format_was_never_sent(monkeypatch, tmp_path, capsys):
    out, saved, bodies = smoke_run(monkeypatch, tmp_path, capsys, "--json-output", json_mode="none")

    assert bodies and all("response_format" not in b for b in bodies)  # the provider never saw it
    assert "JSON output together with tools: not tested (LLM_JSON_MODE=none, so response_format was not sent)" in out
    assert "accepted" not in out.split("JSON output together with tools:")[1].splitlines()[0]
    assert (saved["json_mode"], saved["response_format_sent"], saved["json_output_with_tools"]) == (
        "none", False, "not tested")


def test_the_smoke_script_without_json_output_sends_no_response_format_and_says_nothing_about_it(monkeypatch, tmp_path, capsys):
    out, saved, bodies = smoke_run(monkeypatch, tmp_path, capsys)

    assert all("response_format" not in b for b in bodies) and "JSON output together" not in out
    assert (saved["response_format_sent"], saved["json_output_with_tools"]) == (False, None)


# --- scripts/m2_tool_smoke.py --repair-check -------------------------------------------------------------------

def tool_call_reply():
    return httpx.Response(200, json={
        "id": "c", "object": "chat.completion", "created": 0, "model": "mock-model-1",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "reasoning_content": "look it up",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "lookup_term", "arguments": '{"term": "カゲミザワ"}'}}]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})


def chat_reply(text, finish_reason="stop"):
    return httpx.Response(200, json={
        "id": "c", "object": "chat.completion", "created": 0, "model": "mock-model-1",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 150, "completion_tokens": 10, "total_tokens": 160}})


JSON_ANSWER = '{"translations": [{"id": 0, "zh": "阴泽前辈，今天也在屋顶上吗？"}]}'


def repair_run(monkeypatch, tmp_path, capsys, replies, *args, json_mode=None):
    """The smoke script with --repair-check on a mock provider that plays `replies`. Returns
    (exit code or None, output, saved file, request bodies)."""
    bodies, queue = [], iter(replies)

    def handler(request):
        bodies.append(json.loads(request.content))
        return next(queue)

    monkeypatch.setattr(sys, "argv", ["m2_tool_smoke.py", "--repair-check", *args])
    monkeypatch.setattr(m2, "load_dotenv", lambda path: None)
    monkeypatch.setattr(m2, "ROOT", tmp_path)
    monkeypatch.setattr(m2, "LLMClient", lambda config, meter: LLMClient(
        config, meter, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    env = {"LLM_BASE_URL": "http://provider.test/v1", "LLM_API_KEY": "sk-test", "LLM_MODEL": "mock-model",
           "LLM_PRICE_INPUT_PER_M": "1.0", "LLM_PRICE_OUTPUT_PER_M": "2.0"}
    if json_mode is not None:
        env["LLM_JSON_MODE"] = json_mode
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    code = None
    try:
        m2.main()
    except SystemExit as exited:
        code = exited.code
    (saved,) = (tmp_path / "runs" / "m2").glob("*.json")
    return code, capsys.readouterr().out, json.loads(saved.read_text(encoding="utf-8")), bodies


def test_the_repair_check_dry_run_prints_both_requests_and_sends_nothing(monkeypatch, capsys):
    def no_network(*args, **kwargs):
        raise AssertionError("the dry run sent a request")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(sys, "argv", ["m2_tool_smoke.py", "--repair-check", "--dry-run"])
    monkeypatch.setattr(m2, "load_dotenv", None)  # a dry run reads no .env

    m2.main()

    out = capsys.readouterr().out
    requests = json.loads(out[: out.rindex("}\n}") + 3])
    first, repair = requests["first_request"], requests["repair_request"]
    assert [t["function"]["name"] for t in first["tools"]] == ["lookup_term"]
    assert [m["role"] for m in repair["messages"]] == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert repair["response_format"] == {"type": "json_object"} and "tools" not in repair
    assert repair["messages"][2]["tool_calls"] and repair["messages"][3]["tool_call_id"] == "call_dry_run"
    assert "Your answer could not be used" in repair["messages"][5]["content"]
    assert ("Repair request roles: system, user, assistant, tool, assistant, user; tools: none; "
            "response_format: {'type': 'json_object'}") in out
    assert out.rstrip().endswith("Dry run: nothing was sent.")


def test_the_repair_check_dry_run_follows_llm_json_mode(monkeypatch, capsys):
    monkeypatch.setenv("LLM_JSON_MODE", "none")
    monkeypatch.setattr(sys, "argv", ["m2_tool_smoke.py", "--repair-check", "--dry-run"])

    m2.main()

    out = capsys.readouterr().out
    assert "response_format: not sent" in out and "tools: none" in out


def test_the_repair_check_reports_a_repair_request_the_provider_accepted(monkeypatch, tmp_path, capsys):
    replies = [tool_call_reply(), chat_reply("阴泽前辈，今天也在屋顶上吗？"), chat_reply(JSON_ANSWER)]

    code, out, saved, bodies = repair_run(monkeypatch, tmp_path, capsys, replies)

    assert code is None and len(bodies) == 3
    repair = bodies[2]
    assert [m["role"] for m in repair["messages"]] == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert repair["response_format"] == {"type": "json_object"} and "tools" not in repair
    assert repair["messages"][2]["reasoning_content"] == "look it up"  # the history goes back unchanged
    assert "Repair request: roles system, user, assistant, tool, assistant, user; no tools; response_format was sent" in out
    assert "Repair request: accepted by the provider (HTTP 200); the answer parses as JSON: " + JSON_ANSWER in out
    assert "and it passes parse_translations" in out and "Calls: 3  input tokens: 400" in out
    check = saved["repair_check"]
    assert (check["accepted"], check["error"], check["tools_sent"], check["response_format_sent"]) == (
        True, None, False, True)
    assert (check["answer_is_json"], check["answer_passes_parse_translations"], check["answer"]) == (
        True, True, JSON_ANSWER)
    assert saved["meter"]["calls"] == 3 and "sk-test" not in json.dumps(saved)


def test_the_repair_check_reports_the_providers_error_when_it_rejects_the_request(monkeypatch, tmp_path, capsys):
    rejected = httpx.Response(400, json={"error": {"message": "tool messages need tools"}})
    replies = [tool_call_reply(), chat_reply("阴泽前辈，今天也在屋顶上吗？"), rejected]

    code, out, saved, bodies = repair_run(monkeypatch, tmp_path, capsys, replies)

    assert code is None and len(bodies) == 3  # a finding, not a failure of the script
    assert "Repair request: not accepted (HTTP 400): Bad request:" in out and "tool messages need tools" in out
    check = saved["repair_check"]
    assert (check["accepted"], check["status_code"]) == (False, 400)
    assert "tool messages need tools" in check["error"] and check["answer"] is None
    assert saved["meter"]["calls"] == 3


def test_the_repair_check_does_not_call_an_answer_that_is_not_json_accepted(monkeypatch, tmp_path, capsys):
    replies = [tool_call_reply(), chat_reply("阴泽前辈，今天也在屋顶上吗？"), chat_reply("sorry, plain text")]

    _, out, saved, _ = repair_run(monkeypatch, tmp_path, capsys, replies)

    assert "Repair request: HTTP 200, but the answer is not JSON: 'sorry, plain text'" in out
    assert (saved["repair_check"]["accepted"], saved["repair_check"]["answer_is_json"]) == (False, False)


def test_the_repair_check_says_when_response_format_was_not_sent(monkeypatch, tmp_path, capsys):
    replies = [tool_call_reply(), chat_reply("阴泽前辈，今天也在屋顶上吗？"), chat_reply(JSON_ANSWER)]

    _, out, saved, bodies = repair_run(monkeypatch, tmp_path, capsys, replies, json_mode="none")

    assert "response_format" not in bodies[2]
    assert "response_format was NOT sent (LLM_JSON_MODE=none)" in out
    assert saved["repair_check"]["response_format_sent"] is False


def test_the_repair_check_is_not_sent_when_the_agent_never_answered(monkeypatch, tmp_path, capsys):
    code, out, saved, bodies = repair_run(monkeypatch, tmp_path, capsys, [chat_reply("cut", finish_reason="length")])

    assert code == 1 and len(bodies) == 1
    assert "Repair request: not sent, because the agent did not finish with an answer" in out
    assert saved["repair_check"] is None


def test_without_the_flag_the_smoke_script_saves_no_repair_check(monkeypatch, tmp_path, capsys):
    _, saved, bodies = smoke_run(monkeypatch, tmp_path, capsys)

    assert saved["repair_check"] is None and len(bodies) == 2


def test_the_smoke_script_run_is_one_trace_when_tracing_is_on(monkeypatch, tmp_path, capsys):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from fukidashi import tracing

    exporter = InMemorySpanExporter()
    monkeypatch.setitem(tracing.BACKENDS, "memory", lambda: tracing.Backend(exporter, {}, {"backend": "memory"}))
    monkeypatch.setenv("FUKIDASHI_TRACING", "memory")
    try:
        repair_run(monkeypatch, tmp_path, capsys,
                   [tool_call_reply(), chat_reply("阴泽前辈，今天也在屋顶上吗？"), chat_reply(JSON_ANSWER)])
        tracing.flush()
        spans = exporter.get_finished_spans()
    finally:
        tracing.shutdown()

    assert len({s.context.trace_id for s in spans}) == 1
    (top,) = [s for s in spans if s.parent is None]
    assert top.name == "translate_page m2_tool_smoke/0"
    calls = [s for s in spans if s.instrumentation_scope.name == "openinference.instrumentation.openai"]
    assert len(calls) == 3  # the agent's two calls and the repair request


# --- scripts/m3_review_smoke.py -------------------------------------------------------------------------------

spec = importlib.util.spec_from_file_location("m3_review_smoke", config.ROOT / "scripts" / "m3_review_smoke.py")
m3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3)

import base64  # noqa: E402

from conftest import mock_llm_client  # noqa: E402
from test_agent_loop import Provider, answer, ask, call, upstream_500  # noqa: E402

from fukidashi import runs  # noqa: E402
from fukidashi.agents import reviewer, translator  # noqa: E402
from fukidashi.datasets.importer import import_openmantra  # noqa: E402

M3_TRANSLATIONS = json.dumps({"translations": [{"id": 0, "zh": "你好"}, {"id": 1, "zh": "再见"}]}, ensure_ascii=False)
M3_REVISE = json.dumps({"verdict": "revise", "issues": [
    {"id": 1, "comment": "太生硬", "suggestion": "回头见"}]}, ensure_ascii=False)


def m3_setup(monkeypatch, tmp_path, dataset, db_path, *args):
    monkeypatch.setattr(sys, "argv", ["m3_review_smoke.py", "--book", "book_a", "--page", "1", *args])
    monkeypatch.setattr(m3, "ROOT", tmp_path)  # the ledger and the output file go here, not into runs/
    monkeypatch.setattr(m3, "get_page", lambda book, index: get_page(book, index, dataset))
    monkeypatch.setattr(m3.config, "DB_PATH", db_path)  # the real database is never opened


def m3_run(monkeypatch, tmp_path, dataset, db_path, capsys, provider):
    """The script on a scripted provider. Returns (exit code, output, the saved file, if there is one)."""
    m3_setup(monkeypatch, tmp_path, dataset, db_path)
    monkeypatch.setattr(m3, "load_dotenv", lambda path: None)
    monkeypatch.setattr(m3, "LLMClient", lambda config, meter: LLMClient(
        config, meter, http_client=httpx.Client(transport=httpx.MockTransport(provider))))
    for name, value in {"LLM_BASE_URL": "http://provider.test/v1", "LLM_API_KEY": "sk-test", "LLM_MODEL": "mock-model",
                        "LLM_PRICE_INPUT_PER_M": "1.0", "LLM_PRICE_OUTPUT_PER_M": "2.0"}.items():
        monkeypatch.setenv(name, value)
    code = 0
    try:
        m3.main()
    except SystemExit as exited:
        code = exited.code
    saved = list((tmp_path / "runs" / "smoke").glob("*.json"))
    return code, capsys.readouterr().out, (json.loads(saved[0].read_text(encoding="utf-8")) if saved else None)


def run_count(db_path):
    from fukidashi.db import connect

    c = connect(db_path)
    try:
        return c.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    finally:
        c.close()


def test_the_review_smoke_dry_run_prints_both_first_requests_with_the_image_replaced_and_sends_nothing(
        monkeypatch, tmp_path, dataset, db_path, conn, capsys):
    import_openmantra(conn, dataset)

    def refuse(*args, **kwargs):
        raise AssertionError("a dry run must send nothing")

    m3_setup(monkeypatch, tmp_path, dataset, db_path, "--dry-run")
    monkeypatch.setattr(m3, "LLMClient", refuse)  # no real client,
    monkeypatch.setattr(m3, "load_dotenv", refuse)  # no .env,
    monkeypatch.setattr(httpx.Client, "send", refuse)  # and not a single HTTP request

    m3.main()

    out = capsys.readouterr().out
    shown = json.loads(out[: out.rindex("}") + 1])
    first, second = shown["translator_first_request"], shown["reviewer_first_request"]
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert first["messages"][0]["content"] == translator.FLAG_SYSTEM_PROMPT
    assert [t["function"]["name"] for t in first["tools"]] == ["get_previous_context", "flag_uncertain"]
    assert first["json_output"] is True and "budget" not in first
    assert second["messages"][0]["content"] == reviewer.SYSTEM_PROMPT and second["tools"] is None
    for request in (first, second):
        image, text = request["messages"][1]["content"]
        assert image["image_url"]["url"].endswith("base64 omitted)") and "000.jpg" in image["image_url"]["url"]
        assert "こんにちは" in text["text"]
    assert "(placeholder translation of region 0)" in second["messages"][1]["content"][1]["text"]
    assert "(placeholder flag)" in second["messages"][1]["content"][1]["text"]
    assert "data:image" not in out and base64.b64encode(b"\xff\xd8fake-jpeg").decode() not in out  # the image is gone
    assert "REF-ZH" not in out and "REF-EN" not in out
    assert out.rstrip().endswith("Dry run: nothing was sent.")
    assert not (tmp_path / "runs").exists() and run_count(db_path) == 0  # no file, no ledger, no Run


def test_the_review_smoke_reads_the_earlier_runs_and_does_not_need_a_database(monkeypatch, tmp_path, dataset, db_path, conn):
    m3_setup(monkeypatch, tmp_path, dataset, db_path)
    import_openmantra(conn, dataset)
    assert m3.earlier_pages("book_a", 2) == [translator.PreviousPage(
        1, (translator.PreviousRegion(0, "こんにちは", None), translator.PreviousRegion(1, "さようなら", None)),
        translated=False)]  # page 1 is there, with no translator Run yet
    assert runs.translate_page(
        conn, "book_a", 1, make_client=lambda: mock_llm_client(Provider(answer(M3_TRANSLATIONS))),
        dataset_root=dataset, config="translator").result.ok
    before = run_count(db_path)

    (page,) = m3.earlier_pages("book_a", 2)

    assert (page.page_index, page.translated) == (1, True) and [r.zh for r in page.regions] == ["你好", "再见"]
    assert run_count(db_path) == before  # read only
    monkeypatch.setattr(m3.config, "DB_PATH", tmp_path / "no-such.db")
    assert m3.earlier_pages("book_a", 2) == [] and not (tmp_path / "no-such.db").exists()


def test_the_review_smoke_end_to_end_on_a_mock_provider(monkeypatch, tmp_path, dataset, db_path, conn, capsys):
    import_openmantra(conn, dataset)
    provider = Provider(
        ask(call("f1", {"id": 1, "reason": "迷い"}, name="flag_uncertain"), reasoning_content="SENTINEL-REASONING"),
        answer(M3_TRANSLATIONS),
        answer(M3_REVISE),
    )

    code, out, saved = m3_run(monkeypatch, tmp_path, dataset, db_path, capsys, provider)

    assert code == 0 and len(provider.bodies) == 3
    assert "Translator: stop reason final, 2 steps, 2 calls, 200 input / 40 output tokens, cost $0.0003" in out
    assert "flagged: 迷い" in out and "flags: 1" in out
    assert "Reviewer: stop reason final, 1 steps, 1 calls, 100 input / 20 output tokens, cost $0.0001" in out
    assert "verdict: revise, 1 issue(s)" in out and "[1] 太生硬" in out and "suggestion: 回头见" in out
    assert "Total: 3 calls, 300 input / 60 output tokens, cost $0.0004" in out
    assert "Saved runs/smoke/" in out and "-review-book_a-001.json" in out
    assert saved["translator"]["flags"] == [{"id": 1, "reason": "迷い"}] and saved["translator"]["steps"] == 2
    assert saved["reviewer"]["verdict"] == "revise" and saved["reviewer"]["issues"][0]["suggestion"] == "回头见"
    assert (saved["translator"]["calls"], saved["reviewer"]["calls"], saved["meter"]["calls"]) == (2, 1, 3)
    assert saved["reviewer"]["stop_reason"] == "final" and saved["reviewer"]["latency_s"] >= 0
    assert [r["flags"] for r in saved["regions"]] == [[], ["迷い"]]
    assert "sk-test" not in json.dumps(saved) and "REF-ZH" not in json.dumps(saved)
    assert len((tmp_path / "runs" / "usage.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    assert "SENTINEL-REASONING" not in json.dumps(provider.bodies[2], ensure_ascii=False)  # the Reviewer's own request
    assert run_count(db_path) == 0  # no Run


def test_the_review_smoke_run_is_one_trace_when_tracing_is_on(monkeypatch, tmp_path, dataset, db_path, conn, capsys):
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from fukidashi import tracing

    import_openmantra(conn, dataset)
    exporter = InMemorySpanExporter()
    monkeypatch.setitem(tracing.BACKENDS, "memory", lambda: tracing.Backend(exporter, {}, {"backend": "memory"}))
    monkeypatch.setenv("FUKIDASHI_TRACING", "memory")
    try:
        code, _, _ = m3_run(monkeypatch, tmp_path, dataset, db_path, capsys,
                            Provider(answer(M3_TRANSLATIONS), answer(M3_REVISE)))
        tracing.flush()
        spans = exporter.get_finished_spans()
    finally:
        tracing.shutdown()

    assert code == 0
    assert len({s.context.trace_id for s in spans}) == 1  # the Translator and the Reviewer share one trace
    (top,) = [s for s in spans if s.parent is None]
    assert top.name == "translate_page book_a/1"
    calls = [s for s in spans if s.instrumentation_scope.name == "openinference.instrumentation.openai"]
    assert len(calls) == 2


def test_the_review_smoke_stops_before_the_review_when_the_translator_fails(monkeypatch, tmp_path, dataset, db_path, conn, capsys):
    import_openmantra(conn, dataset)
    provider = Provider(upstream_500(), answer(M3_REVISE))

    code, out, saved = m3_run(monkeypatch, tmp_path, dataset, db_path, capsys, provider)

    assert code == 1 and len(provider.bodies) == 1
    assert "failed: " in out and "API error 500" in out and "Reviewer: not run" in out
    assert saved["reviewer"] is None and saved["translator"]["ok"] is False and saved["translator"]["stop_reason"] == "provider_error"
    assert run_count(db_path) == 0


def test_the_review_smoke_reports_an_unusable_review_and_exits_1(monkeypatch, tmp_path, dataset, db_path, conn, capsys):
    import_openmantra(conn, dataset)
    provider = Provider(answer(M3_TRANSLATIONS), answer("not json"), answer("still not json"), answer(M3_REVISE))

    code, out, saved = m3_run(monkeypatch, tmp_path, dataset, db_path, capsys, provider)

    assert code == 1 and len(provider.bodies) == 3  # the translation, the verdict, its one repair
    assert "review unusable: unparseable" in out
    assert saved["reviewer"]["status"] == "unusable" and saved["reviewer"]["verdict"] is None
    assert saved["reviewer"]["calls"] == 2 and saved["reviewer"]["steps"] == 1
