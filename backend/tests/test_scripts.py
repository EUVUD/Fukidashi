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
