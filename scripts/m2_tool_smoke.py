"""M2 tool smoke test: can the configured model call a tool through the agent loop, with and without JSON output?

One made-up Japanese line with a made-up proper noun, and a `lookup_term` tool with a one-entry glossary for that noun.
Text only: no page image, no OpenMantra text.

Usage:
  uv run python scripts/m2_tool_smoke.py --dry-run       # build the first request, send nothing
  uv run python scripts/m2_tool_smoke.py                 # real run, configured by .env
  uv run python scripts/m2_tool_smoke.py --json-output   # the same with response_format set next to the tools
  uv run python scripts/m2_tool_smoke.py --repair-check  # then one repair request, as the Translator builds it
  uv run python scripts/m2_tool_smoke.py --repair-check --dry-run   # print both requests, send nothing

The repair request carries a history with tool calls and their results but no `tools`. Whether the provider takes that
was never tried (task 007 Review); --repair-check sends it once. The agent's plain-text answer stands in for
a bad answer.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
import httpx
from pydantic import BaseModel, Field

from fukidashi import tracing
from fukidashi.agents.loop import FINAL, AgentResult, Tool, run_agent
from fukidashi.agents.translator import PARSE_ERRORS, repair_messages
from fukidashi.llm.client import LLMClient, LLMConfig
from fukidashi.llm.errors import ProviderError
from fukidashi.llm.metering import BudgetExceeded, Meter, PageBudget
from fukidashi.pipeline.baseline import parse_translations

ROOT = Path(__file__).resolve().parents[1]

LINE = "カゲミザワ先輩、今日も屋上にいるのかな。"  # made up: nobody can know how this name is rendered
GLOSSARY = {"カゲミザワ": "阴泽"}  # the rendering is arbitrary on purpose: only the tool can supply it

SYSTEM_PROMPT = (
    "You translate one line of Japanese manga dialogue into Simplified Chinese (zh-Hans). "
    "Before you answer, call lookup_term for every proper noun in the line and use the rendering it returns exactly. "
    "Reply with the Chinese translation only."
)
JSON_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "Reply with the Chinese translation only.", 'Reply with only a JSON object: {"zh": "<the translation>"}.'
)


class LookupTerm(BaseModel):
    term: str = Field(description="The Japanese term to look up, as written in the line")


def lookup_term(term: str) -> dict:
    for ja, zh in GLOSSARY.items():
        if ja in term:
            return {"term": ja, "zh": zh}
    return {"term": term, "zh": None, "note": "not in the glossary"}


TOOL = Tool(
    name="lookup_term",
    description="Look up the glossary rendering of a Japanese term in Simplified Chinese.",
    args=LookupTerm,
    handler=lookup_term,
)


REPAIR_IDS = [0]  # the one region of the made-up page, as the Translator's repair message lists them


def bad_answer_error(text: str | None) -> str:
    """Why the Translator would reject `text`: what `parse_translations` says about a plain-text translation."""
    try:
        parse_translations(text or "", REPAIR_IDS)
    except PARSE_ERRORS as e:
        return str(e)
    return "the answer is not the JSON this page needs"


def repair_check(
    client, budget: PageBudget, result: AgentResult, *, max_tokens: int, response_format_sent: bool
) -> dict:
    """Send the repair request the Translator would send after `result`'s answer, and say what the provider did."""
    messages = repair_messages(result.messages, bad_answer_error(result.text), REPAIR_IDS)
    check = {"roles": [m["role"] for m in messages], "tools_sent": False, "response_format_sent": response_format_sent,
             "accepted": False, "status_code": None, "error": None, "answer": None, "answer_is_json": None,
             "answer_passes_parse_translations": None}
    try:
        reply = client.chat(messages, budget=budget, max_tokens=max_tokens, json_output=True)
    except ProviderError as e:
        return check | {"status_code": e.status_code, "error": e.description}
    except BudgetExceeded as e:
        return check | {"error": f"not sent: {e}"}
    try:
        json.loads(reply.text)
    except ValueError:
        check["answer_is_json"] = False
    else:
        check["answer_is_json"] = True
    try:
        parse_translations(reply.text, REPAIR_IDS)
    except PARSE_ERRORS:
        check["answer_passes_parse_translations"] = False
    else:
        check["answer_passes_parse_translations"] = True
    # HTTP 200 (the call returned) and an answer that parses as JSON
    return check | {"accepted": check["answer_is_json"], "answer": reply.text}


def describe_repair_check(check: dict) -> list[str]:
    sent = ("response_format was sent" if check["response_format_sent"]
            else "response_format was NOT sent (LLM_JSON_MODE=none)")
    lines = [f"Repair request: roles {', '.join(check['roles'])}; no tools; {sent}"]
    if check["error"] is not None:
        code = f" (HTTP {check['status_code']})" if check["status_code"] else ""
        lines.append(f"Repair request: not accepted{code}: {check['error']}")
    elif check["accepted"]:
        lines.append("Repair request: accepted by the provider (HTTP 200); the answer parses as JSON: "
                     + check["answer"])
        lines.append("  and it passes parse_translations" if check["answer_passes_parse_translations"]
                     else "  but it does not pass parse_translations")
    else:
        lines.append(f"Repair request: HTTP 200, but the answer is not JSON: {check['answer']!r}")
    return lines


def completion(message: dict, finish_reason: str) -> httpx.Response:
    return httpx.Response(200, json={
        "id": "dry-run", "object": "chat.completion", "created": 0, "model": "dry-run",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", **message}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def dry_run_repair_check(args, budget: PageBudget) -> None:
    """Print the first request and the repair request exactly as they would go out (JSON bodies, built by the real
    client and the real agent loop), answering the provider's side with made-up replies on a mock transport."""
    bodies: list[dict] = []
    arguments = json.dumps({"term": "カゲミザワ"}, ensure_ascii=False)
    replies = iter([
        completion({"content": None, "reasoning_content": "(made up)", "tool_calls": [{
            "id": "call_dry_run", "type": "function", "function": {"name": "lookup_term", "arguments": arguments}}]},
            "tool_calls"),
        completion({"content": "阴泽前辈，今天也在屋顶上吗？", "reasoning_content": "(made up)"}, "stop"),
        completion({"content": '{"translations": [{"id": 0, "zh": "..."}]}'}, "stop"),
    ])

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return next(replies)

    json_mode = os.environ.get("LLM_JSON_MODE", "json_object")  # .env is not read in a dry run
    config = LLMConfig("http://dry-run.invalid/v1", "dry-run", "LLM_MODEL", json_mode=json_mode)
    client = LLMClient(config, Meter(), http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = run_agent(client, budget, system_prompt=JSON_SYSTEM_PROMPT if args.json_output else SYSTEM_PROMPT,
                       user_message=LINE, tools=[TOOL], max_tokens=args.max_tokens, json_output=args.json_output,
                       max_steps=args.max_steps)
    check = repair_check(client, budget, result, max_tokens=args.max_tokens,
                         response_format_sent=json_mode == "json_object")
    first, repair = bodies[0], bodies[-1]
    print(json.dumps({"first_request": first, "repair_request": repair}, ensure_ascii=False, indent=2))
    print(f"\nRepair request roles: {', '.join(check['roles'])}; tools: {'sent' if 'tools' in repair else 'none'}; "
          f"response_format: {repair.get('response_format', 'not sent')}")
    print("The history is made up (a tool call, then a plain-text answer); its shape is the real one. "
          "LLM_MODEL and the key are placeholders.")
    print("\nDry run: nothing was sent.")


class FirstRequest(Exception):
    """Raised by the dry-run client when the agent loop makes its first model call."""

    def __init__(self, messages: list[dict], tools: list[dict] | None, options: dict):
        super().__init__("first request")
        self.messages, self.tools, self.options = messages, tools, options


class DryRunClient:
    """Stands in for LLMClient: stops the loop at the first call, so the request printed is the one that would go out."""

    def chat(self, messages, *, tools=None, **options):
        raise FirstRequest(messages, tools, options)


def uses_glossary(text: str | None) -> bool:
    return bool(text) and any(zh in text for zh in GLOSSARY.values())


def describe_steps(result: AgentResult) -> list[str]:
    lines = []
    for n, step in enumerate(result.steps, 1):
        if step["error"]:
            lines.append(f"Step {n}: request failed: {step['error']}")
            continue
        chars = step["reasoning_chars"]
        lines.append(f"Step {n}: finish_reason={step['finish_reason']}  reasoning_content "
                     f"{'did not come back' if chars is None else f'came back ({chars} characters)'}"
                     + (f"  text={step['text']!r}" if step["text"] else ""))
        for c in step["tool_calls"]:
            outcome = f"error: {c['error']}" if c["error"] else f"result: {c['result']}"
            lines.append(f"    tool call {c['name']}({c['arguments']}) -> {outcome}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 tool smoke test on one made-up line.")
    ap.add_argument("--json-output", action="store_true", help="set response_format (JSON mode) next to the tools")
    ap.add_argument("--max-tokens", type=int, default=4_000, help="output cap per call, reasoning included")
    ap.add_argument("--page-budget", type=int, default=20_000, help="token budget for the run")
    ap.add_argument("--max-steps", type=int, default=6, help="model calls per run")
    ap.add_argument("--dry-run", action="store_true", help="print the first request and send nothing")
    ap.add_argument("--repair-check", action="store_true",
                    help="after the run, send one repair request as the Translator Agent builds it (history, repair "
                         "message, JSON output, no tools)")
    args = ap.parse_args()

    system_prompt = JSON_SYSTEM_PROMPT if args.json_output else SYSTEM_PROMPT
    budget = PageBudget("m2/tool-smoke", args.page_budget)
    options = dict(system_prompt=system_prompt, user_message=LINE, tools=[TOOL], max_tokens=args.max_tokens,
                   json_output=args.json_output, max_steps=args.max_steps)

    if args.dry_run and args.repair_check:
        dry_run_repair_check(args, budget)
        return
    if args.dry_run:
        try:
            run_agent(DryRunClient(), budget, **options)
        except FirstRequest as first:
            sent = {k: v for k, v in first.options.items() if k != "budget"}  # the budget is ours, not the request's
            print(json.dumps({"messages": first.messages, "tools": first.tools, **sent}, ensure_ascii=False, indent=2))
            print("\nDry run: nothing was sent.")
        return

    load_dotenv(ROOT / ".env")
    try:
        config = LLMConfig.from_env()
    except RuntimeError as e:
        sys.exit(str(e))
    try:
        tracing.setup()
    except tracing.TracingError as e:
        sys.exit(str(e))
    meter = Meter(ledger_path=ROOT / "runs" / "usage.jsonl")
    client = LLMClient(config, meter)
    repair = None
    with tracing.page_span(book="m2_tool_smoke", page_index=0, config="tool-smoke"):  # one trace for the whole run
        result = run_agent(client, budget, **options)
        if args.repair_check and result.stop_reason == FINAL:
            repair = repair_check(client, budget, result, max_tokens=args.max_tokens,
                                  response_format_sent=config.json_mode == "json_object")
    # LLMClient sends response_format only in JSON mode, so `--json-output` alone proves nothing about the provider
    response_format_sent = args.json_output and config.json_mode == "json_object"
    first = result.steps[0] if result.steps else None
    accepted = first is not None and not first["error"]
    json_with_tools = (None if not args.json_output else "not tested" if not response_format_sent
                       else "accepted" if accepted else "not accepted")

    summary = meter.summary()
    models = sorted({s["record"]["response_model"] for s in result.steps if s["record"] and s["record"]["response_model"]})
    out = {
        "milestone": "M2",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider_host": urlparse(config.base_url).hostname,
        "requested_model": config.model,
        "response_models": models,
        "json_output": args.json_output,
        "json_mode": config.json_mode,
        "response_format_sent": response_format_sent,
        "json_output_with_tools": json_with_tools,  # accepted, not accepted, not tested (no response_format sent) or null
        "line": LINE,
        "uses_glossary": uses_glossary(result.text),
        "meter": summary,
        "page_budget": {"limit": budget.limit_tokens, "used": budget.used_tokens, "verified": budget.verified},
        "result": asdict(result),
        "repair_check": repair,  # null unless --repair-check ran it
    }
    run_dir = ROOT / "runs" / "m2"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{datetime.now():%Y%m%d-%H%M%S}-tool-smoke{'-json' if args.json_output else ''}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model (as returned by the API): {', '.join(models) or '-'}")
    print("\n".join(describe_steps(result)))
    print(f"Stop reason: {result.stop_reason}" + (f"  ({result.error})" if result.error else ""))
    if result.stop_reason == FINAL:
        print(f"Final answer: {result.text}")
        print(f"Uses the glossary rendering {list(GLOSSARY.values())[0]}: {'yes' if out['uses_glossary'] else 'no'}")
    if args.json_output:
        if not response_format_sent:
            print(f"JSON output together with tools: not tested (LLM_JSON_MODE={config.json_mode}, "
                  "so response_format was not sent)")
        else:
            verdict = ("accepted by the provider" if accepted else
                       f"not accepted ({first['error'] if first else 'no request was made'})")
            print(f"JSON output together with tools: {verdict}; response_format was sent")
    if args.repair_check:
        print("\n".join(describe_repair_check(repair) if repair else
                        ["Repair request: not sent, because the agent did not finish with an answer"]))
    cost = f"${summary['cost_usd']:.4f}" if summary["cost_usd"] is not None else "unknown"
    print(f"Calls: {summary['calls']}  input tokens: {summary['input_tokens']}  output tokens: {summary['output_tokens']}"
          f"  unknown-usage calls: {summary['unknown_usage_calls']}  Cost: {cost}")
    print(f"Saved {out_path.relative_to(ROOT)}")
    if result.stop_reason != FINAL:
        sys.exit(1)


if __name__ == "__main__":
    main()
