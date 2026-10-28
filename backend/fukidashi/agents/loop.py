"""The agent loop: LangChain's `create_agent` with `LLMClientChatModel` as its model, and the guardrails of OUTLINE §3.4.

`create_agent` fixes the loop (call the model, run the tools it asks for, pass the results back, repeat). Each
project rule plugs into one of its three extension points:

- the model: `LLMClientChatModel` (model.py), so every call is metered and counted against the `PageBudget`;
- the tools: LangChain tools with a pydantic `args_schema`;
- middleware: the Step limit (`ModelCallLimitMiddleware`), a check that a truncated reply is never an answer, and the
  tool guard (a model's mistakes go back to it as the tool's result, duplicates are not run again).

LangChain stays inside this package: `run_agent` returns an `AgentResult` of plain values. Nothing here retries.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, after_model, wrap_tool_call
from langchain.agents.middleware.model_call_limit import ModelCallLimitExceededError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ValidationError

from ..llm.client import LLMClient
from ..llm.errors import ProviderError
from ..llm.metering import BudgetExceeded, PageBudget
from ..pipeline.baseline import DEFAULT_MAX_TOKENS
from .model import MALFORMED_ARGUMENTS, LLMClientChatModel, raw_arguments, to_openai_messages

DEFAULT_MAX_STEPS = 6  # model calls per run; small, because with unknown usage this and max_tokens are the only cap

FINAL = "final"
STEP_LIMIT = "step_limit"
INCOMPLETE = "incomplete"
BUDGET = "budget"
PROVIDER_ERROR = "provider_error"

# Any other finish reason (length, content_filter, insufficient_system_resource, None) means the reply was cut off.
ANSWER_FINISH_REASONS = ("stop", "tool_calls")

# `langsmith` comes with `langchain-core` and sends every prompt and reply to LangSmith when one of these is on.
TRACING_VARIABLES = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING")
_OFF_VALUES = ("", "0", "false")  # what langchain-core itself reads as off; anything else may switch tracing on


class TracingEnabled(RuntimeError):
    """A LangSmith tracing variable is on. The message names the variables, never their values."""


@dataclass(frozen=True)
class Tool:
    """A tool the model may call, defined without LangChain.

    `args` is a pydantic model: its fields are the tool's parameters, and the model's arguments are validated against
    it. `handler` gets the validated fields as keyword arguments and returns a string or anything JSON-serializable.
    """

    name: str
    description: str
    args: type[BaseModel]
    handler: Callable[..., Any]


@dataclass
class AgentResult:
    text: str | None  # the final answer; None unless stop_reason is "final"
    stop_reason: str  # final, step_limit, incomplete, budget or provider_error
    error: str | None  # None when stop_reason is "final"
    steps: list[dict]  # one per model call, a failed call included; see _steps
    messages: list[dict]  # the conversation as OpenAI-format dicts, as far as the run got


def run_agent(
    client: LLMClient,
    budget: PageBudget,
    *,
    system_prompt: str,
    user_message: str | list[dict],
    tools: Sequence[Tool | BaseTool] = (),
    max_tokens: int = DEFAULT_MAX_TOKENS,
    json_output: bool = False,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AgentResult:
    """Run the model through tool calls to a final answer. Provider failures and spent budgets end up in the result.

    Raises `TracingEnabled`, before any request, when LangSmith tracing is switched on in the environment.
    """
    refuse_tracing()
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    lc_tools = [_langchain_tool(t) for t in tools]
    model = LLMClientChatModel(client=client, budget=budget, max_tokens=max_tokens, json_output=json_output)
    agent = create_agent(
        model,
        tools=lc_tools,
        system_prompt=system_prompt,
        middleware=[
            # "error", not "end": "end" appends an invented AI message to the conversation
            ModelCallLimitMiddleware(run_limit=max_steps, exit_behavior="error"),
            _stop_on_incomplete_reply,
            _tool_guard(lc_tools),
        ],
    )
    # Streaming the state keeps the conversation so far when a node raises: the steps of a failed run come from it.
    # About five graph steps per model call; the Step limit ends the run long before this does.
    config = {"recursion_limit": 6 * max_steps + 10}
    state: dict = {"messages": [HumanMessage(content=user_message)]}
    failed_step: dict | None = None
    try:
        for state in agent.stream(state, config, stream_mode="values"):
            pass
    except ModelCallLimitExceededError as e:
        stop_reason, error = STEP_LIMIT, str(e)
    except BudgetExceeded as e:
        stop_reason, error = BUDGET, f"Stopped: {e}"
    except ProviderError as e:
        stop_reason, error = PROVIDER_ERROR, e.description
        failed_step = {"record": asdict(e.record) if e.record is not None else None, "finish_reason": None,
                       "text": "", "reasoning_chars": None, "tool_calls": [], "error": e.description}
    else:
        reason = _finish_reason(state["messages"][-1])
        if reason in ANSWER_FINISH_REASONS:
            stop_reason, error = FINAL, None
        else:
            stop_reason = INCOMPLETE
            error = f"The model's reply ended with finish reason {reason!r}, so it is not an answer"

    messages = state["messages"]
    steps = _steps(messages) + ([failed_step] if failed_step else [])
    text = messages[-1].content if stop_reason == FINAL else None
    history = [{"role": "system", "content": system_prompt}, *to_openai_messages(messages)]
    return AgentResult(text=text, stop_reason=stop_reason, error=error, steps=steps, messages=history)


def refuse_tracing() -> None:
    """Raise `TracingEnabled` when a LangSmith tracing variable is on. `run_agent` starts with this; so does the eval
    harness, so that a whole eval stops before its first Run instead of failing once per page."""
    on = [name for name in TRACING_VARIABLES if os.environ.get(name, "").strip().lower() not in _OFF_VALUES]
    if on:
        raise TracingEnabled(
            f"{' and '.join(on)} {'is' if len(on) == 1 else 'are'} set, which would send prompts and page images to "
            "LangSmith. Unset it (check .env too) before running an agent."
        )


def _langchain_tool(t: Tool | BaseTool) -> BaseTool:
    if isinstance(t, BaseTool):
        return t
    return StructuredTool.from_function(t.handler, name=t.name, description=t.description, args_schema=t.args)


def _finish_reason(message: BaseMessage) -> str | None:
    return message.response_metadata.get("finish_reason")


@after_model(can_jump_to=["end"], name="StopOnIncompleteReply")
def _stop_on_incomplete_reply(state, runtime):
    """A reply that was cut off is never an answer, and its tool calls are not run."""
    if _finish_reason(state["messages"][-1]) not in ANSWER_FINISH_REASONS:
        return {"jump_to": "end"}
    return None


def _tool_guard(tools: Sequence[BaseTool]):
    """Runs the model's tool calls. Who made a mistake decides what happens (OUTLINE §3.4): the model's mistakes are
    passed back to it as the tool's result and the loop goes on.

    LangChain's ToolNode would also run the tool, but it passes a result that is not JSON to the model as its repr and
    raises when a tool raises. So the tool runs here, and the ToolNode only carries the result into the conversation.
    """
    by_name = {t.name: t for t in tools}

    @wrap_tool_call(name="ToolGuard")
    def guard(request, handler):
        call, tool = request.tool_call, request.tool
        name, args = call["name"], call["args"]
        if MALFORMED_ARGUMENTS in args:
            return _reply(call, f"Error: the arguments of {name} are not a JSON object: {args[MALFORMED_ARGUMENTS]!r}")
        if tool is None:
            return _reply(call, f"Error: there is no tool named {name}. "
                                f"Available tools: {', '.join(by_name) or 'none'}")
        try:
            validated = _validate(tool, args)
        except ValidationError as e:
            return _reply(call, f"Error: the arguments of {name} are not valid: {_describe(e)}")
        if _key(name, validated) in _earlier_keys(request.state["messages"], call["id"], by_name):
            return _reply(call, f"Duplicate call: {name} was already called with these arguments earlier in this "
                                "conversation, and its result is there. Use it instead of calling again.",
                          duplicate=True)
        try:
            output = tool.invoke(args)
        except Exception as e:
            return _reply(call, f"Error: {name} failed with {type(e).__name__}: {e}", ran=True)
        try:
            content = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False)
        except (TypeError, ValueError):
            return _reply(call, f"Error: {name} returned a {type(output).__name__}, which is not JSON-serializable",
                          ran=True)
        return _reply(call, content, ok=True, ran=True)

    return guard


def _reply(call: dict, content: str, *, ok: bool = False, ran: bool = False, duplicate: bool = False) -> ToolMessage:
    return ToolMessage(content=content, name=call["name"], tool_call_id=call["id"],
                       status="success" if ok or duplicate else "error",
                       additional_kwargs={"ran": ran, "duplicate": duplicate})


def _validate(tool: BaseTool, args: dict) -> dict:
    """The arguments as the tool's `args_schema` reads them. Raises ValidationError."""
    schema = tool.args_schema
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        return schema.model_validate(args).model_dump(mode="json")
    return args


def _describe(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, d['loc'])) or 'arguments'}: {d['msg']}" for d in e.errors())


def _key(name: str, validated: dict) -> str:
    return json.dumps([name, validated], sort_keys=True, ensure_ascii=False)


def _earlier_keys(messages: list[BaseMessage], call_id: str, by_name: dict[str, BaseTool]) -> set[str]:
    """The calls made before this one in the run that could run, each as tool name + validated arguments.

    Read from the conversation, not from state kept by the guard: calls of one response run side by side, and this
    way "earlier" means the order the model made them in. A call that failed validation never ran, so it is not here.
    """
    keys: set[str] = set()
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        for c in m.tool_calls:
            if c["id"] == call_id:
                return keys
            tool = by_name.get(c["name"])
            if tool is None or MALFORMED_ARGUMENTS in c["args"]:
                continue
            try:
                keys.add(_key(c["name"], _validate(tool, c["args"])))
            except ValidationError:
                continue
    return keys


def _steps(messages: list[BaseMessage]) -> list[dict]:
    """One entry per model call that returned, each with its tool calls and what became of them.

    A tool call with neither `result` nor `error` was not run: its reply was cut off. `ran` is False for a call
    that was never handed to its tool (a mistake of the model's, or a duplicate).
    """
    replies = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
    steps = []
    for m in messages:
        if not isinstance(m, AIMessage):
            continue
        raw = raw_arguments(m)
        calls = []
        for c in m.tool_calls:
            r = replies.get(c["id"])
            calls.append({
                "id": c["id"], "name": c["name"], "arguments": raw.get(c["id"]),
                "result": r.content if r is not None and r.status != "error" else None,
                "error": r.content if r is not None and r.status == "error" else None,
                "duplicate": bool(r is not None and r.additional_kwargs.get("duplicate")),
                "ran": bool(r is not None and r.additional_kwargs.get("ran")),
            })
        steps.append({
            "record": m.response_metadata.get("record"),
            "finish_reason": _finish_reason(m),
            "text": m.content,
            "reasoning_chars": len(reasoning) if (reasoning := m.additional_kwargs.get("reasoning_content")) is not None
                               else None,  # None: the response had no reasoning_content
            "tool_calls": calls,
            "error": None,
        })
    return steps
