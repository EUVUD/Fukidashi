"""The agent's model: a LangChain chat model that calls `LLMClient`, and the message conversions around it.

LangChain's provider models (`ChatOpenAI`, `ChatDeepSeek`) would bypass `LLMClient`: their calls would not reach the
`Meter`, they retry on their own, and they report missing usage their own way. So every model call of the agent loop
goes through `LLMClient.chat`, exactly once, and this adapter only converts messages in both directions.
DeepSeek rejects a tool-call turn whose `reasoning_content` is not sent back, so it travels in `additional_kwargs`
and comes back out unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ConfigDict

from ..llm.client import LLMResult

# A tool call whose arguments are not a JSON object cannot be a LangChain tool call (`args` is a dict). It is
# carried as {MALFORMED_ARGUMENTS: <the raw string>}, and the tool guard (loop.py) turns it into a tool error.
MALFORMED_ARGUMENTS = "__malformed_arguments__"


class LLMClientChatModel(BaseChatModel):
    """Calls `client.chat` once per model call. `ProviderError` and `BudgetExceeded` leave it unchanged; nothing retries."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    client: Any  # LLMClient
    budget: Any  # PageBudget
    max_tokens: int
    json_output: bool = False

    @property
    def _llm_type(self) -> str:
        return "fukidashi-llm-client"

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any):
        if tool_choice is not None:
            raise ValueError("LLMClientChatModel does not support tool_choice")
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        if stop:
            raise ValueError("LLMClientChatModel does not support stop sequences")
        reply = self.client.chat(
            to_openai_messages(messages),
            budget=self.budget,
            max_tokens=self.max_tokens,
            json_output=self.json_output,
            tools=kwargs.get("tools"),
        )
        return ChatResult(generations=[ChatGeneration(message=_to_ai_message(reply))])


def _to_ai_message(reply: LLMResult) -> AIMessage:
    raw = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
           for c in reply.tool_calls]
    additional: dict[str, Any] = {}
    if raw:
        additional["tool_calls"] = raw  # as received: this is what goes back to the provider
    if reply.reasoning_content is not None:
        additional["reasoning_content"] = reply.reasoning_content
    return AIMessage(
        content=reply.text,
        tool_calls=[{"id": c.id, "name": c.name, "args": _parse_arguments(c.arguments), "type": "tool_call"}
                    for c in reply.tool_calls],
        additional_kwargs=additional,
        response_metadata={"finish_reason": reply.finish_reason, "model_name": reply.record.response_model,
                           "record": asdict(reply.record)},
    )


def _parse_arguments(raw: str) -> dict:
    try:
        args = json.loads(raw)
    except ValueError:
        args = None
    return args if isinstance(args, dict) else {MALFORMED_ARGUMENTS: raw}


def raw_arguments(message: AIMessage) -> dict[str, str]:
    """tool call id -> the arguments string the model sent."""
    return {c["id"]: c["function"]["arguments"] for c in message.additional_kwargs.get("tool_calls", [])}


def to_openai_messages(messages: Sequence[BaseMessage]) -> list[dict]:
    return [to_openai_message(m) for m in messages]


def to_openai_message(m: BaseMessage) -> dict:
    if isinstance(m, SystemMessage):
        return {"role": "system", "content": m.content}
    if isinstance(m, HumanMessage):
        return {"role": "user", "content": m.content}
    if isinstance(m, ToolMessage):
        content = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
        return {"role": "tool", "tool_call_id": m.tool_call_id, "content": content}
    if isinstance(m, AIMessage):
        out: dict[str, Any] = {"role": "assistant", "content": m.content}
        if calls := m.additional_kwargs.get("tool_calls"):
            out["tool_calls"] = calls
        elif m.tool_calls:  # an AIMessage not made by this adapter
            out["tool_calls"] = [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["args"], ensure_ascii=False)}}
                for c in m.tool_calls
            ]
        if (reasoning := m.additional_kwargs.get("reasoning_content")) is not None:
            out["reasoning_content"] = reasoning
        return out
    raise TypeError(f"cannot send a {type(m).__name__} to the provider")
