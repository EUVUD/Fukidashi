"""Minimal OpenAI-compatible chat client with metering and a per-page token budget."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import openai
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from .errors import ProviderError, describe_api_error
from .metering import CallRecord, Meter, PageBudget, Usage

if TYPE_CHECKING:
    import httpx


class LLMNotConfigured(RuntimeError):
    """The <PREFIX>_* variables are missing. The message names them, never their values."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    json_mode: str = "json_object"  # or "none"
    token_param: str = "max_tokens"  # or "max_completion_tokens"
    price_input_per_m: float | None = None
    price_output_per_m: float | None = None
    temperature: float | None = None  # sent only when set, so a request without it is what it was before
    reasoning_effort: str | None = None  # for reasoning models; sent only when set
    env_prefix: str = "LLM"  # the variables this configuration is read from; messages that point at them use it

    @classmethod
    def from_env(cls, prefix: str = "LLM") -> LLMConfig:
        """Read `<prefix>_BASE_URL`, `_API_KEY`, `_MODEL` (required) and the optional ones. The translation system is
        `LLM`, the GEMBA-MQM judge `JUDGE` and its fallback `JUDGE2` (OUTLINE 8)."""

        def var(name: str) -> str:
            return f"{prefix}_{name}"

        missing = [k for k in (var("BASE_URL"), var("API_KEY"), var("MODEL")) if not os.environ.get(k)]
        if missing:
            raise LLMNotConfigured(f"Set {', '.join(missing)} in .env (see .env.example)")

        def optional(key: str) -> str | None:
            value = os.environ.get(key, "").strip()
            return value or None

        def number(key: str) -> float | None:
            value = optional(key)
            return float(value) if value is not None else None

        return cls(
            base_url=os.environ[var("BASE_URL")],
            api_key=os.environ[var("API_KEY")],
            model=os.environ[var("MODEL")],
            json_mode=os.environ.get(var("JSON_MODE"), "json_object"),
            token_param=os.environ.get(var("TOKEN_PARAM"), "max_tokens"),
            price_input_per_m=number(var("PRICE_INPUT_PER_M")),
            price_output_per_m=number(var("PRICE_OUTPUT_PER_M")),
            temperature=number(var("TEMPERATURE")),
            reasoning_effort=optional(var("REASONING_EFFORT")),
            env_prefix=prefix,
        )


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # the raw string the model sent: not parsed, and not guaranteed to be JSON


@dataclass(frozen=True)
class LLMResult:
    text: str
    finish_reason: str | None
    record: CallRecord
    tool_calls: list[ToolCall] = field(default_factory=list)  # in the order the model made them
    reasoning_content: str | None = None  # DeepSeek's thinking; None when the response had none


class LLMClient:
    def __init__(self, config: LLMConfig, meter: Meter, *, http_client: httpx.Client | None = None):
        self.config = config
        self.meter = meter
        # max_retries=0: the SDK would resend a failed request up to twice, and the earlier attempts, which may
        # have been billed, would never reach the Meter. Every request sent is one call the Meter sees.
        self._client = openai.OpenAI(
            base_url=config.base_url, api_key=config.api_key, max_retries=0, http_client=http_client
        )

    def chat(
        self,
        messages: list[dict],
        *,
        budget: PageBudget,
        max_tokens: int = 4096,
        json_output: bool = False,
        tools: list[dict] | None = None,
    ) -> LLMResult:
        budget.check()
        kwargs: dict = {
            "model": self.config.model,
            "messages": messages,
            self.config.token_param: max_tokens,
        }
        if json_output and self.config.json_mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}
        if tools:  # OpenAI function format; an empty list sends nothing, because providers reject `"tools": []`
            kwargs["tools"] = tools
        # Each only when set: the translation requests set neither and stay what they were (test_kb_seams.py).
        if self.config.temperature is not None:
            kwargs["temperature"] = self.config.temperature
        if self.config.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.config.reasoning_effort

        start = time.monotonic()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # The request may have reached the provider and been billed (a timeout, a 5xx): unknown, not zero.
            # A 200 that claims to be JSON and is not makes the SDK raise JSONDecodeError: also the provider's doing.
            if isinstance(e, (openai.OpenAIError, json.JSONDecodeError)):
                raise self._fail(budget, start, describe_api_error(e, self.config.base_url, self.config.env_prefix),
                                 getattr(e, "status_code", None)) from e
            self._record(budget, start, error=self._redact(f"{type(e).__name__}: {e}"))
            raise

        if not isinstance(resp, ChatCompletion):
            # The SDK hands back the body as a str when a 200 is not JSON (say an HTML page from a gateway).
            raise self._fail(
                budget, start,
                f"The response from {self.config.base_url} is not a chat completion (got {type(resp).__name__}, "
                f"not JSON). Check {self.config.env_prefix}_BASE_URL in .env.",
            )

        # From here on the response is read in this one place. The SDK builds it without validating it, so any
        # shape can arrive: whatever is wrong with it, the call is recorded once, with the usage the response
        # carried, and leaves as a ProviderError.
        model = getattr(resp, "model", None)
        model = model if isinstance(model, str) else None
        usage = usage_from_api(getattr(resp, "usage", None))
        try:
            text, finish_reason, reasoning_content, tool_calls = _read_reply(resp)
        except _Unreadable as e:
            raise self._fail(budget, start, f"{self.config.model} returned {e}",
                             response_model=model, usage=usage) from e
        rec = self._record(budget, start, response_model=model, usage=usage)
        return LLMResult(text=text, finish_reason=finish_reason, record=rec, tool_calls=tool_calls,
                         reasoning_content=reasoning_content)

    def _fail(
        self,
        budget: PageBudget,
        start: float,
        description: str,
        status_code: int | None = None,
        *,
        response_model: str | None = None,
        usage: Usage | None = None,
    ) -> ProviderError:
        """Record the failed call and build its ProviderError. The call's `error` and the exception's `description`
        are the same redacted string; the caller raises the result."""
        description = self._redact(description)
        rec = self._record(budget, start, response_model=response_model, usage=usage, error=description)
        return ProviderError(description, status_code, rec)

    def _record(
        self,
        budget: PageBudget,
        start: float,
        *,
        response_model: str | None = None,
        usage: Usage | None = None,
        error: str | None = None,
    ) -> CallRecord:
        """The one place a call enters the Meter and the budget."""
        rec = CallRecord(
            page_id=budget.page_id,
            requested_model=self.config.model,
            response_model=response_model,
            usage=usage,
            latency_s=round(time.monotonic() - start, 3),
            cost_usd=self._cost(usage),
            error=error,
        )
        self.meter.record(rec)
        budget.charge(usage)
        return rec

    def _redact(self, text: str) -> str:
        # A provider may echo the key back in an error body; neither the ledger nor a Run may keep it.
        key = self.config.api_key
        return text.replace(key, "***") if key else text

    def _cost(self, usage: Usage | None) -> float | None:
        c = self.config
        if usage is None or c.price_input_per_m is None or c.price_output_per_m is None:
            return None
        return (usage.input_tokens * c.price_input_per_m + usage.output_tokens * c.price_output_per_m) / 1_000_000


class _Unreadable(Exception):
    """A response that cannot be read. The message finishes the sentence "<model> returned ..."."""


def _read_reply(resp: ChatCompletion) -> tuple[str, str | None, str | None, list[ToolCall]]:
    """(text, finish_reason, reasoning_content, tool_calls) of the first choice. Raises only _Unreadable."""
    try:
        if not resp.choices:
            raise _Unreadable("no choices")
        choice = resp.choices[0]
        message = choice.message
        content = message.content
        if content is not None and not isinstance(content, str):
            raise TypeError(f"message content is {type(content).__name__}, not text")
        finish_reason = choice.finish_reason
        reasoning = getattr(message, "reasoning_content", None)  # not in the SDK's types, kept as an extra field
        raw_calls = getattr(message, "tool_calls", None)
    except _Unreadable:
        raise
    except Exception as e:
        raise _Unreadable("a first choice that cannot be read") from e
    try:
        tool_calls = [_read_tool_call(c) for c in raw_calls or []]
    except Exception as e:
        raise _Unreadable("a tool call that cannot be read") from e
    return (
        content or "",
        finish_reason if isinstance(finish_reason, str) else None,
        reasoning if isinstance(reasoning, str) else None,
        tool_calls,
    )


def _read_tool_call(c) -> ToolCall:
    call_id, name, arguments = c.id, c.function.name, c.function.arguments
    if not (isinstance(call_id, str) and call_id and isinstance(name, str) and name and isinstance(arguments, str)):
        raise TypeError("tool call without an id, a function name or string arguments")
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _is_count(n) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and n >= 0


def usage_from_api(u: CompletionUsage | None) -> Usage | None:
    """Token counts the provider reported, or None when it sent none or none that read as integers (unknown, not 0).

    The SDK builds a response without validating it, so `usage` and its fields can be anything the provider sent.
    """
    if not isinstance(u, CompletionUsage):
        return None
    prompt, completion = getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)
    if not (_is_count(prompt) and _is_count(completion)):
        return None
    cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None)
    return Usage(input_tokens=prompt, output_tokens=completion,
                 cached_input_tokens=cached if _is_count(cached) else None)
