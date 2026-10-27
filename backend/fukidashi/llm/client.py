"""Minimal OpenAI-compatible chat client with metering and a per-page token budget."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import openai
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from .errors import ProviderError, describe_api_error
from .metering import CallRecord, Meter, PageBudget, Usage

if TYPE_CHECKING:
    import httpx


class LLMNotConfigured(RuntimeError):
    """The LLM_* variables are missing. The message names them, never their values."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    json_mode: str = "json_object"  # or "none"
    token_param: str = "max_tokens"  # or "max_completion_tokens"
    price_input_per_m: float | None = None
    price_output_per_m: float | None = None

    @classmethod
    def from_env(cls) -> LLMConfig:
        missing = [k for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL") if not os.environ.get(k)]
        if missing:
            raise LLMNotConfigured(f"Set {', '.join(missing)} in .env (see .env.example)")

        def price(key: str) -> float | None:
            value = os.environ.get(key, "").strip()
            return float(value) if value else None

        return cls(
            base_url=os.environ["LLM_BASE_URL"],
            api_key=os.environ["LLM_API_KEY"],
            model=os.environ["LLM_MODEL"],
            json_mode=os.environ.get("LLM_JSON_MODE", "json_object"),
            token_param=os.environ.get("LLM_TOKEN_PARAM", "max_tokens"),
            price_input_per_m=price("LLM_PRICE_INPUT_PER_M"),
            price_output_per_m=price("LLM_PRICE_OUTPUT_PER_M"),
        )


@dataclass(frozen=True)
class LLMResult:
    text: str
    finish_reason: str | None
    record: CallRecord


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
    ) -> LLMResult:
        budget.check()
        kwargs: dict = {
            "model": self.config.model,
            "messages": messages,
            self.config.token_param: max_tokens,
        }
        if json_output and self.config.json_mode == "json_object":
            kwargs["response_format"] = {"type": "json_object"}

        start = time.monotonic()
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # The request may have reached the provider and been billed (a timeout, a 5xx): unknown, not zero.
            # A 200 that claims to be JSON and is not makes the SDK raise JSONDecodeError: also the provider's doing.
            is_provider_error = isinstance(e, (openai.OpenAIError, json.JSONDecodeError))
            description = (describe_api_error(e, self.config.base_url) if is_provider_error
                           else f"{type(e).__name__}: {e}")
            description = self._redact(description)
            rec = self._record(budget, start, error=description)
            if not is_provider_error:
                raise
            raise ProviderError(description, getattr(e, "status_code", None), rec) from e

        if not isinstance(resp, ChatCompletion):
            # The SDK hands back the body as a str when a 200 is not JSON (say an HTML page from a gateway).
            description = self._redact(
                f"The response from {self.config.base_url} is not a chat completion (got {type(resp).__name__}, "
                "not JSON). Check LLM_BASE_URL in .env."
            )
            raise ProviderError(description, None, self._record(budget, start, error=description))

        # From here on the call is recorded once, with the usage the response carried, error or not.
        text, finish_reason, problem = "", None, None
        if not resp.choices:
            problem = f"{self.config.model} returned no choices"
        else:
            try:
                text, finish_reason = _first_choice(resp)
            except (AttributeError, IndexError, TypeError):
                problem = f"{self.config.model} returned a first choice that cannot be read"
        rec = self._record(budget, start, response_model=resp.model, usage=usage_from_api(resp.usage), error=problem)
        if problem:
            raise ProviderError(self._redact(problem), None, rec)
        return LLMResult(text=text, finish_reason=finish_reason, record=rec)

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


def _first_choice(resp: ChatCompletion) -> tuple[str, str | None]:
    """(text, finish_reason) of the first choice. Raises AttributeError, IndexError or TypeError if it is malformed."""
    choice = resp.choices[0]
    content = choice.message.content
    if content is not None and not isinstance(content, str):
        raise TypeError(f"message content is {type(content).__name__}, not text")
    return content or "", choice.finish_reason


def usage_from_api(u: CompletionUsage | None) -> Usage | None:
    # The SDK builds a response without validating it, so `usage` can be anything the provider sent.
    if not isinstance(u, CompletionUsage) or u.prompt_tokens is None or u.completion_tokens is None:
        return None
    cached = u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else None
    return Usage(input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens, cached_input_tokens=cached)
