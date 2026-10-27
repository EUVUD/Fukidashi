"""Minimal OpenAI-compatible chat client with metering and a per-page token budget."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import openai
from openai.types import CompletionUsage

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
            is_provider_error = isinstance(e, openai.OpenAIError)
            description = self._redact(describe_api_error(e, self.config.base_url) if is_provider_error
                                       else f"{type(e).__name__}: {e}")
            rec = CallRecord(
                page_id=budget.page_id,
                requested_model=self.config.model,
                response_model=None,
                usage=None,
                latency_s=round(time.monotonic() - start, 3),
                cost_usd=None,
                error=description,
            )
            self.meter.record(rec)
            budget.charge(None)
            if not is_provider_error:
                raise
            raise ProviderError(description, getattr(e, "status_code", None)) from e
        latency = time.monotonic() - start

        usage = usage_from_api(resp.usage)
        rec = CallRecord(
            page_id=budget.page_id,
            requested_model=self.config.model,
            response_model=resp.model,
            usage=usage,
            latency_s=round(latency, 3),
            cost_usd=self._cost(usage),
        )
        self.meter.record(rec)
        budget.charge(usage)

        if not resp.choices:
            raise ProviderError(f"{self.config.model} returned no choices")  # already recorded, with its usage
        choice = resp.choices[0]
        return LLMResult(text=choice.message.content or "", finish_reason=choice.finish_reason, record=rec)

    def _redact(self, text: str) -> str:
        # A provider may echo the key back in an error body; neither the ledger nor a Run may keep it.
        key = self.config.api_key
        return text.replace(key, "***") if key else text

    def _cost(self, usage: Usage | None) -> float | None:
        c = self.config
        if usage is None or c.price_input_per_m is None or c.price_output_per_m is None:
            return None
        return (usage.input_tokens * c.price_input_per_m + usage.output_tokens * c.price_output_per_m) / 1_000_000


def usage_from_api(u: CompletionUsage | None) -> Usage | None:
    if u is None or u.prompt_tokens is None or u.completion_tokens is None:
        return None
    cached = u.prompt_tokens_details.cached_tokens if u.prompt_tokens_details else None
    return Usage(input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens, cached_input_tokens=cached)
