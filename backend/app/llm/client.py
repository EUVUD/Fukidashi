"""Minimal OpenAI-compatible chat client with metering and a per-page token budget."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import openai
from openai.types import CompletionUsage

from .metering import CallRecord, Meter, PageBudget, Usage


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
            raise RuntimeError(f"Set {', '.join(missing)} in .env (see .env.example)")

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
    def __init__(self, config: LLMConfig, meter: Meter):
        self.config = config
        self.meter = meter
        self._client = openai.OpenAI(base_url=config.base_url, api_key=config.api_key)

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
        resp = self._client.chat.completions.create(**kwargs)
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
            raise RuntimeError(f"{self.config.model} returned no choices")
        choice = resp.choices[0]
        return LLMResult(text=choice.message.content or "", finish_reason=choice.finish_reason, record=rec)

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
