"""Provider errors: `ProviderError` is what leaves `llm/`; `describe_api_error` words it (CLI scripts and API)."""

from __future__ import annotations

import json

import openai

from .metering import CallRecord


class ProviderError(RuntimeError):
    """A provider request failed or its response could not be read. The call is already in the Meter.

    `description` is readable and never contains the API key; `status_code` is None when it is not known.
    `record` is the call as the Meter has it: unknown usage unless the response carried some.
    """

    def __init__(self, description: str, status_code: int | None = None, record: CallRecord | None = None):
        super().__init__(description)
        self.description = description
        self.status_code = status_code
        self.record = record


def describe_api_error(e: openai.OpenAIError | json.JSONDecodeError, base_url: str | None = None) -> str:
    if isinstance(e, openai.AuthenticationError):
        return f"Authentication failed ({e.status_code}). Check LLM_API_KEY in .env."
    if isinstance(e, openai.NotFoundError):
        return f"Not found ({e.status_code}). Check LLM_BASE_URL and LLM_MODEL in .env. {e.message}"
    if isinstance(e, openai.BadRequestError):
        return (
            f"Bad request: {e.message}\n"
            "If it mentions response_format, set LLM_JSON_MODE=none. "
            "If it mentions max_completion_tokens, set LLM_TOKEN_PARAM=max_completion_tokens. "
            "If it mentions images, this model has no vision support."
        )
    if isinstance(e, openai.RateLimitError):
        return f"Rate limited: {e.message}"
    if isinstance(e, openai.APIStatusError):
        return f"API error {e.status_code}: {e.message}"
    if isinstance(e, openai.APIConnectionError):
        return f"Cannot reach {base_url or 'the provider'}: {e}"
    if isinstance(e, openai.APIResponseValidationError):
        return (
            f"The provider's response could not be read (HTTP {e.status_code}): {e.message} "
            "Check that LLM_BASE_URL in .env points at an OpenAI-compatible endpoint."
        )
    if isinstance(e, json.JSONDecodeError):
        return f"The provider's response was not valid JSON ({e}). Check LLM_BASE_URL in .env."
    return f"LLM call failed: {e}"
