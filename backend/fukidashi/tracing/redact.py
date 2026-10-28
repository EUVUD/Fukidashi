"""Keeps page images and the provider key out of exported spans.

OpenInference's `hide_input_images` drops the structured image attributes, but the LangChain instrumentor also writes
the whole conversation into `input.value` / `output.value` as one JSON string, and that string holds the page image
as a data URL. So every exporter is wrapped in `RedactingExporter`, which rewrites string attributes (and the attributes
of span events, where exceptions go) before the spans leave the process. A provider may also echo the API key back in an
error body, which the OpenAI instrumentor records on the span.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Sequence

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

IMAGE_REPLACEMENT = "[image omitted]"
KEY_REPLACEMENT = "***"
_IMAGE = re.compile(r"data:image/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=_-]*")


def provider_keys() -> list[str]:
    key = os.environ.get("LLM_API_KEY", "")
    return [key] if key else []


class RedactingExporter(SpanExporter):
    def __init__(self, inner: SpanExporter, secrets: Callable[[], Iterable[str]] = provider_keys):
        self._inner = inner
        self._secrets = secrets

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        secrets = [s for s in self._secrets() if s]
        return self._inner.export([_redacted(span, secrets) for span in spans])

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)


def redact(value, secrets: Sequence[str] = ()):
    """`value` with data-URL images and `secrets` replaced, if it is a string or a sequence of strings."""
    if isinstance(value, str):
        value = _IMAGE.sub(IMAGE_REPLACEMENT, value)
        for secret in secrets:
            value = value.replace(secret, KEY_REPLACEMENT)
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(redact(v, secrets) for v in value)
    return value


def _redacted(span: ReadableSpan, secrets: Sequence[str]) -> ReadableSpan:
    attributes = {k: redact(v, secrets) for k, v in (span.attributes or {}).items()}
    events = [Event(e.name, {k: redact(v, secrets) for k, v in (e.attributes or {}).items()}, e.timestamp)
              for e in span.events]
    return ReadableSpan(
        name=span.name, context=span.context, parent=span.parent, resource=span.resource,
        attributes=attributes, events=events, links=span.links, kind=span.kind, status=span.status,
        start_time=span.start_time, end_time=span.end_time, instrumentation_scope=span.instrumentation_scope,
    )
