"""Tracing: one trace per translated page, in a backend chosen by `FUKIDASHI_TRACING`.

Plain OpenTelemetry. The OpenInference instrumentors turn every provider call (`openai`) and the agent graph with its
tool calls (LangChain) into spans, and `page_span` is the root they hang under. Which server receives the spans is
decided in `backends.py` alone. Off by default; `setup()` is called once by an entry point (the API, `eval/run.py`, a
script), never by library code. Tracing never fails a Run: spans leave through a batch processor, and an exporter that
fails is logged by the OpenTelemetry SDK and otherwise ignored.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from .backends import BACKENDS, Backend, TracingError
from .redact import RedactingExporter

__all__ = ["BACKENDS", "Backend", "PageSpan", "TracingError", "describe", "flush", "page_span", "setup", "shutdown"]

ENV_VAR = "FUKIDASHI_TRACING"
ATTRIBUTE_PREFIX = "fukidashi."


@dataclass
class _State:
    name: str
    backend: Backend
    provider: TracerProvider
    tracer: Tracer


_state: _State | None = None


def setup() -> bool:
    """Switch tracing on when `FUKIDASHI_TRACING` names a backend; True when it is on.

    Does nothing (and installs nothing) when the variable is unset or empty, and when tracing is already on.
    Raises `TracingError` for an unknown backend, or when the backend refuses its settings.
    """
    global _state
    name = os.environ.get(ENV_VAR, "").strip().lower()
    if not name:
        return _state is not None
    if _state is not None:
        return True
    if name not in BACKENDS:
        raise TracingError(f"{ENV_VAR}={name} is not a tracing backend. Supported: {', '.join(sorted(BACKENDS))}")
    backend = BACKENDS[name]()

    provider = TracerProvider(resource=Resource.create(backend.resource))
    provider.add_span_processor(BatchSpanProcessor(RedactingExporter(backend.exporter)))
    # Page images stay out of the structured attributes; redact.py covers the JSON strings the instrumentors build.
    trace_config = TraceConfig(hide_input_images=True)
    OpenAIInstrumentor().instrument(tracer_provider=provider, config=trace_config)
    LangChainInstrumentor().instrument(tracer_provider=provider, config=trace_config)
    _state = _State(name, backend, provider, provider.get_tracer("fukidashi"))
    return True


def shutdown() -> None:
    """Switch tracing off: send what is pending, remove the instrumentation and forget the backend."""
    global _state
    if _state is None:
        return
    state, _state = _state, None
    LangChainInstrumentor().uninstrument()
    OpenAIInstrumentor().uninstrument()
    state.provider.shutdown()


def flush(timeout_millis: int = 30000) -> bool:
    """Export the spans that are waiting. True when nothing is on, or when the exporter finished in time."""
    return True if _state is None else _state.provider.force_flush(timeout_millis)


def describe() -> dict:
    """What the eval manifest records: whether tracing is on and where it goes."""
    if _state is None:
        return {"enabled": False}
    return {"enabled": True, **_state.backend.info}


class PageSpan:
    """The root span of one page, as `page_span` hands it out."""

    def __init__(self, span: Span):
        self._span = span

    @property
    def trace_id(self) -> str:
        return format(self._span.get_span_context().trace_id, "032x")

    def set(self, **attributes) -> None:
        """Attributes of the root, under the `fukidashi.` prefix. A value of None is left out."""
        for key, value in attributes.items():
            if value is not None:
                self._span.set_attribute(ATTRIBUTE_PREFIX + key, value)

    def fail(self, description: str) -> None:
        self._span.set_status(Status(StatusCode.ERROR, description))


@contextmanager
def page_span(*, book: str, page_index: int, config: str) -> Iterator[PageSpan | None]:
    """The root span of one `translate_page`; the provider calls, agent steps and tool calls inside it nest under it.

    Yields None, and does nothing, when tracing is off. An exception that leaves the block is recorded on the span
    and raised again.
    """
    if _state is None:
        yield None
        return
    with _state.tracer.start_as_current_span(f"translate_page {book}/{page_index}") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
        page = PageSpan(span)
        page.set(book=book, page_index=page_index, config=config)
        yield page
