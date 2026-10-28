"""Tracing backends: where the spans go.

A backend is a function that reads its own settings from the environment and returns a `Backend`: an OpenTelemetry
exporter, the resource attributes its server reads, and what the manifest should record about it. Everything that is
specific to a tracing server lives in its function. A new backend (Langfuse, a shared server) is one more function
here, plus a decision on whether traces may leave this machine (IDEAS.md); the instrumentation does not change.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from openinference.semconv.resource import ResourceAttributes
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanExporter


class TracingError(RuntimeError):
    """Tracing is switched on but cannot start. The message says what to change."""


@dataclass(frozen=True)
class Backend:
    exporter: SpanExporter
    resource: dict[str, str]  # resource attributes of every span, for the server to sort traces by
    info: dict[str, str]  # recorded in the eval manifest: where the traces go


PHOENIX_DEFAULT_ENDPOINT = "http://localhost:6006/v1/traces"
PHOENIX_DEFAULT_PROJECT = "fukidashi"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def direct_session() -> requests.Session:
    """A session that reads nothing from the environment: no HTTP_PROXY / HTTPS_PROXY / ALL_PROXY / NO_PROXY, no netrc.

    `requests` trusts those by default, so with a proxy in the shell a POST to a local endpoint would go to the proxy,
    with the prompts and translations in it.
    """
    session = requests.Session()
    session.trust_env = False
    return session


def phoenix() -> Backend:
    """Arize Phoenix, run by hand on this machine (`uvx --from arize-phoenix phoenix serve`), reached over OTLP/HTTP.

    `PHOENIX_COLLECTOR_ENDPOINT` must point at this machine: a trace holds prompts, Japanese text and translations,
    and keeping them here is the default. The rule holds on the wire: the exporter is given this endpoint (the
    `OTEL_EXPORTER_OTLP_*ENDPOINT` variables do not apply) and a session that ignores proxy settings.
    """
    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "").strip() or PHOENIX_DEFAULT_ENDPOINT
    project = os.environ.get("PHOENIX_PROJECT_NAME", "").strip() or PHOENIX_DEFAULT_PROJECT
    try:
        host = urlparse(endpoint).hostname
    except ValueError:
        host = None
    if host not in LOCAL_HOSTS:
        raise TracingError(
            f"PHOENIX_COLLECTOR_ENDPOINT ({endpoint}) is not on this machine. Traces hold prompts and translations "
            f"and stay local by default: use a URL whose host is one of {', '.join(LOCAL_HOSTS)}."
        )
    return Backend(
        exporter=OTLPSpanExporter(endpoint=endpoint, session=direct_session()),
        resource={ResourceAttributes.PROJECT_NAME: project},  # the attribute Phoenix reads for the project
        info={"backend": "phoenix", "project": project, "endpoint": endpoint},
    )


BACKENDS: dict[str, Callable[[], Backend]] = {"phoenix": phoenix}
