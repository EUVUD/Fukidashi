"""Helpers for the live trace tests (task 035): a sink that holds the contract on every event it gets, and the checks the
tests share. Every sink in the live trace tests is a `Recorder` or is wrapped by `checked`, so every event they see is
held to the contract once, here."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable

from fukidashi.live_trace import KINDS, RunEvent

# What the contract's checks found wrong, as text. A check cannot raise: `translate_page` never lets a sink fail a Run, so
# an exception here would be swallowed and logged. The live trace tests read this list after every test (an autouse
# fixture) and expect it empty.
VIOLATIONS: list[str] = []


def check_event(kind: str, data: object, run_id: int | None = None, seq: int | None = None) -> None:
    """Hold one event to the contract: a kind from `KINDS`, a positive Run id and a `seq` from 1 up (when it has them),
    and `data` that is a dict `json.dumps(..., allow_nan=False)` takes. Records what is wrong in `VIOLATIONS`."""
    try:
        assert kind in KINDS, f"{kind!r} is not a kind of the contract"
        assert run_id is None or (isinstance(run_id, int) and run_id > 0), f"run_id {run_id!r} is not positive"
        assert seq is None or (isinstance(seq, int) and seq >= 1), f"seq {seq!r} is not 1 or more"
        assert isinstance(data, dict), f"data is a {type(data).__name__}, not a dict"
        json.dumps(data, allow_nan=False)
    except (AssertionError, TypeError, ValueError) as e:
        VIOLATIONS.append(f"{kind}: {e}")


def checked(sink: Callable[[RunEvent], None]) -> Callable[[RunEvent], None]:
    """`sink`, with every event it gets held to the contract first."""
    def hold(event: RunEvent) -> None:
        check_event(event.kind, event.data, event.run_id, event.seq)
        sink(event)

    return hold


class Recorder:
    """A sink for `translate_page(on_event=...)` (call it with a `RunEvent`) and an `Emit` for the agents (`.emit`).
    Each event is held to the contract as it arrives; the thread of each call is kept for the tests that care."""

    def __init__(self):
        self.events: list[RunEvent] = []
        self.emitted: list[tuple[str, dict]] = []
        self.threads: list[int] = []

    def __call__(self, event: RunEvent) -> None:
        check_event(event.kind, event.data, event.run_id, event.seq)
        self.threads.append(threading.get_ident())
        self.events.append(event)

    def emit(self, kind: str, data: dict) -> None:
        check_event(kind, data)
        self.threads.append(threading.get_ident())
        self.emitted.append((kind, data))

    @property
    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]

    @property
    def trail(self) -> list[str]:
        """The kinds in order, a stage's kind after the colon: `stage_start:review`."""
        return [f"{e.kind}:{e.data['stage']}" if e.kind in ("stage_start", "stage_end") else e.kind
                for e in self.events]

    def of(self, kind: str) -> list[RunEvent]:
        return [e for e in self.events if e.kind == kind]


def assert_well_formed(events: list[RunEvent], run_id: int | None = None) -> None:
    """One Run's events: its id on every one, `seq` 1, 2, 3 without a gap, `run_start` first and `run_end` last."""
    assert events, "the Run emitted nothing"
    assert len({e.run_id for e in events}) == 1 and (run_id is None or events[0].run_id == run_id)
    assert [e.seq for e in events] == list(range(1, len(events) + 1))
    assert events[0].kind == "run_start" and events[-1].kind == "run_end"
    assert [e.kind for e in events].count("run_start") == [e.kind for e in events].count("run_end") == 1


def without_latency(value):
    """`value` with every `latency_s` removed, at any depth: the one number that differs from one run to the next."""
    if isinstance(value, dict):
        return {k: without_latency(v) for k, v in value.items() if k != "latency_s"}
    if isinstance(value, list):
        return [without_latency(v) for v in value]
    return value
