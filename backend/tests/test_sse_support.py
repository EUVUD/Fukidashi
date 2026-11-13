"""`Stream.close()` once the response is over (task 044).

When the app finishes a response, `asyncio.run` closes the event loop of the helper's thread. The test thread, which
has just read the end of the response, can reach `close()` before that or after it. After it, `call_soon_threadsafe`
on the closed loop raised `RuntimeError: Event loop is closed`, and the test failed from the end of its `with` block
although every assertion in it had passed. These tests drive apps of their own and use no database, so they show the
helper and nothing else.
"""

from __future__ import annotations

import pytest
from sse_support import Stream


async def one_event_then_done(scope, receive, send):
    """A response that ends by itself: the app returns, and the helper's `asyncio.run` closes its loop."""
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
    await send({"type": "http.response.body", "body": b"id: 1\nevent: run_end\ndata: {}\n\n", "more_body": False})


def finished_stream() -> Stream:
    """A stream read to its end whose thread has finished: `asyncio.run` has returned, so its loop is closed. This is the
    state a slow test thread finds when it reaches `close()`, made certain instead of waited for."""
    stream = Stream(one_event_then_done, "/events")
    assert [e["event"] for e in stream.read_all(timeout=10)] == ["run_end"]
    stream._thread.join(10)
    assert not stream._thread.is_alive() and stream._loop.is_closed()
    return stream


class LoopClosingDuringTheCall:
    """A loop that is open when `close` looks at it and closed when `close` calls it, as when the helper's thread finishes
    in between. `call_soon_threadsafe` refuses the way asyncio's does."""

    closed = False

    def is_closed(self) -> bool:
        return self.closed

    def call_soon_threadsafe(self, callback, *args) -> None:
        self.closed = True
        raise RuntimeError("Event loop is closed")


class LoopRefusingForAnotherReason:
    def is_closed(self) -> bool:
        return False

    def call_soon_threadsafe(self, callback, *args) -> None:
        raise RuntimeError("something else is wrong")


def test_closing_a_stream_whose_loop_has_closed_does_nothing_more_than_join_the_thread():
    stream = finished_stream()

    stream.close()  # was: RuntimeError: Event loop is closed


def test_a_loop_that_closes_between_close_looking_at_it_and_calling_it_is_not_an_error():
    stream = finished_stream()
    stream._loop = LoopClosingDuringTheCall()

    stream.close()


def test_any_other_runtime_error_from_the_call_still_propagates():
    stream = finished_stream()
    stream._loop = LoopRefusingForAnotherReason()

    with pytest.raises(RuntimeError, match="something else is wrong"):
        stream.close()


def test_closing_a_stream_that_is_still_open_delivers_the_disconnect():
    seen: list[str] = []

    async def idle(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
        await receive()  # the request
        seen.append((await receive())["type"])  # nothing more comes until the client goes

    stream = Stream(idle, "/events")
    assert stream._thread.is_alive() and not stream._loop.is_closed()  # idle: the response is not over

    stream.close()

    assert seen == ["http.disconnect"] and not stream._thread.is_alive()
