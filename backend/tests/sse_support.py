"""Reading the API's server-sent events as they arrive (task 036).

Starlette's `TestClient` hands a response back only once the app has finished it, so it cannot show the first events of a
stream whose Run is still running. `Stream` drives the ASGI app itself, in a thread of its own with an event loop of its own,
and gives the test each chunk as the app sends it; closing it is the client going away. (It announces ASGI spec 2.3, as
uvicorn does: Starlette then watches `receive` for the disconnect, so an idle stream ends when the client goes.) `parse_sse` reads the wire format
for both that and a buffered `TestClient` response.
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import time

_END = {"type": "end"}


def parse_sse(raw: bytes | str) -> list[dict]:
    """The events in `raw`, each as {"id", "event", "data"} (data parsed as JSON; a field a frame lacks is None). Comment
    frames (`: ping`) are not events."""
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    events = []
    for frame in text.split("\n\n"):
        fields: dict[str, str] = {}
        for line in frame.split("\n"):
            if line and not line.startswith(":"):
                name, _, value = line.partition(": ")
                fields[name] = value
        if fields:
            events.append({"id": fields.get("id"), "event": fields.get("event"),
                           "data": json.loads(fields["data"]) if "data" in fields else None})
    return events


class Stream:
    """A GET on `app` read chunk by chunk: `status` and `headers` are the response's start; `next_event` waits for the next
    event and gives None when the response is over; `read_all` reads to the end; `close` is the client disconnecting."""

    def __init__(self, app, path: str, *, last_event_id: int | str | None = None):
        headers = [(b"host", b"testserver")]
        if last_event_id is not None:
            headers.append((b"last-event-id", str(last_event_id).encode()))
        self._scope = {
            "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1", "method": "GET",
            "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
            "headers": headers, "client": ("testclient", 50000), "server": ("testserver", 80),
        }
        self._app = app
        self._messages: queue.Queue = queue.Queue()
        self._closed = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._disconnect: asyncio.Event | None = None
        self._buffer = b""
        self._over = False
        self.comments: list[str] = []  # the comment lines seen so far (`: ping`)
        self._thread = threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True)
        self._thread.start()
        start = self._message(10)
        assert start["type"] == "http.response.start", start
        self.status: int = start["status"]
        self.headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._disconnect = asyncio.Event()
        sent_request = False

        async def receive():
            nonlocal sent_request
            if not sent_request:
                sent_request = True
                return {"type": "http.request", "body": b"", "more_body": False}
            await self._disconnect.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if self._closed.is_set():
                raise OSError("the client went away")
            self._messages.put(message)

        try:
            await self._app(self._scope, receive, send)
        except BaseException as e:  # the test sees it when it reads
            self._messages.put({"type": "error", "error": e})
        finally:
            self._messages.put(_END)

    def _message(self, timeout: float) -> dict:
        try:
            message = self._messages.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"nothing arrived in {timeout} s") from None
        if message["type"] == "error":
            raise message["error"]
        return message

    def next_event(self, timeout: float = 10) -> dict | None:
        """The next event, or None once the response is over. Waits at most `timeout` s in all (a `TimeoutError` after that),
        comment frames included: an idle stream that keeps pinging still times out."""
        deadline = time.monotonic() + timeout
        while True:
            head, sep, rest = self._buffer.partition(b"\n\n")
            if sep:
                self._buffer = rest
                if (events := parse_sse(head + sep)):
                    return events[0]
                self.comments.extend(line for line in head.decode("utf-8").split("\n") if line.startswith(":"))
                continue  # a comment frame
            if self._over:
                return None
            message = self._message(max(deadline - time.monotonic(), 0.001))
            if message is _END or message["type"] == "end":
                self._over = True
            elif message["type"] == "http.response.body":
                self._buffer += message.get("body", b"")

    def read_all(self, timeout: float = 10) -> list[dict]:
        events = []
        while (event := self.next_event(timeout)) is not None:
            events.append(event)
        return events

    def close(self) -> None:
        """The client going away: the app is told by a disconnect, then the thread is joined. A response that is already over
        has nobody to tell: `asyncio.run` closes the loop when the app returns, and a test thread that reads the end and gets
        here a moment later finds it closed, so then `close` only joins."""
        self._closed.set()
        loop, disconnect = self._loop, self._disconnect
        if loop is not None and disconnect is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(disconnect.set)
            except RuntimeError:
                if not loop.is_closed():  # it closed between the look and the call; any other RuntimeError is a real one
                    raise
        self._thread.join(10)

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
