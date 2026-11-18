"""The minimal MCP server (task 043; OUTLINE §8, "The minimal MCP server"): one tool, `lookup_term`, over a book's Editor KB,
read-only.

Most tests talk to the server through the SDK's own client over its in-memory transport (the initialize handshake, so the
JSON-RPC framing is real). Two start the real entry point, `python -m fukidashi.mcp_server`, as a subprocess over stdio:
one with the SDK's client, and one with raw JSON-RPC lines to see what reaches stdout. Nothing here uses the network.
"""

from __future__ import annotations

import ast
import importlib.metadata
import json
import os
import sqlite3
import subprocess
import sys
import threading

import anyio
import pytest
from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

from fukidashi import config, mcp_server, store
from fukidashi.agents.kb_tools import lookup_tools
from fukidashi.datasets.importer import import_openmantra
from fukidashi.db import connect, init_db
from fukidashi.kb import KBSnapshot
from fukidashi.kb.loader import load_snapshot
from fukidashi.mcp_server import create_server, open_read_only

PACKAGE = config.ROOT / "backend" / "fukidashi"

# What the Editor KB of book_a holds, and what is next to it.
APPROVED_ENTRY = {"found": True, "source_term": "封印", "target_term": "封印术", "category": "technique",
                  "status": "approved"}
PENDING_ENTRY = {"found": True, "source_term": "結界", "target_term": "结界", "category": "", "status": "pending"}
NOT_IN_THE_KB = {"found": False, "note": "The knowledge base has no entry for this term."}
NO_EDITOR_KB = mcp_server.NO_EDITOR_KB_NOTE


def entry(conn, kb_id: int, source: str, target: str, category: str = "", status: str = "approved") -> int:
    return store.insert_entry(conn, kb_id=kb_id, source_term=source, target_term=target, category=category, status=status,
                              provenance="agent", page_index=1, event={"event": "created"})


@pytest.fixture
def seeded(conn, dataset, db_path):
    """The path of a database with three books:

    - book_a: an Editor KB with an approved entry, a pending one (no rule of the Curator makes a term entry that waits for a
      person yet, but the schema, the snapshot and the lookups allow one), a rejected one and a proposed one, which are not
      readable; and an eval KB with an entry that the Editor KB lacks;
    - book_b: an eval KB with that entry, and no Editor KB;
    - book_c: no KB at all.
    """
    import_openmantra(conn, dataset)
    for title in ("book_b", "book_c"):  # imported books have pages, and the pages are what lists a book's title
        conn.execute("INSERT INTO books (book_title) VALUES (?)", (title,))
        conn.execute("INSERT INTO pages (book_title, page_index, image_path) VALUES (?, 1, ?)",
                     (title, f"images/{title}/ja/000.jpg"))
    editor = store.editor_kb(conn, "book_a")
    entry(conn, editor, "封印", "封印术", "technique")
    entry(conn, editor, "結界", "结界", status="pending_human")
    entry(conn, editor, "王国", "王国", status="rejected")
    entry(conn, editor, "勇者", "勇者", status="proposed")
    entry(conn, editor, "NASA", "美国国家航空航天局")
    for book in ("book_a", "book_b"):
        entry(conn, store.create_kb(conn, book_title=book, purpose="eval", label="run-1"), "魔法", "魔法")
    conn.commit()
    return db_path


def run_session(db_path, scenario, mode: str = "legacy"):
    """Run `async scenario(client)` against a client of the server over `db_path`: `legacy` is the initialize handshake that
    every client of today speaks, `auto` the SDK's newer discovery."""

    async def main():
        async with Client(create_server(db_path), mode=mode) as client:
            return await scenario(client)

    return anyio.run(main)


def call(db_path, book: str, term: str, mode: str = "legacy"):
    return run_session(db_path, lambda client: client.call_tool("lookup_term", {"book_title": book, "term": term}), mode)


def answer_of(result) -> dict:
    """The structured answer of a call that was not an error, which the text content carries as JSON too."""
    assert not result.is_error, [c.text for c in result.content]
    (text,) = result.content
    assert json.loads(text.text) == result.structured_content
    return result.structured_content


def error_of(result) -> str:
    assert result.is_error
    (text,) = result.content
    return text.text


def everything(path) -> dict[str, list[tuple]]:
    """Every row of every table, timestamps included, read without taking part: equal dumps mean nothing was written."""
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")] for t in tables}
    finally:
        conn.close()


# --- what a client sees ---


def test_the_server_has_exactly_one_tool_with_two_required_string_arguments(seeded):
    async def scenario(client):
        return client.server_info, await client.list_tools()

    info, listed = run_session(seeded, scenario)

    assert (info.name, info.version) == ("fukidashi", importlib.metadata.version("fukidashi"))
    (tool,) = listed.tools
    assert tool.name == "lookup_term"
    assert tool.input_schema["required"] == ["book_title", "term"]
    assert {name: spec["type"] for name, spec in tool.input_schema["properties"].items()} == {
        "book_title": "string", "term": "string"}
    assert all(spec["description"] for spec in tool.input_schema["properties"].values())
    assert tool.annotations.read_only_hint is True and tool.annotations.open_world_hint is False


def test_the_description_says_what_the_tool_reads(seeded):
    (tool,) = run_session(seeded, lambda client: client.list_tools()).tools

    for words in ("Editor KB", "approved", "pending", "exactly", "only reads"):
        assert words in tool.description


def test_the_newer_handshake_lists_the_same_tool_and_answers_the_same(seeded):
    async def scenario(client):
        listed = await client.list_tools()
        return [t.name for t in listed.tools], await client.call_tool("lookup_term", {"book_title": "book_a", "term": "封印"})

    names, result = run_session(seeded, scenario, mode="auto")

    assert names == ["lookup_term"]
    assert answer_of(result) == APPROVED_ENTRY


def test_a_term_that_is_approved_is_found_with_its_rendering_and_category(seeded):
    assert answer_of(call(seeded, "book_a", "封印")) == APPROVED_ENTRY


def test_a_term_that_waits_for_a_person_is_found_as_pending(seeded):
    assert answer_of(call(seeded, "book_a", "結界")) == PENDING_ENTRY


@pytest.mark.parametrize("term", [
    "存在しない",  # nothing holds it
    "王国",  # rejected: not readable
    "勇者",  # proposed: not readable
    "魔法",  # only an eval KB of the same book holds it
    "封",  # exact: a part of a term is not the term
    "封印術",  # exact: a longer text is not the term
])
def test_a_term_the_editor_kb_does_not_hold_is_not_found(seeded, term):
    assert answer_of(call(seeded, "book_a", term)) == NOT_IN_THE_KB


@pytest.mark.parametrize("written", [" 封印 ", "\u3000封印\u3000", "封印\n", "ＮＡＳＡ", "NASA"])
def test_the_term_is_matched_after_nfkc_and_trimming(seeded, written):
    found = answer_of(call(seeded, "book_a", written))

    assert found["found"] is True and found["source_term"] in ("封印", "NASA")


def test_a_book_with_no_editor_kb_answers_found_false_and_no_kb_is_made(seeded):
    before, kbs = everything(seeded), everything(seeded)["kbs"]

    for book in ("book_b", "book_c"):  # book_b has an eval KB that holds 魔法: it is not the Editor KB
        for term in ("魔法", "封印"):
            assert answer_of(call(seeded, book, term)) == {"found": False, "note": NO_EDITOR_KB}

    assert "no Editor KB" in NO_EDITOR_KB
    assert everything(seeded)["kbs"] == kbs and everything(seeded) == before


def test_an_unknown_book_is_a_tool_error_that_names_the_books(seeded):
    text = error_of(call(seeded, "no_such_book", "封印"))

    assert "No book is titled 'no_such_book'" in text
    assert "book_a, book_b, book_c" in text


def test_a_database_with_no_book_says_so(tmp_path):
    empty = tmp_path / "empty.db"
    conn = connect(empty)
    init_db(conn)
    conn.close()

    assert "The database holds no book." in error_of(call(empty, "book_a", "封印"))


@pytest.mark.parametrize("blank", ["", " ", "\u3000", " \t\n", "\u3000\u3000"])
def test_an_empty_or_blank_term_is_a_tool_error_even_for_a_book_that_exists(seeded, blank):
    for book in ("book_a", "book_c"):
        assert "The term is empty" in error_of(call(seeded, book, blank))


def test_a_term_or_a_book_that_is_not_text_is_refused_by_the_schema(seeded):
    async def scenario(client):
        return [await client.call_tool("lookup_term", args) for args in (
            {"book_title": "book_a"}, {"term": "封印"}, {"book_title": 1, "term": "封印"}, {"book_title": "book_a", "term": None})]

    results = run_session(seeded, scenario)

    assert all(r.is_error for r in results)


def test_an_unknown_tool_is_refused(seeded):
    result = run_session(seeded, lambda client: client.call_tool("lookup_character", {"name": "ユキ"}))

    assert result.is_error


def test_a_database_that_cannot_be_read_is_a_tool_error_that_says_where_it_looked(tmp_path):
    missing, not_a_database, no_schema = tmp_path / "missing.db", tmp_path / "text.db", tmp_path / "bare.db"
    not_a_database.write_text("this is not a database " * 200)
    sqlite3.connect(no_schema).close()

    for path in (missing, not_a_database, no_schema, tmp_path):
        text = error_of(call(path, "book_a", "封印"))
        assert "The database cannot be read" in text and str(path) in text and "FUKIDASHI_DB" in text

    assert not missing.exists()  # looking at a database that is not there does not make one


def test_a_database_that_appears_after_the_server_started_is_the_one_the_next_call_reads(seeded, conn, tmp_path):
    later = tmp_path / "later.db"

    async def scenario(client):
        ask = lambda: client.call_tool("lookup_term", {"book_title": "book_a", "term": "封印"})  # noqa: E731
        before = await ask()
        copy = sqlite3.connect(later)
        conn.backup(copy)
        copy.execute("PRAGMA journal_mode = DELETE")  # a copy with no -wal / -shm files to read through
        copy.close()
        return before, await ask()

    before, after = run_session(later, scenario)

    assert "The database cannot be read" in error_of(before)
    assert answer_of(after) == APPROVED_ENTRY


# --- the same answer as the agents' ---


def test_the_servers_answer_is_the_agents_lookup_term_answer(seeded):
    conn = connect(seeded)
    try:
        snapshot = load_snapshot(conn, store.find_editor_kb(conn, "book_a")["id"])
    finally:
        conn.close()
    (handler,) = [t.handler for t in lookup_tools(snapshot) if t.name == "lookup_term"]
    terms = ["封印", "結界", "NASA", "ＮＡＳＡ", " 封印 ", "存在しない", "王国", "勇者", "魔法", "封", "封印術"]

    for term in terms:
        expected = handler(term)
        assert mcp_server.answer(snapshot, term) == expected, term
        assert answer_of(call(seeded, "book_a", term)) == expected, term
    assert {handler(t)["found"] for t in terms} == {True, False}  # the terms cover both kinds of answer
    assert {handler(t).get("status") for t in terms} == {"approved", "pending", None}


def test_the_agents_and_the_server_say_the_same_when_the_kb_holds_nothing():
    empty = KBSnapshot(1, "book_a", (), ())
    (handler,) = [t.handler for t in lookup_tools(empty) if t.name == "lookup_term"]

    assert mcp_server.answer(empty, "封印") == handler("封印") == NOT_IN_THE_KB


# --- read-only ---


def test_no_row_of_any_table_changes_whatever_is_asked(seeded):
    before = everything(seeded)
    asked = [("book_a", "封印"), ("book_a", "結界"), ("book_a", "魔法"), ("book_a", "存在しない"), ("book_b", "魔法"),
             ("book_c", "封印"), ("no_such_book", "封印"), ("book_a", " "), ("book_a", "王国")]

    async def scenario(client):
        return [await client.call_tool("lookup_term", {"book_title": b, "term": t}) for b, t in asked]

    results = run_session(seeded, scenario)

    assert [r.is_error for r in results] == [False, False, False, False, False, False, True, True, False]
    assert everything(seeded) == before
    assert len(before["kbs"]) == 3 and {row[1] for row in before["kbs"]} == {"book_a", "book_b"}  # none was made for book_c


def test_every_call_opens_the_database_read_only_and_closes_it_again(seeded, monkeypatch):
    opened = []
    real = sqlite3.connect

    def spy(*args, **kwargs):
        # The tool's worker thread makes the connection. This test uses it from another thread afterwards, so it is made
        # usable from there: what SQLite then says is whether it is closed, not whether it is in the wrong thread (which it
        # checks first, and says of a closed connection too).
        conn = real(*args, **kwargs, check_same_thread=False)
        opened.append((args, kwargs, conn))
        return conn

    asked = [("book_a", "封印"), ("book_a", "結界"), ("book_a", "存在しない"), ("book_c", "封印"), ("no_such_book", "封印")]

    async def scenario(client):
        monkeypatch.setattr(mcp_server.sqlite3, "connect", spy)
        try:
            return [await client.call_tool("lookup_term", {"book_title": b, "term": t}) for b, t in asked]
        finally:
            monkeypatch.undo()

    results = run_session(seeded, scenario)

    assert [r.is_error for r in results] == [False, False, False, False, True]
    assert len(opened) == len(asked)  # one connection for each call, even for the call that was refused
    for args, kwargs, conn in opened:
        assert kwargs == {"uri": True} and args[0].startswith("file:") and args[0].endswith("?mode=ro")
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            conn.execute("SELECT 1")


def test_the_connection_it_opens_cannot_write_and_never_makes_a_file(seeded, tmp_path):
    conn = open_read_only(seeded)
    try:
        for sql in ("INSERT INTO books (book_title) VALUES ('book_d')", "DELETE FROM kbs", "CREATE TABLE t (a)",
                    "UPDATE glossary_entries SET target_term = 'x'"):
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                conn.execute(sql)
    finally:
        conn.close()
    with pytest.raises(sqlite3.OperationalError):
        open_read_only(tmp_path / "nowhere" / "missing.db")
    assert not (tmp_path / "nowhere").exists()


def test_a_call_sees_the_kb_as_it_is_now_and_the_server_remembers_nothing(seeded, conn):
    editor = store.find_editor_kb(conn, "book_a")["id"]

    async def scenario(client):
        ask = lambda term: client.call_tool("lookup_term", {"book_title": "book_a", "term": term})  # noqa: E731
        first = await ask("新語")
        entry(conn, editor, "新語", "新词", "word")  # the editor writes while the server is running
        conn.commit()
        second = await ask("新語")
        conn.execute("UPDATE glossary_entries SET status = 'rejected' WHERE source_term = '新語'")
        conn.commit()
        return first, second, await ask("新語")

    first, second, third = run_session(seeded, scenario)

    assert answer_of(first) == NOT_IN_THE_KB
    assert answer_of(second) == {"found": True, "source_term": "新語", "target_term": "新词", "category": "word",
                                 "status": "approved"}
    assert answer_of(third) == NOT_IN_THE_KB


def test_a_book_that_gets_its_editor_kb_while_the_server_runs_is_answered_from_it(seeded, conn):
    async def scenario(client):
        ask = lambda: client.call_tool("lookup_term", {"book_title": "book_c", "term": "封印"})  # noqa: E731
        before = await ask()
        entry(conn, store.editor_kb(conn, "book_c"), "封印", "封印术", "technique")
        conn.commit()
        return before, await ask()

    before, after = run_session(seeded, scenario)

    assert answer_of(before) == {"found": False, "note": NO_EDITOR_KB}
    assert answer_of(after) == APPROVED_ENTRY


# --- what the module imports ---


def test_importing_the_server_loads_neither_the_agents_nor_a_model_client_nor_an_env_loader():
    code = "import json, sys; import fukidashi.mcp_server; print(json.dumps(sorted(sys.modules)))"
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120, check=True)
    loaded = json.loads(done.stdout)

    assert not {m.split(".")[0] for m in loaded} & {"langchain", "langchain_core", "langgraph", "langsmith", "openai", "dotenv"}
    for module in ("agents", "llm", "runs", "api", "main", "background", "tracing", "evaluation"):
        assert not [m for m in loaded if m == f"fukidashi.{module}" or m.startswith(f"fukidashi.{module}.")], module
    assert "mcp.server" in loaded and "fukidashi.kb.loader" in loaded  # and it did load what it uses


def imports_of(path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module or "")
    return names


def test_the_sdk_is_imported_by_the_server_module_and_nowhere_else_in_the_package():
    importers = {p.relative_to(PACKAGE).as_posix() for p in sorted(PACKAGE.rglob("*.py"))
                 if {n.split(".")[0] for n in imports_of(p)} & {"mcp", "mcp_types"}}

    assert importers == {"mcp_server.py"}
    assert not (PACKAGE / "mcp").exists() and not (PACKAGE / "mcp.py").exists()  # either would shadow the SDK's `mcp`


def test_the_server_never_names_what_would_write_a_kb_make_a_schema_or_build_a_client():
    """A check of the code beside the read-only connection, which is what really stops a write: `store.editor_kb` and
    `store.create_kb` make a `kbs` row, `KB_WRITES` write entries, profiles and proposals, `init_db` runs `schema.sql`."""
    tree = ast.parse((PACKAGE / "mcp_server.py").read_text(encoding="utf-8"))
    named = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    named |= {a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}

    assert not named & {*store.KB_WRITES, "editor_kb", "create_kb", "init_db", "load_dotenv", "make_client", "LLMClient",
                        "insert_human_run", "start_run", "finish_run"}
    assert {"find_editor_kb", "load_snapshot", "reading"} <= named
    assert "db" not in package_modules_imported(tree)  # `db.connect` is the read-write connection


def package_modules_imported(tree: ast.AST) -> set[str]:
    """The modules of the package a module imports with a relative import: `from . import x` and `from .x.y import z` give
    `x`."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            found |= {node.module.split(".")[0]} if node.module else {a.name for a in node.names}
    return found


# --- the real entry point, over stdio ---


def server_parameters(db_path) -> StdioServerParameters:
    return StdioServerParameters(command=sys.executable, args=["-m", "fukidashi.mcp_server"],
                                 env={"FUKIDASHI_DB": str(db_path)})


def test_the_entry_point_serves_the_database_that_fukidashi_db_names_over_stdio(seeded, tmp_path):
    before = everything(seeded)

    async def main():
        with open(tmp_path / "stderr.txt", "w") as errlog:
            async with Client(stdio_client(server_parameters(seeded), errlog=errlog), mode="legacy") as client:
                listed = await client.list_tools()
                found = await client.call_tool("lookup_term", {"book_title": "book_a", "term": "封印"})
                pending = await client.call_tool("lookup_term", {"book_title": "book_a", "term": "結界"})
                none = await client.call_tool("lookup_term", {"book_title": "book_c", "term": "封印"})
                refused = await client.call_tool("lookup_term", {"book_title": "no_such_book", "term": "封印"})
                return client.server_info, listed, found, pending, none, refused

    info, listed, found, pending, none, refused = anyio.run(main)

    assert info.name == "fukidashi" and [t.name for t in listed.tools] == ["lookup_term"]
    assert answer_of(found) == APPROVED_ENTRY and answer_of(pending) == PENDING_ENTRY
    assert answer_of(none) == {"found": False, "note": NO_EDITOR_KB}
    assert "No book is titled" in error_of(refused)
    assert everything(seeded) == before


def test_stdout_carries_only_the_protocol(seeded, tmp_path):
    """Every line the server writes to stdout is a JSON-RPC message, the answers included, and an error and a refusal as well
    as an answer leave it that way."""
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        *({"jsonrpc": "2.0", "id": 3 + i, "method": "tools/call", "params": {"name": "lookup_term", "arguments": args}}
          for i, args in enumerate([{"book_title": "book_a", "term": "封印"}, {"book_title": "book_a", "term": "存在しない"},
                                    {"book_title": "no_such_book", "term": "封印"}, {"book_title": "book_a", "term": " "},
                                    {"book_title": "book_a"}])),
    ]
    env = {"PATH": os.environ["PATH"], "FUKIDASHI_DB": str(seeded)}
    proc = subprocess.Popen([sys.executable, "-m", "fukidashi.mcp_server"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", env=env)
    watchdog = threading.Timer(120, proc.kill)  # a server that hangs must not hang the suite
    watchdog.start()
    lines: list[str] = []
    try:
        for request in requests:  # one at a time: the end of stdin while a call is being answered would cancel the call
            proc.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            while "id" in request:  # read up to this request's answer, whatever comes before it
                line = proc.stdout.readline()
                assert line, "the server closed stdout before it answered"
                lines.append(line)
                if json.loads(line).get("id") == request["id"]:
                    break
        rest, stderr = proc.communicate(timeout=60)  # closes stdin: the end of the session
        lines += rest.splitlines(keepends=True)
    finally:
        watchdog.cancel()
        proc.kill()

    assert proc.returncode == 0
    messages = [json.loads(line) for line in lines]  # every line is JSON, and none is blank: nothing but messages
    assert all(m["jsonrpc"] == "2.0" for m in messages)
    answers = {m["id"]: m for m in messages if "id" in m}
    assert sorted(answers) == [1, 2, 3, 4, 5, 6, 7] and all("error" not in m for m in answers.values())
    assert answers[1]["result"]["serverInfo"]["name"] == "fukidashi"
    assert [t["name"] for t in answers[2]["result"]["tools"]] == ["lookup_term"]
    assert answers[3]["result"]["structuredContent"] == APPROVED_ENTRY
    assert answers[4]["result"]["structuredContent"] == NOT_IN_THE_KB
    assert all(answers[i]["result"]["isError"] for i in (5, 6, 7))  # an unknown book, a blank term, a missing argument
    assert '"jsonrpc"' not in stderr  # and the protocol is not on stderr
