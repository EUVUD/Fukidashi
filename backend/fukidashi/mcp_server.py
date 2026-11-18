"""A minimal MCP server over a book's Editor KB: one tool, `lookup_term`, read-only (OUTLINE §2, §8, "The minimal MCP
server"; `SCOPE.md` §1).

  uv run python -m fukidashi.mcp_server

It speaks MCP over stdio, so an MCP client starts it as a subprocess, and it reads the database at `config.DB_PATH`
(`FUKIDASHI_DB` overrides it). It writes nothing:

- Each call opens the database with SQLite's `mode=ro` URI, reads the book's Editor KB in one read-only view
  (`store.reading`) and closes the connection. So a call sees the KB as it is now, and the editor may write it between two
  calls. (Reading a WAL database makes SQLite touch its `-shm` file, and create the `-wal` and `-shm` files when none
  exist and the directory lets it, even in `mode=ro`. The database file itself is never written.)
- The Editor KB is found with `store.find_editor_kb`, never made: `store.editor_kb` would insert a row. An eval run's KB
  of the same book is never read.
- It is not the API. It never loads `.env`, builds no LLM client and never runs `schema.sql` (`db.init_db`).

The answer is the one the agents' `lookup_term` gives (`agents/kb_tools.py`), built from the same `KBSnapshot.term()`. That
module imports the agent loop and so LangChain, which a server over a database has no use for, so the dict is built again
here; `test_mcp_server.py` checks that the two agree, and that importing this module loads neither LangChain, `openai`
nor `fukidashi.agents`.

This is the only module that imports `mcp`, the official MCP Python SDK. It is not called `mcp.py`, and there is no `mcp/`
directory beside it, because a module or package of that name would shadow the SDK's own.

stdout carries only the protocol: this module prints nothing, and what the SDK logs goes to stderr.
"""

from __future__ import annotations

import importlib.metadata
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from . import config, store
from .kb.entities import APPROVED, normalise
from .kb.loader import load_snapshot
from .kb.snapshot import KBSnapshot

SERVER_NAME = "fukidashi"
TOOL_NAME = "lookup_term"

TOOL_DESCRIPTION = (
    "Look up one Japanese term in a book's Editor KB: the knowledge base that the Fukidashi editor keeps for the book. "
    "It reads the term entries that are approved (in the knowledge base) or pending (waiting for a person to confirm "
    "them), and it matches the whole term exactly, after Unicode NFKC normalisation and trimming. Returns the term's "
    "Simplified Chinese rendering, its category and its status, or says there is none. It only reads."
)
BOOK_TITLE_HELP = "The book's title as the database has it, for example tojime_no_siora."
TERM_HELP = "The term exactly as it is written in the Japanese text."

# The agents' words for a term the KB does not have (`kb_tools.lookup_tools`): the two answers must agree.
NO_ENTRY_NOTE = "The knowledge base has no entry for this term."
NO_EDITOR_KB_NOTE = ("This book has no Editor KB yet: the editor makes it the first time it translates a page of the book "
                     "with the full config.")


def open_read_only(path: Path) -> sqlite3.Connection:
    """A connection that cannot write: SQLite's `mode=ro`, which also never makes the database file (a missing one is an
    error, not an empty database). `db.connect` is the read-write one, and makes the directory too."""
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row  # the store's reads index their rows by column name
    return conn


def answer(snapshot: KBSnapshot, term: str) -> dict:
    """What the agents' `lookup_term` answers over `snapshot` (`agents/kb_tools.py`), word for word."""
    entry = snapshot.term(term)
    if entry is None:
        return {"found": False, "note": NO_ENTRY_NOTE}
    return {"found": True, "source_term": entry.source_term, "target_term": entry.target_term,
            "category": entry.category, "status": "approved" if entry.status == APPROVED else "pending"}


def _unknown_book(conn: sqlite3.Connection, book_title: str) -> ToolError:
    titles = sorted({r["book_title"] for r in store.list_page_summaries(conn)})
    known = f" The books in this database are: {', '.join(titles)}." if titles else " The database holds no book."
    return ToolError(f"No book is titled {book_title!r}.{known}")


def lookup_term_in(db_path: Path, book_title: str, term: str) -> dict:
    """The tool's work: `book_title`'s Editor KB in the database at `db_path`, read as it is now, and the answer for `term`.

    Raises ToolError, with a sentence for the client, for a blank term, a book the database does not have, and a database
    that cannot be read. A book with no Editor KB is an answer (`found: false`), not an error.
    """
    if not normalise(term):
        raise ToolError("The term is empty. Give one Japanese term, written as in the book.")
    try:
        with closing(open_read_only(db_path)) as conn, store.reading(conn):
            if store.find_book(conn, book_title) is None:
                raise _unknown_book(conn, book_title)
            kb = store.find_editor_kb(conn, book_title)
            if kb is None:
                return {"found": False, "note": NO_EDITOR_KB_NOTE}
            snapshot = load_snapshot(conn, kb["id"])
    except sqlite3.Error as e:
        raise ToolError(f"The database cannot be read ({e}). The server reads {db_path}; set FUKIDASHI_DB to read "
                        f"another file.") from e
    return answer(snapshot, term)


def create_server(db_path: Path) -> MCPServer:
    """The MCP server over the database at `db_path`, with its one tool. The database is opened when a call comes in, not
    here, so the server starts whether or not the file exists yet, and the next call reads whatever file is there."""
    # WARNING: a refused call (an unknown book, a blank term) is the tool's normal answer, not something to log.
    server = MCPServer(SERVER_NAME, version=importlib.metadata.version("fukidashi"), log_level="WARNING")

    @server.tool(name=TOOL_NAME, description=TOOL_DESCRIPTION,
                 annotations=ToolAnnotations(read_only_hint=True, open_world_hint=False))
    def lookup_term(
        book_title: Annotated[str, Field(description=BOOK_TITLE_HELP)],
        term: Annotated[str, Field(description=TERM_HELP)],
    ) -> dict[str, Any]:
        return lookup_term_in(db_path, book_title, term)

    return server


def main() -> None:
    create_server(config.DB_PATH).run("stdio")


if __name__ == "__main__":
    main()
