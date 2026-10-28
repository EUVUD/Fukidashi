"""Helpers shared by the Knowledge Base tests (a module of its own: conftest.py is not touched by this task)."""

from __future__ import annotations

from fukidashi import store
from fukidashi.kb.curator import Decision
from fukidashi.kb.entities import Proposal, make_proposal

KB_TABLES = ("kbs", "glossary_entries", "character_profiles", "proposals")


def dump(conn) -> dict[str, list[tuple]]:
    """Every row of the KB tables, timestamps included: equal dumps mean nothing was written in between."""
    return {t: [tuple(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY rowid")] for t in KB_TABLES}


def new_kb(conn, label: str = "run-1", book: str = "book_a", purpose: str = "eval") -> int:
    return store.create_kb(conn, book_title=book, purpose=purpose, label=label)


def new_run(conn) -> int:
    """A Run on page 1 of book_a: proposals reference the Run that made them."""
    page = store.find_page(conn, "book_a", 1)
    return store.start_run(conn, page_id=page["id"], config="baseline", params={}, model="m")


def term(kb_id: int, page: int, source: str, target: str, **extra) -> Proposal:
    return make_proposal(kb_id, page, "term", {"source_term": source, "target_term": target, **extra}, "translator")


def name(kb_id: int, page: int, source: str, target: str, **extra) -> Proposal:
    return make_proposal(kb_id, page, "character", {"name_src": source, "name_tgt": target, **extra}, "translator")


def accept(reason: str = "worth keeping") -> Decision:
    return Decision(True, reason)


def reject(reason: str = "too generic") -> Decision:
    return Decision(False, reason)
