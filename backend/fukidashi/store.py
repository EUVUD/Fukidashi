"""SQL for pages, segments, runs, translations and the Knowledge Base. No HTTP and no LLM in here."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_page(conn: sqlite3.Connection, book_title: str, page_index: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM pages WHERE book_title = ? AND page_index = ?", (book_title, page_index)
    ).fetchone()


def page_segments(conn: sqlite3.Connection, page_id: int) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM segments WHERE page_id = ? ORDER BY idx", (page_id,)).fetchall()


def list_page_summaries(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT p.book_title, p.page_index,
                  (SELECT COUNT(*) FROM segments s WHERE s.page_id = p.id) AS segment_count,
                  EXISTS (SELECT 1 FROM runs r WHERE r.page_id = p.id AND r.status = 'succeeded') AS translated
           FROM pages p ORDER BY p.id"""
    ).fetchall()


def earlier_pages(conn: sqlite3.Connection, book_title: str, page_index: int, count: int) -> list[sqlite3.Row]:
    """The pages of the book numbered `page_index - count` .. `page_index - 1` that exist, oldest first."""
    return conn.execute(
        "SELECT * FROM pages WHERE book_title = ? AND page_index >= ? AND page_index < ? ORDER BY page_index",
        (book_title, page_index - count, page_index),
    ).fetchall()


def latest_succeeded_run(conn: sqlite3.Connection, page_id: int, config: str | None = None) -> sqlite3.Row | None:
    """The page's latest succeeded Run, of any config, or of `config` when given."""
    return conn.execute(
        "SELECT * FROM runs WHERE page_id = ? AND status = 'succeeded' AND (? IS NULL OR config = ?)"
        " ORDER BY id DESC LIMIT 1", (page_id, config, config)
    ).fetchone()


def get_run(conn: sqlite3.Connection, run_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()


def segments_with_translations(conn: sqlite3.Connection, page_id: int, run_id: int | None) -> list[sqlite3.Row]:
    """Every segment of the page, with its zh-Hans text from `run_id` (NULL when there is no run)."""
    return conn.execute(
        """SELECT s.*, t.text_zh_hans FROM segments s
           LEFT JOIN translations t ON t.segment_id = s.id AND t.run_id = ?
           WHERE s.page_id = ? ORDER BY s.idx""",
        (run_id, page_id),
    ).fetchall()


def start_run(conn: sqlite3.Connection, *, page_id: int, config: str, params: dict, model: str) -> int:
    with conn:  # committed right away, so the LLM call that follows holds no write lock
        return conn.execute(
            """INSERT INTO runs (page_id, config, config_json, model, status, started_at)
               VALUES (?, ?, ?, ?, 'running', ?)""",
            (page_id, config, json.dumps(params), model, _now()),
        ).lastrowid


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    error: str | None,
    response_models: list[str],
    usage: dict,  # Meter.summary()
    attempts: list[dict] | dict,  # baseline: one dict per call; translator: one record of the agent, repair, fallback
    translations: dict[int, str] | None,  # segment row id -> zh-Hans; None means the run failed
) -> None:
    # tokens are summed over calls with known usage; with none known they stay NULL rather than 0
    known = usage["calls"] - usage["unknown_usage_calls"]
    with conn:
        conn.execute(
            """UPDATE runs SET status = ?, error = ?, response_model = ?, calls = ?, unknown_usage_calls = ?,
                      input_tokens = ?, output_tokens = ?, cost_usd = ?, attempts_json = ?, finished_at = ?
               WHERE id = ?""",
            ("succeeded" if translations is not None else "failed", error,
             ", ".join(response_models) or None, usage["calls"], usage["unknown_usage_calls"],
             usage["input_tokens"] if known else None, usage["output_tokens"] if known else None,
             usage["cost_usd"], json.dumps(attempts, ensure_ascii=False), _now(), run_id),
        )
        conn.executemany(
            "INSERT INTO translations (run_id, segment_id, text_zh_hans) VALUES (?, ?, ?)",
            [(run_id, seg_id, text) for seg_id, text in (translations or {}).items()],
        )


# --- Knowledge Base (OUTLINE 3.3) ---
# Reads are open to everyone. The functions named in KB_WRITES write entries, profiles and proposals, and only
# kb/curator.py calls them (a test scans the code for any other caller). They do not commit: the Curator's `apply`
# opens the transaction with `transaction` and so owns all-or-nothing. `create_kb` only makes the empty container,
# so it is not in KB_WRITES.

KB_WRITES = (
    "insert_proposal", "set_proposal_status", "insert_entry", "insert_profile",
    "add_entry_evidence", "add_profile_evidence",
)

# The states a lookup or a KB snippet may return (OUTLINE 3.3): 'proposed' and 'rejected' are not shown.
READABLE_STATUSES = ("approved", "pending_human")


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """One transaction that takes the write lock at once (what is read inside cannot change before it is written),
    committed when the block ends and rolled back when it raises."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def create_kb(conn: sqlite3.Connection, *, book_title: str, purpose: str, label: str) -> int:
    """A new, empty KB. Raises sqlite3.IntegrityError for an unknown book, a purpose other than eval / editor, or a
    (book, purpose, label) that exists."""
    with conn:
        return conn.execute(
            "INSERT INTO kbs (book_title, purpose, label, created_at) VALUES (?, ?, ?, ?)",
            (book_title, purpose, label, _now()),
        ).lastrowid


def find_kb(conn: sqlite3.Connection, *, book_title: str, purpose: str, label: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM kbs WHERE book_title = ? AND purpose = ? AND label = ?", (book_title, purpose, label)
    ).fetchone()


def get_kb(conn: sqlite3.Connection, kb_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM kbs WHERE id = ?", (kb_id,)).fetchone()


def _in(statuses: Sequence[str]) -> str:
    return ", ".join("?" * len(statuses))


def kb_entries(conn: sqlite3.Connection, kb_id: int, statuses: Sequence[str]) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT * FROM glossary_entries WHERE kb_id = ? AND status IN ({_in(statuses)}) ORDER BY id",
        (kb_id, *statuses),
    ).fetchall()


def kb_profiles(conn: sqlite3.Connection, kb_id: int, statuses: Sequence[str]) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT * FROM character_profiles WHERE kb_id = ? AND status IN ({_in(statuses)}) ORDER BY id",
        (kb_id, *statuses),
    ).fetchall()


def kb_proposals(conn: sqlite3.Connection, kb_id: int, status: str | None = None) -> list[sqlite3.Row]:
    """The KB's proposals in the order they were queued, or only those with `status`."""
    return conn.execute(
        "SELECT * FROM proposals WHERE kb_id = ? AND (? IS NULL OR status = ?) ORDER BY rowid",
        (kb_id, status, status),
    ).fetchall()


def find_entry(conn: sqlite3.Connection, kb_id: int, source_term: str) -> sqlite3.Row | None:
    """The readable entry for `source_term`: an approved one before a pending one, then the lowest id."""
    return conn.execute(
        f"SELECT * FROM glossary_entries WHERE kb_id = ? AND source_term = ? AND status IN ({_in(READABLE_STATUSES)})"
        " ORDER BY status = 'approved' DESC, id LIMIT 1", (kb_id, source_term, *READABLE_STATUSES)
    ).fetchone()


def find_profile(conn: sqlite3.Connection, kb_id: int, name_src: str) -> sqlite3.Row | None:
    """The readable profile whose `name_src` is `name_src`, picked like `find_entry`."""
    return conn.execute(
        f"SELECT * FROM character_profiles WHERE kb_id = ? AND name_src = ? AND status IN ({_in(READABLE_STATUSES)})"
        " ORDER BY status = 'approved' DESC, id LIMIT 1", (kb_id, name_src, *READABLE_STATUSES)
    ).fetchone()


def _dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def insert_proposal(
    conn: sqlite3.Connection, *, proposal_id: str, kb_id: int, kind: str, payload: dict, proposer: str,
    page_index: int, run_id: int,
) -> bool:
    """Queue a proposal as 'proposed'. False, and nothing written, when the id is already there. (Not OR IGNORE: that
    would also swallow a failed CHECK, and report a bad row as a duplicate.)"""
    return conn.execute(
        """INSERT INTO proposals (proposal_id, kb_id, kind, payload, proposer, page_index, run_id, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'proposed', ?) ON CONFLICT (proposal_id) DO NOTHING""",
        (proposal_id, kb_id, kind, _dumps(payload), proposer, page_index, run_id, _now()),
    ).rowcount == 1


def set_proposal_status(conn: sqlite3.Connection, proposal_id: str, *, status: str, reason: str) -> None:
    conn.execute(
        "UPDATE proposals SET status = ?, reason = ?, decided_at = ? WHERE proposal_id = ?",
        (status, reason, None if status == "proposed" else _now(), proposal_id),
    )


def insert_entry(
    conn: sqlite3.Connection, *, kb_id: int, source_term: str, target_term: str, category: str, status: str,
    provenance: str, page_index: int, event: dict,
) -> int:
    """A new entry with `page_index` as its evidence and `event` as its first history line."""
    now = _now()
    return conn.execute(
        """INSERT INTO glossary_entries (kb_id, source_term, target_term, category, status, provenance,
                                         evidence_pages, history, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (kb_id, source_term, target_term, category, status, provenance, _dumps([page_index]),
         _dumps([{"at": now, **event}]), now, now),
    ).lastrowid


def insert_profile(
    conn: sqlite3.Connection, *, kb_id: int, name_src: str, name_tgt: str, aliases: Sequence[str], notes: str,
    status: str, page_index: int, event: dict,
) -> int:
    now = _now()
    return conn.execute(
        """INSERT INTO character_profiles (kb_id, name_src, name_tgt, aliases, notes, status,
                                           evidence_pages, history, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (kb_id, name_src, name_tgt, _dumps(list(aliases)), notes, status, _dumps([page_index]),
         _dumps([{"at": now, **event}]), now, now),
    ).lastrowid


def _add_evidence(conn: sqlite3.Connection, table: str, row_id: int, page_index: int, event: dict) -> bool:
    row = conn.execute(f"SELECT evidence_pages, history FROM {table} WHERE id = ?", (row_id,)).fetchone()
    pages = json.loads(row["evidence_pages"])
    if page_index in pages:
        return False
    now = _now()
    history = [*json.loads(row["history"]), {"at": now, **event}]
    conn.execute(
        f"UPDATE {table} SET evidence_pages = ?, history = ?, updated_at = ? WHERE id = ?",
        (_dumps(sorted([*pages, page_index])), _dumps(history), now, row_id),
    )
    return True


def add_entry_evidence(conn: sqlite3.Connection, entry_id: int, page_index: int, event: dict) -> bool:
    """Add `page_index` to the entry's evidence and `event` to its history. False, and nothing written, when the page
    is in the evidence already."""
    return _add_evidence(conn, "glossary_entries", entry_id, page_index, event)


def add_profile_evidence(conn: sqlite3.Connection, profile_id: int, page_index: int, event: dict) -> bool:
    return _add_evidence(conn, "character_profiles", profile_id, page_index, event)
