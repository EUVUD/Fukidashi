"""SQL for pages, segments, runs and translations. No HTTP and no LLM in here."""

from __future__ import annotations

import json
import sqlite3
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


def latest_succeeded_run(conn: sqlite3.Connection, page_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM runs WHERE page_id = ? AND status = 'succeeded' ORDER BY id DESC LIMIT 1", (page_id,)
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
    attempts: list[dict],
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
