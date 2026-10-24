"""Load OpenMantra into SQLite. Safe to run repeatedly: rows are keyed on natural keys and updated in place."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .openmantra import DEFAULT_ROOT, load_pages


def import_openmantra(conn: sqlite3.Connection, root: Path = DEFAULT_ROOT) -> dict[str, int]:
    pages = load_pages(root)
    with conn:  # one transaction: either the whole dataset is imported or nothing is
        for page in pages:
            conn.execute("INSERT OR IGNORE INTO books (book_title) VALUES (?)", (page.book_title,))
            # ON CONFLICT DO UPDATE keeps row ids stable, so runs and translations keep pointing at them.
            # INSERT OR REPLACE would delete and re-insert, which breaks those references.
            page_id = conn.execute(
                """INSERT INTO pages (book_title, page_index, image_path) VALUES (?, ?, ?)
                   ON CONFLICT (book_title, page_index) DO UPDATE SET image_path = excluded.image_path
                   RETURNING id""",
                (page.book_title, page.page_index, page.image_path.relative_to(root).as_posix()),
            ).fetchone()["id"]
            conn.executemany(
                """INSERT INTO segments (page_id, idx, x, y, w, h, text_ja, text_zh_hant, text_en)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (page_id, idx) DO UPDATE SET
                       x = excluded.x, y = excluded.y, w = excluded.w, h = excluded.h,
                       text_ja = excluded.text_ja, text_zh_hant = excluded.text_zh_hant,
                       text_en = excluded.text_en""",
                [(page_id, s.index, s.x, s.y, s.w, s.h, s.ja, s.zh_hant, s.en) for s in page.segments],
            )
    return {
        "books": conn.execute("SELECT COUNT(*) FROM books").fetchone()[0],
        "pages": conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0],
        "segments": conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0],
    }
