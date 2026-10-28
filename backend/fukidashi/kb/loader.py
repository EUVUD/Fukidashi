"""Reading a KB out of the database into a snapshot (OUTLINE 3.3).

This is, with `curator.py`, the only module of `kb/` that imports the store. `snapshot.py` and the rest are plain values,
so an agent can import the snapshot type without the database module coming with it. Whoever opens the database (the
caller of the agents, not an agent) imports `load_snapshot` from here.
"""

from __future__ import annotations

import json
import sqlite3

from .. import store
from .entities import CharacterProfile, GlossaryEntry, KBNotFound, frozen_json
from .snapshot import KBSnapshot


def load_snapshot(conn: sqlite3.Connection, kb_id: int) -> KBSnapshot:
    """Copy the KB's readable entries and profiles out of the database. Raises KBNotFound for an unknown id."""
    kb = store.get_kb(conn, kb_id)
    if kb is None:
        raise KBNotFound(f"no knowledge base has id {kb_id}")
    entries = tuple(
        GlossaryEntry(r["id"], r["source_term"], r["target_term"], r["category"], r["status"], r["provenance"],
                      tuple(json.loads(r["evidence_pages"])), frozen_json(r["history"]))
        for r in store.kb_entries(conn, kb_id, store.READABLE_STATUSES)
    )
    profiles = tuple(
        CharacterProfile(r["id"], r["name_src"], r["name_tgt"], tuple(json.loads(r["aliases"])), r["notes"],
                         r["status"], tuple(json.loads(r["evidence_pages"])), frozen_json(r["history"]))
        for r in store.kb_profiles(conn, kb_id, store.READABLE_STATUSES)
    )
    return KBSnapshot(kb_id, kb["book_title"], entries, profiles)
