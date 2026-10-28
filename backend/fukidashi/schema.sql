-- Fukidashi SQLite schema. Applied by fukidashi.db.init_db; every statement is idempotent.

-- Imported from OpenMantra (oracle boxes + source text + reference translations).
CREATE TABLE IF NOT EXISTS books (
    book_title TEXT PRIMARY KEY            -- romaji slug, e.g. tojime_no_siora
);

CREATE TABLE IF NOT EXISTS pages (
    id         INTEGER PRIMARY KEY,
    book_title TEXT    NOT NULL REFERENCES books(book_title),
    page_index INTEGER NOT NULL,           -- 1-based, as in annotation.json
    image_path TEXT    NOT NULL,           -- relative to the dataset root
    UNIQUE (book_title, page_index)
);

CREATE TABLE IF NOT EXISTS segments (
    id           INTEGER PRIMARY KEY,
    page_id      INTEGER NOT NULL REFERENCES pages(id),
    idx          INTEGER NOT NULL,         -- position in annotation.json; also the region id in prompts
    x            INTEGER NOT NULL,         -- oracle box, in pixels of the original image
    y            INTEGER NOT NULL,
    w            INTEGER NOT NULL,
    h            INTEGER NOT NULL,
    text_ja      TEXT    NOT NULL,
    text_zh_hant TEXT    NOT NULL,         -- reference translation (Traditional Chinese)
    text_en      TEXT    NOT NULL,         -- reference translation
    UNIQUE (page_id, idx)
);

-- One translation attempt on one page. Written when the request starts, finished when it ends.
CREATE TABLE IF NOT EXISTS runs (
    id                  INTEGER PRIMARY KEY,
    page_id             INTEGER NOT NULL REFERENCES pages(id),
    config              TEXT    NOT NULL,  -- baseline
    config_json         TEXT    NOT NULL,  -- parameters of this run (page budget, max tokens)
    model               TEXT    NOT NULL,  -- requested model
    response_model      TEXT,              -- as returned by the API
    status              TEXT    NOT NULL CHECK (status IN ('running', 'succeeded', 'failed')),
    error               TEXT,
    calls               INTEGER NOT NULL DEFAULT 0,
    unknown_usage_calls INTEGER NOT NULL DEFAULT 0,  -- calls whose provider returned no usage
    input_tokens        INTEGER,           -- sums over calls with known usage; NULL when none is known
    output_tokens       INTEGER,
    cost_usd            REAL,              -- NULL when not computable, never a partial sum
    attempts_json       TEXT,              -- raw responses and parse errors, for debugging
    started_at          TEXT    NOT NULL,
    finished_at         TEXT
);
CREATE INDEX IF NOT EXISTS runs_page ON runs(page_id, id);

CREATE TABLE IF NOT EXISTS translations (
    run_id       INTEGER NOT NULL REFERENCES runs(id),
    segment_id   INTEGER NOT NULL REFERENCES segments(id),
    text_zh_hans TEXT    NOT NULL,
    PRIMARY KEY (run_id, segment_id)
);

-- The Knowledge Base (OUTLINE 3.3). One KB is one row: an eval run starts every book with a fresh one, the editor keeps
-- one per book. Entries, profiles and proposals belong to exactly one KB, so two KBs never see each other.
CREATE TABLE IF NOT EXISTS kbs (
    id         INTEGER PRIMARY KEY,
    book_title TEXT    NOT NULL REFERENCES books(book_title),
    purpose    TEXT    NOT NULL CHECK (purpose IN ('eval', 'editor')),
    label      TEXT    NOT NULL,           -- for an eval run its id; names the KB within (book, purpose)
    created_at TEXT    NOT NULL,
    UNIQUE (book_title, purpose, label)
);

-- Glossary entries. Only the Curator's rules write them (kb/curator.py); status 'pending_human' is the Human Review Queue.
CREATE TABLE IF NOT EXISTS glossary_entries (
    id             INTEGER PRIMARY KEY,
    kb_id          INTEGER NOT NULL REFERENCES kbs(id),
    source_term    TEXT    NOT NULL,       -- NFKC, trimmed
    target_term    TEXT    NOT NULL,
    category       TEXT    NOT NULL DEFAULT '',
    status         TEXT    NOT NULL CHECK (status IN ('proposed', 'approved', 'rejected', 'pending_human')),
    provenance     TEXT    NOT NULL CHECK (provenance IN ('agent', 'human')),
    evidence_pages TEXT    NOT NULL DEFAULT '[]',  -- JSON: page indexes, ascending
    history        TEXT    NOT NULL DEFAULT '[]',  -- JSON: one object per change
    created_at     TEXT    NOT NULL,
    updated_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS glossary_entries_source ON glossary_entries(kb_id, source_term);

-- Character Profiles. A name goes to the Human Review Queue, so a profile has a status like an entry does.
CREATE TABLE IF NOT EXISTS character_profiles (
    id             INTEGER PRIMARY KEY,
    kb_id          INTEGER NOT NULL REFERENCES kbs(id),
    name_src       TEXT    NOT NULL,       -- NFKC, trimmed
    name_tgt       TEXT    NOT NULL,
    aliases        TEXT    NOT NULL DEFAULT '[]',  -- JSON: source-language names the character is also called
    notes          TEXT    NOT NULL DEFAULT '',    -- speech style and so on
    status         TEXT    NOT NULL CHECK (status IN ('proposed', 'approved', 'rejected', 'pending_human')),
    evidence_pages TEXT    NOT NULL DEFAULT '[]',
    history        TEXT    NOT NULL DEFAULT '[]',
    created_at     TEXT    NOT NULL,
    updated_at     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS character_profiles_name ON character_profiles(kb_id, name_src);

-- The Proposal Queue. proposal_id is a hash of the KB, the page index, the kind and the normalised payload, so a retried
-- page proposes the same ids and inserting one that exists writes nothing.
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT    PRIMARY KEY,
    kb_id       INTEGER NOT NULL REFERENCES kbs(id),
    kind        TEXT    NOT NULL CHECK (kind IN ('term', 'character')),
    payload     TEXT    NOT NULL,          -- JSON
    proposer    TEXT    NOT NULL,
    page_index  INTEGER NOT NULL,
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    status      TEXT    NOT NULL CHECK (status IN ('proposed', 'approved', 'rejected', 'pending_human')),
    reason      TEXT,                      -- the Curator's reason; NULL while the proposal is undecided
    created_at  TEXT    NOT NULL,
    decided_at  TEXT                       -- NULL while the proposal is undecided
);
CREATE INDEX IF NOT EXISTS proposals_kb ON proposals(kb_id, page_index, status);
