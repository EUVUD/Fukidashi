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
