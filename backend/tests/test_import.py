from fukidashi.datasets.importer import import_openmantra


def table_counts(conn):
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("books", "pages", "segments")}


def test_import_twice_does_not_duplicate_rows(conn, dataset):
    first = import_openmantra(conn, dataset)
    ids = [r["id"] for r in conn.execute("SELECT id FROM segments ORDER BY id")]
    second = import_openmantra(conn, dataset)

    assert first == second == {"books": 1, "pages": 2, "segments": 2}
    assert table_counts(conn) == {"books": 1, "pages": 2, "segments": 2}
    assert [r["id"] for r in conn.execute("SELECT id FROM segments ORDER BY id")] == ids


def test_reimport_keeps_runs_and_translations(conn, dataset):
    import_openmantra(conn, dataset)
    page_id = conn.execute("SELECT id FROM pages WHERE page_index = 1").fetchone()["id"]
    seg_id = conn.execute("SELECT id FROM segments WHERE page_id = ? AND idx = 0", (page_id,)).fetchone()["id"]
    with conn:
        run_id = conn.execute(
            "INSERT INTO runs (page_id, config, config_json, model, status, started_at)"
            " VALUES (?, 'baseline', '{}', 'm', 'succeeded', 't')", (page_id,)
        ).lastrowid
        conn.execute("INSERT INTO translations VALUES (?, ?, '你好')", (run_id, seg_id))

    import_openmantra(conn, dataset)

    row = conn.execute("SELECT text_zh_hans FROM translations WHERE run_id = ? AND segment_id = ?",
                       (run_id, seg_id)).fetchone()
    assert row["text_zh_hans"] == "你好"
