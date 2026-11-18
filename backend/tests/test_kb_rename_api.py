"""Renaming a KB entry's or profile's rendering over HTTP (task 042; OUTLINE §8, "Rename → whole-book sync"): the four routes
`POST /api/books/{book}/kb/{entries|profiles}/{id}/rename[/preview]`, with their 404, 409 and 422. The rename itself is tested
in `test_kb_rename.py`, the rule that picks the boxes in `test_kb_rename_boxes.py`."""

from __future__ import annotations

import pytest
from kb_support import term
from rename_support import BOOK, ENTRY_SOURCE, attempts_of, build_world, displayed, snapshot, translations_of, write_world

from fukidashi import store
from fukidashi.kb.curator import apply

GET_KB = f"/api/books/{BOOK}/kb"
ROUTES = [("entries", "preview"), ("entries", "write"), ("profiles", "preview"), ("profiles", "write")]
TARGETS = {"entries": "屏障", "profiles": "雪乃"}
SINGULAR = {"entries": "entry", "profiles": "profile"}
PREVIEW_FIELDS = {"kind", "id", "before", "after", "changes", "listed", "settles"}
BOX_FIELDS = {"page_index", "idx", "text_ja", "before", "after"}


@pytest.fixture
def dataset(tmp_path):
    return write_world(tmp_path / "world")


@pytest.fixture
def world(api, conn):
    """The book's Runs and its Editor KB (the `api` fixture imported the book)."""
    return build_world(conn)


def row_id(world, route: str) -> int:
    return {"entries": world.entry, "profiles": world.profile}[route]


def rename_url(route: str, what: str, rid, book: str = BOOK) -> str:
    return f"/api/books/{book}/kb/{route}/{rid}/rename" + ("/preview" if what == "preview" else "")


def post(client, route: str, what: str, rid, target=None, *, book: str = BOOK, body=None):
    """POST a rename (`what` is "preview" or "write"); `target` is sent as the body unless `body` is."""
    return client.post(rename_url(route, what, rid, book), json={"target": target} if body is None else body)


def boxes(items) -> list[tuple]:
    return [(b["page_index"], b["idx"], b["before"], b["after"]) for b in items]


def add_conflicts(conn, world):
    """Two term conflicts on 結界 that propose 屏障, which renaming the entry to 屏障 settles, and one that proposes 封锁."""
    agree = [term(world.kb, 2, ENTRY_SOURCE, "屏障"), term(world.kb, 3, ENTRY_SOURCE, "屏障")]
    other = term(world.kb, 4, ENTRY_SOURCE, "封锁")
    for p in (*agree, other):
        apply(conn, world.kb, p.page_index, world.runs[p.page_index], [p])
    return [p.proposal_id for p in agree], other.proposal_id


# --- the preview ---


def test_the_preview_of_an_entry_answers_the_boxes_that_change_and_the_ones_only_listed(api, conn, world):
    client, _ = api

    resp = post(client, "entries", "preview", world.entry, "屏障")

    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == PREVIEW_FIELDS
    assert (body["kind"], body["id"], body["before"], body["after"], body["settles"]) == ("entry", world.entry, "结界", "屏障", [])
    assert boxes(body["changes"]) == [
        (1, 0, "布下结界！", "布下屏障！"), (1, 1, "小雪，到结界外面去", "小雪，到屏障外面去"),
        (2, 0, "结界就是结界", "屏障就是屏障"), (3, 0, "结界里很安全", "屏障里很安全")]
    assert boxes(body["listed"]) == [(3, 2, "什么？", None), (4, 0, None, None)]  # an untranslated box: before is null too
    assert all(set(b) == BOX_FIELDS for b in (*body["changes"], *body["listed"]))
    assert body["changes"][0]["text_ja"] == "結界を張る！" and body["listed"][1]["text_ja"] == "結界とユキ"
    assert (1, 2) not in {(b["page_index"], b["idx"]) for b in (*body["changes"], *body["listed"])}  # the decoy is not named


def test_the_preview_of_a_profile_matches_through_its_aliases_and_names_the_proposal_it_settles(api, conn, world):
    client, _ = api

    body = post(client, "profiles", "preview", world.profile, "雪乃").json()

    assert set(body) == PREVIEW_FIELDS
    assert (body["kind"], body["id"], body["before"], body["after"]) == ("profile", world.profile, "小雪", "雪乃")
    assert boxes(body["changes"]) == [
        (1, 1, "小雪，到结界外面去", "雪乃，到结界外面去"), (2, 1, "小雪笑了", "雪乃笑了"), (3, 1, "那是小雪的声音", "那是雪乃的声音")]
    assert boxes(body["listed"]) == [(4, 0, None, None)]
    assert body["settles"] == [world.new_name]


def test_the_preview_of_an_entry_names_the_conflicts_the_write_would_settle(api, conn, world):
    client, _ = api
    agree, _other = add_conflicts(conn, world)

    body = post(client, "entries", "preview", world.entry, "屏障").json()

    assert body["settles"] == agree  # in the order they were queued; the conflict for 封锁 is not one of them


@pytest.mark.parametrize("route", ["entries", "profiles"])
def test_a_preview_writes_nothing(api, conn, world, route):
    client, _ = api
    add_conflicts(conn, world)
    before = snapshot(conn)

    for target in (TARGETS[route], "  " + TARGETS[route], "别的名字"):
        assert post(client, route, "preview", row_id(world, route), target).status_code == 200

    assert snapshot(conn) == before  # the KB rows and their histories, every Run and Translation, the events
    assert client.get(f"/api/pages/{BOOK}/1").json()["run"]["config"] == "full"  # and the pages show what they showed


# --- the write ---


@pytest.mark.parametrize("route", ["entries", "profiles"])
def test_the_write_answers_what_the_preview_answered_on_the_same_data(api, conn, world, route):
    client, _ = api
    add_conflicts(conn, world)
    previewed = post(client, route, "preview", row_id(world, route), TARGETS[route]).json()

    resp = post(client, route, "write", row_id(world, route), TARGETS[route])

    assert resp.status_code == 200
    written = resp.json()
    assert set(written) == PREVIEW_FIELDS | {"runs", "kb"}
    assert {k: written[k] for k in PREVIEW_FIELDS} == previewed  # changes, listed and settles, and the rest
    assert [(r["page_index"], set(r)) for r in written["runs"]] == [(i, {"page_index", "run_id"}) for i in (1, 2, 3)]
    assert written["kb"] == client.get(GET_KB).json()  # the KB after it, as the GET gives it


def test_a_written_entry_is_in_the_answers_kb_with_its_history_and_the_conflicts_settled(api, conn, world):
    client, _ = api
    agree, other = add_conflicts(conn, world)

    kb = post(client, "entries", "write", world.entry, "屏障").json()["kb"]

    (entry,) = kb["entries"]
    assert (entry["target_term"], entry["status"], entry["provenance"], entry["evidence_pages"]) == ("屏障", "approved", "human", [1, 2, 3])
    assert [h["event"] for h in entry["history"]] == ["created", "renamed", "confirmed", "confirmed"]
    assert entry["history"][1] | {"at": None} == {
        "at": None, "event": "renamed", "by": "human", "from": "approved", "before": "结界", "after": "屏障"}
    assert [q["proposal_id"] for q in kb["human_review_queue"] if q["kind"] == "term"] == [other]  # the two that agreed left
    assert kb["proposal_counts"]["approved"] == 3 and kb["proposal_counts"]["pending_human"] == 2  # +2 approved; the new name and 封锁


def test_a_written_profile_is_approved_in_the_answers_kb_and_its_new_name_proposal_leaves_the_queue(api, conn, world):
    client, _ = api

    kb = post(client, "profiles", "write", world.profile, "雪乃").json()["kb"]

    (profile,) = kb["profiles"]
    assert (profile["name_src"], profile["name_tgt"], profile["status"], profile["provenance"]) == (
        "ユキ", "雪乃", "approved", "human")
    assert profile["aliases"] == ["ゆき"] and profile["history"][-1]["event"] == "renamed"
    assert kb["human_review_queue"] == []  # the new name's proposal is settled with its profile


def test_the_changed_pages_show_the_new_text_and_a_human_run_and_the_rest_do_not(api, conn, world):
    client, _ = api
    written = post(client, "entries", "write", world.entry, "屏障").json()

    shown = {p: client.get(f"/api/pages/{BOOK}/{p}").json() for p in (1, 2, 3, 4)}

    assert [s["text_zh_hans"] for s in shown[1]["segments"]] == ["布下屏障！", "小雪，到屏障外面去", "结界消失了"]
    assert [s["text_zh_hans"] for s in shown[2]["segments"]] == ["屏障就是屏障", "小雪笑了"]
    assert [s["text_zh_hans"] for s in shown[3]["segments"]] == ["屏障里很安全", "那是小雪的声音", "什么？"]
    assert shown[4]["run"] is None and [s["text_zh_hans"] for s in shown[4]["segments"]] == [None]
    for run in written["runs"]:
        page = shown[run["page_index"]]
        assert (page["run"]["id"], page["run"]["config"], page["run"]["model"], page["run"]["calls"]) == (
            run["run_id"], "human", "human", 0)
        assert page["run"]["cost_usd"] == 0.0


def test_a_page_whose_displayed_run_is_a_renames_human_edit_over_a_full_run_reports_the_renamed_boxes_as_edited(api, conn, world):
    client, _ = api
    before = client.get(f"/api/pages/{BOOK}/1").json()
    assert before["review"]["edited"] == [] and before["review"]["config"] == "full"
    post(client, "entries", "write", world.entry, "屏障")

    after = client.get(f"/api/pages/{BOOK}/1").json()

    assert after["run"]["config"] == "human"
    review = after["review"]
    assert (review["run_id"], review["config"]) == (world.runs[1], "full")  # the review is still the `full` Run's
    assert review["edited"] == [0, 1]  # exactly the boxes the rename changed
    assert [(f["region"], f["current"]) for f in review["flags"]] == [(0, False), (1, False)]
    assert [(i["region"], i["current"]) for i in review["issues"]] == [(1, False), (2, True)]
    assert client.get(f"/api/pages/{BOOK}/3").json()["review"] is None  # page 3 shows a `baseline` Run: nothing to review
    # and the stored record is what the answer says
    (run_1,) = [r["run_id"] for r in post(client, "profiles", "write", world.profile, "雪乃").json()["runs"] if r["page_index"] == 1]
    assert attempts_of(conn, run_1)["rename"] == {"kind": "profile", "id": world.profile, "kb_id": world.kb, "before": "小雪", "after": "雪乃"}
    assert client.get(f"/api/pages/{BOOK}/1").json()["review"]["edited"] == [0, 1]  # a chain of two Human Runs, still two boxes


def test_a_rename_that_is_written_twice_is_the_second_time_a_rename_from_the_first_ones_rendering(api, conn, world):
    client, _ = api
    post(client, "entries", "write", world.entry, "屏障")

    same = post(client, "entries", "preview", world.entry, "屏障")
    again = post(client, "entries", "preview", world.entry, "结界")  # back: the boxes it changed are the ones that change

    assert same.status_code == 409 and same.json()["detail"] == f"entry {world.entry} already has the rendering 屏障: a rename needs another one"
    assert again.status_code == 200
    assert boxes(again.json()["changes"]) == [
        (1, 0, "布下屏障！", "布下结界！"), (1, 1, "小雪，到屏障外面去", "小雪，到结界外面去"),
        (2, 0, "屏障就是屏障", "结界就是结界"), (3, 0, "屏障里很安全", "结界里很安全")]
    assert displayed(conn, 1) == ["布下屏障！", "小雪，到屏障外面去", "结界消失了"]  # the preview changed nothing


def test_the_write_is_planned_again_so_a_page_that_changed_since_the_preview_is_right(api, conn, world):
    client, _ = api
    previewed = post(client, "entries", "preview", world.entry, "屏障").json()
    assert (1, 0) in {(b["page_index"], b["idx"]) for b in previewed["changes"]}
    edited = client.put(f"/api/pages/{BOOK}/1/segments/0/translation", json={"text_zh_hans": "张开防护罩！"})  # a person's edit
    assert edited.status_code == 200

    written = post(client, "entries", "write", world.entry, "屏障").json()

    assert (1, 0) not in {(b["page_index"], b["idx"]) for b in written["changes"]}  # no longer holds 结界: it is only listed
    assert (1, 0) in {(b["page_index"], b["idx"]) for b in written["listed"]}
    assert displayed(conn, 1) == ["张开防护罩！", "小雪，到屏障外面去", "结界消失了"]
    assert translations_of(conn, written["runs"][0]["run_id"])[0] == "张开防护罩！"


def test_the_write_normalises_the_new_rendering_as_the_kb_does(api, conn, world):
    client, _ = api

    body = post(client, "entries", "write", world.entry, "  屏障\u3000").json()

    assert body["after"] == "屏障" and body["kb"]["entries"][0]["target_term"] == "屏障"


def test_an_eval_kbs_rows_are_the_same_before_and_after_a_rename(api, conn, world):
    client, _ = api
    eval_kb = store.create_kb(conn, book_title=BOOK, purpose="eval", label="run-1")
    apply(conn, eval_kb, 1, world.runs[1], [term(eval_kb, 1, ENTRY_SOURCE, "结界")], {})
    rows = "SELECT * FROM {} WHERE kb_id = ? ORDER BY rowid"
    before = {t: [tuple(r) for r in conn.execute(rows.format(t), (eval_kb,))] for t in ("glossary_entries", "character_profiles", "proposals")}
    assert before["proposals"]

    post(client, "entries", "write", world.entry, "屏障")
    post(client, "profiles", "write", world.profile, "雪乃")

    assert {t: [tuple(r) for r in conn.execute(rows.format(t), (eval_kb,))] for t in before} == before


# --- the 404s: nothing is written ---


@pytest.mark.parametrize("route, what", ROUTES)
def test_an_unknown_book_is_a_404(api, conn, world, route, what):
    client, _ = api

    resp = post(client, route, what, row_id(world, route), TARGETS[route], book="no_such_book")

    assert (resp.status_code, resp.json()["detail"]) == (404, "no book no_such_book")


@pytest.mark.parametrize("route, what", ROUTES)
def test_a_book_with_no_editor_kb_is_a_404_and_a_rename_makes_none(api, conn, world, route, what):
    client, _ = api
    conn.execute("INSERT INTO books (book_title) VALUES ('book_b')")
    conn.commit()
    kbs, before = conn.execute("SELECT COUNT(*) FROM kbs").fetchone()[0], snapshot(conn)

    resp = post(client, route, what, row_id(world, route), TARGETS[route], book="book_b")

    assert (resp.status_code, resp.json()["detail"]) == (404, "book book_b has no Editor KB yet, so there is nothing to decide")
    assert conn.execute("SELECT COUNT(*) FROM kbs").fetchone()[0] == kbs and snapshot(conn) == before


@pytest.mark.parametrize("route, what", ROUTES)
@pytest.mark.parametrize("bad", [424242, 0, -1, 10**30, -(10**30)])
def test_an_id_the_editor_kb_does_not_have_is_a_404(api, conn, world, route, what, bad):
    client, _ = api
    before = snapshot(conn)

    resp = post(client, route, what, bad, TARGETS[route])

    assert resp.status_code == 404
    assert snapshot(conn) == before


@pytest.mark.parametrize("route, what", ROUTES)
def test_an_eval_kbs_row_or_another_books_row_is_a_404_through_this_books_route(api, conn, world, route, what):
    client, _ = api
    eval_kb = store.create_kb(conn, book_title=BOOK, purpose="eval", label="run-1")
    conn.execute("INSERT INTO books (book_title) VALUES ('book_b')")
    conn.commit()
    other_kb = store.editor_kb(conn, "book_b")
    foreign = []
    for kb in (eval_kb, other_kb):
        if route == "entries":
            foreign.append(store.insert_entry(conn, kb_id=kb, source_term="魔法", target_term="魔法", category="", status="approved",
                                              provenance="agent", page_index=1, event={"event": "created"}))
        else:
            foreign.append(store.insert_profile(conn, kb_id=kb, name_src="ハル", name_tgt="小春", aliases=[], notes="",
                                                status="approved", page_index=1, event={"event": "created"}))
    conn.commit()
    before = snapshot(conn)

    for rid in foreign:
        resp = post(client, route, what, rid, TARGETS[route])
        assert resp.status_code == 404, (route, what, rid)
        assert resp.json()["detail"] == f"no {SINGULAR[route]} {rid} in the KB"

    assert snapshot(conn) == before
    # the other book's own route finds its row (a rename of a row with no box to follow: only the row is written)
    assert post(client, route, "write", foreign[1], TARGETS[route], book="book_b").status_code == 200


# --- the 409s: nothing is written, and the preview says the same ---


@pytest.mark.parametrize("route, what", ROUTES)
@pytest.mark.parametrize("status", ["rejected", "proposed"])
def test_a_row_that_is_not_readable_is_a_409_that_says_to_approve_it_first(api, conn, world, route, what, status):
    client, _ = api
    table = {"entries": "glossary_entries", "profiles": "character_profiles"}[route]
    conn.execute(f"UPDATE {table} SET status = ? WHERE id = ?", (status, row_id(world, route)))
    conn.commit()
    before = snapshot(conn)

    resp = post(client, route, what, row_id(world, route), TARGETS[route])

    assert (resp.status_code, resp.json()["detail"]) == (
        409, f"{SINGULAR[route]} {row_id(world, route)} is {status}, so it cannot be renamed: approve it first")
    assert snapshot(conn) == before


@pytest.mark.parametrize("route, what", ROUTES)
@pytest.mark.parametrize("same", ["结界", " 结界 ", "结界\u3000"])
def test_a_rendering_equal_to_the_old_one_after_normalising_is_a_409(api, conn, world, route, what, same):
    client, _ = api
    old = {"entries": "结界", "profiles": "小雪"}[route]
    before = snapshot(conn)

    resp = post(client, route, what, row_id(world, route), same.replace("结界", old))

    assert (resp.status_code, resp.json()["detail"]) == (
        409, f"{SINGULAR[route]} {row_id(world, route)} already has the rendering {old}: a rename needs another one")
    assert snapshot(conn) == before


@pytest.mark.parametrize("what", ["preview", "write"])
def test_a_collision_an_approval_would_refuse_is_a_409(api, conn, world, what):
    client, _ = api
    store.insert_entry(conn, kb_id=world.kb, source_term="魔法", target_term="魔术", category="", status="approved",
                       provenance="agent", page_index=1, event={"event": "created"})
    pending = store.insert_entry(conn, kb_id=world.kb, source_term="魔法", target_term="魔法", category="", status="pending_human",
                                 provenance="agent", page_index=1, event={"event": "created"})
    rival = store.insert_profile(conn, kb_id=world.kb, name_src="ユキ", name_tgt="雪", aliases=[], notes="", status="approved",
                                 page_index=1, event={"event": "created"})
    conn.commit()
    before = snapshot(conn)

    entry = post(client, "entries", what, pending, "咒语")
    profile = post(client, "profiles", what, world.profile, "雪乃")  # the pending new name, and a readable profile for ユキ

    assert entry.status_code == 409 and "two readable entries" in entry.json()["detail"]
    assert entry.json()["detail"].startswith(f"entry {pending} cannot be approved: entry ")
    assert profile.status_code == 409
    assert profile.json()["detail"].startswith(f"profile {world.profile} cannot be approved: profile {rival} (雪)")
    assert snapshot(conn) == before


@pytest.mark.parametrize("route, what", ROUTES)
def test_a_book_with_a_running_run_is_a_409_and_nothing_is_written_until_it_ends(api, conn, world, route, what):
    client, _ = api
    page_id = store.find_page(conn, BOOK, 2)["id"]
    running = store.start_run(conn, page_id=page_id, config="full", params={}, model="m")
    before = snapshot(conn)

    resp = post(client, route, what, row_id(world, route), TARGETS[route])

    assert resp.status_code == 409
    assert resp.json()["detail"] == (f"a Run is still running on page 2 (Run {running}) of {BOOK}, and a Run that finishes "
                                     f"writes the KB: rename after it ends")
    assert snapshot(conn) == before
    conn.execute("UPDATE runs SET status = 'failed', error = 'stopped' WHERE id = ?", (running,))
    conn.commit()
    assert post(client, route, what, row_id(world, route), TARGETS[route]).status_code == 200


# --- the 422s: nothing is written ---

BAD_BODIES = [
    pytest.param({}, id="empty-object"),
    pytest.param({"target": "屏障", "extra": 1}, id="extra-key"),
    pytest.param({"rendering": "屏障"}, id="another-key"),
    pytest.param({"target": None}, id="null"),
    pytest.param({"target": 5}, id="number"),
    pytest.param({"target": ["屏障"]}, id="list"),
    pytest.param({"target": {"a": 1}}, id="object"),
    pytest.param([], id="array"),
    pytest.param("屏障", id="string"),
    pytest.param(None, id="null-body"),
]
BAD_RENDERINGS = [
    pytest.param("", id="empty"),
    pytest.param("   ", id="spaces"),
    pytest.param("\u3000", id="ideographic-space"),
    pytest.param("雪" * 61, id="too-long"),
    pytest.param("a\nb", id="newline"),
    pytest.param("a\tb", id="tab"),
    pytest.param("a\u2028b", id="line-separator"),
    pytest.param("a\x00b", id="nul"),
]


@pytest.mark.parametrize("route, what", ROUTES)
@pytest.mark.parametrize("body", BAD_BODIES)
def test_a_body_that_is_not_exactly_a_target_is_a_422_and_writes_nothing(api, conn, world, route, what, body):
    client, _ = api
    before = snapshot(conn)

    resp = post(client, route, what, row_id(world, route), body=body)

    assert resp.status_code == 422
    assert snapshot(conn) == before


@pytest.mark.parametrize("route, what", ROUTES)
def test_a_missing_or_unreadable_body_is_a_422(api, conn, world, route, what):
    client, _ = api
    before = snapshot(conn)
    url = rename_url(route, what, row_id(world, route))

    assert client.post(url).status_code == 422
    assert client.post(url, content=b"{not json", headers={"content-type": "application/json"}).status_code == 422
    assert snapshot(conn) == before


@pytest.mark.parametrize("route, what", ROUTES)
@pytest.mark.parametrize("bad", BAD_RENDERINGS)
def test_a_rendering_the_kb_would_not_store_is_a_422_and_writes_nothing(api, conn, world, route, what, bad):
    client, _ = api
    before = snapshot(conn)

    resp = post(client, route, what, row_id(world, route), bad)

    assert resp.status_code == 422
    assert resp.json()["detail"].startswith("invalid new rendering: ")  # a sentence, as the 409s have, not a list of errors
    assert snapshot(conn) == before


@pytest.mark.parametrize("route, what", ROUTES)
def test_an_id_that_is_not_an_integer_is_a_422(api, conn, world, route, what):
    client, _ = api

    for bad in ("abc", "1.5"):
        assert post(client, route, what, bad, TARGETS[route]).status_code == 422


def test_there_is_no_rename_route_for_a_proposal(api, conn, world):
    client, _ = api

    resp = client.post(f"/api/books/{BOOK}/kb/proposals/{world.new_name}/rename", json={"target": "雪乃"})

    assert resp.status_code in (404, 405)
