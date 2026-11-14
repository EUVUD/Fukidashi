"""The review of a page: the Reviewer's Issues and the Translator's Flags, read from the stored Run (`fukidashi.review_notes`,
task 039). The builder is tested over attempts records written here; the API over Runs a scripted provider makes through
the real Orchestrator, so that the record the builder reads is the one the Orchestrator writes. No network."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import mock_llm_client
from fastapi.testclient import TestClient
from test_agent_loop import Provider, answer, ask
from test_reviewer import APPROVE, issue, verdict_json
from test_translator import BOOK, flag, imported, root, translate, zh, zh_json  # noqa: F401 (fixtures)

from fukidashi import orchestrator, review_notes, runs, store
from fukidashi.api.deps import get_dataset_root, get_db, get_db_path
from fukidashi.db import connect
from fukidashi.llm.metering import Meter
from fukidashi.main import app
from fukidashi.review_notes import build_review

# --- Records written by hand. They have the Orchestrator's shape (`orchestrator/flow.py`, `run_orchestrator`). --------------


def stage(kind, status="ok", **fields):
    return {"kind": kind, "status": status, "error": None, **fields, "attempts": {}}


def translated(*flags, status="ok"):
    """The first pass. A flag is (region, reason)."""
    return stage("translate", status, flags=[{"id": r, "reason": why} for r, why in flags])


def revised(*flags, status="ok"):
    return stage("revise", status, flags=[{"id": r, "reason": why} for r, why in flags])


def reviewed(verdict, *issues, status="ok"):
    """A review. An issue is (region, comment, suggestion)."""
    return stage("review", status, verdict=verdict,
                 issues=[{"id": r, "comment": c, "suggestion": s} for r, c, s in issues])


def record(stages, stop, produced_by, *, config="full"):
    """An Orchestrator Run's attempts: `produced_by` is the index of the stage whose translation was stored."""
    return {"config": config, "stages": stages, "send_backs": sum(s["kind"] == "revise" for s in stages),
            "final": {"stage": produced_by, "kind": stages[produced_by]["kind"], "reviewed": stop in
                      ("approved", "revision_unusable"), "stop": stop}}


def row(run_id, config, attempts):
    """A Run as the builder reads it: `id`, `config`, and the attempts as the JSON text the database holds."""
    return {"id": run_id, "config": config, "attempts_json": json.dumps(attempts, ensure_ascii=False)}


def human(run_id, base, *edited):
    """A `human` Run on `base` that edited these regions (`runs.edit_translation`'s record)."""
    return row(run_id, "human", {"config": "human", "base_run_id": base,
                                 "edits": [{"idx": i, "before": "旧", "after": "新"} for i in edited]})


def lookup(*rows):
    """`get_run` for the builder: the Runs by id."""
    by_id = {r["id"]: r for r in rows}
    return by_id.get


def build(shown, *others):
    return build_review(shown, lookup(shown, *others))


def flag_of(region, reason, pass_, current):
    return {"region": region, "reason": reason, "pass": pass_, "current": current}


def issue_of(region, comment, suggestion, verdict, review, current):
    return {"region": region, "comment": comment, "suggestion": suggestion, "verdict": verdict, "review": review,
            "current": current}


# The page of `tojime_no_siora/018` as it was stored (Run 448): a flag on the first pass, a review that sent the page back,
# a revision that flagged nothing, and a second review that approved with notes on three regions.
SENT_BACK_ONCE = record([
    translated((9, "指代不明")),
    reviewed("revise", (9, "「もっとも」译成了附和", "不过……")),
    revised(),
    reviewed("approve", (3, "总算略带润色", None), (6, "补出「画里」无误", None), (8, "「数刻」略偏长", None)),
], "approved", 2)


# --- The builder: the Run's own stages -----------------------------------------------------------------------------------


def test_approved_on_the_first_review_every_note_is_current():
    full = row(7, "full", record([
        translated((2, "语气不明")),
        reviewed("approve", (2, "可以更口语", None), (5, "译法合理", None)),
    ], "approved", 0))

    assert build(full) == {
        "run_id": 7, "config": "full", "send_backs": 0, "final_stop": "approved", "edited": [],
        "flags": [flag_of(2, "语气不明", 0, True)],
        "issues": [issue_of(2, "可以更口语", None, "approve", 1, True), issue_of(5, "译法合理", None, "approve", 1, True)],
    }


def test_approved_after_one_send_back_only_the_last_review_and_the_revisions_flags_are_current():
    full = row(448, "full", SENT_BACK_ONCE)

    assert build(full) == {
        "run_id": 448, "config": "full", "send_backs": 1, "final_stop": "approved", "edited": [],
        "flags": [flag_of(9, "指代不明", 0, False)],  # about the first translation, which the revision replaced
        "issues": [
            issue_of(9, "「もっとも」译成了附和", "不过……", "revise", 1, False),  # why the page was sent back
            issue_of(3, "总算略带润色", None, "approve", 2, True),
            issue_of(6, "补出「画里」无误", None, "approve", 2, True),
            issue_of(8, "「数刻」略偏长", None, "approve", 2, True),
        ],
    }


def test_a_flag_of_the_revision_that_was_stored_is_current_and_the_first_passs_is_not():
    full = row(5, "full", record([
        translated((1, "初译不确定")),
        reviewed("revise", (1, "译错了", "改")),
        revised((1, "仍不确定"), (4, "新的疑问")),
        reviewed("approve"),
    ], "approved", 2))

    review = build(full)

    assert review["flags"] == [flag_of(1, "初译不确定", 0, False), flag_of(1, "仍不确定", 1, True),
                               flag_of(4, "新的疑问", 1, True)]
    assert review["issues"] == [issue_of(1, "译错了", "改", "revise", 1, False)]


def test_an_unusable_review_gives_no_verdict_and_the_translation_stands_with_its_flags_current():
    full = row(8, "full", record([translated((0, "存疑")), reviewed(None, status="unusable")], "review_unusable", 0))

    assert build(full) == {
        "run_id": 8, "config": "full", "send_backs": 0, "final_stop": "review_unusable", "edited": [],
        "flags": [flag_of(0, "存疑", 0, True)], "issues": [],
    }


def test_an_unusable_review_after_a_send_back_leaves_the_revision_standing():
    full = row(9, "full", record([
        translated((0, "初译存疑")),
        reviewed("revise", (0, "译错了", "改")),
        revised((1, "修改后存疑")),
        reviewed(None, status="unusable"),
    ], "review_unusable", 2))

    review = build(full)

    assert (review["send_backs"], review["final_stop"]) == (1, "review_unusable")
    assert review["flags"] == [flag_of(0, "初译存疑", 0, False), flag_of(1, "修改后存疑", 1, True)]
    assert review["issues"] == [issue_of(0, "译错了", "改", "revise", 1, False)]


def test_an_unusable_revision_leaves_the_reviewed_translation_so_the_revise_verdict_on_it_is_current():
    full = row(10, "full", record([
        translated((3, "存疑")),
        reviewed("revise", (3, "译错了", "改")),
        revised(status="unusable"),
    ], "revision_unusable", 0))

    assert build(full) == {
        "run_id": 10, "config": "full", "send_backs": 1, "final_stop": "revision_unusable", "edited": [],
        "flags": [flag_of(3, "存疑", 0, True)],  # the translation that stands is the first pass's
        "issues": [issue_of(3, "译错了", "改", "revise", 1, True)],
    }


def test_after_the_send_back_limit_no_review_follows_the_stored_translation_so_no_issue_is_current():
    full = row(866, "full", record([
        translated((1, "一")),
        reviewed("revise", (1, "第一次退回", "甲")),
        revised((2, "二")),
        reviewed("revise", (2, "第二次退回", "乙")),
        revised((4, "三")),
    ], "send_back_limit", 4))

    review = build(full)

    assert (review["send_backs"], review["final_stop"]) == (2, "send_back_limit")
    assert review["flags"] == [flag_of(1, "一", 0, False), flag_of(2, "二", 1, False), flag_of(4, "三", 2, True)]
    assert review["issues"] == [issue_of(1, "第一次退回", "甲", "revise", 1, False),
                                issue_of(2, "第二次退回", "乙", "revise", 2, False)]


def test_a_first_pass_that_fell_back_to_baseline_has_no_flags_and_its_issues_are_current():
    first = translated()  # the agent's answer was not used, so the stage holds no flags
    first["attempts"] = {"config": "translator", "fallback": {"reason": "step_limit"}}
    full = row(11, "full", record([first, reviewed("approve", (0, "基线译文可以接受", None))], "approved", 0))

    review = build(full)

    assert review["flags"] == []
    assert review["issues"] == [issue_of(0, "基线译文可以接受", None, "approve", 1, True)]


def test_a_full_no_kb_run_has_a_review_too():
    full_no_kb = row(12, "full_no_kb", record([translated(), reviewed("approve")], "approved", 0, config="full_no_kb"))

    assert build(full_no_kb)["config"] == "full_no_kb"


@pytest.mark.parametrize("attempts, config", [
    ([{"request": "…", "response": "…"}], "baseline"),  # a baseline Run's record is a list: one attempt per call
    ({"config": "translator", "agent": {"steps": []}, "repair": None, "fallback": None}, "translator"),
], ids=["baseline", "translator"])
def test_a_baseline_run_and_a_translator_run_have_no_review(attempts, config):
    assert build(row(1, config, attempts)) is None


def test_a_page_with_no_succeeded_run_has_no_review():
    assert build_review(None, lookup()) is None


# --- The builder: a person's edits ---------------------------------------------------------------------------------------


def test_two_human_runs_over_a_full_run_editing_different_regions_edit_both_and_their_notes_are_not_current():
    full, first, second = row(448, "full", SENT_BACK_ONCE), human(449, 448, 8), human(450, 449, 3)

    review = build(second, first, full)

    assert (review["run_id"], review["config"]) == (448, "full")  # the Orchestrator's Run, not the person's
    assert review["edited"] == [3, 8]
    assert [(i["region"], i["review"], i["current"]) for i in review["issues"]] == [
        (9, 1, False), (3, 2, False), (6, 2, True), (8, 2, False)]
    assert [(f["region"], f["current"]) for f in review["flags"]] == [(9, False)]
    assert build(first, full)["edited"] == [8]  # the page before the second edit, as the one edit left it


def test_an_edit_makes_a_current_flag_of_that_region_not_current_and_leaves_the_others():
    full = row(5, "full", record([translated((1, "存疑"), (2, "也存疑")), reviewed("approve")], "approved", 0))

    review = build(human(6, 5, 2), full)

    assert review["flags"] == [flag_of(1, "存疑", 0, True), flag_of(2, "也存疑", 0, False)]


def test_one_human_run_that_edited_several_regions_lists_them_sorted_and_once():
    full = row(5, "full", record([translated(), reviewed("approve", (4, "c", None))], "approved", 0))
    rename = row(6, "human", {"config": "human", "base_run_id": 5, "edits": [
        {"idx": 7, "before": "a", "after": "b"}, {"idx": 4, "before": "a", "after": "b"},
        {"idx": 7, "before": "b", "after": "c"}]})

    review = build(rename, full)

    assert review["edited"] == [4, 7]
    assert review["issues"][0]["current"] is False


def test_a_region_that_has_no_notes_is_still_listed_as_edited():
    full = row(5, "full", record([translated(), reviewed("approve")], "approved", 0))

    assert build(human(6, 5, 1), full)["edited"] == [1]


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_a_human_run_over_a_baseline_run_has_no_review_however_many_human_runs_are_on_it(depth):
    baseline = row(1, "baseline", [{"attempt": 1}])
    chain = [baseline]
    for n in range(depth):
        chain.append(human(2 + n, 1 + n, n))

    assert build(chain[-1], *chain[:-1]) is None


def test_a_human_run_over_a_translator_run_has_no_review():
    translator = row(1, "translator", {"config": "translator"})

    assert build(human(2, 1, 0), translator) is None


def test_a_chain_that_does_not_reach_a_run_has_no_review():
    full = row(5, "full", record([translated(), reviewed("approve")], "approved", 0))

    assert build(human(7, 6, 0), human(6, 99, 1)) is None  # Run 99 is not there
    assert build(row(7, "human", {"config": "human", "edits": []}), full) is None  # no base_run_id
    assert build(row(7, "human", None), full) is None  # no record at all


def test_a_chain_that_leads_back_to_itself_ends_rather_than_looping():
    assert build(human(7, 6, 0), human(6, 7, 1)) is None
    assert build(human(7, 7, 0)) is None


UNREADABLE = {  # the `attempts_json` of a `full` Run that the builder cannot read
    "null": None,
    "not-json": "not json",
    "list": json.dumps(["a list"]),
    "empty": json.dumps({}),
    "no-stages": json.dumps({"config": "full"}),
    "stage-out-of-range": json.dumps(
        {"config": "full", "stages": [], "send_backs": 0, "final": {"stage": 0, "stop": "approved"}}),
    "failed-run": json.dumps({"config": "full", "stages": [translated()], "send_backs": 0,
                              "final": {"stage": None, "kind": None, "stop": "failed"}}),
    "no-stop": json.dumps({"config": "full", "stages": [translated()], "send_backs": 0, "final": {"stage": 0}}),
}


@pytest.mark.parametrize("attempts_json", UNREADABLE.values(), ids=UNREADABLE.keys())
def test_a_record_that_cannot_be_read_has_no_review_and_is_no_error(attempts_json):
    broken = {"id": 3, "config": "full", "attempts_json": attempts_json}

    assert build_review(broken, lookup()) is None


def test_an_entry_that_names_no_region_is_left_out():
    odd = record([translated(), reviewed("approve")], "approved", 0)
    odd["stages"][0]["flags"] = [{"reason": "no id"}, "text", {"id": True, "reason": "a boolean"}, {"id": 2, "reason": "ok"}]
    odd["stages"][1]["issues"] = [None, {"id": "3", "comment": "a string id"}, {"id": 4, "comment": "ok"}]

    review = build(row(3, "full", odd))

    assert [f["region"] for f in review["flags"]] == [2]
    assert [i["region"] for i in review["issues"]] == [4] and review["issues"][0]["suggestion"] is None


def test_the_builder_is_given_a_sqlite_row_as_well_as_a_dict(imported):
    """`api/pages.py` hands it `sqlite3.Row`s; the builder reads `id`, `config` and `attempts_json` by name only."""
    page_id = store.find_page(imported, BOOK, 3)["id"]
    run_id = store.start_run(imported, page_id=page_id, config="full", params={}, model="m")
    store.finish_run(imported, run_id, error=None, response_models=[], usage=Meter().summary(),
                     attempts=record([translated((1, "存疑")), reviewed("approve")], "approved", 0), translations={})

    review = build_review(store.get_run(imported, run_id), lambda rid: store.get_run(imported, rid))

    assert review["run_id"] == run_id and review["flags"] == [flag_of(1, "存疑", 0, True)]


# --- What the builder shares with the modules that write the records ----------------------------------------------------


def test_the_words_of_the_records_are_the_ones_the_writing_modules_use():
    assert review_notes.HUMAN == store.HUMAN == runs.HUMAN
    assert set(review_notes.ORCHESTRATOR_CONFIGS) == {orchestrator.CONFIG_NAME, orchestrator.KB_CONFIG_NAME}
    assert (review_notes.TRANSLATE, review_notes.REVIEW, review_notes.REVISE, review_notes.OK) == (
        orchestrator.TRANSLATE, orchestrator.REVIEW, orchestrator.REVISE, orchestrator.OK)


def test_the_builder_loads_no_database_and_imports_neither_sqlite3_nor_store():
    code = ("import fukidashi.review_notes, sys; "
            "print([n for n in ('fukidashi.store', 'fukidashi.db', 'sqlite3') if n in sys.modules])")
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert ast.literal_eval(done.stdout.strip()) == []  # a fresh interpreter: what this process loaded does not count
    tree = ast.parse(Path(review_notes.__file__).read_text(encoding="utf-8"))
    modules = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    modules |= {(n.module or "") + "." + a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not [m for m in modules if m.split(".")[0] == "sqlite3" or "store" in m.split(".")]


# --- The API: Runs the real Orchestrator made on a scripted provider --------------------------------------------------------

REVISE_ONE = verdict_json("revise", issue(1, "第二句译得生硬", "更自然的第二句"))
APPROVE_TWO = verdict_json("approve", issue(0, "第一句可以更口语", None), issue(1, "第二句已经自然", None))


@pytest.fixture
def client(imported, root, db_path):
    """The API on `book_t`'s database (the Runs these tests make are read by their own connections)."""

    def override_db():
        c = connect(db_path)
        try:
            yield c
        finally:
            c.close()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_db_path] = lambda: db_path
    app.dependency_overrides[get_dataset_root] = lambda: root
    yield TestClient(app)
    app.dependency_overrides.clear()


def full_no_kb_run(conn, root, *replies, page_index=3):
    """A `full_no_kb` Run on book_t's page through the real Orchestrator; the provider answers with `replies`."""
    outcome = translate(conn, root, mock_llm_client(Provider(*replies)), page_index, config="full_no_kb")
    assert outcome.result.ok
    return outcome


def sent_back_once(conn, root):
    """Flag on region 1, a send-back about region 1, a revision that flags region 0, then an approval with a note on each."""
    return full_no_kb_run(
        conn, root, ask(flag("f1", 1, "初译不确定")), answer(zh_json(3)), answer(REVISE_ONE),
        ask(flag("f2", 0, "改后存疑")), answer(zh_json(3, "改")), answer(APPROVE_TWO))


def edit(client, region, text, *, page=3):
    return client.put(f"/api/pages/{BOOK}/{page}/segments/{region}/translation", json={"text_zh_hans": text})


def test_the_page_answers_with_the_review_the_orchestrator_stored(imported, root, client):
    outcome = sent_back_once(imported, root)

    body = client.get(f"/api/pages/{BOOK}/3").json()

    assert body["run"]["id"] == outcome.run_id and body["review"] == {
        "run_id": outcome.run_id, "config": "full_no_kb", "send_backs": 1, "final_stop": "approved", "edited": [],
        "flags": [flag_of(1, "初译不确定", 0, False), flag_of(0, "改后存疑", 1, True)],
        "issues": [
            issue_of(1, "第二句译得生硬", "更自然的第二句", "revise", 1, False),
            issue_of(0, "第一句可以更口语", None, "approve", 2, True),
            issue_of(1, "第二句已经自然", None, "approve", 2, True),
        ],
    }
    assert [s["text_zh_hans"] for s in body["segments"]] == [zh(3, 0, "改"), zh(3, 1, "改")]


def test_the_edit_answers_with_the_review_and_the_edited_regions_notes_are_not_current(imported, root, client):
    outcome = sent_back_once(imported, root)

    resp = edit(client, 0, "我改的第一句")

    assert resp.status_code == 200
    body = resp.json()
    assert body["run"]["config"] == "human" and body["run"]["id"] != outcome.run_id
    review = body["review"]
    assert (review["run_id"], review["config"], review["edited"]) == (outcome.run_id, "full_no_kb", [0])
    assert review["flags"] == [flag_of(1, "初译不确定", 0, False), flag_of(0, "改后存疑", 1, False)]  # region 0's: edited
    assert review["issues"] == [
        issue_of(1, "第二句译得生硬", "更自然的第二句", "revise", 1, False),
        issue_of(0, "第一句可以更口语", None, "approve", 2, False),  # region 0's: edited
        issue_of(1, "第二句已经自然", None, "approve", 2, True),  # region 1's: still about the stored translation
    ]
    assert client.get(f"/api/pages/{BOOK}/3").json() == body  # a reload shows the same page


def test_a_second_edit_through_the_api_adds_its_region_to_the_edited_ones(imported, root, client):
    outcome = sent_back_once(imported, root)

    edit(client, 0, "第一句")
    review = edit(client, 1, "第二句").json()["review"]

    assert (review["run_id"], review["edited"]) == (outcome.run_id, [0, 1])
    assert [n["current"] for n in review["flags"] + review["issues"]] == [False] * 5


def test_an_edit_with_the_current_text_records_no_run_and_the_review_is_as_before(imported, root, client):
    sent_back_once(imported, root)
    before = client.get(f"/api/pages/{BOOK}/3").json()

    resp = edit(client, 1, zh(3, 1, "改"))  # the text it already has

    assert resp.status_code == 200 and resp.json() == before and before["review"]["edited"] == []


def test_a_newer_run_supersedes_the_edit_and_the_review_is_the_newer_runs(imported, root, client):
    sent_back_once(imported, root)
    edit(client, 0, "我改的第一句")

    newer = full_no_kb_run(imported, root, answer(zh_json(3, "新")), answer(APPROVE))
    review = client.get(f"/api/pages/{BOOK}/3").json()["review"]

    assert review == {"run_id": newer.run_id, "config": "full_no_kb", "send_backs": 0, "final_stop": "approved",
                      "edited": [], "flags": [], "issues": []}


def test_a_page_with_no_succeeded_run_answers_with_no_review(imported, client):
    body = client.get(f"/api/pages/{BOOK}/1").json()

    assert body["run"] is None and body["review"] is None


@pytest.mark.parametrize("config", ["baseline", "translator"])
def test_a_baseline_or_translator_run_answers_with_no_review_and_so_does_an_edit_of_it(imported, root, client, config):
    outcome = translate(imported, root, mock_llm_client(Provider(answer(zh_json(2)))), 2, config=config)
    assert outcome.result.ok

    shown = client.get(f"/api/pages/{BOOK}/2").json()
    edited = edit(client, 0, "我改的", page=2).json()

    assert shown["run"]["config"] == config and shown["review"] is None
    assert edited["run"]["config"] == "human" and edited["review"] is None


def test_the_review_is_part_of_the_pydantic_model_with_the_key_pass(imported, root, client):
    sent_back_once(imported, root)

    flags = client.get(f"/api/pages/{BOOK}/3").json()["review"]["flags"]

    assert set(flags[0]) == {"region", "reason", "pass", "current"}  # `pass`, not `pass_`
