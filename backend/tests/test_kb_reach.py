"""How often the Knowledge Base reaches a page (task 018): the counts the eval harness reads from a stored `full` Run.

The Runs come from `runs.translate_page` on a scripted provider (the real LLMClient, SDK and agent loop), and the counts
are read back from the database, as the harness reads them.
"""

from __future__ import annotations

import json

from kb_support import term
from test_agent_loop import Provider, answer, ask, call
from test_curator_agent import decision, decisions_json
from test_full import CONFIG, kb, run_page, seed_page_three, who  # noqa: F401 (fixture)
from test_kb_tools import lookup_character, lookup_term, propose
from test_reviewer import APPROVE, issue, verdict_json
from test_translator import attempts_of, evaluate, imported, root, zh_json  # noqa: F401 (fixtures)

from fukidashi import store
from fukidashi.agents.kb_tools import lookup_tools
from fukidashi.evaluation import harness
from fukidashi.kb.loader import load_snapshot

REVISE = verdict_json("revise", issue(0))


def counts(calls, found):
    return {"calls": calls, "found": found, "not_found": calls - found}


NO_LOOKUPS = {agent: {tool: counts(0, 0) for tool in ("lookup_term", "lookup_character")}
              for agent in ("translator", "reviewer")}


def reach_of(conn, run_id):
    return harness._kb_reach(store.get_run(conn, run_id))


def test_the_lookup_names_and_result_keys_the_harness_reads_are_the_tools_own(imported, kb):
    seed_page_three(imported, kb)
    tools = {t.name: t for t in lookup_tools(load_snapshot(imported, kb))}

    assert set(tools) == set(harness.LOOKUP_TOOLS)
    assert tools["lookup_term"].handler("日本語")[harness.LOOKUP_TOOLS["lookup_term"][1]] == "日本語"
    assert tools["lookup_character"].handler("3-0")[harness.LOOKUP_TOOLS["lookup_character"][1]] == "ユキ"  # an alias


# A `full` page with every kind of lookup call, for both agents and for a Revision. KB: 日本語 (on the page, approved),
# 3-1 (on it, pending), ユキ with the alias 3-0 (on it), and off the page 宇宙船 and ハル. The conversations:
#   Translator   a reply with seven calls: three that ran and found something (a term on the page, a term off the page,
#                a name off the page), one that ran and found nothing, a repeat of the first, a call with invalid
#                arguments and a call to a tool that does not exist; then the answer
#   Reviewer     a term found and a name not found; asks for a revision
#   Revision     the same term found again (a new conversation: not a repeat) and a term not found; then the answer
#   Reviewer     the name found by an alias; asks for a revision again
#   Revision     a reply cut off in the middle of a lookup call: it never ran, and the revision is unusable
def lookup_everything():
    return Provider(
        ask(lookup_term("t1", "日本語"), lookup_term("t2", "宇宙船"), lookup_character("t3", "ハル"),
            lookup_term("t4", "未知"), lookup_term("t5", "日本語"), call("t6", {}, name="lookup_term"),
            call("t7", {"name": "x"}, name="lookup_place")),
        answer(zh_json(3)),
        ask(lookup_term("r1", "日本語"), lookup_character("r2", "ナシ")), answer(REVISE),
        ask(lookup_term("v1", "日本語"), lookup_term("v2", "未知")), answer(zh_json(3, "改")),
        ask(lookup_character("r3", "3-0")), answer(REVISE),
        ask(lookup_term("v3", "未知"), finish_reason="length"),
    )


def test_the_lookups_that_ran_are_counted_for_each_agent_and_tool_found_and_not_found(imported, root, kb):
    seed_page_three(imported, kb)
    provider = lookup_everything()

    outcome, _ = run_page(imported, root, provider, kb)

    assert outcome.result.ok and outcome.result.final_stop == "revision_unusable"
    assert [who(b) for b in provider.bodies] == [
        "translate", "translate", "review", "review", "revise", "revise", "review", "review", "revise"]
    attempts = attempts_of(imported, outcome.run_id)
    assert [s["kind"] for s in attempts["stages"]] == ["translate", "review", "revise", "review", "revise"]
    first_calls = attempts["stages"][0]["attempts"]["agent"]["steps"][0]["tool_calls"]
    assert [(c["id"], c["ran"], c["duplicate"], c["error"] is not None) for c in first_calls] == [
        ("t1", True, False, False), ("t2", True, False, False), ("t3", True, False, False),
        ("t4", True, False, False), ("t5", False, True, False),    # the repeat
        ("t6", False, False, True), ("t7", False, False, True)]    # invalid arguments; no such tool
    (unran,) = attempts["stages"][4]["attempts"]["agent"]["steps"][0]["tool_calls"]
    assert (unran["ran"], unran["duplicate"], unran["result"], unran["error"]) == (False, False, None, None)

    lookups = reach_of(imported, outcome.run_id)["lookups"]
    assert lookups == {
        "translator": {"lookup_term": counts(5, 3),        # first pass: 3 ran (2 found); revision: 2 ran (1 found)
                       "lookup_character": counts(1, 1)},
        "reviewer": {"lookup_term": counts(1, 1), "lookup_character": counts(2, 1)}}   # not the 7 + 1 calls made


def test_the_items_that_reached_the_page_are_distinct_snippets_first_then_the_lookups_that_found_something(
        imported, root, kb):
    seed_page_three(imported, kb)

    outcome, _ = run_page(imported, root, lookup_everything(), kb)

    reach = reach_of(imported, outcome.run_id)
    assert reach["snippets"] == {"items": 3, "by_kind": {"term": 2, "character": 1},
                                 "by_status": {"approved": 2, "pending": 1}}
    assert reach["reached"] == [   # 日本語 was shown and found twice, ユキ was shown and found by its alias: once each
        {"kind": "term", "key": "日本語"}, {"kind": "character", "key": "ユキ"}, {"kind": "term", "key": "3-1"},
        {"kind": "term", "key": "宇宙船"}, {"kind": "character", "key": "ハル"}]


def test_a_page_the_kb_did_not_reach_has_zero_counts_and_no_items(imported, root, kb):
    provider = Provider(answer(zh_json(3)), answer(APPROVE))

    outcome, _ = run_page(imported, root, provider, kb)

    assert reach_of(imported, outcome.run_id) == {
        "snippets": {"items": 0, "by_kind": {"term": 0, "character": 0}, "by_status": {"approved": 0, "pending": 0}},
        "lookups": NO_LOOKUPS, "reached": []}


def test_a_run_stored_before_snippets_were_recorded_has_unknown_snippets_but_countable_lookups(imported, root, kb):
    seed_page_three(imported, kb)
    outcome, _ = run_page(imported, root, lookup_everything(), kb)
    before = reach_of(imported, outcome.run_id)
    attempts = attempts_of(imported, outcome.run_id)
    del attempts["kb_snippets"]   # as a Run of the M3 evals has it
    imported.execute("UPDATE runs SET attempts_json = ? WHERE id = ?", (json.dumps(attempts), outcome.run_id))

    after = reach_of(imported, outcome.run_id)

    assert after["snippets"] is None   # unknown, not 0
    assert after["lookups"] == before["lookups"]
    assert after["reached"] == [item("日本語"), item("宇宙船"), item("ハル", "character"),
                                item("ユキ", "character")]   # what the found lookups reached, in their order


def row(snippets, reached, lookups=NO_LOOKUPS):
    return {"kb_reach": {"snippets": snippets, "lookups": lookups, "reached": reached}}


def snippets_of(n):
    return {"items": n, "by_kind": {"term": n, "character": 0}, "by_status": {"approved": n, "pending": 0}}


def item(key, kind="term"):
    return {"kind": kind, "key": key}


def test_the_totals_count_an_item_once_however_many_pages_it_reached_and_unknown_pages_apart_from_empty_ones():
    pages = [row(snippets_of(2), [item("a"), item("b")]),
             row(snippets_of(1), [item("a")]),   # a again, on a second page
             row(snippets_of(0), []),
             row(None, [item("b"), item("a", "character")]),   # unknown snippets; a name called a is another item
             row(snippets_of(0), [])]
    reviewer = NO_LOOKUPS["reviewer"]
    pages[0]["kb_reach"]["lookups"] = {**NO_LOOKUPS, "reviewer": {**reviewer, "lookup_term": counts(2, 1)}}
    pages[3]["kb_reach"]["lookups"] = {**NO_LOOKUPS, "reviewer": {**reviewer, "lookup_term": counts(3, 3)},
                                       "translator": {**NO_LOOKUPS["translator"], "lookup_character": counts(1, 0)}}

    totals = harness._kb_reach_totals(pages)

    assert totals["snippet_pages"] == {"nonempty": 2, "empty": 2, "unknown": 1}   # not "3 pages had none"
    assert totals["items_reached"] == 3   # a, b, and the character a
    assert totals["lookups"] == {
        "translator": {"lookup_term": counts(0, 0), "lookup_character": counts(1, 0)},
        "reviewer": {"lookup_term": counts(5, 4), "lookup_character": counts(0, 0)}}


def test_the_totals_of_no_pages_are_zeros():
    assert harness._kb_reach_totals([]) == {
        "snippet_pages": {"nonempty": 0, "empty": 0, "unknown": 0}, "lookups": NO_LOOKUPS, "items_reached": 0}


# --- an eval: the rows, the totals and the summary line ---------------------------------------------------------------

def two_pages_where_the_second_one_reads_what_the_first_wrote(kb_id=1):
    """Page 1 writes a term, which the Curator accepts, and a name, which waits for a person. Page 2 meets the term in
    its text and looks the name up (a pending name is found) and a term up that is not there; its Reviewer looks the
    term up."""
    return Provider(
        ask(propose("a", "term", source_term="日本語", target_term="日语"),
            propose("b", "character", name_src="ユキ", name_tgt="小雪")),
        answer(zh_json(1)), answer(APPROVE),
        answer(decisions_json(decision(term(kb_id, 1, "日本語", "日语")))),
        ask(lookup_character("c", "ユキ"), lookup_term("d", "未知")), answer(zh_json(2)),
        ask(lookup_term("e", "日本語")), answer(APPROVE))


def test_an_eval_reports_the_reach_per_page_in_total_and_in_one_summary_line(imported, root, tmp_path):
    result = evaluate(imported, root, tmp_path, two_pages_where_the_second_one_reads_what_the_first_wrote(), [1, 2],
                      config=CONFIG)

    first, second = result.scores["pages"]
    assert first["kb_reach"] == {
        "snippets": {"items": 0, "by_kind": {"term": 0, "character": 0}, "by_status": {"approved": 0, "pending": 0}},
        "lookups": NO_LOOKUPS, "reached": []}   # the first page's KB was empty
    assert second["kb_reach"] == {
        "snippets": {"items": 1, "by_kind": {"term": 1, "character": 0}, "by_status": {"approved": 1, "pending": 0}},
        "lookups": {"translator": {"lookup_term": counts(1, 0), "lookup_character": counts(1, 1)},
                    "reviewer": {"lookup_term": counts(1, 1), "lookup_character": counts(0, 0)}},
        "reached": [item("日本語"), item("ユキ", "character")]}
    on_disk = json.loads((result.out_dir / "scores.json").read_text(encoding="utf-8"))
    assert on_disk["kb_reach"] == result.scores["kb_reach"] == {
        "snippet_pages": {"nonempty": 1, "empty": 1, "unknown": 0},
        "lookups": {"translator": {"lookup_term": counts(1, 0), "lookup_character": counts(1, 1)},
                    "reviewer": {"lookup_term": counts(1, 1), "lookup_character": counts(0, 0)}},
        "items_reached": 2}
    assert on_disk["pages"][1]["kb_reach"] == second["kb_reach"]
    assert on_disk["kb"]["entries"]["approved"] == 1 and on_disk["kb"]["profiles"]["pending_human"] == 1   # KB size

    lines = harness.format_summary(result).splitlines()
    (reach_line,) = [line for line in lines if line.startswith("kb reach")]
    assert reach_line == ("kb reach  snippets on 1 of 2 pages; lookups found 2, not found 1; "
                          "2 items reached a later page")
    assert lines.index(reach_line) == [i for i, line in enumerate(lines) if line.startswith("knowledge base")][0] + 1


def test_the_summary_line_says_how_many_pages_were_unknown_rather_than_counting_them_as_empty(
        imported, root, tmp_path):
    result = evaluate(imported, root, tmp_path, two_pages_where_the_second_one_reads_what_the_first_wrote(), [1, 2],
                      config=CONFIG)

    result.scores["kb_reach"]["snippet_pages"] = {"nonempty": 1, "empty": 0, "unknown": 1}
    assert "snippets on 1 of 1 pages (1 unknown);" in harness.format_summary(result)
    result.scores["kb_reach"]["snippet_pages"] = {"nonempty": 0, "empty": 0, "unknown": 2}
    assert "kb reach  snippets unknown on 2 pages;" in harness.format_summary(result)


def test_the_other_configs_have_no_kb_reach(imported, root, tmp_path):
    reviewed = evaluate(imported, root, tmp_path, Provider(answer(zh_json(1)), answer(APPROVE)), [1],
                        config="full_no_kb")

    assert "kb_reach" not in reviewed.scores and all("kb_reach" not in p for p in reviewed.scores["pages"])
    assert "kb reach" not in harness.format_summary(reviewed)
