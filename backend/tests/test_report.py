"""The report that pools finished eval runs (OUTLINE 4.3, 4.4): `evaluation/report.py` and eval/report.py, on made-up runs.

Nothing here needs data/, the database or a model. `make_eval` writes an eval run's directory the way the harness and the
two scorers leave it; the numbers in the fixtures are small enough to work out by hand, and where a test cannot do that
(chrF) it compares with `score_corpus` over the pooled segments, which is what the report is meant to equal.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from datetime import datetime, timezone

import pytest

from fukidashi import config, runs
from fukidashi.agents import curator as curator_agent
from fukidashi.evaluation import harness, report
from fukidashi.evaluation.metrics import cost_summary, latency_summary, score_corpus
from fukidashi.kb import curator as curator_rules
from fukidashi.orchestrator import flow

A, B = "tojime_no_siora", "boureisougi"  # the two dev books; the glossaries here are made-up files
DROP = object()  # in a page's `row`: the field is not there (what an older run's file looks like)
FINISHED = "2026-10-29T00:01:00+00:00"
COMET = report.COMET_MODEL
SNAPSHOT = "2760a223ac957f30acfb18c8aa649b01cf1d75f2"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edit_json(path, change):
    data = read_json(path)
    change(data)
    write_json(path, data)


def outcomes(**counts):
    return dict.fromkeys(report.PROPOSAL_RESULTS, 0) | counts


def kb_reach(snippet_items, reached=(), lookups=None):
    """A page row's `kb_reach` as the harness writes it. `lookups` is {(agent, tool): (calls, found, not_found)}."""
    counts = {agent: {tool: {"calls": 0, "found": 0, "not_found": 0} for tool in report.LOOKUP_TOOLS}
              for agent in report.LOOKUP_AGENTS}
    for (agent, tool), (calls, found, not_found) in (lookups or {}).items():
        counts[agent][tool] = {"calls": calls, "found": found, "not_found": not_found}
    snippets = None if snippet_items is None else {
        "items": snippet_items, "by_kind": {"term": snippet_items, "character": 0},
        "by_status": {"approved": snippet_items, "pending": 0}}
    return {"snippets": snippets, "lookups": counts, "reached": [{"kind": k, "key": key} for k, key in reached]}


def default_row(config_name):
    """The fields of a scored page's row that a run of `config_name` has."""
    row = {"calls": 1, "unknown_usage_calls": 0, "cost_usd": 0.01, "latency_s": 2.0, "trace_id": None,
           "fell_back": False, "fallback_reason": None}
    if config_name in report.REVIEWED_CONFIGS:
        row |= {"send_backs": 0, "final_stop": "approved", "final_reviewed": True, "review_statuses": ["ok"],
                "review_verdicts": ["approve"]}
    if config_name == report.KB_CONFIG:
        row |= {"curator_status": "ok", "proposal_outcomes": outcomes(), "kb_reach": kb_reach(0)}
    return row


def page(index, status="scored", segments=(("甲", "甲"),), *, error=None, **row):
    """A page of an eval run: `segments` are (hyp, ref_zh_hans) pairs; `row` overrides its row in scores.json."""
    return {"index": index, "status": status, "segments": list(segments), "error": error, "row": row}


def make_glossary_dir(tmp_path, books=(A, B)):
    directory = tmp_path / "glossaries"
    directory.mkdir(exist_ok=True)
    for book in books:
        write_json(directory / f"{book}.json", {"book_title": book, "entries": []})
    return directory


def make_eval(tmp_path, config_name, book, pages, *, glossary_dir, entries=(), comet=None, comet_reversed=False,
              git_commit="c0ffee", git_dirty=False, models=("deepseek-flash",), finished_at=FINISHED, revision=SNAPSHOT):
    """An eval run's directory as the harness, eval/terms.py and eval/comet.py leave it.

    `entries` are the glossary entries of the terms file: {"id", "borderline", "occurrences": [(page, idx, hit)]}.
    `comet` maps (page_index, idx) to the segment's COMET score (0.5 where it is not given); `comet_reversed` writes the
    COMET file's segments in the opposite order to segments.jsonl.
    """
    eval_id = f"20261029T000000Z-{config_name}-{book}"
    run_dir = tmp_path / eval_id
    run_dir.mkdir()
    manifest_pages, rows, failed, skipped, lines = [], [], [], [], []
    for p in pages:
        index, status, run_id = p["index"], p["status"], 1000 + p["index"]
        manifest_pages.append({"book": book, "page_index": index, "status": status,
                               "run_id": None if status == "skipped" else run_id, "trace_id": None})
        if status == "scored":
            row = {"book": book, "page_index": index, "run_id": run_id, "segments": len(p["segments"]),
                   **default_row(config_name), **p["row"]}
            rows.append({k: v for k, v in row.items() if v is not DROP})
            lines += [{"book": book, "page_index": index, "idx": idx, "run_id": run_id, "ja": f"日{index}-{idx}",
                       "hyp": hyp, "ref_zh_hans": ref, "ref_zh_hant": f"{ref}-hant"}
                      for idx, (hyp, ref) in enumerate(p["segments"])]
        elif status == "failed":
            failed.append({"book": book, "page_index": index, "run_id": run_id, "trace_id": None, "error": p["error"],
                           "fell_back": False, "fallback_reason": None, "latency_s": 1.0,
                           "cost_usd": p["row"].get("cost_usd", 0.0)})
        else:
            skipped.append({"book": book, "page_index": index, "reason": "no text"})
    write_json(run_dir / "manifest.json", {
        "eval_id": eval_id, "started_at": "2026-10-29T00:00:00+00:00", "finished_at": finished_at,
        "config_name": config_name, "config_params": {}, "book": book, "pages": manifest_pages,
        "annotation_sha256": "annotation", "requested_models": list(models), "response_models": list(models),
        "git_commit": git_commit, "git_dirty": git_dirty, "tracing": {"enabled": False}})
    write_json(run_dir / "scores.json", {
        "config": config_name, "book": book, "pages_scored": len(rows), "pages": rows, "failed_pages": failed,
        "skipped_pages": skipped})
    (run_dir / "segments.jsonl").write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines),
                                            encoding="utf-8")
    write_json(run_dir / "terms.json", {
        "eval_id": eval_id, "book": book, "glossary_sha256": sha256_of(glossary_dir / f"{book}.json"),
        "rules": "OUTLINE §4.7 (v0.22)", "scores": {},
        "entries": [{"id": e["id"], "borderline": e["borderline"], "occurrences": len(e["occurrences"]),
                     "segments": [{"page_index": pi, "idx": i, "hit": hit} for pi, i, hit in e["occurrences"]]}
                    for e in entries]})
    scored = [{"book": l["book"], "page_index": l["page_index"], "idx": l["idx"], "run_id": l["run_id"],
               "score": (comet or {}).get((l["page_index"], l["idx"]), 0.5)} for l in lines]
    write_json(run_dir / report.COMET_FILE, {
        "eval_id": eval_id, "model": COMET, "model_revision": revision, "segments_sha256": sha256_of(run_dir / "segments.jsonl"),
        "n_segments": len(scored), "system_score": 0.5, "segments": scored[::-1] if comet_reversed else scored,
        "versions": {"unbabel-comet": "2.2.7"}, "device": "cpu"})
    return run_dir


# --- a fixture of 2 books x 2 configs, small enough to work out by hand -----------------------------------------------
#
# Book A has pages 1-3 and 9 (9 has no text, so every config skipped it); book B has pages 1 and 2. `full_no_kb` failed
# page 3 of book A, so the common pages are A: 1, 2 and B: 1, 2, and `baseline`'s page 3 is left out of its numbers.

REFS = {(A, 1): ["今天天气很好", "我们一起去吧"], (A, 2): ["龙来了", "好的"], (A, 3): ["我在这里等你回来", "我也是这样想的"],
        (B, 1): ["他走了。"], (B, 2): ["你好吗"]}
BASELINE_HYPS = {(A, 1): REFS[(A, 1)], (A, 2): REFS[(A, 2)], (A, 3): ["完全不同的句子啊", "毫无关系的文字"],
                 (B, 1): ["嗯"], (B, 2): ["啊"]}
FULL_NO_KB_HYPS = {(A, 1): ["今天天气不好", "我们一起去吧"], (A, 2): ["龙来了", "好吧"], (B, 1): ["他走了。"], (B, 2): ["你好"]}
BASELINE_COMET = {A: {(1, 0): 0.2, (1, 1): 0.4, (2, 0): 0.6, (2, 1): 0.8, (3, 0): 0.99, (3, 1): 0.99},
                  B: {(1, 0): 0.1, (2, 0): 0.5}}
FULL_NO_KB_COMET = {A: {(1, 0): 0.3, (1, 1): 0.5, (2, 0): 0.7, (2, 1): 0.9}, B: {(1, 0): 0.2, (2, 0): 0.6}}
TERMS = {  # (config, book) -> glossary entries; the baseline's page 3 occurrences are off the common pages
    ("baseline", A): [
        {"id": "ryu", "borderline": False, "occurrences": [(1, 0, True), (2, 0, True), (2, 1, False), (3, 0, True)]},
        {"id": "word", "borderline": True, "occurrences": [(1, 1, True), (3, 1, True)]},
        {"id": "ghost", "borderline": False, "occurrences": [(3, 0, False)]}],
    ("baseline", B): [{"id": "shiora", "borderline": False, "occurrences": [(1, 0, True), (2, 0, False)]}],
    ("full_no_kb", A): [
        {"id": "ryu", "borderline": False, "occurrences": [(1, 0, True), (2, 0, True), (2, 1, True)]},
        {"id": "word", "borderline": True, "occurrences": [(1, 1, False)]},
        {"id": "ghost", "borderline": False, "occurrences": []}],
    ("full_no_kb", B): [{"id": "shiora", "borderline": False, "occurrences": [(1, 0, True), (2, 0, True)]}],
}
BASELINE_COST = {(A, 1): 0.001, (A, 2): 0.002, (A, 3): 0.04, (B, 1): 0.003, (B, 2): 0.004}
BASELINE_LATENCY = {(A, 1): 1.0, (A, 2): 2.0, (A, 3): 50.0, (B, 1): 3.0, (B, 2): 4.0}
FULL_NO_KB_COST = {(A, 1): 0.011, (A, 2): 0.012, (B, 1): 0.013, (B, 2): 0.014}


def two_by_two(tmp_path, **options):
    """({(config, book): run dir}, glossary dir). `options` reach `make_eval` for every run."""
    glossary_dir = make_glossary_dir(tmp_path)
    dirs = {}
    for book, indexes in ((A, (1, 2, 3, 9)), (B, (1, 2))):
        def baseline_page(i):
            if i == 9:
                return page(9, "skipped")
            return page(i, segments=list(zip(BASELINE_HYPS[(book, i)], REFS[(book, i)])),
                        cost_usd=BASELINE_COST[(book, i)], latency_s=BASELINE_LATENCY[(book, i)])

        def full_page(i):
            if (book, i) == (A, 9):
                return page(9, "skipped")
            if (book, i) == (A, 3):
                return page(3, "failed", error="ProviderError: HTTP 500", cost_usd=0.02)
            return page(i, segments=list(zip(FULL_NO_KB_HYPS[(book, i)], REFS[(book, i)])),
                        cost_usd=FULL_NO_KB_COST[(book, i)], latency_s=10.0 * i)

        dirs["baseline", book] = make_eval(
            tmp_path, "baseline", book, [baseline_page(i) for i in indexes], glossary_dir=glossary_dir,
            entries=TERMS["baseline", book], comet=BASELINE_COMET[book], comet_reversed=True, **options)
        dirs["full_no_kb", book] = make_eval(
            tmp_path, "full_no_kb", book, [full_page(i) for i in indexes], glossary_dir=glossary_dir,
            entries=TERMS["full_no_kb", book], comet=FULL_NO_KB_COMET[book], **options)
    return dirs, glossary_dir


def build(dirs, glossary_dir):
    return report.build_report(list(dirs.values()) if isinstance(dirs, dict) else dirs, glossary_dir)


# --- the common pages ---------------------------------------------------------------------------------------------------

def test_the_common_pages_are_the_pages_every_config_scored_and_a_page_nobody_translated_is_not_listed(tmp_path):
    result = build(*two_by_two(tmp_path))

    assert result["common_pages"] == {A: [1, 2], B: [1, 2]}
    assert result["left_out"]["baseline"] == [{"book": A, "page_index": 3, "status": "scored"}]  # scored, but not by all
    assert result["left_out"]["full_no_kb"] == [{"book": A, "page_index": 3, "status": "failed",
                                                 "error": "ProviderError: HTTP 500"}]
    assert all(entry["page_index"] != 9 for entries in result["left_out"].values() for entry in entries)  # no text


def test_a_page_a_config_was_not_asked_to_translate_is_left_out_as_absent(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    dirs = [make_eval(tmp_path, "baseline", A, [page(1), page(2)], glossary_dir=glossary_dir),
            make_eval(tmp_path, "full_no_kb", A, [page(1)], glossary_dir=glossary_dir)]

    result = report.build_report(dirs, glossary_dir)

    assert result["common_pages"] == {A: [1]}
    assert result["left_out"]["baseline"] == [{"book": A, "page_index": 2, "status": "scored"}]
    assert result["left_out"]["full_no_kb"] == [{"book": A, "page_index": 2, "status": "absent"}]


def test_a_page_that_failed_in_one_config_is_left_out_of_every_configs_numbers(tmp_path):
    result = build(*two_by_two(tmp_path))

    for config_name in ("baseline", "full_no_kb"):
        assert result["pooled"][config_name]["pages"] == 4 and result["pooled"][config_name]["segments"] == 6
        assert result["by_book"][A][config_name]["pages"] == 2 and result["by_book"][A][config_name]["segments"] == 4
    # baseline's own page 3 (4 segments of its worst text) does not count, so its chrF is that of pages 1-2 and book B
    common = [(A, 1), (A, 2), (B, 1), (B, 2)]
    expected = score_corpus([h for k in common for h in BASELINE_HYPS[k]], [r for k in common for r in REFS[k]])
    everything = score_corpus([h for k in BASELINE_HYPS for h in BASELINE_HYPS[k]], [r for k in REFS for r in REFS[k]])
    assert result["pooled"]["baseline"]["chrf"]["score"] == expected.chrf != everything.chrf


# --- pooling -------------------------------------------------------------------------------------------------------------

def test_chrf_and_bleu_are_recomputed_from_the_pooled_segments_not_averaged_over_books(tmp_path):
    result = build(*two_by_two(tmp_path))

    for config_name, hyps in (("baseline", BASELINE_HYPS), ("full_no_kb", FULL_NO_KB_HYPS)):
        common = [(A, 1), (A, 2), (B, 1), (B, 2)]
        pooled = score_corpus([h for k in common for h in hyps[k]], [r for k in common for r in REFS[k]])
        per_book = {book: score_corpus([h for k in common if k[0] == book for h in hyps[k]],
                                       [r for k in common if k[0] == book for r in REFS[k]]) for book in (A, B)}
        got = result["pooled"][config_name]
        assert got["chrf"] == {"score": pooled.chrf, "signature": pooled.chrf_signature}
        assert got["bleu"] == {"score": pooled.bleu, "signature": pooled.bleu_signature}
        for book in (A, B):
            assert result["by_book"][book][config_name]["chrf"]["score"] == per_book[book].chrf
            assert result["by_book"][book][config_name]["bleu"]["score"] == per_book[book].bleu
        assert abs(got["chrf"]["score"] - (per_book[A].chrf + per_book[B].chrf) / 2) > 1  # the mean of books is not it
        assert abs(got["bleu"]["score"] - (per_book[A].bleu + per_book[B].bleu) / 2) > 1


def test_comet_is_the_mean_of_the_segment_scores_on_the_common_pages_pooled_over_the_books(tmp_path):
    result = build(*two_by_two(tmp_path))

    # baseline: A pages 1-2 .2 .4 .6 .8 (mean .5), B .1 .5 (mean .3); page 3's .99 is off the common pages.
    # Pooled over the 6 segments: 2.6 / 6, which is not the mean of the books' means (.4).
    baseline = result["pooled"]["baseline"]["comet"]
    assert (baseline["model"], baseline["revision"], baseline["n"]) == (COMET, SNAPSHOT, 6)
    assert baseline["mean"] == pytest.approx(2.6 / 6) and baseline["mean"] != pytest.approx(0.4)
    assert result["by_book"][A]["baseline"]["comet"]["mean"] == pytest.approx(0.5)
    assert result["by_book"][B]["baseline"]["comet"]["mean"] == pytest.approx(0.3)
    # full_no_kb: A .3 .5 .7 .9 (mean .6), B .2 .6 (mean .4): 3.2 / 6
    assert result["pooled"]["full_no_kb"]["comet"]["mean"] == pytest.approx(3.2 / 6)


def test_comet_is_joined_by_book_page_and_idx_whatever_the_order_of_the_two_files(tmp_path):
    """The fixture writes baseline's COMET segments in the opposite order to its segments.jsonl; a join by position would
    give A page 1 the scores of B's last page, and so on."""
    dirs, glossary_dir = two_by_two(tmp_path)
    reversed_scores = read_json(dirs["baseline", A] / report.COMET_FILE)["segments"]
    in_file_order = read_json(dirs["full_no_kb", A] / report.COMET_FILE)["segments"]
    assert [(s["page_index"], s["idx"]) for s in reversed_scores][:2] == [(3, 1), (3, 0)]
    assert [(s["page_index"], s["idx"]) for s in in_file_order][:2] == [(1, 0), (1, 1)]

    result = build(dirs, glossary_dir)

    assert result["by_book"][A]["baseline"]["comet"]["mean"] == pytest.approx(0.5)
    assert result["by_book"][A]["baseline"]["comet"]["n"] == 4


def test_term_accuracy_counts_only_the_occurrences_on_the_common_pages(tmp_path):
    result = build(*two_by_two(tmp_path))

    # baseline, primary (ryu, ghost, shiora): ryu 2 of 3 on pages 1-2 (its page 3 hit is off), ghost has none left,
    # shiora 1 of 2. Pooled 3/5; the mean over the two entries that have an occurrence is (2/3 + 1/2) / 2, not .6.
    primary = result["pooled"]["baseline"]["terms"]["primary"]
    assert (primary["entries"], primary["entries_with_occurrences"], primary["occurrences"], primary["hits"]) == (3, 2, 5, 3)
    assert primary["accuracy"] == pytest.approx(0.6)
    assert primary["mean_over_entries"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    # all entries add word (borderline): 1 of 1 on the common pages -> 4/6, mean (2/3 + 1/2 + 1) / 3
    every = result["pooled"]["baseline"]["terms"]["all_entries"]
    assert (every["entries"], every["entries_with_occurrences"], every["occurrences"], every["hits"]) == (4, 3, 6, 4)
    assert every["accuracy"] == pytest.approx(4 / 6) and every["mean_over_entries"] == pytest.approx((2 / 3 + 0.5 + 1) / 3)
    # per book: A's primary is ryu alone (ghost has no occurrence), B's is shiora alone
    assert result["by_book"][A]["baseline"]["terms"]["primary"]["accuracy"] == pytest.approx(2 / 3)
    assert result["by_book"][A]["baseline"]["terms"]["primary"]["entries_with_occurrences"] == 1
    assert result["by_book"][B]["baseline"]["terms"]["primary"]["accuracy"] == pytest.approx(0.5)
    # full_no_kb: ryu 3/3, shiora 2/2 -> 5/5; word 0/1
    assert result["pooled"]["full_no_kb"]["terms"]["primary"]["accuracy"] == 1.0
    assert result["pooled"]["full_no_kb"]["terms"]["all_entries"]["accuracy"] == pytest.approx(5 / 6)


def test_a_set_of_entries_with_no_occurrence_scores_none_not_zero(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    only_off_page = [{"id": "ghost", "borderline": False, "occurrences": [(3, 0, True)]}]
    run_dir = make_eval(tmp_path, "baseline", A, [page(1), page(2)], glossary_dir=glossary_dir, entries=only_off_page)

    primary = report.build_report([run_dir], glossary_dir)["pooled"]["baseline"]["terms"]["primary"]

    assert primary["accuracy"] is None and primary["mean_over_entries"] is None
    assert (primary["entries"], primary["entries_with_occurrences"], primary["occurrences"]) == (1, 0, 0)


def test_the_costs_and_the_latencies_are_those_of_the_common_pages(tmp_path):
    result = build(*two_by_two(tmp_path))

    # baseline: pages A1 A2 B1 B2 cost .001 .002 .003 .004 and took 1 2 3 4 s; page A3 (.04, 50 s) is off the common pages
    assert result["pooled"]["baseline"]["cost"] == cost_summary([(0.001, 0), (0.002, 0), (0.003, 0), (0.004, 0)])
    assert result["pooled"]["baseline"]["cost"]["total_usd"] == 0.01 and result["pooled"]["baseline"]["cost"]["per_page_usd"] == 0.0025
    assert result["pooled"]["baseline"]["latency"] == latency_summary([1.0, 2.0, 3.0, 4.0])
    assert (result["pooled"]["baseline"]["latency"]["n"], result["pooled"]["baseline"]["latency"]["p50_s"]) == (4, 2.5)
    assert result["by_book"][B]["baseline"]["cost"]["total_usd"] == 0.007


def test_spend_covers_all_of_a_configs_runs_the_failed_ones_and_the_off_common_pages_included(tmp_path):
    result = build(*two_by_two(tmp_path))

    # baseline: .001 + .002 + .04 + .003 + .004 over 5 runs; full_no_kb: .011 + .012 + .013 + .014 and the failed page's .02
    assert result["spend"]["baseline"] == cost_summary([(c, 0) for c in BASELINE_COST.values()]) | {"runs": 5, "failed_runs": 0}
    assert result["spend"]["baseline"]["total_usd"] == 0.05
    assert result["spend"]["full_no_kb"]["total_usd"] == 0.07
    assert (result["spend"]["full_no_kb"]["runs"], result["spend"]["full_no_kb"]["failed_runs"]) == (5, 1)


def test_a_failed_page_with_no_run_recorded_is_not_a_run_in_the_spend(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    run_dir = make_eval(tmp_path, "baseline", A, [page(1), page(2, "failed", error="LLMNotConfigured")],
                        glossary_dir=glossary_dir)
    edit_json(run_dir / "scores.json", lambda d: d["failed_pages"][0].update(run_id=None, cost_usd=None))

    spend = report.build_report([run_dir], glossary_dir)["spend"]["baseline"]

    assert (spend["total_usd"], spend["runs"], spend["failed_runs"]) == (0.01, 1, 0)


def test_one_unknown_cost_makes_the_cost_none_not_a_partial_sum(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    pages = [page(1), page(2, cost_usd=None, unknown_usage_calls=2), page(3)]
    run_dir = make_eval(tmp_path, "baseline", A, pages, glossary_dir=glossary_dir)

    result = report.build_report([run_dir], glossary_dir)

    cost = result["pooled"]["baseline"]["cost"]
    assert (cost["total_usd"], cost["per_page_usd"], cost["unknown_cost_runs"], cost["unknown_usage_calls"]) == (None, None, 1, 2)
    assert result["spend"]["baseline"]["total_usd"] is None
    assert "n/a (1 unknown)" in report.render(result)


def test_a_cost_that_has_a_price_but_a_call_without_usage_is_unknown_too(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    run_dir = make_eval(tmp_path, "baseline", A, [page(1, unknown_usage_calls=1)], glossary_dir=glossary_dir)

    cost = report.build_report([run_dir], glossary_dir)["pooled"]["baseline"]["cost"]

    assert cost["total_usd"] is None and cost["unknown_cost_runs"] == 1


def test_a_missing_field_is_unknown_not_zero(tmp_path):
    """An older run's page rows: no `fell_back` before the `translator` config, no `kb_reach` before #40."""
    glossary_dir = make_glossary_dir(tmp_path)
    old_baseline = make_eval(tmp_path, "baseline", A, [page(1, fell_back=DROP, fallback_reason=DROP), page(2)],
                             glossary_dir=glossary_dir)
    m3_full = make_eval(tmp_path, "full", A, [page(1, kb_reach=DROP), page(2, kb_reach=DROP)], glossary_dir=glossary_dir)

    result = report.build_report([old_baseline, m3_full], glossary_dir)

    assert result["pooled"]["baseline"]["fallback_pages"] is None  # one page lacks the field: the count is unknown
    assert result["pooled"]["full"]["fallback_pages"] == 0  # the other config has it on every page
    reach = result["pooled"]["full"]["kb_reach"]
    assert reach["snippet_pages"] == {"nonempty": 0, "empty": 0, "unknown": 2}
    assert reach["lookups"] is None and reach["items_reached"] is None
    text = report.render(result)
    assert "kb snippet pages unknown" in text and "kb lookups found" in text
    assert [l for l in text.splitlines() if l.startswith("kb items reached")][0].split() == ["kb", "items", "reached", "-", "n/a"]


def test_one_page_without_kb_reach_makes_the_lookups_unknown_and_the_known_pages_still_count(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    pages = [page(1, kb_reach=kb_reach(2, [("term", "a")], {("translator", "lookup_term"): (1, 1, 0)})),
             page(2, kb_reach=DROP)]
    run_dir = make_eval(tmp_path, "full", A, pages, glossary_dir=glossary_dir)

    reach = report.build_report([run_dir], glossary_dir)["pooled"]["full"]["kb_reach"]

    assert reach["snippet_pages"] == {"nonempty": 1, "empty": 0, "unknown": 1}
    assert reach["lookups"] is None and reach["items_reached"] is None


@pytest.mark.parametrize("separator", [" ", " ", "\u0085"], ids=["U+2028", "U+2029", "U+0085"])
def test_a_translation_with_a_unicode_line_separator_is_read_as_one_line(tmp_path, separator):
    """The harness leaves these characters unescaped in segments.jsonl; `splitlines()` would cut the line in two."""
    glossary_dir = make_glossary_dir(tmp_path)
    hyp = f"龙{separator}来了"
    run_dir = make_eval(tmp_path, "baseline", A, [page(1, segments=[(hyp, "龙来了"), ("好的", "好的")])],
                        glossary_dir=glossary_dir)
    assert separator in (run_dir / "segments.jsonl").read_text(encoding="utf-8")

    pooled = report.build_report([run_dir], glossary_dir)["pooled"]["baseline"]

    assert pooled["segments"] == 2
    assert pooled["chrf"]["score"] == score_corpus([hyp, "好的"], ["龙来了", "好的"]).chrf


# --- the review loop, the Curator, the KB's reach --------------------------------------------------------------------------

def full_set(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    lookups = {("translator", "lookup_term"): (3, 2, 1), ("reviewer", "lookup_character"): (1, 0, 1)}
    a_pages = [
        page(1, send_backs=0, final_stop="approved", curator_status="skipped", kb_reach=kb_reach(0, [], lookups)),
        page(2, send_backs=2, final_stop="send_back_limit", curator_status="ok",
             proposal_outcomes=outcomes(accepted=2, pending_human=1),
             kb_reach=kb_reach(2, [("term", "a"), ("term", "b")], {("translator", "lookup_term"): (1, 1, 0)})),
        page(3, send_backs=1, final_stop="approved", curator_status="failed",
             proposal_outcomes=outcomes(undecided=1, accepted=1),
             kb_reach=kb_reach(1, [("term", "a")], {("reviewer", "lookup_term"): (2, 2, 0)}))]
    b_pages = [page(1, send_backs=0, final_stop="review_unusable", curator_status="unusable",
                    proposal_outcomes=outcomes(rejected=1), kb_reach=kb_reach(None, [("term", "a")]))]
    return (glossary_dir, {A: make_eval(tmp_path, "full", A, a_pages, glossary_dir=glossary_dir),
                           B: make_eval(tmp_path, "full", B, b_pages, glossary_dir=glossary_dir)})


def test_the_review_loop_the_curator_and_the_kb_reach_are_summed_over_the_common_pages(tmp_path):
    glossary_dir, dirs = full_set(tmp_path)

    pooled = report.build_report(list(dirs.values()), glossary_dir)["pooled"]["full"]

    assert pooled["review_loop"]["final_stops"] == {
        "approved": 2, "review_unusable": 1, "revision_unusable": 0, "send_back_limit": 1}
    assert pooled["review_loop"]["send_backs"] == 3
    assert pooled["curator"]["statuses"] == {"ok": 1, "unusable": 1, "failed": 1, "skipped": 1}
    assert pooled["curator"]["proposal_outcomes"] == outcomes(accepted=3, pending_human=1, undecided=1, rejected=1)
    reach = pooled["kb_reach"]
    # snippets: A1 empty, A2 and A3 nonempty, B1 unknown (the Run's record is null). Items are distinct within a book
    # (A: a, b; B: a) and summed over books: 3, not 2.
    assert reach["snippet_pages"] == {"nonempty": 2, "empty": 1, "unknown": 1}
    assert (reach["lookups"]["found"], reach["lookups"]["not_found"], reach["lookups"]["calls"]) == (5, 2, 7)
    assert reach["lookups"]["by_agent"]["translator"]["lookup_term"] == {"calls": 4, "found": 3, "not_found": 1}
    assert reach["lookups"]["by_agent"]["reviewer"]["lookup_term"] == {"calls": 2, "found": 2, "not_found": 0}
    assert reach["lookups"]["by_agent"]["reviewer"]["lookup_character"] == {"calls": 1, "found": 0, "not_found": 1}
    assert reach["items_reached"] == 3


def test_the_kb_reach_per_book_counts_each_books_distinct_items(tmp_path):
    glossary_dir, dirs = full_set(tmp_path)

    by_book = report.build_report(list(dirs.values()), glossary_dir)["by_book"]

    assert by_book[A]["full"]["kb_reach"]["items_reached"] == 2 and by_book[B]["full"]["kb_reach"]["items_reached"] == 1
    assert by_book[B]["full"]["kb_reach"]["snippet_pages"] == {"nonempty": 0, "empty": 0, "unknown": 1}


def test_the_kb_reach_agrees_with_the_harnesss_own_totals_for_one_book(tmp_path):
    glossary_dir, dirs = full_set(tmp_path)
    rows = read_json(dirs[A] / "scores.json")["pages"]

    reach = report.build_report([dirs[A]], glossary_dir)["pooled"]["full"]["kb_reach"]

    totals = harness._kb_reach_totals(rows)
    assert reach["snippet_pages"] == totals["snippet_pages"]
    assert reach["lookups"]["by_agent"] == totals["lookups"]
    assert reach["items_reached"] == totals["items_reached"]


def test_a_config_without_a_review_loop_has_no_such_section_and_a_reviewed_one_without_a_curator_has_none(tmp_path):
    result = build(*two_by_two(tmp_path))

    assert "review_loop" not in result["pooled"]["baseline"] and "curator" not in result["pooled"]["full_no_kb"]
    assert result["pooled"]["full_no_kb"]["review_loop"]["final_stops"]["approved"] == 4
    assert "kb_reach" not in result["pooled"]["full_no_kb"]


def off_common_set(tmp_path):
    """Book A, pages 1-3, three configs. `baseline` failed page 3, so page 3 is off the common pages; `full_no_kb` and
    `full` scored it, with values that show if they are counted anyway."""
    glossary_dir = make_glossary_dir(tmp_path)
    dirs = [
        make_eval(tmp_path, "baseline", A, [page(1), page(2), page(3, "failed", error="E", cost_usd=0.5)],
                  glossary_dir=glossary_dir),
        make_eval(tmp_path, "full_no_kb", A, [
            page(1), page(2, fell_back=True, fallback_reason="step_limit", send_backs=1),
            page(3, fell_back=True, fallback_reason="step_limit", send_backs=2, final_stop="send_back_limit")],
            glossary_dir=glossary_dir),
        make_eval(tmp_path, "full", A, [
            page(1, proposal_outcomes=outcomes(accepted=1), kb_reach=kb_reach(1, [("term", "a")], {("translator", "lookup_term"): (1, 1, 0)})),
            page(2, curator_status="skipped", proposal_outcomes=outcomes(pending_human=2), kb_reach=kb_reach(0)),
            page(3, fell_back=True, send_backs=2, final_stop="send_back_limit", curator_status="failed",
                 proposal_outcomes=outcomes(undecided=7),
                 kb_reach=kb_reach(5, [("term", "x"), ("term", "y")], {("reviewer", "lookup_term"): (9, 9, 0)}))],
            glossary_dir=glossary_dir)]
    return dirs, glossary_dir


def test_the_review_loop_the_curator_and_the_kb_reach_leave_out_a_page_off_the_common_pages(tmp_path):
    dirs, glossary_dir = off_common_set(tmp_path)

    pooled = report.build_report(dirs, glossary_dir)["pooled"]

    assert pooled["full_no_kb"]["review_loop"] == {
        "final_stops": {"approved": 2, "review_unusable": 0, "revision_unusable": 0, "send_back_limit": 0}, "send_backs": 1}
    full = pooled["full"]
    assert full["review_loop"] == {
        "final_stops": {"approved": 2, "review_unusable": 0, "revision_unusable": 0, "send_back_limit": 0}, "send_backs": 0}
    assert full["curator"]["statuses"] == {"ok": 1, "unusable": 0, "failed": 0, "skipped": 1}
    assert full["curator"]["proposal_outcomes"] == outcomes(accepted=1, pending_human=2)
    assert full["kb_reach"]["snippet_pages"] == {"nonempty": 1, "empty": 1, "unknown": 0}
    assert (full["kb_reach"]["lookups"]["calls"], full["kb_reach"]["lookups"]["found"]) == (1, 1)
    assert full["kb_reach"]["items_reached"] == 1  # page 3's x and y are not counted
    assert [pooled[c]["pages"] for c in ("baseline", "full_no_kb", "full")] == [2, 2, 2]


def test_a_page_that_fell_back_is_counted_when_it_is_common_and_not_when_it_is_off_the_common_pages(tmp_path):
    dirs, glossary_dir = off_common_set(tmp_path)

    result = report.build_report(dirs, glossary_dir)

    # full_no_kb fell back on pages 2 (common) and 3 (off the common pages); full on page 3 only; baseline never
    assert [result["pooled"][c]["fallback_pages"] for c in ("baseline", "full_no_kb", "full")] == [0, 1, 0]
    row = [l for l in report.render(result).splitlines() if l.startswith("fallback to baseline")][0]
    assert row.split() == ["fallback", "to", "baseline", "(pages)", "0", "1", "0"]


def test_the_spend_of_all_runs_includes_the_page_off_the_common_pages_and_the_failed_run(tmp_path):
    dirs, glossary_dir = off_common_set(tmp_path)

    spend = report.build_report(dirs, glossary_dir)["spend"]

    assert spend["baseline"]["total_usd"] == 0.52 and (spend["baseline"]["runs"], spend["baseline"]["failed_runs"]) == (3, 1)
    assert spend["full_no_kb"]["total_usd"] == 0.03 and spend["full"]["total_usd"] == 0.03


def test_a_failed_run_whose_cost_is_unknown_makes_the_spend_unknown_not_a_cheaper_total(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    run_dir = make_eval(tmp_path, "baseline", A, [page(1), page(2, "failed", error="E", cost_usd=None)],
                        glossary_dir=glossary_dir)

    spend = report.build_report([run_dir], glossary_dir)["spend"]["baseline"]

    assert (spend["total_usd"], spend["unknown_cost_runs"], spend["runs"], spend["failed_runs"]) == (None, 1, 2, 1)


def test_a_review_loop_or_curator_field_a_row_lacks_is_unknown_not_a_default(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    no_loop = make_eval(tmp_path, "full_no_kb", A, [page(1, final_stop=DROP, send_backs=DROP), page(2)],
                        glossary_dir=glossary_dir)
    no_curator = make_eval(tmp_path, "full", A, [page(1, curator_status=DROP, proposal_outcomes=DROP), page(2)],
                           glossary_dir=glossary_dir)

    pooled = report.build_report([no_loop, no_curator], glossary_dir)["pooled"]

    assert pooled["full_no_kb"]["review_loop"] == {"final_stops": None, "send_backs": None}
    assert pooled["full"]["review_loop"]["final_stops"]["approved"] == 2
    assert pooled["full"]["curator"] == {"statuses": None, "proposal_outcomes": None}
    text = report.render(report.build_report([no_loop, no_curator], glossary_dir))
    assert [l for l in text.splitlines() if l.startswith("send-backs")][0].split() == ["send-backs", "n/a", "0"]


def test_a_stop_or_result_the_report_does_not_know_is_counted_so_that_the_counts_add_up(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    run_dir = make_eval(tmp_path, "full", A, [
        page(1, final_stop="a_new_stop", curator_status="a_new_status", proposal_outcomes=outcomes(a_new_result=2)),
        page(2)], glossary_dir=glossary_dir)

    full = report.build_report([run_dir], glossary_dir)["pooled"]["full"]

    assert full["review_loop"]["final_stops"] == {**dict.fromkeys(report.FINAL_STOPS, 0), "approved": 1, "a_new_stop": 1}
    assert full["curator"]["statuses"]["a_new_status"] == 1 and full["curator"]["statuses"]["ok"] == 1
    assert full["curator"]["proposal_outcomes"]["a_new_result"] == 2


def test_merged_is_pooled_and_a_run_stored_before_it_existed_counts_it_as_0(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    before_merged = {r: n for r, n in outcomes(pending_human=2).items() if r != "merged"}   # a row written before #026
    run_dir = make_eval(tmp_path, "full", A, [
        page(1, proposal_outcomes=outcomes(merged=2, conflict=1)), page(2, proposal_outcomes=before_merged)],
        glossary_dir=glossary_dir)

    result = report.build_report([run_dir], glossary_dir)

    assert "merged" in report.PROPOSAL_RESULTS
    assert result["pooled"]["full"]["curator"]["proposal_outcomes"] == outcomes(merged=2, conflict=1, pending_human=2)
    assert [l.split() for l in report.render(result).splitlines() if l.startswith("proposals merged")] == [
        ["proposals", "merged", "2"]]


# --- provenance -------------------------------------------------------------------------------------------------------------

def test_each_run_is_listed_with_its_provenance(tmp_path):
    dirs, glossary_dir = two_by_two(tmp_path, git_commit="abc123", models=("deepseek-flash",))

    runs_listed = build(dirs, glossary_dir)["runs"]

    assert [(r["config"], r["book"]) for r in runs_listed] == [("baseline", A), ("baseline", B), ("full_no_kb", A), ("full_no_kb", B)]
    first = runs_listed[0]
    assert first["eval_id"] == f"20261029T000000Z-baseline-{A}" and first["git_commit"] == "abc123"
    assert (first["git_dirty"], first["requested_models"], first["response_models"]) == (False, ["deepseek-flash"], ["deepseek-flash"])
    assert first["wall_s"] == 60.0 and first["pages"] == {"scored": 3, "failed": 0, "skipped": 1, "pending": 0}
    assert first["comet"]["model_revision"] == SNAPSHOT and first["terms"]["rules"] == "OUTLINE §4.7 (v0.22)"


def test_runs_that_agree_and_are_clean_give_no_warning(tmp_path):
    result = build(*two_by_two(tmp_path))

    assert result["warnings"] == []
    assert "warnings: none" in report.render(result)


def test_a_dirty_tree_different_commits_and_different_models_are_warnings_that_name_the_runs(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    clean = make_eval(tmp_path, "baseline", A, [page(1)], glossary_dir=glossary_dir, git_commit="aaaaaaaaaaaaaaaa")
    dirty = make_eval(tmp_path, "full_no_kb", A, [page(1)], glossary_dir=glossary_dir, git_commit="bbbbbbbbbbbbbbbb",
                      git_dirty=True, models=("other-model",))

    result = report.build_report([clean, dirty], glossary_dir)

    text = "\n".join(result["warnings"])
    assert f"20261029T000000Z-full_no_kb-{A} (full_no_kb on {A}) ran on a dirty working tree" in text
    assert "different commits" in text and "aaaaaaaaaaaa" in text and "bbbbbbbbbbbb" in text
    assert "differ in requested_models" in text and "differ in response_models" in text and "other-model" in text
    assert "dirty" not in "\n".join(w for w in result["warnings"] if "baseline" in w and "ran on" in w)
    assert "warnings:\n  " in report.render(result)


def test_an_unknown_git_state_is_a_warning_not_a_clean_tree(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    run_dir = make_eval(tmp_path, "baseline", A, [page(1)], glossary_dir=glossary_dir, git_dirty=None)

    warnings = report.build_report([run_dir], glossary_dir)["warnings"]

    assert len(warnings) == 1 and "unknown" in warnings[0]


def test_comet_scored_with_different_checkpoint_revisions_is_a_warning_and_the_revision_is_unknown(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    one = make_eval(tmp_path, "baseline", A, [page(1)], glossary_dir=glossary_dir, revision=SNAPSHOT)
    two = make_eval(tmp_path, "full_no_kb", A, [page(1)], glossary_dir=glossary_dir, revision="b" * 40)

    result = report.build_report([one, two], glossary_dir)

    assert any("different checkpoint revisions" in w for w in result["warnings"])
    assert result["pooled"]["baseline"]["comet"]["revision"] == SNAPSHOT
    pooled_over_books = report.build_report([one, make_eval(tmp_path, "baseline", B, [page(1)], glossary_dir=glossary_dir,
                                                            revision="b" * 40)], glossary_dir)
    assert pooled_over_books["pooled"]["baseline"]["comet"]["revision"] is None


def test_a_comet_file_without_a_revision_is_not_a_warning_and_its_revision_is_none(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    run_dir = make_eval(tmp_path, "baseline", A, [page(1)], glossary_dir=glossary_dir)
    edit_json(run_dir / report.COMET_FILE, lambda d: d.pop("model_revision"))

    result = report.build_report([run_dir], glossary_dir)

    assert result["warnings"] == [] and result["pooled"]["baseline"]["comet"]["revision"] is None


# --- the printed summary -----------------------------------------------------------------------------------------------------

def test_the_summary_has_one_column_per_config_in_the_order_of_the_configs_and_the_pooled_rows_come_first(tmp_path):
    glossary_dir = make_glossary_dir(tmp_path)
    # given in the opposite order to the columns
    dirs = [make_eval(tmp_path, c, A, [page(1)], glossary_dir=glossary_dir) for c in ("full", "translator", "baseline", "full_no_kb")]

    text = report.render(report.build_report(dirs, glossary_dir))

    header = [l for l in text.splitlines() if l.strip().startswith("baseline")][0]
    assert header.split() == ["baseline", "translator", "full_no_kb", "full"]
    lines = text.splitlines()
    assert lines.index(header) < next(i for i, l in enumerate(lines) if l.startswith("chrF")) < next(
        i for i, l in enumerate(lines) if l.startswith(A))  # pooled chrF before the per-book blocks


def test_the_summary_shows_the_pages_left_out_with_their_status_and_error_and_n_a_for_unknown(tmp_path):
    result = build(*two_by_two(tmp_path))

    text = report.render(result)

    assert f"  full_no_kb: {A}/3 failed: ProviderError: HTTP 500" in text
    assert f"  baseline: {A}/3 scored" in text
    chrf = [l for l in text.splitlines() if l.startswith("chrF")][0]
    assert f"{result['pooled']['baseline']['chrf']['score']:.2f}" in chrf
    assert "-" in [l for l in text.splitlines() if l.startswith("final_stop approved")][0].split()  # baseline has none
    assert [l for l in text.splitlines() if l.startswith("spend")][0].split() == ["spend,", "all", "runs", "$0.050000", "$0.070000"]


def row_tokens(text, label):
    """The label's words and then the cells of the first table row that has exactly this label (the pooled rows come
    first; a per-book block repeats a few labels)."""
    for l in text.splitlines():
        if l[:report.LABEL_WIDTH].rstrip() == label:
            return label.split() + l[report.LABEL_WIDTH:].split()
    raise AssertionError(f"no row labelled {label!r}")


def block_after(text, header_start):
    """The lines under the line that starts with `header_start`, up to the next blank line."""
    lines = text.splitlines()
    first = next(i for i, l in enumerate(lines) if l.startswith(header_start)) + 1
    last = next((i for i in range(first, len(lines)) if not lines[i].strip()), len(lines))
    return lines[first:last]


def test_every_row_of_the_summary_shows_its_own_number_under_its_own_label(tmp_path):
    result = build(*two_by_two(tmp_path))
    text = report.render(result)
    p = result["pooled"]

    # the pooled rows, one column for each of baseline and full_no_kb; the values differ from row to row and column to
    # column, so a row that shows another row's number, or the columns swapped, is caught
    assert row_tokens(text, "pages") == ["pages", "4", "4"] and row_tokens(text, "segments") == ["segments", "6", "6"]
    assert row_tokens(text, "chrF")[1:] == [f"{p[c]['chrf']['score']:.2f}" for c in ("baseline", "full_no_kb")]
    assert row_tokens(text, "BLEU")[1:] == [f"{p[c]['bleu']['score']:.2f}" for c in ("baseline", "full_no_kb")]
    assert row_tokens(text, "BLEU")[1:] != row_tokens(text, "chrF")[1:]
    assert row_tokens(text, "COMET (wmt22-comet-da)") == ["COMET", "(wmt22-comet-da)", f"{2.6 / 6:.4f}", f"{3.2 / 6:.4f}"]
    assert row_tokens(text, "term accuracy, primary") == ["term", "accuracy,", "primary", "60.0%", "(3/5)", "100.0%", "(5/5)"]
    assert row_tokens(text, "term accuracy, all entries") == ["term", "accuracy,", "all", "entries", "66.7%", "(4/6)", "83.3%", "(5/6)"]
    assert row_tokens(text, "cost per page") == ["cost", "per", "page", "$0.002500", "$0.012500"]
    assert row_tokens(text, "latency p50 / p95 (s)") == ["latency", "p50", "/", "p95", "(s)", "2.5", "/", f"{p['baseline']['latency']['p95_s']:.1f}",
                                           "15.0", "/", "20.0"]
    assert row_tokens(text, "fallback to baseline (pages)") == ["fallback", "to", "baseline", "(pages)", "0", "0"]
    assert row_tokens(text, "spend, all runs") == ["spend,", "all", "runs", "$0.050000", "$0.070000"]
    assert row_tokens(text, "  runs (failed)") == ["runs", "(failed)", "5", "(0)", "5", "(1)"]
    # the same columns again for each book, with that book's own numbers
    book_a, book_b = block_after(text, f"{A}  (2 common pages)"), block_after(text, f"{B}  (2 common pages)")
    assert book_a[0].split() == ["baseline", "full_no_kb"]
    assert book_a[1].split() == ["chrF"] + [f"{result['by_book'][A][c]['chrf']['score']:.2f}" for c in ("baseline", "full_no_kb")]
    assert book_a[2].split() == ["COMET", "(wmt22-comet-da)", "0.5000", "0.6000"]
    assert book_b[2].split() == ["COMET", "(wmt22-comet-da)", "0.3000", "0.4000"]
    assert book_a[3].split() == ["term", "accuracy,", "primary", "66.7%", "(2/3)", "100.0%", "(3/3)"]
    assert book_b[3].split() == ["term", "accuracy,", "primary", "50.0%", "(1/2)", "100.0%", "(2/2)"]


def test_a_long_cell_never_touches_the_column_before_it(tmp_path):
    """`n/a (16 unknown)` is wider than a column of 16: the layout widens the columns instead of running cells together."""
    glossary_dir = make_glossary_dir(tmp_path)
    unknown = [page(i, cost_usd=None, unknown_usage_calls=1) for i in range(1, 17)]
    dirs = [make_eval(tmp_path, "baseline", A, [page(i) for i in range(1, 17)], glossary_dir=glossary_dir),
            make_eval(tmp_path, "full_no_kb", A, unknown, glossary_dir=glossary_dir)]

    text = report.render(report.build_report(dirs, glossary_dir))

    assert row_tokens(text, "cost per page") == ["cost", "per", "page", "$0.010000", "n/a", "(16", "unknown)"]
    widths = {len(l) for l in text.splitlines() if l.startswith(("segments", "chrF", "cost per page", "spend"))}
    assert len(widths) == 1  # every row of the table is as wide as the others
    assert "  n/a (16 unknown)" in text  # at least two spaces before it


# --- reading the files: refusals -----------------------------------------------------------------------------------------------

@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("eval_report_under_test", config.ROOT / "eval" / "report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(cli, dirs, glossary_dir, out_root):
    return cli.main([*map(str, dirs), "--glossary-dir", str(glossary_dir), "--out-root", str(out_root)])


def snapshot(*directories):
    """Every file under the directories with its bytes: a refusal must change none of them."""
    return {str(p): p.read_bytes() for d in directories if d.exists() for p in sorted(d.rglob("*")) if p.is_file()}


def refusals(tmp_path):
    """[(what is wrong, run directories, glossary dir, text the message must hold, the directory it must name)]."""
    cases = []

    def case(what, dirs, glossary_dir, expected, names):
        cases.append((what, dirs, glossary_dir, expected, names))

    def fresh(name):
        sub = tmp_path / name
        sub.mkdir()
        glossary_dir = make_glossary_dir(sub)
        return sub, glossary_dir, make_eval(sub, "baseline", A, [page(1), page(2)], glossary_dir=glossary_dir)

    sub, g, run_dir = fresh("no-dir")
    case("a missing directory", [sub / "nowhere"], g, "it is not a directory", sub / "nowhere")

    sub, g, run_dir = fresh("no-manifest")
    (run_dir / "manifest.json").unlink()
    case("a missing manifest", [run_dir], g, "manifest.json is missing", run_dir)

    sub, g, run_dir = fresh("unfinished")
    edit_json(run_dir / "manifest.json", lambda d: d.update(finished_at=None))
    case("an unfinished run", [run_dir], g, "not finished", run_dir)

    for name, expected in (("terms.json", "terms.json is missing"), (report.COMET_FILE, f"{report.COMET_FILE} is missing"),
                           ("scores.json", "scores.json is missing"), ("segments.jsonl", "segments.jsonl is missing")):
        sub, g, run_dir = fresh(f"no-{name}")
        (run_dir / name).unlink()
        case(f"a missing {name}", [run_dir], g, expected, run_dir)

    for name in ("terms.json", report.COMET_FILE):
        sub, g, run_dir = fresh(f"other-eval-id-{name}")
        edit_json(run_dir / name, lambda d: d.update(eval_id="20260101T000000Z-baseline"))
        case(f"an eval_id in {name} that is not the manifest's", [run_dir], g, f"{name} is of eval run", run_dir)

    sub, g, run_dir = fresh("changed-segments")
    text = (run_dir / "segments.jsonl").read_text(encoding="utf-8")
    (run_dir / "segments.jsonl").write_text(text.replace("甲", "乙", 1), encoding="utf-8")
    case("a COMET file that scored another segments.jsonl", [run_dir], g, "segments_sha256", run_dir)

    sub, g, run_dir = fresh("other-model")
    edit_json(run_dir / report.COMET_FILE, lambda d: d.update(model="Unbabel/XCOMET-XL"))
    case("a COMET model other than wmt22-comet-da", [run_dir], g, "scored with 'Unbabel/XCOMET-XL'", run_dir)

    sub, g, run_dir = fresh("changed-glossary")
    write_json(g / f"{A}.json", {"book_title": A, "entries": [{"id": "new"}]})
    case("a terms file scored against another glossary", [run_dir], g, "glossary_sha256", run_dir)

    sub, g, run_dir = fresh("no-glossary")
    (g / f"{A}.json").unlink()
    case("a book with no glossary file", [run_dir], g, f"no glossary for book '{A}'", run_dir)

    sub, g, run_dir = fresh("same-config-and-book")
    (sub / "again").mkdir()
    twin = make_eval(sub / "again", "baseline", A, [page(1)], glossary_dir=g)  # another directory, the same config and book
    case("two directories with the same config and book", [run_dir, twin], g, f"both baseline on {A}", twin)

    sub, g, run_dir = fresh("different-books")
    other = make_eval(sub, "full_no_kb", B, [page(1)], glossary_dir=g)
    case("configs that cover different books", [run_dir, other], g, "do not cover the same books", run_dir)

    sub, g, run_dir = fresh("page-without-lines")
    edit_json(run_dir / "manifest.json", lambda d: d["pages"].append(
        {"book": A, "page_index": 5, "status": "scored", "run_id": 5, "trace_id": None}))
    edit_json(run_dir / "scores.json", lambda d: d["pages"].append({"book": A, "page_index": 5, "cost_usd": 0.0,
                                                                     "unknown_usage_calls": 0, "latency_s": 1.0}))
    case("a scored page with no line in segments.jsonl", [run_dir], g, "page 5 is scored in manifest.json but has no line", run_dir)

    sub, g, run_dir = fresh("page-without-row")
    edit_json(run_dir / "scores.json", lambda d: d["pages"].pop())
    case("a scored page with no row in scores.json", [run_dir], g, "page 2 is scored in manifest.json but has no row", run_dir)

    sub, g, run_dir = fresh("segment-without-comet-score")
    edit_json(run_dir / report.COMET_FILE, lambda d: d["segments"].pop())
    case("a segment with no COMET score", [run_dir], g, "has no score for segment", run_dir)

    sub, g, run_dir = fresh("row-without-latency")
    edit_json(run_dir / "scores.json", lambda d: d["pages"][0].pop("latency_s"))
    case("a page row without latency_s", [run_dir], g, "the row of page 1 in scores.json has no `latency_s`", run_dir)

    sub, g, run_dir = fresh("repeated-segment")
    lines = (run_dir / "segments.jsonl").read_text(encoding="utf-8").splitlines()
    (run_dir / "segments.jsonl").write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
    edit_json(run_dir / report.COMET_FILE, lambda d: d.update(segments_sha256=sha256_of(run_dir / "segments.jsonl")))
    case("a repeated segment", [run_dir], g, "repeats page_index 1 idx 0", run_dir)

    sub, g, run_dir = fresh("segment-of-another-book")
    text = (run_dir / "segments.jsonl").read_text(encoding="utf-8")
    (run_dir / "segments.jsonl").write_text(text.replace(A, B, 1), encoding="utf-8")
    edit_json(run_dir / report.COMET_FILE, lambda d: d.update(segments_sha256=sha256_of(run_dir / "segments.jsonl")))
    case("a segment of another book", [run_dir], g, f"is of book '{B}'", run_dir)

    sub, g, run_dir = fresh("comet-score-twice")
    edit_json(run_dir / report.COMET_FILE, lambda d: d["segments"].append(d["segments"][0]))
    case("a segment scored twice by COMET", [run_dir], g, "scores segment", run_dir)

    sub, g, run_dir = fresh("comet-nan")
    (run_dir / report.COMET_FILE).write_text(
        (run_dir / report.COMET_FILE).read_text(encoding="utf-8").replace('"score": 0.5', '"score": NaN', 1), encoding="utf-8")
    case("a COMET score that is not a finite number", [run_dir], g, "not a finite number", run_dir)

    sub, g, run_dir = fresh("nan-cost")
    edit_json(run_dir / "scores.json", lambda d: d["pages"][0].update(cost_usd=float("nan")))
    case("a page row whose cost is NaN", [run_dir], g, "`cost_usd` that is neither a finite number nor null", run_dir)

    sub, g, run_dir = fresh("boolean-cost")
    edit_json(run_dir / "scores.json", lambda d: d["pages"][0].update(cost_usd=True))
    case("a page row whose cost is a boolean", [run_dir], g, "`cost_usd` that is neither a finite number nor null", run_dir)

    sub, g, run_dir = fresh("infinite-latency")
    edit_json(run_dir / "scores.json", lambda d: d["pages"][0].update(latency_s=float("inf")))
    case("a page row whose latency is Infinity", [run_dir], g, "`latency_s` that is not a finite number", run_dir)

    sub, g, run_dir = fresh("text-unknown-calls")
    edit_json(run_dir / "scores.json", lambda d: d["pages"][0].update(unknown_usage_calls="0"))
    case("a page row whose unknown_usage_calls is not a number", [run_dir], g,
         "`unknown_usage_calls` that is not an integer", run_dir)

    sub, g, run_dir = fresh("nan-failed-cost")
    (sub / "failed").mkdir()
    failed_run = make_eval(sub / "failed", "baseline", B, [page(1), page(2, "failed", error="E", cost_usd=0.1)],
                           glossary_dir=g)
    edit_json(failed_run / "scores.json", lambda d: d["failed_pages"][0].update(cost_usd=float("nan")))
    case("a failed page row whose cost is NaN", [failed_run], g, "`cost_usd` that is neither a finite number nor null",
         failed_run)

    sub, g, run_dir = fresh("broken-json")
    (run_dir / "scores.json").write_text("{not json", encoding="utf-8")
    case("a file that is not JSON", [run_dir], g, "scores.json cannot be read as JSON", run_dir)

    sub, g, run_dir = fresh("broken-terms")
    edit_json(run_dir / "terms.json", lambda d: d.update(entries=[{"id": "x"}]))
    case("a terms file of another shape", [run_dir], g, "not shaped as eval/terms.py writes it", run_dir)
    return cases


def test_every_refusal_exits_2_names_the_directory_and_the_cause_and_writes_nothing(cli, tmp_path, capsys):
    cases = refusals(tmp_path)
    assert len(cases) >= 22

    for what, dirs, glossary_dir, expected, names in cases:
        out_root = tmp_path / f"report-{abs(hash(what))}"
        before = snapshot(*[d for d in dirs], glossary_dir)

        code = run_cli(cli, dirs, glossary_dir, out_root)

        err = capsys.readouterr().err
        assert code == 2, what
        assert err.startswith("Nothing was written: "), what
        assert expected in err, (what, err)
        assert str(names) in err, (what, err)
        assert not out_root.exists(), what  # no report directory, not even an empty one
        assert snapshot(*[d for d in dirs], glossary_dir) == before, what


def test_the_library_raises_refused_for_the_same_cases(tmp_path):
    for what, dirs, glossary_dir, expected, names in refusals(tmp_path):
        with pytest.raises(report.Refused, match=expected.replace("`", ".").replace("(", ".")) as raised:
            report.build_report(dirs, glossary_dir)
        assert str(names) in str(raised.value), what


# --- writing the report ---------------------------------------------------------------------------------------------------------

class FixedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 29, 12, 0, 0, tzinfo=tz or timezone.utc)


def test_the_cli_writes_report_json_into_a_new_directory_and_prints_the_summary(cli, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(report, "datetime", FixedClock)
    dirs, glossary_dir = two_by_two(tmp_path)
    out_root = tmp_path / "out" / "report"  # does not exist yet

    code = run_cli(cli, dirs.values(), glossary_dir, out_root)

    assert code == 0
    (out_dir,) = list(out_root.iterdir())
    assert out_dir.name == "20261029T120000Z-report" and sorted(p.name for p in out_dir.iterdir()) == ["report.json"]
    written = read_json(out_dir / "report.json")
    assert written == json.loads(json.dumps(build(dirs, glossary_dir) | {"generated_at": written["generated_at"]}))
    assert list(written) == ["generated_at", "configs", "books", "common_pages", "left_out", "pooled", "by_book", "spend",
                             "runs", "warnings"]
    out = capsys.readouterr().out
    assert out.endswith(f"Wrote {out_dir / 'report.json'}\n") and "chrF" in out and "warnings: none" in out


def test_a_second_report_in_the_same_second_gets_a_directory_of_its_own_and_never_writes_into_an_existing_one(
        cli, tmp_path, monkeypatch):
    monkeypatch.setattr(report, "datetime", FixedClock)
    dirs, glossary_dir = two_by_two(tmp_path)
    out_root = tmp_path / "report"
    first = report.write_report(build(dirs, glossary_dir), out_root)
    (first / "report.json").write_text("first", encoding="utf-8")
    (out_root / "20261029T120000Z-report-2").mkdir()  # taken, too

    second = report.write_report(build(dirs, glossary_dir), out_root)

    assert (first.name, second.name) == ("20261029T120000Z-report", "20261029T120000Z-report-3")
    assert (first / "report.json").read_text(encoding="utf-8") == "first"
    assert sorted(p.name for p in second.iterdir()) == ["report.json"]


def test_a_failing_write_leaves_no_directory_and_no_temporary_file(tmp_path, monkeypatch):
    dirs, glossary_dir = two_by_two(tmp_path)
    document = build(dirs, glossary_dir)
    monkeypatch.setattr(report.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk on fire")))

    with pytest.raises(OSError, match="disk on fire"):
        report.write_report(document, tmp_path / "report")

    assert list((tmp_path / "report").iterdir()) == []


def test_the_cli_exits_1_when_the_report_cannot_be_written(cli, tmp_path, capsys):
    dirs, glossary_dir = two_by_two(tmp_path)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file", encoding="utf-8")

    code = run_cli(cli, dirs.values(), glossary_dir, blocker / "report")

    assert code == 1 and "Nothing was written: could not write" in capsys.readouterr().err


def test_the_default_glossary_dir_and_report_dir_are_the_projects(cli):
    assert cli.GLOSSARY_DIR == config.ROOT / "eval" / "gold_glossary"
    assert cli.REPORT_DIR == config.ROOT / "runs" / "report"


# --- what the module is -----------------------------------------------------------------------------------------------------------

def test_the_names_the_report_repeats_are_the_ones_the_harness_and_the_agents_use():
    assert report.CONFIG_ORDER == runs.CONFIGS
    assert report.REVIEWED_CONFIGS == harness.REVIEWED_CONFIGS and report.KB_CONFIG == runs.FULL
    assert report.FINAL_STOPS == flow.FINAL_STOPS
    assert report.CURATOR_STATUSES == curator_agent.STATUSES == harness.CURATOR_STATUSES
    assert report.PROPOSAL_RESULTS == curator_rules.RESULTS
    assert report.LOOKUP_AGENTS == harness.LOOKUP_AGENTS and report.LOOKUP_TOOLS == tuple(harness.LOOKUP_TOOLS)


def test_the_comet_model_and_file_name_are_the_ones_eval_comet_writes():
    spec = importlib.util.spec_from_file_location("comet_script_for_the_report_test", config.ROOT / "eval" / "comet.py")
    comet_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(comet_script)

    assert report.COMET_MODEL == comet_script.DEFAULT_MODEL
    assert report.COMET_FILE == comet_script.output_name(comet_script.DEFAULT_MODEL)


def test_the_report_reads_files_only():
    names = set()
    for path in (config.ROOT / "backend" / "fukidashi" / "evaluation" / "report.py", config.ROOT / "eval" / "report.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(part for alias in node.names for part in alias.name.split("."))
            elif isinstance(node, ast.ImportFrom):
                names.update((node.module or "").split("."))
                names.update(alias.name for alias in node.names)
    assert not names & {"sqlite3", "store", "db", "runs", "llm", "openai", "httpx", "requests", "socket", "subprocess"}


def test_the_script_runs_as_a_program(tmp_path):
    import subprocess
    import sys

    dirs, glossary_dir = two_by_two(tmp_path)
    script = str(config.ROOT / "eval" / "report.py")
    args = [sys.executable, script, *map(str, dirs.values()), "--glossary-dir", str(glossary_dir), "--out-root", str(tmp_path / "r")]

    done = subprocess.run(args, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "baseline" in done.stdout and (next((tmp_path / "r").iterdir()) / "report.json").is_file()

    refused = subprocess.run([sys.executable, script, str(tmp_path / "nowhere")], capture_output=True, text=True,
                             env={"PATH": "/usr/bin:/bin"}, timeout=60)
    assert refused.returncode == 2 and "Nothing was written" in refused.stderr
