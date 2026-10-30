"""The paired bootstrap (OUTLINE 4.3, 8; eval/PROTOCOL.md, "Statistics"): `evaluation/bootstrap.py` and eval/bootstrap.py, on
made-up runs.

Nothing here needs data/, the database or a model, and no test book is named: the eval runs are written with
`test_report`'s helpers (`make_eval`, `write_consistency`, `write_mqm`) for the two dev books' names, whose glossaries are
made-up files. The numbers are small enough to work out by hand, and where a test cannot do that (chrF) it compares with
`score_corpus` over the segment lists, which is what the bootstrap's sums are meant to equal.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import random
import re
import subprocess
import sys
from collections import Counter

import pytest

from fukidashi import config
from fukidashi.evaluation import bootstrap, mqm, report
from fukidashi.evaluation.metrics import percentile, score_corpus
from test_report import (A, B, BASELINE_HYPS, JUDGE_MODEL, MQM_MODEL, REFS, FixedClock, edit_json, make_eval,
                         make_glossary_dir, occ, page, read_json, refusals, sha256_of, snapshot, write_consistency,
                         write_mqm)

CONFIGS = bootstrap.CONFIGS  # baseline, full_no_kb, full
R = 40  # resamples in most tests: what they check does not depend on how many
REAL_GIT_STATE = bootstrap.git_state


@pytest.fixture(autouse=True)
def known_code(monkeypatch):
    """The output records the code's commit; the tests do not depend on the checkout they run in."""
    monkeypatch.setattr(bootstrap, "git_state", lambda repo: {"commit": "c0ffee1", "dirty": False})


# --- three configs x two books, small enough to work out by hand ------------------------------------------------------
#
# Book A has pages 1-3 and 9 (9 has no text, so every config skipped it); book B has pages 1 and 2. Every segment is on
# a page of REFS; the numbers below are written so that the configs differ everywhere a metric looks.

PAGES = {A: (1, 2, 3, 9), B: (1, 2)}
HYPS = {
    "baseline": BASELINE_HYPS,
    "full_no_kb": {(A, 1): ["今天天气不好", "我们一起去吧"], (A, 2): ["龙来了", "好吧"], (A, 3): ["我在这里等你", "我也这样想"],
                   (B, 1): ["他走了。"], (B, 2): ["你好"]},
    "full": {(A, 1): ["今天天气很好", "我们一起去吧啊"], (A, 2): ["龙来了", "好的"], (A, 3): ["我在这里等你回来", "我也是这样想"],
             (B, 1): ["他走了"], (B, 2): ["你好吗"]},
}
COMET_SCORES = {
    "baseline": {A: {(1, 0): .2, (1, 1): .4, (2, 0): .6, (2, 1): .8, (3, 0): .3, (3, 1): .5}, B: {(1, 0): .1, (2, 0): .5}},
    "full_no_kb": {A: {(1, 0): .3, (1, 1): .5, (2, 0): .7, (2, 1): .9, (3, 0): .4, (3, 1): .6}, B: {(1, 0): .2, (2, 0): .6}},
    "full": {A: {(1, 0): .4, (1, 1): .6, (2, 0): .8, (2, 1): .7, (3, 0): .9, (3, 1): .5}, B: {(1, 0): .6, (2, 0): .3}},
}
OCCURRENCES = {  # book -> entry -> (borderline, [(page, idx)]): where the Japanese text has the term and the reference its target
    A: {"ryu": (False, [(1, 0), (2, 0), (2, 1), (3, 0)]), "word": (True, [(1, 1), (3, 1)]), "ghost": (False, [(3, 0)])},
    B: {"shiora": (False, [(1, 0), (2, 0)])},
}
HITS = {  # config -> entry -> whether the Translation has the term, for each of the entry's occurrences
    "baseline": {"ryu": [1, 1, 0, 1], "word": [1, 1], "ghost": [0], "shiora": [1, 0]},
    "full_no_kb": {"ryu": [1, 1, 1, 0], "word": [0, 1], "ghost": [1], "shiora": [1, 1]},
    "full": {"ryu": [1, 1, 1, 1], "word": [1, 0], "ghost": [1], "shiora": [1, 1]},
}
TARGETS = {"ryu": "龙", "ghost": "鬼", "shiora": "希奥拉", "kururu": "枢"}
STATUSES = ("omitted", "not_this_term", "invalid")  # what an occurrence without a rendering says
RENDERED = {  # config -> book -> entry -> [(page, idx, the Translation's rendering, or a status that has none)]
    "baseline": {A: {"ryu": [(1, 0, "龙"), (2, 0, "龙"), (2, 1, "飞龙"), (3, 0, "龙王")],
                     "ghost": [(1, 1, "omitted"), (3, 1, "鬼")]},
                 B: {"shiora": [(1, 0, "希奥拉"), (2, 0, "西奥拉")], "kururu": [(1, 0, "invalid"), (2, 0, "枢")]}},
    "full_no_kb": {A: {"ryu": [(1, 0, "龙"), (2, 0, "龙"), (2, 1, "龙"), (3, 0, "龙")],
                       "ghost": [(1, 1, "not_this_term"), (3, 1, "鬼")]},
                   B: {"shiora": [(1, 0, "希奥拉"), (2, 0, "西奥拉")], "kururu": [(1, 0, "枢"), (2, 0, "枢")]}},
    "full": {A: {"ryu": [(1, 0, "龙"), (2, 0, "龙"), (2, 1, "飞龙"), (3, 0, "龙")], "ghost": [(2, 0, "鬼")]},
             B: {"shiora": [(1, 0, "希奥拉"), (2, 0, "希奥拉")], "kururu": [(2, 0, "枢")]}},
}
MQM_JUDGED = {  # config -> book -> (page, idx) -> the segment's errors [(severity, category)], or "failed" / "skipped"
    "baseline": {A: {(1, 0): [("major", "accuracy")], (1, 1): [("minor", "style"), ("minor", "fluency")], (2, 0): "skipped",
                     (2, 1): [], (3, 0): [("critical", "accuracy")], (3, 1): "failed"},
                 B: {(1, 0): [("critical", "accuracy")], (2, 0): "failed"}},
    "full_no_kb": {A: {(1, 0): [("minor", "style")], (1, 1): [], (2, 0): "skipped", (2, 1): [("major", "fluency")],
                       (3, 0): [("minor", "style")], (3, 1): []},
                   B: {(1, 0): [], (2, 0): [("minor", "style")]}},
    "full": {A: {(1, 0): [], (1, 1): [("minor", "style")], (2, 0): "skipped", (2, 1): [("minor", "style")], (3, 0): [],
                 (3, 1): [("major", "accuracy")]},
             B: {(1, 0): [], (2, 0): [("minor", "style")]}},
}


def term_entries(name, book, kept):
    return [{"id": entry, "borderline": borderline,
             "occurrences": [(p, i, bool(hit)) for (p, i), hit in zip(places, HITS[name][entry]) if p in kept]}
            for entry, (borderline, places) in OCCURRENCES[book].items()]


def consistency_entries(name, book, kept, incorrect=()):
    """The consistency file's entries; `incorrect` are (entry, page, idx) occurrences whose extractor check fails."""
    return {entry: [occ(p, i, rendering, TARGETS[entry], True, (entry, p, i) not in incorrect)
                    for p, i, rendering in places if p in kept]
            for entry, places in RENDERED[name][book].items()}


def three_by_two(tmp_path, *, failed=(), incorrect=(), books=(A, B)):
    """({(config, book): run directory}, glossary dir): baseline, full_no_kb and full on both books, each with every file the
    bootstrap reads. `failed` is a set of (config, book, page) the config failed; `incorrect` is a set of (config, book, entry,
    page, idx) whose extractor check fails."""
    glossary_dir = make_glossary_dir(tmp_path)
    dirs = {}
    for name in CONFIGS:
        for book in books:
            def make_page(i):
                if (book, i) == (A, 9):
                    return page(9, "skipped")
                if (name, book, i) in failed:
                    return page(i, "failed", error="ProviderError: HTTP 500", cost_usd=0.02)
                return page(i, segments=list(zip(HYPS[name][book, i], REFS[book, i])))

            kept = {i for i in PAGES[book] if i != 9 and (name, book, i) not in failed}
            wrong = {key[2:] for key in incorrect if key[:2] == (name, book)}
            run_dir = make_eval(tmp_path, name, book, [make_page(i) for i in PAGES[book]], glossary_dir=glossary_dir,
                                entries=term_entries(name, book, kept), comet=COMET_SCORES[name][book])
            write_consistency(run_dir, glossary_dir, consistency_entries(name, book, kept, wrong))
            write_mqm(run_dir, {k: v for k, v in MQM_JUDGED[name][book].items() if k[0] in kept})
            dirs[name, book] = run_dir
    return dirs, glossary_dir


def tiny(tmp_path, books=(A,), *, mqm_judged=None):
    """baseline, full_no_kb and full on `books`, each book with two pages. Page 1 (two segments) holds everything the metrics
    need: a term entry with two occurrences, a consistency entry with two rendered occurrences, a scored MQM judgement. Page 2
    (one segment) has no occurrence and a failed MQM judgement, so a resample that draws only page 2 is undefined for term
    accuracy, term consistency and MQM, and for no other metric."""
    glossary_dir = make_glossary_dir(tmp_path, books=books)
    dirs = {}
    for name in CONFIGS:
        hyps = {"baseline": ("甲", "乙", "丙"), "full_no_kb": ("甲", "丁", "丙"), "full": ("甲", "乙", "戊")}[name]
        for book in books:
            pages = [page(1, segments=[(hyps[0], "甲"), (hyps[1], "乙")]), page(2, segments=[(hyps[2], "丙")])]
            run_dir = make_eval(
                tmp_path, name, book, pages, glossary_dir=glossary_dir,
                entries=[{"id": "x", "borderline": False, "occurrences": [(1, 0, True), (1, 1, name == "full")]}],
                comet={(1, 0): .2, (1, 1): .4, (2, 0): .9})
            write_consistency(run_dir, glossary_dir,
                              {"x": [occ(1, 0, "甲", "甲", True, True), occ(1, 1, "甲", "甲", True, True)]})
            write_mqm(run_dir, mqm_judged if mqm_judged is not None else {(1, 0): [("minor", "style")], (2, 0): "failed"})
            dirs[name, book] = run_dir
    return dirs, glossary_dir


def build(dirs, glossary_dir, *, resamples=R, **options):
    return bootstrap.build_bootstrap(list(dirs.values()) if isinstance(dirs, dict) else dirs, glossary_dir,
                                     resamples=resamples, **options)


def comparisons(document):
    return {c["name"]: c for c in document["comparisons"]}


def without_time(document):
    return {k: v for k, v in document.items() if k != "generated_at"}


def by_hand(name, sample):
    """Every metric of config `name` on `sample` (a list of (book, page), a page as often as it appears) worked out from the
    fixture's raw numbers above, with no code of the bootstrap or of the report."""
    places = [(book, p, idx) for book, p in sample for idx in range(len(REFS[book, p]))]
    corpus = score_corpus([HYPS[name][book, p][idx] for book, p, idx in places], [REFS[book, p][idx] for book, p, idx in places])
    comet = [COMET_SCORES[name][book][p, idx] for book, p, idx in places]
    hits = occurrences = 0
    for book, p in sample:
        for entry, (borderline, entry_places) in OCCURRENCES[book].items():
            if not borderline:
                for (entry_page, _), hit in zip(entry_places, HITS[name][entry]):
                    if entry_page == p:
                        occurrences += 1
                        hits += hit
    entries = rendered = most_common = 0
    for book in (A, B):
        for entry, entry_places in RENDERED[name][book].items():
            counts: Counter = Counter()
            for sampled_book, p in sample:
                if sampled_book == book:
                    counts.update(r for entry_page, _, r in entry_places if entry_page == p and r not in STATUSES)
            if sum(counts.values()) >= 2:
                entries += 1
                rendered += sum(counts.values())
                most_common += max(counts.values())
    scores = []
    for book, p, idx in places:
        judged = MQM_JUDGED[name][book].get((p, idx), [])
        if isinstance(judged, list):
            scores.append(mqm.mqm_score(s for s, _ in judged))
    return {"chrf": corpus.chrf, "bleu": corpus.bleu, "comet": sum(comet) / len(comet),
            "term_accuracy": hits / occurrences if occurrences else None,
            "term_consistency": most_common / rendered if rendered else None,
            "mqm": sum(scores) / len(scores) if scores else None}


def prepared(dirs, glossary_dir, name, other):
    """(the comparison's pages, each config's material, the scorer) of the bootstrap's own building blocks."""
    runs, _ = bootstrap.load_runs(list(dirs.values()), glossary_dir)
    scorer = bootstrap.SacreBLEU()
    pages = bootstrap.comparison_pages(runs, name, other)
    return pages, {c: bootstrap.build_material([r for r in runs if r.config == c], pages, scorer) for c in (name, other)}, scorer


# --- the pages ----------------------------------------------------------------------------------------------------------

def test_a_comparisons_pages_are_those_both_of_its_configs_scored_and_a_page_only_full_no_kb_failed_stays_in_full_vs_baseline(
        tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path, failed={("full_no_kb", A, 3)})

    result = comparisons(build(dirs, glossary_dir))

    assert list(result) == ["full vs baseline", "full vs full_no_kb"]
    assert result["full vs baseline"]["pages"] == {A: [1, 2, 3], B: [1, 2]} and result["full vs baseline"]["n_pages"] == 5
    assert result["full vs full_no_kb"]["pages"] == {A: [1, 2], B: [1, 2]} and result["full vs full_no_kb"]["n_pages"] == 4
    # the report's common pages are every config's, so they are the second comparison's and not the first's
    assert report.build_report(list(dirs.values()), glossary_dir)["common_pages"] == {A: [1, 2], B: [1, 2]}


def test_a_page_every_config_skipped_is_in_no_comparison(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)

    for comparison in build(dirs, glossary_dir)["comparisons"]:
        assert comparison["pages"] == {A: [1, 2, 3], B: [1, 2]}  # A9 has no text


@pytest.mark.parametrize("books", [(B, A), (A, "zz_book"), ("zz_book", B)], ids=["reversed", "unknown-last", "unknown-and-known"])
def test_the_pages_are_pooled_as_one_list_sorted_by_the_reports_book_order_and_then_by_page_index(tmp_path, books):
    dirs, glossary_dir = tiny(tmp_path, books)

    for ordered in (list(dirs.values()), list(reversed(dirs.values()))):
        result = build(ordered, glossary_dir)
        for comparison in result["comparisons"]:
            assert list(comparison["pages"]) == sorted(books, key=report._book_order)
            assert all(pages == [1, 2] for pages in comparison["pages"].values()) and comparison["n_pages"] == 4
    assert sorted(books, key=report._book_order)[0] in (A, B)  # a book the report knows comes before one it does not


def test_the_page_list_of_a_comparison_is_sorted_whatever_the_order_of_the_runs_it_is_made_from(tmp_path):
    dirs, glossary_dir = tiny(tmp_path, (B, A))
    loaded = [report.load_run(run_dir, glossary_dir) for run_dir in dirs.values()]  # baseline on B comes first

    for runs in (loaded, loaded[::-1]):  # and then full on A
        assert bootstrap.comparison_pages(runs, "full", "baseline") == [(A, 1), (A, 2), (B, 1), (B, 2)]
        assert bootstrap.comparison_pages(runs, "full", "full_no_kb") == [(A, 1), (A, 2), (B, 1), (B, 2)]
    assert (loaded[0].config, loaded[0].book) == ("baseline", B) and (loaded[-1].config, loaded[-1].book) == ("full", A)


# --- the numbers on the full sample ---------------------------------------------------------------------------------------

def reports_value(pooled, key):
    return {"chrf": pooled["chrf"]["score"], "bleu": pooled["bleu"]["score"], "comet": pooled["comet"]["mean"],
            "term_accuracy": pooled["terms"]["primary"]["accuracy"],
            "term_consistency": pooled["consistency"]["translation"]["consistency"], "mqm": pooled["mqm"]["score"]}[key]


def reports_support(pooled, key):
    consistency, primary = pooled["consistency"]["translation"], pooled["terms"]["primary"]
    return {"chrf": {"segments": pooled["segments"]}, "bleu": {"segments": pooled["segments"]},
            "comet": {"segments": pooled["comet"]["n"]},
            "term_accuracy": {"hits": primary["hits"], "occurrences": primary["occurrences"]},
            "term_consistency": {"entries": consistency["entries"], "rendered": consistency["rendered"],
                                 "most_common_count": consistency["most_common_count"]},
            "mqm": {"scored": pooled["mqm"]["scored"], "segments": pooled["mqm"]["segments"]}}[key]


def test_on_the_full_sample_every_configs_value_of_every_metric_is_the_reports_pooled_value(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)
    pooled = report.build_report(list(dirs.values()), glossary_dir)["pooled"]

    result = comparisons(build(dirs, glossary_dir))

    seen = 0
    for comparison in result.values():
        assert comparison["pages"] == {A: [1, 2, 3], B: [1, 2]}  # every config scored the same pages
        for key, metric in comparison["metrics"].items():
            for name, value in metric["values"].items():
                assert value == pytest.approx(reports_value(pooled[name], key), abs=1e-9), (comparison["name"], key, name)
                assert metric["support"][name] == reports_support(pooled[name], key), (comparison["name"], key, name)
                seen += 1
            assert metric["point_estimate"] == pytest.approx(
                metric["values"][comparison["config"]] - metric["values"][comparison["other"]], abs=1e-12)
    assert seen == 16  # six metrics and two metrics, two configs each


def test_chrf_and_bleu_on_the_full_sample_are_score_corpus_bit_for_bit_and_the_signatures_are_its(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)

    document = build(dirs, glossary_dir)

    sample = [(A, 1), (A, 2), (A, 3), (B, 1), (B, 2)]
    for name in ("full", "baseline"):
        expected = by_hand(name, sample)
        values = comparisons(document)["full vs baseline"]["metrics"]
        assert values["chrf"]["values"][name] == expected["chrf"] and values["bleu"]["values"][name] == expected["bleu"]
    corpus = score_corpus(HYPS["full"][A, 1], REFS[A, 1])
    assert document["method"]["sacrebleu"] == {"chrf": corpus.chrf_signature, "bleu": corpus.bleu_signature}


def test_a_comparison_counts_the_pages_both_configs_scored_even_where_the_report_does_not(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path, failed={("full_no_kb", A, 3)})
    pooled = report.build_report(list(dirs.values()), glossary_dir)["pooled"]

    result = comparisons(build(dirs, glossary_dir))

    chrf = result["full vs baseline"]["metrics"]["chrf"]["values"]
    with_page_3 = [(A, 1), (A, 2), (A, 3), (B, 1), (B, 2)]
    without = [(A, 1), (A, 2), (B, 1), (B, 2)]
    assert chrf["full"] == by_hand("full", with_page_3)["chrf"] != by_hand("full", without)["chrf"]
    assert chrf["full"] != pooled["full"]["chrf"]["score"]  # the report leaves page A3 out of every config's numbers
    assert pooled["full"]["chrf"]["score"] == by_hand("full", without)["chrf"]
    ablation = result["full vs full_no_kb"]["metrics"]["term_accuracy"]["values"]
    assert ablation["full"] == pooled["full"]["terms"]["primary"]["accuracy"]  # the common pages are this comparison's
    assert ablation["full_no_kb"] == pooled["full_no_kb"]["terms"]["primary"]["accuracy"]


# --- a page drawn twice ------------------------------------------------------------------------------------------------------

def test_a_page_drawn_twice_counts_each_time_in_every_metric(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)
    pages, material, scorer = prepared(dirs, glossary_dir, "full", "baseline")
    assert pages == [(A, 1), (A, 2), (A, 3), (B, 1), (B, 2)]
    drawn = [pages.index((A, 2)), pages.index((A, 2)), pages.index((B, 1))]  # A2 twice, B1 once: 5 segments
    keys = bootstrap.METRICS["full", "baseline"]

    full, baseline = (bootstrap.values_on(material[name], drawn, scorer, keys) for name in ("full", "baseline"))

    # chrF and BLEU: score_corpus over the duplicated segment lists
    for name, got in (("full", full), ("baseline", baseline)):
        hyps = HYPS[name][A, 2] * 2 + HYPS[name][B, 1]
        expected = score_corpus(hyps, REFS[A, 2] * 2 + REFS[B, 1])
        assert (got["chrf"], got["bleu"]) == (expected.chrf, expected.bleu)
    # COMET: full (.8 + .7) * 2 + .6, baseline (.6 + .8) * 2 + .1, over 5 segments
    assert full["comet"] == pytest.approx(3.6 / 5) and baseline["comet"] == pytest.approx(2.9 / 5)
    # term accuracy: full has ryu on A2 twice (2 hits of 2) and shiora on B1 (1 of 1): 5 of 5; baseline 1 of 2 on A2 and 1 of 1
    assert (full["term_accuracy"], baseline["term_accuracy"]) == (5 / 5, 3 / 5)
    # MQM: full scores A2's second segment -1 (its first was skipped) twice and B1's 0; baseline 0 twice and -25
    assert full["mqm"] == pytest.approx(-2 / 3) and baseline["mqm"] == pytest.approx(-25 / 3)
    # consistency: full has an entry "ghost" with ONE rendered occurrence, on A2, so on this resample it has two and counts:
    #   ryu 龙 x2 + 飞龙 x2 (4 rendered, most common 2), ghost 鬼 x2 (2, 2); shiora has 希奥拉 once (B1 is drawn once) and
    #   does not count, kururu's page is not drawn  ->  (2 + 2) / (4 + 2).
    # baseline has no such entry: ryu 龙 x2 + 飞龙 x2 (4, 2) only  ->  2 / 4.
    sums = bootstrap.sample_sums(material["full"], drawn, ["term_consistency"])
    assert sums["term_consistency"] == (2, 6, 4) and full["term_consistency"] == pytest.approx(4 / 6)
    assert bootstrap.sample_sums(material["baseline"], drawn, ["term_consistency"])["term_consistency"] == (1, 4, 2)
    assert baseline["term_consistency"] == pytest.approx(2 / 4)
    # the same numbers from the independent arithmetic above, so that this test's own helper is checked too
    sample = [(A, 2), (A, 2), (B, 1)]
    for name, got in (("full", full), ("baseline", baseline)):
        assert got == pytest.approx(by_hand(name, sample), abs=1e-12)


def test_an_entry_rendered_once_on_a_page_drawn_twice_counts_but_not_when_the_page_is_drawn_once(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)
    pages, material, scorer = prepared(dirs, glossary_dir, "full", "baseline")
    a2 = pages.index((A, 2))

    once = bootstrap.sample_sums(material["full"], [a2], ["term_consistency"])["term_consistency"]
    twice = bootstrap.sample_sums(material["full"], [a2, a2], ["term_consistency"])["term_consistency"]

    assert once == (1, 2, 1)  # only ryu counts (龙, 飞龙: 2 rendered, most common 1); ghost has one rendered occurrence
    assert twice == (2, 6, 4)  # ghost's 鬼 twice counts too, and ryu's counts double


def test_pages_are_summed_not_deduplicated_so_a_sample_of_one_page_repeated_has_the_pages_own_corpus_scores(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)
    pages, material, scorer = prepared(dirs, glossary_dir, "full", "baseline")
    b1 = pages.index((B, 1))

    got = bootstrap.values_on(material["baseline"], [b1] * 3, scorer, ["chrf", "bleu", "comet"])

    expected = by_hand("baseline", [(B, 1)] * 3)
    assert (got["chrf"], got["bleu"]) == (expected["chrf"], expected["bleu"])
    assert got["comet"] == pytest.approx(.1)


# --- the draws and the intervals --------------------------------------------------------------------------------------------

def test_the_draws_are_choices_of_the_page_list_from_a_new_random_seeded_2027_in_order():
    pages = ["a", "b", "c", "d", "e"]
    rng = random.Random(2027)
    expected = [[pages.index(p) for p in rng.choices(pages, k=len(pages))] for _ in range(7)]

    assert list(bootstrap.draws(len(pages), 7, bootstrap.SEED)) == expected
    assert list(bootstrap.draws(len(pages), 7, bootstrap.SEED)) == expected  # a new generator each time
    assert bootstrap.SEED == 2027 and bootstrap.RESAMPLES == 10_000


def test_the_protocol_says_the_resamples_and_the_seed_the_code_has():
    protocol = (config.ROOT / "eval" / "PROTOCOL.md").read_text(encoding="utf-8")

    assert f"{bootstrap.RESAMPLES:,} resamples, seed {bootstrap.SEED}" in protocol


def recomputed_interval(name, other, pages, key, resamples):
    """The interval of `key` worked out the long way: the protocol's draws of (book, page) pairs, every metric of both configs
    on the concatenated segments of the drawn pages (by_hand), the percentiles of the differences."""
    rng = random.Random(2027)
    differences = []
    for _ in range(resamples):
        sample = rng.choices(pages, k=len(pages))
        mine, theirs = by_hand(name, sample)[key], by_hand(other, sample)[key]
        if mine is not None and theirs is not None:
            differences.append(mine - theirs)
    return percentile(differences, 2.5), percentile(differences, 97.5), len(differences)


def test_every_interval_is_the_2_5th_and_97_5th_percentile_of_the_differences_of_the_protocols_resamples(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path, failed={("full_no_kb", A, 3)})
    resamples = 120

    result = comparisons(build(dirs, glossary_dir, resamples=resamples))

    pages = {"full vs baseline": [(A, 1), (A, 2), (A, 3), (B, 1), (B, 2)], "full vs full_no_kb": [(A, 1), (A, 2), (B, 1), (B, 2)]}
    for comparison_name, comparison in result.items():
        for key, metric in comparison["metrics"].items():
            low, high, defined = recomputed_interval(comparison["config"], comparison["other"], pages[comparison_name], key,
                                                     resamples)
            assert metric["interval"]["low"] == pytest.approx(low, abs=1e-12), (comparison_name, key)
            assert metric["interval"]["high"] == pytest.approx(high, abs=1e-12), (comparison_name, key)
            assert metric["defined_resamples"] == defined and metric["undefined_resamples"] == resamples - defined
            assert metric["interval"]["low"] <= metric["interval"]["high"]


def test_each_comparison_draws_from_a_generator_of_its_own_so_equal_page_counts_give_equal_draws(tmp_path, monkeypatch):
    dirs, glossary_dir = three_by_two(tmp_path)  # both comparisons have the same five pages
    log = []
    real = bootstrap.draws

    def spy(size, resamples, seed):
        drawn = list(real(size, resamples, seed))
        log.append((size, seed, drawn))
        return iter(drawn)

    monkeypatch.setattr(bootstrap, "draws", spy)

    build(dirs, glossary_dir, resamples=5)

    (first_size, first_seed, first), (second_size, second_seed, second) = log
    assert (first_size, first_seed, second_size, second_seed) == (5, 2027, 5, 2027)
    assert first == second  # the second comparison did not continue the first one's stream


def test_the_point_estimate_is_the_difference_on_the_comparisons_pages_without_resampling(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)

    result = comparisons(build(dirs, glossary_dir))

    sample = [(A, 1), (A, 2), (A, 3), (B, 1), (B, 2)]
    for comparison in result.values():
        mine, theirs = by_hand(comparison["config"], sample), by_hand(comparison["other"], sample)
        for key, metric in comparison["metrics"].items():
            assert metric["point_estimate"] == pytest.approx(mine[key] - theirs[key], abs=1e-12), (comparison["name"], key)


def test_the_same_inputs_and_resamples_give_identical_numbers_twice(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path, failed={("full_no_kb", A, 3)})

    first, second = (build(dirs, glossary_dir, resamples=75) for _ in range(2))

    assert without_time(first) == without_time(second)
    assert json.dumps(without_time(first)) == json.dumps(without_time(second))


def test_other_resamples_give_other_intervals_and_the_output_records_the_number_used(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)

    few, more = build(dirs, glossary_dir, resamples=20), build(dirs, glossary_dir, resamples=200)

    assert (few["method"]["resamples"], more["method"]["resamples"]) == (20, 200)
    assert few["comparisons"][0]["metrics"]["comet"]["interval"] != more["comparisons"][0]["metrics"]["comet"]["interval"]
    assert few["comparisons"][0]["metrics"]["comet"]["values"] == more["comparisons"][0]["metrics"]["comet"]["values"]
    with pytest.raises(ValueError):
        build(dirs, glossary_dir, resamples=0)


# --- an undefined metric -------------------------------------------------------------------------------------------------------

def test_a_resample_on_which_a_metric_is_undefined_is_counted_and_does_not_crash(tmp_path):
    dirs, glossary_dir = tiny(tmp_path)
    resamples = 200
    rng = random.Random(2027)
    pages = [(A, 1), (A, 2)]
    only_page_2 = sum(rng.choices(pages, k=2) == [(A, 2), (A, 2)] for _ in range(resamples))
    assert 0 < only_page_2 < resamples

    result = comparisons(build(dirs, glossary_dir, resamples=resamples))

    for comparison in result.values():
        for key, metric in comparison["metrics"].items():
            undefined = only_page_2 if key in ("term_accuracy", "term_consistency", "mqm") else 0
            assert metric["undefined_resamples"] == undefined, (comparison["name"], key)
            assert metric["defined_resamples"] == resamples - undefined
            assert metric["interval"] is not None and metric["point_estimate"] is not None


def test_a_metric_that_is_undefined_on_the_full_sample_has_no_value_no_estimate_and_no_interval(tmp_path):
    dirs, glossary_dir = tiny(tmp_path, mqm_judged={(1, 0): "failed", (1, 1): "skipped", (2, 0): "failed"})

    result = build(dirs, glossary_dir, resamples=30)
    mqm_metric = comparisons(result)["full vs baseline"]["metrics"]["mqm"]

    assert mqm_metric["values"] == {"full": None, "baseline": None}
    assert mqm_metric["point_estimate"] is None and mqm_metric["interval"] is None
    assert (mqm_metric["defined_resamples"], mqm_metric["undefined_resamples"]) == (0, 30)
    assert mqm_metric["support"]["full"] == {"scored": 0, "segments": 3}
    row = [l for l in bootstrap.render(result).splitlines() if l.startswith("GEMBA-MQM")][0]
    assert row.count("n/a") == 4 and "0/30" in row  # both values, the difference and the interval


# --- the labels ----------------------------------------------------------------------------------------------------------------

def test_mqm_is_labelled_for_reference_only_with_the_judges_model_and_the_other_metrics_are_results(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)

    result = comparisons(build(dirs, glossary_dir))

    label = result["full vs baseline"]["metrics"]["mqm"]["label"]
    assert label["use"] == "for reference only" and label["judge_model"] == MQM_MODEL
    assert "agreement check" in label["why"] and "4.5" in label["why"]
    for key in ("comet", "chrf", "bleu", "term_accuracy"):
        assert result["full vs baseline"]["metrics"][key]["label"] == {"use": "result", "why": None}
    assert result["full vs full_no_kb"]["metrics"]["term_accuracy"]["label"] == {"use": "result", "why": None}
    assert "mqm" not in result["full vs full_no_kb"]["metrics"]


def test_consistency_is_a_result_when_the_extractor_is_right_on_90_percent_of_the_checked_occurrences_in_every_run(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)

    result = comparisons(build(dirs, glossary_dir))

    for comparison in result.values():
        label = comparison["metrics"]["term_consistency"]["label"]
        assert label["use"] == "result" and label["why"] is None and label["judge_model"] == JUDGE_MODEL
        assert [(row["config"], row["book"]) for row in label["extractor"]] == [
            (name, book) for name in ("baseline", "full_no_kb", "full") if name in (comparison["config"], comparison["other"])
            for book in (A, B)]
        assert all(row["reportable"] and row["accuracy"] == 1.0 for row in label["extractor"])


def test_consistency_is_for_reference_only_for_a_comparison_when_a_run_of_it_is_under_the_threshold(tmp_path):
    wrong = {("baseline", B, "shiora", 1, 0)}  # baseline on book B: 3 of its 4 checked occurrences are correct
    dirs, glossary_dir = three_by_two(tmp_path, incorrect=wrong)

    result = comparisons(build(dirs, glossary_dir))

    label = result["full vs baseline"]["metrics"]["term_consistency"]["label"]
    assert label["use"] == "for reference only" and f"baseline on {B}" in label["why"] and "90%" in label["why"]
    row = next(row for row in label["extractor"] if (row["config"], row["book"]) == ("baseline", B))
    assert (row["checked"], row["correct"], row["accuracy"], row["reportable"]) == (4, 3, 0.75, False)
    # baseline is not in the other comparison, whose runs are all fine
    assert result["full vs full_no_kb"]["metrics"]["term_consistency"]["label"]["use"] == "result"
    # nothing else is relabelled by it
    assert result["full vs baseline"]["metrics"]["comet"]["label"]["use"] == "result"


def test_a_run_decides_not_the_pooled_counts_of_its_config(tmp_path):
    """The report decides `reportable` on a config's counts pooled over its books; the comparison's label is stricter, as
    Scope says: any run (one config on one book) under the threshold makes the interval for reference only. Baseline
    pools 9 of 10 checked occurrences, which is exactly 90% and so reportable for the report, but on book B it has 3 of 4."""
    dirs, glossary_dir = three_by_two(tmp_path, incorrect={("baseline", B, "shiora", 1, 0)})

    pooled = report.build_report(list(dirs.values()), glossary_dir)["pooled"]["baseline"]["consistency"]["extractor"]
    result = comparisons(build(dirs, glossary_dir))

    assert (pooled["checked"], pooled["correct"], pooled["reportable"]) == (10, 9, True)
    label = result["full vs baseline"]["metrics"]["term_consistency"]["label"]
    assert label["use"] == "for reference only" and f"baseline on {B}" in label["why"]


def test_a_run_with_no_page_in_a_comparison_does_not_decide_its_label(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path, failed={("full", A, 1), ("full", A, 2), ("full", A, 3)})

    result = comparisons(build(dirs, glossary_dir))

    # full has no scored page on book A, so neither comparison has a page there, and its run on A adds nothing to either
    for comparison in result.values():
        assert comparison["pages"] == {A: [], B: [1, 2]}
        label = comparison["metrics"]["term_consistency"]["label"]
        assert label["use"] == "result"
        assert {row["book"] for row in label["extractor"]} == {B}


# --- what the output records ----------------------------------------------------------------------------------------------------

def test_the_output_records_the_inputs_the_method_the_code_and_the_reports_warnings(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)

    document = build(dirs, glossary_dir, resamples=25)

    assert list(document) == ["generated_at", "code", "inputs", "method", "comparisons", "warnings"]
    assert document["code"] == {"git_commit": "c0ffee1", "git_dirty": False}
    assert document["warnings"] == report.build_report(list(dirs.values()), glossary_dir)["warnings"] == []
    inputs = document["inputs"]
    assert [(r["config"], r["book"]) for r in inputs["runs"]] == [(name, book) for name in CONFIGS for book in (A, B)]
    assert inputs["runs"][0] == {
        "directory": str(dirs["baseline", A]), "eval_id": f"20261029T000000Z-baseline-{A}", "config": "baseline", "book": A,
        "segments_sha256": sha256_of(dirs["baseline", A] / "segments.jsonl"),
        "glossary_sha256": sha256_of(glossary_dir / f"{A}.json"),
        "files": {"comet": report.COMET_FILE, "terms": "terms.json", "consistency": f"consistency-{JUDGE_MODEL}.json",
                  "mqm": f"mqm-{MQM_MODEL}.json"}}
    assert inputs["consistency"] == {"judge_model": JUDGE_MODEL, "rules": read_json(
        dirs["baseline", A] / f"consistency-{JUDGE_MODEL}.json")["rules"], "min_rendered": 2, "threshold": 0.9}
    assert inputs["mqm"] == {"judge_model": MQM_MODEL, "file": f"mqm-{MQM_MODEL}.json"}
    assert inputs["glossary_dir"] == str(glossary_dir) and inputs["mqm_judge"] is None
    method = document["method"]
    assert (method["resamples"], method["seed"], method["percentiles"]) == (25, 2027, [2.5, 97.5])
    assert method["protocol"].startswith("eval/PROTOCOL.md")
    assert "random.Random(seed)" in method["generator"] and "rng.choices(pages, k=len(pages))" in method["generator"]
    assert "new one for each comparison" in method["generator"]
    assert "counts each time" in method["duplicated_pages"] and "sorted by the report's book order" in method["pages"]
    assert method["difference"] == "full minus the other config, on each resample"
    assert "linear interpolation" in method["interval"] and "without resampling" in method["point_estimate"]
    assert "gives no difference" in method["undefined"]
    first = document["comparisons"][0]
    assert list(first) == ["name", "config", "other", "n_pages", "pages", "metrics"]
    assert list(first["metrics"]) == ["comet", "chrf", "bleu", "term_accuracy", "term_consistency", "mqm"]
    assert list(first["metrics"]["comet"]) == ["definition", "role", "values", "support", "point_estimate", "interval",
                                               "defined_resamples", "undefined_resamples", "label"]


def test_only_the_decided_metrics_get_an_interval_and_the_primary_ones_are_marked(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)

    result = comparisons(build(dirs, glossary_dir))

    assert list(result["full vs baseline"]["metrics"]) == ["comet", "chrf", "bleu", "term_accuracy", "term_consistency", "mqm"]
    assert list(result["full vs full_no_kb"]["metrics"]) == ["term_accuracy", "term_consistency"]
    roles = {(name, key): metric["role"] for name, c in result.items() for key, metric in c["metrics"].items()}
    assert [k for k, role in roles.items() if role == "primary"] == [("full vs baseline", "comet"),
                                                                    ("full vs full_no_kb", "term_accuracy")]
    document = json.dumps(build(dirs, glossary_dir))
    assert "cost" not in document and "latency" not in document  # neither gets an interval


def test_the_code_state_is_the_projects_git_state(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "git_state", REAL_GIT_STATE)
    dirs, glossary_dir = three_by_two(tmp_path)

    code = build(dirs, glossary_dir, resamples=2)["code"]

    assert bootstrap.ROOT == config.ROOT
    assert code["git_commit"] is None or re.fullmatch("[0-9a-f]{40}", code["git_commit"])
    assert code["git_dirty"] in (True, False, None)


def test_the_bootstrap_changes_none_of_its_input_files(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)
    before = snapshot(*dirs.values(), glossary_dir)

    build(dirs, glossary_dir, resamples=10)

    assert snapshot(*dirs.values(), glossary_dir) == before


def test_a_judge_is_chosen_when_a_run_has_the_files_of_two_judges(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)
    for run_dir in dirs.values():
        write_mqm(run_dir, {}, model="gpt-5.1-2025-11-13")  # no error anywhere: every score is 0

    with pytest.raises(bootstrap.Refused, match="more than one MQM file"):
        build(dirs, glossary_dir)
    default = comparisons(build(dirs, glossary_dir, mqm_judge=MQM_MODEL))["full vs baseline"]["metrics"]["mqm"]
    other = comparisons(build(dirs, glossary_dir, mqm_judge="gpt-5.1-2025-11-13"))["full vs baseline"]["metrics"]["mqm"]

    assert default["values"]["baseline"] != 0.0 and other["values"] == {"full": 0.0, "baseline": 0.0}
    assert default["label"]["judge_model"] == MQM_MODEL and other["label"]["judge_model"] == "gpt-5.1-2025-11-13"


# --- refusals -----------------------------------------------------------------------------------------------------------------

@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("eval_bootstrap_under_test", config.ROOT / "eval" / "bootstrap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(cli, dirs, glossary_dir, out_root, *extra):
    return cli.main([*map(str, dirs), "--glossary-dir", str(glossary_dir), "--out-root", str(out_root), *extra])


def assert_refused(cli, capsys, dirs, glossary_dir, out_root, expected, names, *extra):
    before = snapshot(*dirs, glossary_dir)

    code = run_cli(cli, dirs, glossary_dir, out_root, *extra)

    err = capsys.readouterr().err
    assert code == 2, err
    assert err.startswith("Nothing was written: "), err
    assert expected in err and str(names) in err, err
    assert not out_root.exists()  # no bootstrap directory, not even an empty one
    assert snapshot(*dirs, glossary_dir) == before


def test_every_input_the_report_refuses_is_refused_with_exit_2_the_reports_message_and_nothing_written(cli, tmp_path, capsys):
    cases = refusals(tmp_path)
    assert len(cases) >= 22

    for what, dirs, glossary_dir, expected, names in cases:
        assert_refused(cli, capsys, dirs, glossary_dir, tmp_path / f"bootstrap-{abs(hash(what))}", expected, names)


def test_the_library_raises_refused_with_the_reports_own_message(tmp_path):
    for what, dirs, glossary_dir, expected, names in refusals(tmp_path):
        with pytest.raises(bootstrap.Refused) as raised:
            bootstrap.build_bootstrap(dirs, glossary_dir, resamples=2)
        with pytest.raises(report.Refused) as from_report:
            report.build_report(dirs, glossary_dir)
        assert str(raised.value) == str(from_report.value), what
    assert bootstrap.Refused is report.Refused


def test_two_directories_of_one_config_and_book_are_refused(cli, tmp_path, capsys):
    dirs, glossary_dir = three_by_two(tmp_path)
    (tmp_path / "again").mkdir()
    twin = make_eval(tmp_path / "again", "baseline", A, [page(1)], glossary_dir=glossary_dir)

    assert_refused(cli, capsys, [*dirs.values(), twin], glossary_dir, tmp_path / "out", f"both baseline on {A}", twin)


def test_a_run_without_its_mqm_file_is_refused(cli, tmp_path, capsys):
    dirs, glossary_dir = three_by_two(tmp_path)
    (dirs["full", B] / f"mqm-{MQM_MODEL}.json").unlink()

    assert_refused(cli, capsys, list(dirs.values()), glossary_dir, tmp_path / "out", "the report pools GEMBA-MQM",
                   dirs["full", B])
    assert_refused(cli, capsys, list(dirs.values()), glossary_dir, tmp_path / "out", f"it has no mqm-{MQM_MODEL}.json",
                   dirs["full", B], "--mqm-judge", MQM_MODEL)


def test_a_run_without_its_consistency_file_is_refused(cli, tmp_path, capsys):
    dirs, glossary_dir = three_by_two(tmp_path)
    (dirs["baseline", A] / f"consistency-{JUDGE_MODEL}.json").unlink()

    assert_refused(cli, capsys, list(dirs.values()), glossary_dir, tmp_path / "out", "term consistency for every run or for none",
                   dirs["baseline", A])


def test_runs_that_have_no_consistency_file_at_all_are_refused_though_the_report_accepts_them(cli, tmp_path, capsys):
    dirs, glossary_dir = three_by_two(tmp_path)
    for run_dir in dirs.values():
        (run_dir / f"consistency-{JUDGE_MODEL}.json").unlink()
    report.build_report(list(dirs.values()), glossary_dir)  # the report is fine with it

    assert_refused(cli, capsys, list(dirs.values()), glossary_dir, tmp_path / "out", f"it has no {report.CONSISTENCY_GLOB}",
                   dirs["baseline", A])


def test_runs_that_have_no_mqm_file_at_all_are_refused_though_the_report_accepts_them(cli, tmp_path, capsys):
    dirs, glossary_dir = three_by_two(tmp_path)
    for run_dir in dirs.values():
        (run_dir / f"mqm-{MQM_MODEL}.json").unlink()
    report.build_report(list(dirs.values()), glossary_dir)

    assert_refused(cli, capsys, list(dirs.values()), glossary_dir, tmp_path / "out", f"it has no {report.MQM_GLOB}",
                   dirs["baseline", A])


def test_the_configs_must_be_exactly_baseline_full_no_kb_and_full(cli, tmp_path, capsys):
    dirs, glossary_dir = three_by_two(tmp_path)
    without_full = [d for (name, _), d in dirs.items() if name != "full"]
    report.build_report(without_full, glossary_dir)  # two configs: the report pools them

    assert_refused(cli, capsys, without_full, glossary_dir, tmp_path / "out",
                   "the bootstrap takes the configs baseline, full_no_kb, full, one eval run for each book; it got baseline (",
                   dirs["baseline", A])

    (tmp_path / "more").mkdir()
    translator = [make_eval(tmp_path / "more", "translator", book, [page(1)], glossary_dir=glossary_dir) for book in (A, B)]
    for run_dir in translator:
        write_consistency(run_dir, glossary_dir, {"x": [occ(1, 0, "甲", "甲")]})
        write_mqm(run_dir, {})
    report.build_report([*dirs.values(), *translator], glossary_dir)
    assert_refused(cli, capsys, [*dirs.values(), *translator], glossary_dir, tmp_path / "out", "; translator (", translator[0])


def test_a_comparison_whose_configs_share_no_scored_page_is_refused(cli, tmp_path, capsys):
    dirs, glossary_dir = three_by_two(tmp_path, failed={("full", A, 1), ("full", A, 2), ("full", A, 3),
                                                       ("full", B, 1), ("full", B, 2)})

    assert_refused(cli, capsys, list(dirs.values()), glossary_dir, tmp_path / "out",
                   "full vs baseline: no page was scored by both full and baseline", "full vs baseline")


def test_a_bad_resample_count_is_a_usage_error_and_writes_nothing(cli, tmp_path, capsys):
    dirs, glossary_dir = three_by_two(tmp_path)

    for bad in ("0", "-3", "many"):
        with pytest.raises(SystemExit) as raised:
            run_cli(cli, dirs.values(), glossary_dir, tmp_path / "out", "--resamples", bad)
        assert raised.value.code == 2
    assert not (tmp_path / "out").exists() and "at least 1" in capsys.readouterr().err


# --- writing the output and the command ---------------------------------------------------------------------------------------

def test_the_cli_writes_bootstrap_json_into_a_new_directory_and_prints_the_table(cli, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(bootstrap, "datetime", FixedClock)
    dirs, glossary_dir = three_by_two(tmp_path, failed={("full_no_kb", A, 3)})
    out_root = tmp_path / "out" / "bootstrap"  # does not exist yet

    code = run_cli(cli, dirs.values(), glossary_dir, out_root, "--resamples", "30")

    assert code == 0
    (out_dir,) = list(out_root.iterdir())
    assert out_dir.name == "20261029T120000Z-bootstrap" and sorted(p.name for p in out_dir.iterdir()) == ["bootstrap.json"]
    written = read_json(out_dir / "bootstrap.json")
    assert written == json.loads(json.dumps(build(dirs, glossary_dir, resamples=30) | {"generated_at": written["generated_at"]}))
    captured = capsys.readouterr()
    assert captured.out.endswith(f"Wrote {out_dir / 'bootstrap.json'}\n") and "Computed in " in captured.out
    assert "full vs baseline: 5 pages" in captured.out and "full vs full_no_kb: 4 pages" in captured.out
    assert "30 resamples" in captured.err and "--resamples is for tests" in captured.err  # not the protocol's number


def test_with_the_protocols_resamples_the_command_does_not_warn(cli, tmp_path, capsys, monkeypatch):
    dirs, glossary_dir = tiny(tmp_path)
    monkeypatch.setattr(bootstrap, "RESAMPLES", 12)  # the command compares with it; 10,000 would only make the test slow

    assert cli.main([*map(str, dirs.values()), "--glossary-dir", str(glossary_dir), "--out-root", str(tmp_path / "out"),
                     "--resamples", "12"]) == 0
    assert capsys.readouterr().err == ""


def test_the_command_twice_gives_the_same_numbers_in_two_directories(cli, tmp_path, capsys):
    dirs, glossary_dir = three_by_two(tmp_path)
    out_root = tmp_path / "bootstrap"

    assert run_cli(cli, dirs.values(), glossary_dir, out_root, "--resamples", "50") == 0
    assert run_cli(cli, dirs.values(), glossary_dir, out_root, "--resamples", "50") == 0

    first, second = sorted(out_root.iterdir())
    assert without_time(read_json(first / "bootstrap.json")) == without_time(read_json(second / "bootstrap.json"))


def test_a_second_bootstrap_in_the_same_second_gets_a_directory_of_its_own_and_never_writes_into_an_existing_one(
        tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "datetime", FixedClock)
    dirs, glossary_dir = tiny(tmp_path)
    document = build(dirs, glossary_dir, resamples=3)
    out_root = tmp_path / "bootstrap"
    first = bootstrap.write_bootstrap(document, out_root)
    (first / "bootstrap.json").write_text("first", encoding="utf-8")
    (out_root / "20261029T120000Z-bootstrap-2").mkdir()  # taken, too

    second = bootstrap.write_bootstrap(document, out_root)

    assert (first.name, second.name) == ("20261029T120000Z-bootstrap", "20261029T120000Z-bootstrap-3")
    assert (first / "bootstrap.json").read_text(encoding="utf-8") == "first"
    assert sorted(p.name for p in second.iterdir()) == ["bootstrap.json"]


def test_a_failing_write_leaves_no_directory_and_no_temporary_file(tmp_path, monkeypatch):
    dirs, glossary_dir = tiny(tmp_path)
    document = build(dirs, glossary_dir, resamples=3)
    monkeypatch.setattr(bootstrap.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk on fire")))

    with pytest.raises(OSError, match="disk on fire"):
        bootstrap.write_bootstrap(document, tmp_path / "bootstrap")

    assert list((tmp_path / "bootstrap").iterdir()) == []


def test_the_cli_exits_1_when_the_output_cannot_be_written(cli, tmp_path, capsys):
    dirs, glossary_dir = tiny(tmp_path)
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file", encoding="utf-8")

    code = run_cli(cli, dirs.values(), glossary_dir, blocker / "bootstrap", "--resamples", "3")

    assert code == 1 and "Nothing was written: could not write" in capsys.readouterr().err


def test_the_default_glossary_dir_and_output_dir_are_the_projects(cli):
    assert cli.GLOSSARY_DIR == config.ROOT / "eval" / "gold_glossary"
    assert cli.BOOTSTRAP_DIR == config.ROOT / "runs" / "bootstrap"


def test_the_script_runs_as_a_program(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)
    script = str(config.ROOT / "eval" / "bootstrap.py")
    args = [sys.executable, script, *map(str, dirs.values()), "--glossary-dir", str(glossary_dir),
            "--out-root", str(tmp_path / "r"), "--resamples", "5"]

    done = subprocess.run(args, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "full vs baseline" in done.stdout and (next((tmp_path / "r").iterdir()) / "bootstrap.json").is_file()

    refused = subprocess.run([sys.executable, script, str(tmp_path / "nowhere")], capture_output=True, text=True,
                             env={"PATH": "/usr/bin:/bin"}, timeout=120)
    assert refused.returncode == 2 and "Nothing was written" in refused.stderr


# --- the printed table ---------------------------------------------------------------------------------------------------------

def test_the_table_has_a_row_for_each_interval_marks_the_primary_metric_and_flags_mqm(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path, failed={("full_no_kb", A, 3)})
    document = build(dirs, glossary_dir, resamples=25)

    lines = bootstrap.render(document).splitlines()

    assert "25 resamples, seed 2027" in lines[1] and "c0ffee1 (clean tree)" in lines[1]
    first = lines.index("full vs baseline: 5 pages (tojime_no_siora 3, boureisougi 2)")
    second = lines.index("full vs full_no_kb: 4 pages (tojime_no_siora 2, boureisougi 2)")
    rows = {l.split("  ")[0].strip(): l for l in lines[first + 1:second]}
    assert list(rows) == ["", "COMET (wmt22-comet-da)", "chrF", "BLEU", "term accuracy (non-borderline)",
                          "term consistency (strict)", "GEMBA-MQM"]
    assert rows["COMET (wmt22-comet-da)"].endswith("primary metric") and "primary" not in rows["chrF"]
    assert rows["GEMBA-MQM"].endswith(f"for reference only ({MQM_MODEL})")
    assert "25/25" in rows["chrF"]
    assert "pp" in rows["term accuracy (non-borderline)"] and "pp" in rows["term consistency (strict)"]
    assert [l for l in lines[second + 1:] if l.startswith("term accuracy")][0].endswith("primary metric")
    assert any(l.startswith("sacreBLEU  chrF nrefs:1") for l in lines) and lines[-1] == "warnings: none"


def test_the_tables_numbers_are_the_documents(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)
    document = build(dirs, glossary_dir, resamples=25)
    comet = comparisons(document)["full vs baseline"]["metrics"]["comet"]

    row = [l for l in bootstrap.render(document).splitlines() if l.startswith("COMET")][0]

    assert f"{comet['values']['full']:.4f}" in row and f"{comet['values']['baseline']:.4f}" in row
    assert f"{comet['point_estimate']:+.4f}" in row
    assert f"[{comet['interval']['low']:+.4f}, {comet['interval']['high']:+.4f}]" in row


def test_the_warnings_of_the_report_are_in_the_output_and_the_table(tmp_path):
    dirs, glossary_dir = three_by_two(tmp_path)
    edit_json(dirs["baseline", A] / "manifest.json", lambda d: d.update(git_commit="deadbee"))

    document = build(dirs, glossary_dir, resamples=3)

    assert len(document["warnings"]) == 1 and "different commits" in document["warnings"][0]
    assert "warnings:\n  the runs were made at different commits" in bootstrap.render(document)


# --- what the module is -----------------------------------------------------------------------------------------------

def test_the_names_the_bootstrap_repeats_are_the_reports_and_the_protocols():
    assert bootstrap.CONFIGS == tuple(c for c in report.CONFIG_ORDER if c != "translator")
    assert bootstrap.TEXT in report.CONSISTENCY_TEXTS
    assert bootstrap.COMPARISONS == (("full", "baseline"), ("full", "full_no_kb"))
    assert report.KB_ABLATION["comparison"] == "full vs full_no_kb" and report.KB_CONFIG == "full"
    assert set(bootstrap.PRIMARY) == set(bootstrap.COMPARISONS) == set(bootstrap.METRICS)
    assert bootstrap.PERCENTILES == (2.5, 97.5)


def imported_names(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(part for alias in node.names for part in alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            names.update((node.module or "").split("."))
            names.update(alias.name for alias in node.names)
    return names


def test_the_bootstrap_reads_files_only():
    for path in (config.ROOT / "backend" / "fukidashi" / "evaluation" / "bootstrap.py", config.ROOT / "eval" / "bootstrap.py"):
        names = imported_names(path)
        assert not names & {"sqlite3", "store", "db", "runs", "llm", "openai", "httpx", "requests", "socket", "subprocess"}, path


def test_the_bootstrap_has_no_dependency_beyond_the_standard_library_the_project_and_sacrebleu():
    top_level = set()
    for path in (config.ROOT / "backend" / "fukidashi" / "evaluation" / "bootstrap.py", config.ROOT / "eval" / "bootstrap.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                top_level.add((node.module or "").split(".")[0])
    assert top_level <= set(sys.stdlib_module_names) | {"fukidashi", "sacrebleu"}, top_level
