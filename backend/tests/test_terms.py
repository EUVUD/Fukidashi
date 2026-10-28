"""Term accuracy (OUTLINE 4.7, v0.22): the scorer over `segments.jsonl` lines and eval/terms.py, on made-up text.

Nothing here needs data/, the database or a model. Lines carry the fields of the harness' `segments.jsonl`; where a test
says so, they also carry the Traditional reference, which the scorer must ignore.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import pytest

from fukidashi import config
from fukidashi.evaluation import glossary, terms

BOOK = "boureisougi"  # a dev book in splits.py; the glossaries here hold made-up terms


def entry(id, source, target, *, borderline=False, aliases=(), variants=()):
    return {"id": id, "category": "term", "source": source, "target": target, "borderline": borderline,
            "source_aliases": list(aliases), "target_variants": list(variants)}


def glossary_of(*entries):
    loaded = glossary.parse_glossary({
        "book_title": BOOK, "split": "dev", "opencc_config": "t2s", "drafted_by": "test",
        "reviewed_by_author": None, "entries": list(entries)})
    assert loaded.problems == ()
    return loaded


def line(page, idx, ja, ref, hyp, **more):
    return {"book": BOOK, "page_index": page, "idx": idx, "run_id": 1, "ja": ja, "hyp": hyp, "ref_zh_hans": ref, **more}


def scored(entries, lines):
    return {e.id: e for e in terms.score_terms(glossary_of(*entries), lines).entries}


def places(found):
    return [(o.page_index, o.idx) for o in found.occurrences]


# --- the occurrence rule ------------------------------------------------------------------------------------------

def test_a_segment_whose_reference_lacks_the_term_is_left_out_and_listed():
    lines = [line(1, 0, "リュウが来た", "龙来了", "龙来了"),
             line(1, 1, "リュウの学校", "他的学校", "龙的学校"),  # the reference does not use the term
             line(2, 0, "学校へ行く", "去学校", "龙去学校")]  # the Japanese does not contain the source
    (ryu,) = scored([entry("ryu", "リュウ", "龙")], lines).values()

    assert places(ryu) == [(1, 0)]
    assert ryu.left_out == ((1, 1),)
    assert ryu.hits == 1


def test_an_alias_in_the_japanese_counts():
    lines = [line(1, 0, "ドラゴンが来た", "龙来了", "龙来了"), line(1, 1, "リュウが来た", "龙走了", "走了")]
    (ryu,) = scored([entry("ryu", "リュウ", "龙", aliases=["ドラゴン"])], lines).values()

    assert [(o.idx, o.hit) for o in ryu.occurrences] == [(0, True), (1, False)]


def test_a_variant_in_the_reference_counts_and_so_does_one_in_the_translation():
    lines = [line(1, 0, "シオラが来た", "西奥拉来了", "西奥拉来了"),  # the reference has the variant only
             line(1, 1, "シオラです", "我是希奥拉", "我是西奥拉"),  # the translation has the variant, the reference the target
             line(1, 2, "シオラ!", "希奥拉!", "希奥!")]
    (shiora,) = scored([entry("shiora", "シオラ", "希奥拉", variants=["西奥拉"])], lines).values()

    assert [(o.idx, o.hit) for o in shiora.occurrences] == [(0, True), (1, True), (2, False)]
    assert shiora.left_out == ()


def test_matching_is_by_substring_so_a_longer_word_around_the_term_still_matches():
    lines = [line(1, 0, "ガンマンたち", "枪手们", "那些枪手")]
    (gunman,) = scored([entry("gunman", "ガンマン", "枪手")], lines).values()

    assert [(o.idx, o.hit) for o in gunman.occurrences] == [(0, True)]


def test_a_full_width_half_width_difference_matches_after_nfkc():
    lines = [
        line(1, 0, "ガンマンだ", "是ＡＢＣ型", "是ABC型"),  # full-width letters in the reference, half-width in the translation
        line(1, 1, "ｶﾞﾝﾏﾝだ", "是ABC型", "是ＡＢＣ型"),  # half-width katakana in the Japanese, the other way round
    ]
    (gunman,) = scored([entry("gunman", "ガンマン", "ABC")], lines).values()
    assert [(o.idx, o.hit) for o in gunman.occurrences] == [(0, True), (1, True)]

    # and the glossary's own forms are normalised too
    (half,) = scored([entry("gunman", "ｶﾞﾝﾏﾝ", "ＡＢＣ")], lines).values()
    assert [(o.idx, o.hit) for o in half.occurrences] == [(0, True), (1, True)]


def test_an_empty_alias_or_variant_matches_nothing():
    lines = [line(1, 0, "リュウ", "他", "他")]
    loaded = glossary.Glossary("b", "dev", "t2s", "t", None,
                               (glossary.Entry("e", "term", "リュウ", "龙", ("",), ("",)),))

    (found,) = terms.score_terms(loaded, lines).entries
    assert found.occurrences == () and found.left_out == ((1, 0),)


def test_occurrences_come_in_reading_order_whatever_the_order_of_lines():
    lines = [line(2, 0, "リュウ", "龙", "龙"), line(1, 1, "リュウ", "龙", "龙"), line(1, 0, "リュウ", "龙", "x")]
    (ryu,) = scored([entry("ryu", "リュウ", "龙")], lines).values()

    assert places(ryu) == [(1, 0), (1, 1), (2, 0)]


# --- hits and scores ----------------------------------------------------------------------------------------------

ENTRIES = [
    entry("alpha", "アルファ", "甲"),  # 4 occurrences, 3 hits, and a fifth segment the reference leaves out
    entry("beta", "ベータ", "乙"),  # 1 occurrence, no hit
    entry("gamma", "ガンマ", "丙"),  # no occurrence at all
    entry("delta", "デルタ", "丁", borderline=True),  # 3 occurrences, 3 hits
    entry("epsilon", "イプシ", "戊", borderline=True),  # no occurrence
]
LINES = [
    line(1, 0, "アルファ", "甲", "甲"), line(1, 1, "アルファ", "甲", "甲"), line(2, 0, "アルファ", "甲", "甲"),
    line(2, 1, "アルファ", "甲", "错"),
    line(2, 2, "アルファ", "没有", "甲"),  # left out, although the translation has the term
    line(3, 0, "ベータ", "乙", "错"),
    line(3, 1, "デルタ", "丁", "丁"), line(3, 2, "デルタ", "丁", "丁"), line(4, 0, "デルタ", "丁", "丁"),
]


def test_the_scores_equal_the_hand_computed_values():
    scores = terms.score_terms(glossary_of(*ENTRIES), LINES)

    # primary: alpha 3/4, beta 0/1 (gamma has no occurrence): pooled 3/5, mean (0.75 + 0) / 2
    assert scores.primary == terms.SetScore(
        entries=3, entries_with_occurrences=2, occurrences=5, hits=3, accuracy=0.6, mean_over_entries=0.375)
    # all entries: add delta 3/3 (epsilon has none): pooled 6/8, mean (0.75 + 0 + 1) / 3
    assert scores.all_entries.entries == 5 and scores.all_entries.entries_with_occurrences == 3
    assert (scores.all_entries.occurrences, scores.all_entries.hits) == (8, 6)
    assert scores.all_entries.accuracy == 0.75
    assert scores.all_entries.mean_over_entries == pytest.approx(1.75 / 3)


def test_each_entry_keeps_its_counts_its_hits_and_its_left_out_segments():
    found = {e.id: e for e in terms.score_terms(glossary_of(*ENTRIES), LINES).entries}

    assert [(o.page_index, o.idx, o.hit) for o in found["alpha"].occurrences] == [
        (1, 0, True), (1, 1, True), (2, 0, True), (2, 1, False)]
    assert (found["alpha"].hits, found["alpha"].accuracy, found["alpha"].left_out) == (3, 0.75, ((2, 2),))
    assert (found["beta"].hits, found["beta"].accuracy) == (0, 0.0)
    assert (found["gamma"].hits, found["gamma"].accuracy, found["gamma"].occurrences) == (0, None, ())
    assert [(e.id, e.borderline) for e in terms.score_terms(glossary_of(*ENTRIES), LINES).entries] == [
        ("alpha", False), ("beta", False), ("gamma", False), ("delta", True), ("epsilon", True)]


def test_an_entry_with_no_occurrence_is_left_out_of_the_mean_not_counted_as_zero():
    with_gamma = terms.score_terms(glossary_of(*ENTRIES), LINES)
    without = terms.score_terms(glossary_of(*[e for e in ENTRIES if e["id"] != "gamma"]), LINES)

    assert with_gamma.primary.mean_over_entries == without.primary.mean_over_entries == 0.375
    assert with_gamma.primary.entries == 3 and without.primary.entries == 2


def test_a_set_with_no_occurrence_scores_none_not_zero():
    nothing = terms.score_terms(glossary_of(entry("gamma", "ガンマ", "丙")), LINES)
    assert nothing.primary.accuracy is None and nothing.primary.mean_over_entries is None
    assert nothing.all_entries.accuracy is None and nothing.all_entries.mean_over_entries is None
    assert (nothing.primary.occurrences, nothing.primary.hits) == (0, 0)

    only_borderline = terms.score_terms(glossary_of(*ENTRIES[3:]), LINES)
    assert only_borderline.primary.accuracy is None and only_borderline.primary.entries == 0
    assert only_borderline.all_entries.accuracy == 1.0

    shown = nothing.to_dict()["scores"]["primary"]
    assert shown["accuracy"] is None and shown["mean_over_entries"] is None
    assert json.loads(json.dumps(nothing.to_dict()))["scores"]["all_entries"]["accuracy"] is None


def test_the_pooled_score_and_the_mean_over_entries_differ_when_entries_are_unequal():
    lines = [line(1, i, "アルファ", "甲", "甲") for i in range(9)] + [line(2, 0, "ベータ", "乙", "错")]
    scores = terms.score_terms(glossary_of(*ENTRIES[:2]), lines)

    assert scores.primary.accuracy == 0.9 and scores.primary.mean_over_entries == 0.5


def test_to_dict_holds_the_scores_and_every_entrys_evidence():
    shown = terms.score_terms(glossary_of(*ENTRIES), LINES).to_dict()

    assert shown["scores"]["primary"] == {
        "entries": 3, "entries_with_occurrences": 2, "occurrences": 5, "hits": 3, "accuracy": 0.6,
        "mean_over_entries": 0.375}
    alpha = shown["entries"][0]
    assert alpha == {
        "id": "alpha", "borderline": False, "occurrences": 4, "hits": 3, "accuracy": 0.75,
        "left_out": [{"page_index": 2, "idx": 2}],
        "segments": [{"page_index": 1, "idx": 0, "hit": True}, {"page_index": 1, "idx": 1, "hit": True},
                     {"page_index": 2, "idx": 0, "hit": True}, {"page_index": 2, "idx": 1, "hit": False}]}
    assert [e["id"] for e in shown["entries"]] == [e["id"] for e in ENTRIES]


# --- what the scorer reads ----------------------------------------------------------------------------------------

def test_the_traditional_reference_is_never_read():
    plain = terms.score_terms(glossary_of(*ENTRIES), LINES)
    # a Traditional reference that would change every answer if it were read: it has the term where the Simplified one
    # does not, and the other way round
    trap = [{**l, "ref_zh_hant": "甲乙丙丁戊" if "没有" in l["ref_zh_hans"] else "没有"} for l in LINES]

    assert terms.score_terms(glossary_of(*ENTRIES), trap) == plain
    assert all("ref_zh_hant" not in l for l in LINES)  # and lines without it are scored, as `plain` shows


@pytest.mark.parametrize("field", ["page_index", "idx", "ja", "hyp", "ref_zh_hans"])
def test_a_line_without_a_field_it_reads_is_an_error(field):
    broken = {k: v for k, v in line(1, 0, "リュウ", "龙", "龙").items() if k != field}

    with pytest.raises(terms.TermsError, match=field):
        terms.score_terms(glossary_of(entry("ryu", "リュウ", "龙")), [broken])


@pytest.mark.parametrize("change", [{"hyp": None}, {"idx": "0"}, {"page_index": True}, {"ja": 3}])
def test_a_line_with_a_field_of_the_wrong_type_is_an_error(change):
    with pytest.raises(terms.TermsError):
        terms.score_terms(glossary_of(entry("ryu", "リュウ", "龙")), [{**line(1, 0, "リュウ", "龙", "龙"), **change}])
    with pytest.raises(terms.TermsError, match="not a JSON object"):
        terms.score_terms(glossary_of(entry("ryu", "リュウ", "龙")), ["text"])


def imported_names(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(part for alias in node.names for part in alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            names.update((node.module or "").split("."))
            names.update(alias.name for alias in node.names)
    return names


SCORER_CODE = [config.ROOT / "backend" / "fukidashi" / "evaluation" / "terms.py", config.ROOT / "eval" / "terms.py"]


def test_the_scorer_cannot_reach_the_database_or_the_traditional_reference():
    for path in SCORER_CODE:
        assert not imported_names(path) & {"sqlite3", "store", "db", "runs"}, path
        text = path.read_text(encoding="utf-8")
        for word in ("store", "sqlite", "ref_zh_hant"):
            assert word not in text, (path, word)


# --- eval/terms.py ------------------------------------------------------------------------------------------------

@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("eval_terms_under_test", config.ROOT / "eval" / "terms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_run(tmp_path, lines=LINES, entries=ENTRIES, *, manifest=None, name="run-1"):
    """An eval directory and a glossary directory in tmp_path; returns (run_dir, glossary_dir)."""
    run_dir, gloss_dir = tmp_path / name, tmp_path / f"glossaries-{name}"
    run_dir.mkdir()
    gloss_dir.mkdir()
    (gloss_dir / f"{BOOK}.json").write_text(json.dumps({
        "book_title": BOOK, "split": "dev", "opencc_config": "t2s", "drafted_by": "test",
        "reviewed_by_author": "2026-10-28", "entries": entries}, ensure_ascii=False), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({
        "eval_id": "20261028T000000Z-baseline", "book": BOOK, "finished_at": "2026-10-28T00:01:00+00:00",
        "pages": [{"book": BOOK, "page_index": p, "status": "scored"} for p in (1, 2, 3, 4)],
        **(manifest or {})}), encoding="utf-8")
    (run_dir / "segments.jsonl").write_text(
        "".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
    return run_dir, gloss_dir


def run(cli, run_dir, gloss_dir):
    return cli.main([str(run_dir), "--glossary-dir", str(gloss_dir)])


def test_the_cli_writes_terms_json_with_every_field(cli, tmp_path, capsys):
    run_dir, gloss_dir = make_run(tmp_path)

    assert run(cli, run_dir, gloss_dir) == 0

    assert sorted(p.name for p in run_dir.iterdir()) == ["manifest.json", "segments.jsonl", "terms.json"]  # no .tmp
    written = json.loads((run_dir / "terms.json").read_text(encoding="utf-8"))
    assert list(written)[:5] == ["eval_id", "book", "glossary_sha256", "rules", "pages_scored"]
    assert written["eval_id"] == "20261028T000000Z-baseline" and written["book"] == BOOK
    assert written["glossary_sha256"] == hashlib.sha256((gloss_dir / f"{BOOK}.json").read_bytes()).hexdigest()
    assert written["rules"] == "OUTLINE §4.7 (v0.22)"
    assert written["pages_scored"] == [1, 2, 3, 4] and written["pages_not_scored"] == []
    assert written["scores"]["primary"]["accuracy"] == 0.6
    assert written["scores"]["primary"]["mean_over_entries"] == 0.375
    assert written["scores"]["all_entries"]["accuracy"] == 0.75
    assert written["scores"]["all_entries"]["hits"] == 6 and written["scores"]["all_entries"]["occurrences"] == 8
    assert [e["id"] for e in written["entries"]] == [e["id"] for e in ENTRIES]
    alpha = written["entries"][0]
    assert (alpha["borderline"], alpha["occurrences"], alpha["hits"]) == (False, 4, 3)
    assert alpha["left_out"] == [{"page_index": 2, "idx": 2}]
    assert alpha["segments"][-1] == {"page_index": 2, "idx": 1, "hit": False}
    assert written["entries"][2]["accuracy"] is None  # gamma: no occurrence
    assert "primary (not borderline): 3/5 = 60.0% pooled" in capsys.readouterr().out


def test_the_cli_replaces_terms_json_through_a_temporary_file(cli, tmp_path, monkeypatch):
    run_dir, gloss_dir = make_run(tmp_path)
    (run_dir / "terms.json").write_text("old", encoding="utf-8")
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: replaced.append((src, dst)) or real_replace(src, dst))

    assert run(cli, run_dir, gloss_dir) == 0

    ((src, dst),) = replaced
    assert os.path.basename(src) == "terms.json.tmp" and os.path.basename(dst) == "terms.json"
    assert json.loads((run_dir / "terms.json").read_text(encoding="utf-8"))["book"] == BOOK
    assert not (run_dir / "terms.json.tmp").exists()


def test_the_cli_names_the_pages_the_run_did_not_score(cli, tmp_path):
    pages = [{"book": BOOK, "page_index": 1, "status": "scored"}, {"book": BOOK, "page_index": 2, "status": "failed"}]
    run_dir, gloss_dir = make_run(tmp_path, [line(1, 0, "アルファ", "甲", "甲")], manifest={"pages": pages})

    assert run(cli, run_dir, gloss_dir) == 0

    written = json.loads((run_dir / "terms.json").read_text(encoding="utf-8"))
    assert written["pages_scored"] == [1] and written["pages_not_scored"] == [{"page_index": 2, "status": "failed"}]


def refused_cases(tmp_path):
    """(what is wrong, run dir, glossary dir, text expected in the message), each in a directory of its own."""
    cases = []

    run_dir, gloss_dir = make_run(tmp_path, name="no-manifest")
    (run_dir / "manifest.json").unlink()
    cases.append(("no manifest", run_dir, gloss_dir, "manifest.json"))

    run_dir, gloss_dir = make_run(tmp_path, manifest={"finished_at": None}, name="unfinished")
    cases.append(("unfinished", run_dir, gloss_dir, "not finished"))

    run_dir, gloss_dir = make_run(tmp_path, [], name="empty")
    cases.append(("empty segments", run_dir, gloss_dir, "no segments"))

    run_dir, gloss_dir = make_run(tmp_path, name="no-segments-file")
    (run_dir / "segments.jsonl").unlink()
    cases.append(("no segments file", run_dir, gloss_dir, "segments.jsonl"))

    run_dir, gloss_dir = make_run(tmp_path, manifest={"book": "tojime_no_siora"}, name="no-glossary")
    cases.append(("book without a glossary", run_dir, gloss_dir, "no glossary for book 'tojime_no_siora'"))

    bad = [{**entries, "borderline": "yes"} for entries in ENTRIES[:1]] + ENTRIES[1:]
    run_dir, gloss_dir = make_run(tmp_path, entries=bad, name="bad-glossary")
    cases.append(("glossary with problems", run_dir, gloss_dir, "borderline"))

    run_dir, gloss_dir = make_run(tmp_path, entries=[{k: v for k, v in e.items() if k != "borderline"}
                                                     for e in ENTRIES], name="missing-field")
    cases.append(("borderline missing", run_dir, gloss_dir, "borderline"))

    run_dir, gloss_dir = make_run(tmp_path, name="broken-line")
    (run_dir / "segments.jsonl").write_text("{not json\n", encoding="utf-8")
    cases.append(("a line that is not JSON", run_dir, gloss_dir, "line 1"))

    run_dir, gloss_dir = make_run(tmp_path, [{"page_index": 1, "idx": 0, "ja": "x"}], name="short-line")
    cases.append(("a line without its fields", run_dir, gloss_dir, "hyp"))

    cases.append(("a directory that is not there", tmp_path / "nowhere", gloss_dir, "not a directory"))
    return cases


def test_the_cli_refuses_with_exit_code_2_and_writes_nothing(cli, tmp_path, capsys):
    for what, run_dir, gloss_dir, message in refused_cases(tmp_path):
        before = sorted(p.name for p in run_dir.iterdir()) if run_dir.exists() else None

        assert run(cli, run_dir, gloss_dir) == 2, what

        assert capsys.readouterr().err.count(message) >= 1, what
        assert (sorted(p.name for p in run_dir.iterdir()) if run_dir.exists() else None) == before, what
        assert not (run_dir / "terms.json").exists(), what


def test_a_glossary_of_another_book_is_refused(cli, tmp_path, capsys):
    run_dir, gloss_dir = make_run(tmp_path)
    path = gloss_dir / f"{BOOK}.json"
    path.write_text(path.read_text(encoding="utf-8").replace(f'"{BOOK}"', '"tojime_no_siora"', 1), encoding="utf-8")

    assert run(cli, run_dir, gloss_dir) == 2
    assert "glossary of 'tojime_no_siora'" in capsys.readouterr().err
    assert not (run_dir / "terms.json").exists()


def test_the_script_runs_as_a_program(tmp_path):
    run_dir, gloss_dir = make_run(tmp_path)
    script = str(config.ROOT / "eval" / "terms.py")
    env = {"PATH": "/usr/bin:/bin"}

    done = subprocess.run([sys.executable, script, str(run_dir), "--glossary-dir", str(gloss_dir)],
                          capture_output=True, text=True, env=env, timeout=60)
    assert done.returncode == 0, done.stderr
    assert (run_dir / "terms.json").is_file()

    refused = subprocess.run([sys.executable, script, str(tmp_path / "nowhere")],
                             capture_output=True, text=True, env=env, timeout=60)
    assert refused.returncode == 2 and "Cannot score" in refused.stderr
