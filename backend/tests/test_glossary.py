"""The gold glossary's validator, occurrences, candidates and review sheet, on a tiny dataset built in tmp_path.

The fixture's references are Traditional, as OpenMantra's are; nothing here needs data/ and nothing reads the database.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import socket
from pathlib import Path

import pytest

from fukidashi import config
from fukidashi.datasets.openmantra import Page, Segment, load_pages
from fukidashi.evaluation import glossary, splits

BOOK = "boureisougi"  # a dev book in splits.py; in this fixture it holds made-up text

# (text_ja, text_zh in Traditional). 龍/學/謝/斷 are the characters t2s changes.
PAGES = {
    1: [("リュウが来た", "龍來了"), ("リュウの学校", "龍的學校")],
    2: [("リュウさん!", "龍先生"), ("学校へ行く", "去學校")],
    3: [("ありがとう", "謝謝"), ("杖が折れた", "杖斷了")],
}


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "open-mantra-dataset"
    root.mkdir()
    pages = [
        {"page_index": index, "image_paths": {"ja": f"images/{BOOK}/ja/{index - 1:03d}.jpg"}, "frame": [],
         "text": [{"x": 1, "y": 2, "w": 3, "h": 4, "text_ja": ja, "text_en": "EN", "text_zh": zh}
                  for ja, zh in segments]}
        for index, segments in PAGES.items()
    ]
    (root / "annotation.json").write_text(json.dumps([{"book_title": BOOK, "pages": pages}]), encoding="utf-8")
    return root


@pytest.fixture
def pages(root):
    return load_pages(root)


RYU = {"id": "ryu", "category": "character", "source": "リュウ", "target": "龙"}
SCHOOL = {"id": "school", "category": "term", "source": "学校", "target": "学校"}


def raw(*entries, **header):
    return {"book_title": BOOK, "split": "dev", "opencc_config": "t2s", "drafted_by": "test",
            "reviewed_by_author": None, "entries": list(entries), **header}


def problems(pages, *entries, **header):
    return glossary.check(glossary.parse_glossary(raw(*entries, **header)), pages)


def codes(found):
    return [p.code for p in found]


def entry(**fields):
    return glossary.Entry(**{"id": "e", "category": "term", "source": "学校", "target": "学校", **fields})


# --- splits ------------------------------------------------------------------------------------------------------

def test_the_dev_and_test_books_are_the_ones_in_outline_8():
    assert splits.DEV_BOOKS == ("tojime_no_siora", "boureisougi")
    assert set(splits.TEST_BOOKS) == {"balloon_dream", "tencho_isoro", "rasetugari"}
    assert {splits.split_of(b) for b in splits.DEV_BOOKS} == {"dev"}
    assert {splits.split_of(b) for b in splits.TEST_BOOKS} == {"test"}
    assert splits.split_of("no_such_book") is None


# --- check: one test per rule -------------------------------------------------------------------------------------

def test_a_valid_glossary_has_no_problems(pages):
    assert problems(pages, RYU, SCHOOL) == []


def test_an_unknown_category_is_a_problem(pages):
    assert codes(problems(pages, {**RYU, "category": "place"})) == ["category"]


def test_a_duplicate_id_is_a_problem(pages):
    assert codes(problems(pages, RYU, {**SCHOOL, "id": "ryu"})) == ["duplicate_id"]


def test_a_source_or_alias_found_in_no_segment_is_a_problem(pages):
    only_alias = problems(pages, {**RYU, "source_aliases": ["ドラゴン"]})
    assert codes(only_alias) == ["source_not_found"]
    assert "ドラゴン" in only_alias[0].message
    assert "source_not_found" in codes(problems(pages, {**RYU, "source": "ネコ"}))


def test_an_entry_in_fewer_than_two_segments_is_a_problem(pages):
    once = {"id": "thanks", "category": "term", "source": "ありがとう", "target": "谢谢"}
    assert codes(problems(pages, once)) == ["too_few_segments"]
    # the alias adds a second segment, so the entry is fine
    assert problems(pages, {**once, "source_aliases": ["杖"], "target_variants": ["断"]}) == []


def test_a_target_or_variant_found_in_no_reference_of_its_segments_is_a_problem(pages):
    assert codes(problems(pages, {**RYU, "target": "云"})) == ["target_not_found"]
    assert codes(problems(pages, {**RYU, "target_variants": ["云"]})) == ["target_not_found"]


def test_a_target_found_only_in_segments_without_the_source_is_a_problem(pages):
    # 谢谢 is in the converted reference of page 3, which does not contain リュウ
    assert codes(problems(pages, {**RYU, "target": "谢谢"})) == ["target_not_found"]


def test_a_source_or_alias_used_by_two_entries_is_a_problem(pages):
    twice = problems(pages, SCHOOL, {**SCHOOL, "id": "school-2"})
    assert codes(twice) == ["term_reused"]
    assert "school" in twice[0].where and "school-2" in twice[0].where
    assert codes(problems(pages, {**SCHOOL, "source_aliases": ["リュウ"]}, {**RYU, "id": "other"})) == ["term_reused"]
    assert sorted(codes(problems(pages, {**SCHOOL, "source_aliases": ["学校"]}))) == [
        "alias_contains_source", "term_reused"]


def test_a_term_longer_than_20_characters_is_a_problem(pages):
    assert "term_too_long" in codes(problems(pages, {**RYU, "target": "龙" * 21}))
    assert "term_too_long" not in codes(problems(pages, {**RYU, "target": "龙" * 20}))
    assert "term_too_long" in codes(problems(pages, {**RYU, "source": "リュウ" * 7}))
    assert "term_too_long" in codes(problems(pages, {**RYU, "target_variants": ["龙" * 21]}))


def test_a_split_that_does_not_match_splits_py_is_a_problem(pages):
    wrong = problems(pages, RYU, SCHOOL, split="test")
    assert codes(wrong) == ["split"]
    assert "'dev'" in wrong[0].message


def test_a_book_the_dataset_does_not_have_is_a_problem(pages):
    assert "book" in codes(problems(pages, book_title="balloon_dream", split="test"))


def test_a_book_outside_splits_py_is_a_problem(root, pages):
    data = json.loads((root / "annotation.json").read_text(encoding="utf-8"))
    data[0]["book_title"] = "unlisted_book"
    (root / "annotation.json").write_text(json.dumps(data), encoding="utf-8")
    found = problems(load_pages(root), book_title="unlisted_book")
    assert codes(found) == ["split"] and "neither split" in found[0].message


def test_another_opencc_config_is_a_problem(pages):
    assert codes(problems(pages, RYU, SCHOOL, opencc_config="tw2sp")) == ["opencc"]


def test_an_alias_that_contains_the_source_is_a_problem(pages):
    assert codes(problems(pages, {**RYU, "source_aliases": ["リュウさん"]})) == ["alias_contains_source"]


def test_a_variant_that_repeats_the_target_is_a_problem(pages):
    assert codes(problems(pages, {**RYU, "target_variants": ["龙"]})) == ["variant_repeats_target"]


def test_an_empty_alias_is_a_problem(pages):
    assert codes(problems(pages, {**RYU, "source_aliases": [""]})) == ["term_empty"]


def test_each_problem_is_reported_separately(pages):
    found = problems(pages, {**RYU, "category": "place", "target": "云"}, {**SCHOOL, "id": "ryu"})
    assert sorted(codes(found)) == ["category", "duplicate_id", "target_not_found"]


@pytest.mark.parametrize("change, field", [
    ({"id": None}, "id"), ({"category": ""}, "category"), ({"source": 5}, "source"), ({"target": None}, "target"),
    ({"source_aliases": "x"}, "source_aliases"), ({"target_variants": [1]}, "target_variants"),
    ({"note": 3}, "note"), ({"sourse": "x"}, "sourse"),
])
def test_an_entry_of_the_wrong_shape_is_a_problem_and_left_out(pages, change, field):
    loaded = glossary.parse_glossary(raw({**RYU, **change}, SCHOOL))

    assert [e.id for e in loaded.entries] == ["school"]
    assert [p.code for p in loaded.problems] == ["shape"]
    assert field in loaded.problems[0].message


@pytest.mark.parametrize("header", [
    {"book_title": ""}, {"split": None}, {"opencc_config": 1}, {"drafted_by": " "},
    {"reviewed_by_author": "yesterday"}, {"entries": {}}, {"extra": 1},
])
def test_a_header_of_the_wrong_shape_is_a_problem(header):
    assert "shape" in [p.code for p in glossary.parse_glossary(raw(RYU, **header)).problems]


def test_a_missing_reviewed_by_author_is_a_problem():
    data = raw(RYU)
    del data["reviewed_by_author"]
    assert [p.code for p in glossary.parse_glossary(data).problems] == ["shape"]


def test_a_date_in_reviewed_by_author_is_accepted(pages):
    reviewed = glossary.parse_glossary(raw(RYU, SCHOOL, reviewed_by_author="2026-10-28"))
    assert reviewed.reviewed_by_author == "2026-10-28"
    assert glossary.check(reviewed, pages) == []


def test_a_file_that_is_not_a_glossary_is_a_problem_not_an_exception(tmp_path, pages):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert codes(glossary.check(glossary.load_glossary(bad), pages)) == ["shape"]
    assert codes(glossary.check(glossary.load_glossary(tmp_path / "missing.json"), pages)) == ["shape"]
    bad.write_text("[]", encoding="utf-8")
    assert codes(glossary.check(glossary.load_glossary(bad), pages)) == ["shape"]


# --- occurrences --------------------------------------------------------------------------------------------------

def test_occurrences_are_the_segments_containing_the_source_or_an_alias(pages):
    found = glossary.occurrences(entry(source="リュウ", source_aliases=("学校",), target="龙"), pages)

    assert [(o.page_index, o.region_id) for o in found] == [(1, 0), (1, 1), (2, 0), (2, 1)]
    assert glossary.occurrences(entry(source="リュウ", target="龙"), pages)[-1].page_index == 2
    assert glossary.occurrences(entry(source="ネコ", target="猫"), pages) == []


def test_occurrences_are_marked_by_the_target_or_a_variant(pages):
    found = glossary.occurrences(entry(source="学校", target="学校"), pages)
    assert [o.covered for o in found] == [True, True]

    mixed = glossary.occurrences(entry(source="リュウ", source_aliases=("学校",), target="龙"), pages)
    assert [(o.region_id, o.covered) for o in mixed] == [(0, True), (1, True), (0, True), (1, False)]
    variant = glossary.occurrences(
        entry(source="リュウ", source_aliases=("学校",), target="龙", target_variants=("学校",)), pages)
    assert all(o.covered for o in variant)


def test_matching_uses_the_converted_reference(pages):
    # the reference holds 龍 and 學校 only in Traditional characters
    [first, *_] = glossary.occurrences(entry(source="リュウ", target="龙"), pages)
    assert "龍" not in first.text_zh and "龙" in first.text_zh and first.covered

    assert not any(o.covered for o in glossary.occurrences(entry(source="リュウ", target="龍"), pages))
    assert codes(problems(pages, {**RYU, "target": "龍"})) == ["target_not_found"]


def test_occurrences_come_in_reading_order_whatever_the_order_of_pages(pages):
    shuffled = list(reversed(pages))
    found = glossary.occurrences(entry(source="リュウ", target="龙"), shuffled)
    assert [(o.page_index, o.region_id) for o in found] == [(1, 0), (1, 1), (2, 0)]


def test_coverage_is_the_share_of_covered_segments(pages):
    found = glossary.occurrences(entry(source="リュウ", source_aliases=("学校",), target="龙"), pages)
    assert glossary.coverage(found) == 0.75
    assert glossary.coverage([]) is None


def test_summary_counts_entries_by_category_and_pages(pages):
    stats = glossary.summarize(glossary.parse_glossary(raw(RYU, SCHOOL)), pages)

    assert stats.entries == 2 and stats.by_category == {"character": 1, "term": 1}
    assert stats.multi_page == 2
    assert stats.mean_coverage == 1.0 and stats.pooled_coverage == 1.0


# --- candidates and the review sheet ------------------------------------------------------------------------------

def test_candidates_are_found_by_pattern_and_listed_from_two_segments(pages):
    found = {c.text: c for c in glossary.find_candidates(pages)}

    assert found["リュウ"].segments == 3 and found["リュウ"].pages == 2
    assert found["リュウ"].kinds == ("honorific", "katakana")
    assert found["学校"].kinds == ("kanji",) and found["学校"].segments == 2
    assert "ありがとう" not in found  # one segment, and hiragana
    assert [o.region_id for o in found["リュウ"].examples] == [0, 1, 0]
    assert all(not o.covered for o in found["リュウ"].examples)


def test_candidates_come_in_a_fixed_order(pages):
    texts = [c.text for c in glossary.find_candidates(pages)]
    assert texts == ["リュウ", "学校"]
    assert glossary.render_candidates(BOOK, pages) == glossary.render_candidates(BOOK, list(reversed(pages)))


def test_the_review_sheet_has_a_row_per_entry_and_quotes_the_evidence(pages):
    loaded = glossary.parse_glossary(raw({**RYU, "source_aliases": ["学校"], "note": "A test note."}))
    sheet = glossary.render_sheet(loaded, pages)

    assert "| 1 | ryu | character | リュウ (学校) | 龙 | 4 | 2 | 75% |" in sheet
    assert "Reviewed by author: not yet" in sheet
    assert "Note: A test note." in sheet
    assert "- 2#1 [NOT covered] ja: 学校へ行く / zh: 去学校" in sheet  # quoted converted, uncovered ones included
    assert "龙 3/4" in sheet


def test_the_sheet_quotes_a_few_segments_spread_over_the_book():
    # ten pages with the term: the reference has the target on pages 1-6 and something else on pages 7-10
    book = [Page(BOOK, i, Path("x.jpg"), [Segment(0, 0, 0, 0, 0, "リュウ", "EN", "龍" if i <= 6 else "別人")])
            for i in range(1, 11)]
    sheet = glossary.render_sheet(glossary.parse_glossary(raw(RYU)), book)

    quoted = [line for line in sheet.splitlines() if line.startswith("- ")]
    assert [line.split("#")[0] for line in quoted] == ["- 1", "- 3", "- 6", "- 7", "- 10"]
    assert [("NOT covered" in line) for line in quoted] == [False, False, False, True, True]
    assert "| 1 | ryu | character | リュウ | 龙 | 10 | 10 | 60% |" in sheet


# --- the commands -------------------------------------------------------------------------------------------------

@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("eval_glossary", config.ROOT / "eval" / "glossary.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def write_glossary(directory, data):
    directory.mkdir(exist_ok=True)
    path = directory / f"{BOOK}.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_candidates_writes_one_markdown_file_and_is_repeatable(cli, root, tmp_path, no_network):
    out = tmp_path / "out"
    argv = ["--dataset", str(root), "--out", str(out), "candidates", "--book", BOOK]

    assert cli.main(argv) == 0
    first = (out / f"{BOOK}-candidates.md").read_text(encoding="utf-8")
    assert cli.main(argv) == 0

    assert sorted(p.name for p in out.iterdir()) == [f"{BOOK}-candidates.md"]
    assert (out / f"{BOOK}-candidates.md").read_text(encoding="utf-8") == first
    assert "### リュウ: 3 segments, 2 pages" in first


def test_sheet_writes_one_markdown_file(cli, root, tmp_path, no_network):
    out, directory = tmp_path / "out", tmp_path / "glossaries"
    write_glossary(directory, raw(RYU, SCHOOL))

    code = cli.main(["--dataset", str(root), "--out", str(out), "--glossary-dir", str(directory),
                     "sheet", "--book", BOOK])

    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == [f"{BOOK}-review.md"]
    assert "| 1 | ryu |" in (out / f"{BOOK}-review.md").read_text(encoding="utf-8")
    assert sorted(p.name for p in directory.iterdir()) == [f"{BOOK}.json"]  # the glossary itself is not touched


def test_sheet_and_candidates_refuse_what_they_cannot_use(cli, root, tmp_path, capsys):
    out, directory = tmp_path / "out", tmp_path / "glossaries"
    base = ["--dataset", str(root), "--out", str(out), "--glossary-dir", str(directory)]

    assert cli.main([*base, "candidates", "--book", "no_such_book"]) == 2
    assert cli.main([*base, "sheet", "--book", BOOK]) == 2  # no glossary file
    write_glossary(directory, raw({**RYU, "category": None}))
    assert cli.main([*base, "sheet", "--book", BOOK]) == 2  # a file of the wrong shape
    assert not out.exists()
    assert "category" in capsys.readouterr().err


def test_check_exits_zero_for_a_valid_glossary_and_reports_its_counts(cli, root, tmp_path, capsys, no_network):
    path = write_glossary(tmp_path / "glossaries", raw(RYU, SCHOOL, reviewed_by_author="2026-10-28"))

    assert cli.main(["--dataset", str(root), "check", str(path)]) == 0

    output = capsys.readouterr().out
    assert "2 entries (character 1, term 1); 2 occur on 2 or more pages" in output
    assert "reviewed by author: 2026-10-28" in output and "problems: none" in output


def test_check_exits_non_zero_and_lists_each_problem(cli, root, tmp_path, capsys):
    path = write_glossary(tmp_path / "glossaries", raw({**RYU, "target": "云"}, {**SCHOOL, "category": "place"}))

    assert cli.main(["--dataset", str(root), "check", str(path)]) == 1

    output = capsys.readouterr().out
    assert "problems: 2" in output and "[target_not_found]" in output and "[category]" in output


def test_check_without_paths_reads_every_file_of_the_glossary_directory(cli, root, tmp_path, capsys):
    directory = tmp_path / "glossaries"
    write_glossary(directory, raw(RYU, SCHOOL))
    (directory / "second.json").write_text(json.dumps(raw(RYU, {**SCHOOL, "target": "云"})), encoding="utf-8")

    assert cli.main(["--dataset", str(root), "--glossary-dir", str(directory), "check"]) == 1

    output = capsys.readouterr().out
    assert f"{BOOK}.json" in output and "second.json" in output


def test_check_cannot_start_without_the_dataset_or_any_glossary(cli, root, tmp_path, capsys):
    assert cli.main(["--dataset", str(tmp_path / "nowhere"), "check", str(tmp_path / "x.json")]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["--dataset", str(root), "--glossary-dir", str(empty), "check"]) == 2
    assert "No glossary files" in capsys.readouterr().err


# --- the protocol (OUTLINE 4.7) and the files in git ----------------------------------------------------------------

FORBIDDEN_IMPORTS = {"sqlite3", "store", "db", "runs"}
GLOSSARY_CODE = [config.ROOT / "backend" / "fukidashi" / "evaluation" / "glossary.py",
                 config.ROOT / "backend" / "fukidashi" / "evaluation" / "splits.py",
                 config.ROOT / "eval" / "glossary.py"]


def imported_names(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(part for alias in node.names for part in alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            names.update((node.module or "").split("."))
            names.update(alias.name for alias in node.names)
    return names


def test_the_glossary_code_cannot_reach_runs_or_translations():
    for path in GLOSSARY_CODE:
        assert not imported_names(path) & FORBIDDEN_IMPORTS, path


def test_nothing_outside_evaluation_refers_to_the_gold_glossary():
    package = config.ROOT / "backend" / "fukidashi"
    for path in package.rglob("*.py"):
        if "evaluation" not in path.relative_to(package).parts:
            assert "gold_glossary" not in path.read_text(encoding="utf-8"), path


def test_the_committed_glossaries_hold_terms_for_their_own_book():
    files = sorted((config.ROOT / "eval" / "gold_glossary").glob("*.json"))
    assert {f.stem for f in files} >= set(splits.DEV_BOOKS)
    for path in files:
        loaded = glossary.load_glossary(path)
        assert loaded.problems == (), path
        assert loaded.book_title == path.stem and loaded.split == splits.split_of(loaded.book_title)
        assert loaded.opencc_config == "t2s"
        assert 1 <= len(loaded.entries) <= 40, path
        assert all(len(t) <= glossary.MAX_TERM_CHARS for e in loaded.entries for t in (*e.sources, *e.targets)), path
