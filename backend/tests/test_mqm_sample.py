"""The GEMBA-MQM human sample (OUTLINE 4.5): `evaluation/mqm_sample.py` and eval/mqm_sample.py on small made-up runs.

Nothing here needs data/, the database or a model.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from datetime import datetime, timezone

import pytest

from fukidashi import config
from fukidashi.evaluation import mqm, mqm_sample

BIG, SMALL = "tojime_no_siora", "boureisougi"  # 20 and 17 segments: 15 splits into 8 and 7, as 334 and 274 do
SIZES = {BIG: 20, SMALL: 17}
EMPTY_JA = {BIG: {3}, SMALL: set()}  # positions whose Japanese text is empty
SAME = {BIG: {0, 5, 9}, SMALL: {2, 4}}  # positions where baseline and full translated alike
NOW = datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)


def lines_of(book, config):
    out = []
    for n in range(SIZES[book]):
        same = n in SAME[book]
        out.append({"book": book, "page_index": n // 4 + 1, "idx": n % 4, "run_id": (10 if config == "baseline" else 20)
                    + n // 4, "ja": "" if n in EMPTY_JA[book] else f"{book}の台詞{n}",
                    "hyp": f"{book}译文{n}" if same else f"{book}译文{n}{'甲' if config == 'baseline' else '乙'}",
                    "ref_zh_hans": "REF", "ref_zh_hant": "REF"})
    return out


def make_run(tmp_path, book, config, *, finished=True, lines=None):
    directory = tmp_path / "eval" / f"20261029T00000{0 if config == 'baseline' else 1}Z-{config}-{book}"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({
        "eval_id": directory.name, "book": book, "config_name": config,
        "finished_at": "2026-10-29T01:00:00+00:00" if finished else None}), encoding="utf-8")
    body = "".join(json.dumps(l, ensure_ascii=False) + "\n" for l in (lines or lines_of(book, config)))
    (directory / "segments.jsonl").write_text(body, encoding="utf-8")
    return directory


@pytest.fixture
def runs(tmp_path):
    return {"baseline": [make_run(tmp_path, BIG, "baseline"), make_run(tmp_path, SMALL, "baseline")],
            "full": [make_run(tmp_path, BIG, "full"), make_run(tmp_path, SMALL, "full")]}


@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("eval_mqm_sample_under_test", config.ROOT / "eval" / "mqm_sample.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def draw(runs, seed=0):
    return mqm_sample.draw(runs["baseline"], runs["full"], seed)


def source(entry):
    return entry["book"], entry["page_index"], entry["idx"]


# --- draw ----------------------------------------------------------------------------------------------------------

def test_the_allocation_follows_the_books_segment_counts():
    assert mqm_sample.allocate({"tojime_no_siora": 334, "boureisougi": 274}, 15) == {"tojime_no_siora": 8,
                                                                                     "boureisougi": 7}
    assert mqm_sample.allocate(SIZES, 15) == {BIG: 8, SMALL: 7}
    assert mqm_sample.allocate({"a": 1, "b": 1}, 3) == {"a": 2, "b": 1}  # a tie goes to the first name


def test_the_same_seed_gives_the_same_30_items(runs):
    first, second = draw(runs), draw(runs)

    assert first == second and len(first.rows) == 30
    assert draw(runs, seed=1).key["items"] != first.key["items"]


def test_the_items_are_15_source_segments_8_and_7_by_book_each_with_a_baseline_and_a_full_translation(runs):
    sample = draw(runs)
    items = sample.key["items"]

    assert [e["item"] for e in items] == list(range(1, 31)) and [r["item"] for r in sample.rows] == list(range(1, 31))
    sources = {source(e) for e in items}
    assert len(sources) == 15
    assert sum(1 for s in sources if s[0] == BIG) == 8 and sum(1 for s in sources if s[0] == SMALL) == 7
    for s in sources:
        assert sorted(e["config"] for e in items if source(e) == s) == ["baseline", "full"]
    by_place = {(c, l["book"], l["page_index"], l["idx"]): l for c in ("baseline", "full") for b in (BIG, SMALL)
                for l in lines_of(b, c)}
    for entry, row in zip(items, sample.rows):
        line = by_place[(entry["config"], *source(entry))]
        assert (row["ja"], row["translation"]) == (line["ja"], line["hyp"])
        assert entry["run_id"] == line["run_id"] and entry["eval_id"].endswith(f"-{entry['config']}-{entry['book']}")
        assert line["ja"]  # never a segment with an empty Japanese text
        other = by_place[("full" if entry["config"] == "baseline" else "baseline", *source(entry))]
        assert other["hyp"] != line["hyp"]  # never one whose two translations are the same
    assert sample.key["allocation"] == {SMALL: 7, BIG: 8}
    assert sample.key["candidates"] == {SMALL: 15, BIG: 16} and sample.key["segments"] == {SMALL: 17, BIG: 20}


def test_the_items_are_shuffled_not_grouped_by_config_or_book(runs):
    configs = [e["config"] for e in draw(runs).key["items"]]

    assert configs != sorted(configs) and configs != ["baseline", "full"] * 15


def test_the_written_sheet_is_blind_and_every_item_is_in_the_key(runs, tmp_path):
    out_dir = mqm_sample.write_sample(draw(runs), tmp_path / "human", now=NOW)

    assert out_dir.name == "20261029T120000Z-sample"
    assert sorted(p.name for p in out_dir.iterdir()) == ["README.txt", "key.json", "sheet.csv"]
    data = (out_dir / "sheet.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
    assert tuple(rows[0]) == mqm_sample.SHEET_COLUMNS == ("item", "ja", "translation", "severity", "category", "span",
                                                          "note")
    assert all(r["severity"] == r["category"] == r["span"] == r["note"] == "" for r in rows)
    text = data.decode("utf-8-sig")
    key = json.loads((out_dir / "key.json").read_text(encoding="utf-8"))
    for entry in key["items"]:
        assert entry["eval_id"] not in text
    for word in ("baseline", "full", "config", "eval_id", "run_id", "page_index"):
        assert word not in text
    assert sorted(int(r["item"]) for r in rows) == [e["item"] for e in key["items"]]
    assert key["sheet_sha256"] == hashlib.sha256(data).hexdigest() and key["seed"] == 0
    assert key["drawn_at"] == "2026-10-29T12:00:00+00:00"
    readme = (out_dir / "README.txt").read_text(encoding="utf-8")
    assert "docs/mqm-annotation.md" in readme and "no-error" in readme and "same item number" in readme


def test_an_existing_output_directory_is_never_written_into(runs, tmp_path):
    taken = tmp_path / "human" / "20261029T120000Z-sample"
    taken.mkdir(parents=True)
    (taken / "sheet.csv").write_text("annotated by hand", encoding="utf-8")

    out_dir = mqm_sample.write_sample(draw(runs), tmp_path / "human", now=NOW)

    assert out_dir.name == "20261029T120000Z-sample-2"
    assert sorted(p.name for p in taken.iterdir()) == ["sheet.csv"]
    assert (taken / "sheet.csv").read_text(encoding="utf-8") == "annotated by hand"


@pytest.mark.parametrize("problem, message", [
    ("unfinished", "not finished"),
    ("wrong config", "not 'full'"),
    ("two runs of a book", "two baseline runs"),
    ("books differ", "give one run of each config per book"),
    ("japanese differs", "disagree on the Japanese text"),
    ("too few candidates", "fewer than"),
])
def test_runs_that_cannot_give_the_sample_are_refused(tmp_path, problem, message):
    baseline = [make_run(tmp_path, BIG, "baseline"), make_run(tmp_path, SMALL, "baseline")]
    full = [make_run(tmp_path, BIG, "full")]
    if problem == "unfinished":
        full.append(make_run(tmp_path / "x", SMALL, "full", finished=False))
    elif problem == "wrong config":
        full.append(make_run(tmp_path / "x", SMALL, "baseline"))
    elif problem == "two runs of a book":
        baseline[1] = make_run(tmp_path / "x", BIG, "baseline")
        full.append(make_run(tmp_path / "y", SMALL, "full"))
    elif problem == "books differ":
        pass  # no full run of SMALL
    elif problem == "japanese differs":
        changed = lines_of(SMALL, "full")
        changed[1] = {**changed[1], "ja": "別の台詞"}
        full.append(make_run(tmp_path / "x", SMALL, "full", lines=changed))
    elif problem == "too few candidates":
        same = [{**l, "hyp": b["hyp"]} for l, b in zip(lines_of(SMALL, "full"), lines_of(SMALL, "baseline"))]
        full.append(make_run(tmp_path / "x", SMALL, "full", lines=same))

    with pytest.raises(mqm_sample.Refused, match=message):
        mqm_sample.draw(baseline, full, 0)


def test_the_draw_cli_writes_the_sample_and_refuses_an_unfinished_run(cli, runs, tmp_path, capsys):
    args = ["draw", "--baseline", *map(str, runs["baseline"]), "--full", *map(str, runs["full"]), "--seed", "0",
            "--out", str(tmp_path / "human")]

    assert cli.main(args) == 0

    (out_dir,) = (tmp_path / "human").iterdir()
    assert sorted(p.name for p in out_dir.iterdir()) == ["README.txt", "key.json", "sheet.csv"]
    out = capsys.readouterr().out
    assert "Drew 30 items from 15 source segments (boureisougi 7, tojime_no_siora 8), seed 0." in out

    manifest = runs["full"][1] / "manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace('"finished_at": "2026-10-29T01:00:00+00:00"',
                                                                     '"finished_at": null'), encoding="utf-8")
    assert cli.main(args) == 2
    assert "not finished" in capsys.readouterr().err
    assert len(list((tmp_path / "human").iterdir())) == 1  # nothing new written


# --- check ---------------------------------------------------------------------------------------------------------

@pytest.fixture
def sheet(runs, tmp_path):
    return mqm_sample.write_sample(draw(runs), tmp_path / "human", now=NOW) / "sheet.csv"


def read_rows(path):
    return list(csv.DictReader(io.StringIO(path.read_bytes().decode("utf-8-sig"))))


def write_rows(path, rows):
    path.write_bytes(mqm_sample.sheet_bytes(rows))


def label(rows, item, severity, category="", span="", *, extra=False):
    if extra:
        rows.append({"item": item, "ja": "", "translation": "", "severity": severity, "category": category,
                     "span": span, "note": ""})
        return
    for row in rows:
        if int(row["item"]) == item:
            row.update(severity=severity, category=category, span=span)
            return


def filled(path):
    """Every item labelled: item 1 has a major and a minor error (on two rows), item 2 a critical one, the rest none."""
    rows = read_rows(path)
    label(rows, 1, "major", "accuracy/mistranslation", "译文")
    label(rows, 1, "Minor", " fluency/punctuation ", "。", extra=True)  # case and spaces do not matter
    label(rows, 2, "critical", "non-translation", "")
    for item in range(3, 31):
        label(rows, item, "no-error")
    return rows


def test_check_accepts_a_well_filled_sheet_and_prints_the_scores_and_the_sha256(cli, sheet, capsys):
    write_rows(sheet, filled(sheet))

    assert cli.main(["check", str(sheet)]) == 0

    out = capsys.readouterr().out
    assert "item 1: -6 (major 1, minor 1)" in out and "item 2: -25 (critical 1)" in out
    assert "item 30: 0 (no-error)" in out
    assert f"sha256 {hashlib.sha256(sheet.read_bytes()).hexdigest()}" in out
    result = mqm_sample.check_sheet(sheet)
    assert result.ok and result.score(1) == mqm.mqm_score(["major", "minor"])


def test_check_rejects_the_blank_sheet_with_every_item_unlabelled(cli, sheet, capsys):
    assert cli.main(["check", str(sheet)]) == 1

    err = capsys.readouterr().err
    assert "30 of 30 items have no label: 1, 2, 3" in err and "sha256" in err


@pytest.mark.parametrize("change, message", [
    (lambda rows: [r for r in rows if int(r["item"]) != 7], "1 of 30 items have no label: 7"),
    (lambda rows: label(rows, 3, "serious", "style/awkward"), "unknown severity 'serious'"),
    (lambda rows: label(rows, 3, "major", "accuracy/wrong"), "unknown category 'accuracy/wrong'"),
    (lambda rows: label(rows, 3, "major", "style/awkward", extra=True), "item 3 is marked no-error and has an error"),
    (lambda rows: label(rows, 3, "minor", "non-translation"), "non-translation is always critical"),
    (lambda rows: label(rows, 3, "", "style/awkward"), "a category or span without a severity"),
    (lambda rows: label(rows, 3, "no-error", "style/awkward"), "no-error with a category or span"),
    (lambda rows: label(rows, 99, "minor", "other", extra=True), "no such item"),
    (lambda rows: label(rows, "x", "minor", "other", extra=True), "item 'x' is not a number"),
])
def test_check_rejects_a_sheet_with_a_problem_and_exits_1(cli, sheet, capsys, change, message):
    rows = filled(sheet)
    changed = change(rows)
    write_rows(sheet, changed if changed is not None else rows)

    assert cli.main(["check", str(sheet)]) == 1

    assert message in capsys.readouterr().err


def test_check_needs_the_key_beside_the_sheet_to_know_the_items(cli, sheet, capsys):
    write_rows(sheet, filled(sheet))
    (sheet.parent / "key.json").unlink()

    assert cli.main(["check", str(sheet)]) == 1
    assert "cannot read the item numbers" in capsys.readouterr().err


def test_check_refuses_a_sheet_that_is_not_there(cli, tmp_path, capsys):
    assert cli.main(["check", str(tmp_path / "nowhere.csv")]) == 2
    assert "Cannot read" in capsys.readouterr().err


def test_the_typology_is_the_prompts_categories():
    for category in mqm.TYPOLOGY:
        top = category.split("/")[0]
        assert top in mqm.USER_TEMPLATE
        if "/" in category:
            assert category.split("/")[1] in mqm.USER_TEMPLATE
    assert len(mqm.TYPOLOGY) == 15
