"""GEMBA-MQM's human agreement check (OUTLINE 4.5): `evaluation/agreement.py` and eval/mqm_agreement.py on made-up
samples and MQM files.

Nothing here needs data/, the database or a model. The MQM files are written the way eval/mqm.py writes them, as far as
the check reads them; the tau_b values are worked out by hand in the comments.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction

import pytest

from fukidashi import config
from fukidashi.evaluation import agreement, mqm, mqm_sample

A, B = "tojime_no_siora", "boureisougi"
MODEL = "gpt-4o-mini-2024-07-18"
NOW = datetime(2026, 10, 29, 21, 0, 0, tzinfo=timezone.utc)
RUNS = (("baseline", B, "20261029T043245Z-baseline"), ("baseline", A, "20261029T042715Z-baseline"),
        ("full", B, "20261029T054332Z-full"), ("full", A, "20261029T051514Z-full"))


# --- Kendall's tau_b (AC1) --------------------------------------------------------------------------------------------

def test_perfect_agreement_gives_1_and_reversed_order_gives_minus_1():
    assert agreement.kendall_tau_b([1, 2, 3, 4], [10, 20, 30, 40]).value == 1.0
    assert agreement.kendall_tau_b([1, 2, 3, 4], [40, 30, 20, 10]).value == -1.0
    assert agreement.kendall_tau_b([-25, -5, -1, 0], [-6, -2, -1, 0]).value == 1.0  # only the order counts


def test_ties_give_tau_b_not_tau_a():
    # x = 1 1 2 3, y = 1 2 2 3. Pairs: (0,1) tied in x; (1,2) tied in y; the other 4 concordant.
    # tau_b = (4 - 0) / sqrt((6 - 1) * (6 - 1)) = 0.8 (tau_a would be 4 / 6).
    tau = agreement.kendall_tau_b([1, 1, 2, 3], [1, 2, 2, 3])
    assert (tau.pairs, tau.concordant, tau.discordant, tau.tied_x, tau.tied_y, tau.tied_both) == (6, 4, 0, 1, 1, 0)
    assert tau.value == pytest.approx(0.8)


def test_a_pair_tied_in_both_counts_in_both_tie_counts():
    # x = y = 0 0 1: (0,1) tied in both, (0,2) and (1,2) concordant: 2 / sqrt((3 - 1) * (3 - 1)) = 1
    tau = agreement.kendall_tau_b([0, 0, 1], [0, 0, 1])
    assert (tau.concordant, tau.tied_x, tau.tied_y, tau.tied_both) == (2, 1, 1, 1)
    assert tau.value == 1.0


def test_hand_worked_examples():
    # x = 1 2 3 4 5, y = 3 1 2 5 4: discordant (0,1), (0,2), (3,4); the other 7 concordant: (7 - 3) / 10 = 0.4
    tau = agreement.kendall_tau_b([1, 2, 3, 4, 5], [3, 1, 2, 5, 4])
    assert (tau.concordant, tau.discordant) == (7, 3) and tau.value == pytest.approx(0.4)
    # x = 1 2 2 3, y = 3 1 2 2: discordant (0,1), (0,2), (0,3); concordant (1,3); tied in x (1,2); tied in y (2,3):
    # (1 - 3) / sqrt(5 * 5) = -0.4
    tau = agreement.kendall_tau_b([1, 2, 2, 3], [3, 1, 2, 2])
    assert (tau.concordant, tau.discordant, tau.tied_x, tau.tied_y) == (1, 3, 1, 1)
    assert tau.value == pytest.approx(-0.4)


def test_tau_b_is_undefined_when_one_list_is_all_ties_or_there_are_fewer_than_two_items():
    assert agreement.kendall_tau_b([0, 0, 0], [1, 2, 3]).value is None
    assert agreement.kendall_tau_b([1, 2, 3], [-5, -5, -5]).value is None
    assert agreement.kendall_tau_b([1], [1]).value is None
    assert agreement.kendall_tau_b([], []).value is None
    with pytest.raises(ValueError):
        agreement.kendall_tau_b([1, 2], [1])


def the_samples_shape(judge_ties):
    """The recorded sample's shape (OUTLINE 4.5): 30 items, the annotator's item 0 scores -1 and the other 29 score 0.
    The judge scores item 0 below every other item, so all 29 pairs with it are concordant, and gives the other 29 items
    scores in groups of the sizes `judge_ties` (the rest one score each)."""
    x = [-1] + [0] * 29
    y = [-100]  # scores for tau_b only, not MQM scores
    score = -1
    for size in judge_ties:
        y += [score] * size
        score -= 1
    while len(y) < 30:
        y.append(score)
        score -= 1
    assert len(y) == 30 and min(y[1:]) > y[0]
    return x, y


def test_with_the_samples_shape_a_judge_whose_scores_all_differ_cannot_reach_0_30():
    tau = agreement.kendall_tau_b(*the_samples_shape([]))
    # 29 / sqrt((435 - 406) * 435) = sqrt(29 / 435)
    assert (tau.concordant, tau.discordant, tau.tied_x, tau.tied_y) == (29, 0, 406, 0)
    assert tau.value == pytest.approx(math.sqrt(29 / 435)) and tau.value < 0.30
    assert not tau.at_least(agreement.TAU_B_MIN)


# --- the decision (AC2) ------------------------------------------------------------------------------------------------

def test_the_thresholds_are_outline_4_5s():
    assert agreement.TAU_B_MIN == Fraction(3, 10) and float(agreement.TAU_B_MIN) == 0.30
    assert agreement.RECALL_MIN == Fraction(3, 5) and float(agreement.RECALL_MIN) == 0.60
    assert agreement.SERIOUS == ("critical", "major")


def test_tau_b_on_the_threshold_passes_and_just_below_it_fails():
    # 113 tied judge pairs (groups of 15, 4, 2, 2: 105 + 6 + 1 + 1): 29 / sqrt(29 * 322) = 0.30010; 112 (15, 4, 2): 0.29964
    above = agreement.kendall_tau_b(*the_samples_shape([15, 4, 2, 2]))
    below = agreement.kendall_tau_b(*the_samples_shape([15, 4, 2]))
    assert above.tied_y == 113 and above.value == pytest.approx(math.sqrt(29 / 322)) and above.value > 0.30
    assert below.tied_y == 112 and below.value == pytest.approx(math.sqrt(29 / 323)) and below.value < 0.30

    assert agreement.decide(above, 0, 0)["pass"] is True
    assert agreement.decide(below, 0, 0)["pass"] is False


def test_tau_b_is_compared_exactly_not_after_rounding():
    # Counts built directly (no score lists give them): (6 - 3) / sqrt(10 * 10) is exactly 3/10, on the threshold.
    tau = agreement.TauB(n=5, pairs=10, concordant=6, discordant=3, tied_x=0, tied_y=0, tied_both=0)
    assert tau.value == pytest.approx(0.3)
    assert tau.at_least(Fraction(3, 10)) and not tau.at_least(Fraction(31, 100))
    assert not agreement.TauB(n=5, pairs=10, concordant=3, discordant=6, tied_x=0, tied_y=0, tied_both=0).at_least(
        Fraction(0))  # a negative tau_b is below any threshold that is not negative


def test_recall_on_the_threshold_passes_and_below_it_fails():
    good_tau = agreement.kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4])

    assert agreement.decide(good_tau, 5, 3)["pass"] is True  # 60%
    assert agreement.decide(good_tau, 7, 4)["pass"] is False  # 57.1%
    assert agreement.decide(good_tau, 5, 3)["recall_ok"] is True
    assert agreement.decide(good_tau, 7, 4)["recall_ok"] is False


def test_a_pass_needs_both():
    good_tau = agreement.kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4])
    bad_tau = agreement.kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1])

    assert agreement.decide(good_tau, 2, 2)["pass"] is True
    assert agreement.decide(good_tau, 2, 1)["pass"] is False  # recall 50%
    assert agreement.decide(bad_tau, 2, 2)["pass"] is False  # recall 100%, tau_b -1
    assert agreement.decide(bad_tau, 2, 2)["tau_b_ok"] is False and agreement.decide(bad_tau, 2, 2)["recall_ok"] is True


def test_with_no_major_or_critical_error_from_the_annotator_tau_b_decides_alone():
    good_tau = agreement.kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4])
    bad_tau = agreement.kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1])

    decision = agreement.decide(good_tau, 0, 0)
    assert decision["pass"] is True and decision["recall_ok"] is None and "tau_b alone" in decision["decided_by"]
    assert agreement.decide(bad_tau, 0, 0)["pass"] is False


def test_an_undefined_tau_b_fails():
    undefined = agreement.kendall_tau_b([0, 0, 0], [1, 2, 3])
    assert agreement.decide(undefined, 0, 0)["pass"] is False
    assert agreement.decide(undefined, 1, 1)["pass"] is False


def test_hits_cannot_exceed_the_serious_items():
    with pytest.raises(ValueError):
        agreement.decide(agreement.kendall_tau_b([1, 2], [1, 2]), 1, 2)


# --- a made-up sample and its MQM files ------------------------------------------------------------------------------

def item_place(item):
    """(config, book, eval_id, page_index, idx, run_id) of item `item` (1-based): the items go round the four runs."""
    config_name, book, eval_id = RUNS[(item - 1) % 4]
    return config_name, book, eval_id, 2 + (item - 1) // 4, (item * 7) % 5, 100 + RUNS.index((config_name, book, eval_id))


def write_sample(tmp_path, labels):
    """A sample directory with sheet.csv (labelled with `labels`: item -> [(severity, category)], [] for no error, None
    for no label) and key.json. Returns (directory, the sheet's sha256)."""
    directory = tmp_path / "20261029T172846Z-sample"
    directory.mkdir()
    rows = []
    for item in sorted(labels):
        errors = labels[item]
        if errors is None:
            rows.append({"item": item, "ja": f"台詞{item}", "translation": f"译文{item}", "severity": "",
                         "category": "", "span": "", "note": ""})
            continue
        first = errors[0] if errors else ("no-error", "")
        rows.append({"item": item, "ja": f"台詞{item}", "translation": f"译文{item}", "severity": first[0],
                     "category": first[1], "span": "x" if errors else "", "note": ""})
        rows += [{"item": item, "ja": "", "translation": "", "severity": s, "category": c, "span": "x", "note": ""}
                 for s, c in errors[1:]]
    sheet = mqm_sample.sheet_bytes(rows)
    (directory / "sheet.csv").write_bytes(sheet)
    items = []
    for item in sorted(labels):
        config_name, book, eval_id, page_index, idx, run_id = item_place(item)
        items.append({"item": item, "book": book, "page_index": page_index, "idx": idx, "config": config_name,
                      "eval_id": eval_id, "run_id": run_id})
    key = {"about": "made up", "seed": 0, "runs": [
        {"config": c, "book": b, "eval_id": e, "dir": f"runs/eval/{e}", "segments_sha256": f"segments-of-{e}"}
        for c, b, e in RUNS], "items": items, "drawn_at": "2026-10-29T17:28:46+00:00", "sheet_sha256": "blank"}
    (directory / "key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    return directory, hashlib.sha256(sheet).hexdigest()


def segment(book, page_index, idx, run_id, judged):
    """One MQM file segment: `judged` is [(severity, category)] for a scored segment, or "failed" / "skipped"."""
    if judged in ("failed", "skipped"):
        return {"book": book, "page_index": page_index, "idx": idx, "run_id": run_id, "status": judged, "errors": [],
                "score": None, "answer": None, "attempts": 2 if judged == "failed" else 0, "problems": []}
    return {"book": book, "page_index": page_index, "idx": idx, "run_id": run_id, "status": "scored",
            "errors": [{"severity": s, "category": c, "span": None, "text": f"{c}"} for s, c in judged],
            "score": mqm.mqm_score(s for s, _ in judged), "answer": "(an answer)", "attempts": 1, "problems": []}


def write_mqm_files(tmp_path, judged, *, model=MODEL, runs=RUNS):
    """One MQM file per run, under tmp_path/eval/<eval_id>/mqm-<model>.json; `judged` maps an item to its judgement.
    Each file also holds a segment that is not in the sample. Returns the paths in RUNS order."""
    paths = []
    for config_name, book, eval_id in runs:
        run_id = 100 + RUNS.index((config_name, book, eval_id)) if (config_name, book, eval_id) in RUNS else 200
        segments = [segment(book, 90, 0, run_id, "skipped")]
        segments += [segment(book, item_place(i)[3], item_place(i)[4], run_id, judged[i])
                     for i in sorted(judged) if item_place(i)[2] == eval_id]
        directory = tmp_path / "eval" / eval_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"mqm-{model.rsplit('/', 1)[-1]}.json"
        path.write_text(json.dumps({
            "eval_id": eval_id, "book": book, "segments_sha256": f"segments-of-{eval_id}",
            "judge": {"prefix": "JUDGE", "model": model, "response_model": model, "temperature": 0.0,
                      "reasoning_effort": None, "max_tokens": 500},
            "gemba": {"commit": mqm.GEMBA["commit"], "template_sha256": "template"}, "weights": dict(mqm.WEIGHTS),
            "segments": segments, "scored_at": "2026-10-29T20:00:00+00:00"}, ensure_ascii=False), encoding="utf-8")
        paths.append(path)
    return paths


MAJOR = [("major", "accuracy/mistranslation")]
MINOR = [("minor", "fluency/punctuation")]
CRITICAL = [("critical", "accuracy/omission")]
# Eight items. The annotator: items 1 and 2 major, 3 minor + major (two rows), 4 minor, the rest no error.
LABELS = {1: MAJOR, 2: MAJOR, 3: [*MINOR, *MAJOR], 4: MINOR, 5: [], 6: [], 7: [], 8: []}
# The judge: item 1 critical (a hit), 2 minor only (a miss), 3 major (a hit), 4 major, 5 minor, the rest no error.
JUDGED = {1: [("critical", "accuracy-mistranslation")], 2: [("minor", "style")], 3: [("major", "fluency-grammar")],
          4: [("major", "accuracy-omission")], 5: [("minor", "fluency-punctuation")], 6: [], 7: [], 8: []}


@pytest.fixture
def sample(tmp_path):
    directory, sha = write_sample(tmp_path, LABELS)
    return directory, sha, write_mqm_files(tmp_path, JUDGED)


def test_the_check_joins_each_item_to_its_segment_and_computes_tau_b_and_the_recall(sample):
    directory, sha, paths = sample

    document = agreement.compute(directory, sha, paths, now=NOW)

    by_item = {i["item"]: i for i in document["items"]}
    assert [by_item[i]["annotator"]["score"] for i in range(1, 9)] == [-5, -5, -6, -1, 0, 0, 0, 0]
    assert [by_item[i]["judge"]["score"] for i in range(1, 9)] == [-25, -1, -5, -5, -1, 0, 0, 0]
    assert [by_item[i]["annotator"]["major_or_critical"] for i in range(1, 9)] == [True] * 3 + [False] * 5
    assert [by_item[i]["judge"]["major_or_critical"] for i in range(1, 9)] == [True, False, True, True] + [False] * 4
    assert by_item[3]["annotator"]["errors"] == [["minor", "fluency/punctuation"], ["major", "accuracy/mistranslation"]]
    assert by_item[1]["judge"]["errors"] == [["critical", "accuracy-mistranslation"]]
    assert by_item[5] | {"annotator": None, "judge": None} == {
        "item": 5, "book": item_place(5)[1], "page_index": item_place(5)[3], "idx": item_place(5)[4],
        "config": "baseline", "eval_id": RUNS[0][2], "run_id": 100, "annotator": None, "judge": None}
    expected = agreement.kendall_tau_b([-5, -5, -6, -1, 0, 0, 0, 0], [-25, -1, -5, -5, -1, 0, 0, 0])
    assert document["tau_b"] == expected.to_dict()
    assert document["recall"] == {"items_with_major_or_critical": 3, "hits": 2, "value": pytest.approx(2 / 3)}
    assert document["decision"]["recall_ok"] is True
    assert document["decision"]["pass"] is expected.at_least(agreement.TAU_B_MIN)


def test_the_tau_b_of_the_fixture_worked_out_by_hand(sample):
    # item        1   2   3   4   5   6   7   8
    # annotator  -5  -5  -6  -1   0   0   0   0
    # judge     -25  -1  -5  -5  -1   0   0   0
    # 28 pairs:
    # - tied in the annotator's scores only: (1,2), (5,6), (5,7), (5,8); in the judge's only: (2,5), (3,4);
    #   in both: (6,7), (6,8), (7,8). So 7 tied in the annotator's, 5 in the judge's, 3 in both.
    # - discordant: (1,3) (annotator 1 above 3, judge 1 below 3) and (2,4) (annotator 2 below 4, judge 2 above 4).
    # - concordant: the other 28 - 4 - 2 - 3 - 2 = 17.
    # tau_b = (17 - 2) / sqrt((28 - 7) * (28 - 5)) = 15 / sqrt(483) = 0.6825
    directory, sha, paths = sample
    tau = agreement.compute(directory, sha, paths)["tau_b"]
    assert (tau["pairs"], tau["tied_annotator"], tau["tied_judge"], tau["tied_both"]) == (28, 7, 5, 3)
    assert (tau["concordant"], tau["discordant"]) == (17, 2)
    assert tau["value"] == pytest.approx(15 / math.sqrt(483)) == pytest.approx(0.6825, abs=1e-4)
    decision = agreement.compute(directory, sha, paths)["decision"]
    assert decision == {"pass": True, "tau_b_ok": True, "recall_ok": True,  # 2 of 3 is 66.7%
                        "decided_by": "tau_b and the recall: both must reach their thresholds",
                        "outcome": decision["outcome"]} and decision["outcome"].startswith("the judge passes")


def test_the_file_holds_the_thresholds_the_rule_the_sha256_the_annotator_and_the_inputs(sample):
    directory, sha, paths = sample

    document = agreement.compute(directory, sha, paths, now=NOW)

    assert document["thresholds"] == {"tau_b": 0.30, "recall": 0.60}
    assert "0.30" in document["rule"] and "60%" in document["rule"] and "tau_b decides alone" in document["rule"]
    assert document["sheet"] == {"path": str(directory / "sheet.csv"), "sha256": sha, "expected_sha256": sha}
    assert document["annotator"]["kind"] == "llm" and "not a person" in document["annotator"]["note"]
    assert document["judge"]["model"] == MODEL and document["judge"]["response_model"] == MODEL
    assert [i["eval_id"] for i in document["inputs"]] == [e for _, _, e in RUNS]
    assert document["inputs"][0] == {
        "path": str(paths[0]), "sha256": hashlib.sha256(paths[0].read_bytes()).hexdigest(), "eval_id": RUNS[0][2],
        "book": B, "config": "baseline", "segments_sha256": f"segments-of-{RUNS[0][2]}",
        "scored_at": "2026-10-29T20:00:00+00:00"}
    assert document["n_items"] == 8 and document["computed_at"] == "2026-10-29T21:00:00+00:00"
    assert agreement.compute(directory, sha, paths, annotator="person")["annotator"]["kind"] == "person"


def test_the_recorded_samples_case_one_minor_error_and_tau_b_alone(tmp_path):
    # 30 items, one minor error from the annotator (OUTLINE 4.5): no major or critical error, so tau_b decides alone.
    labels = {i: (MINOR if i == 1 else []) for i in range(1, 31)}
    directory, sha = write_sample(tmp_path, labels)
    distinct = {i: ([("critical", "x")] if i == 1 else [("minor", "x")] * (1 + i % 5)) for i in range(1, 31)}
    paths = write_mqm_files(tmp_path, distinct)

    document = agreement.compute(directory, sha, paths)

    assert document["recall"] == {"items_with_major_or_critical": 0, "hits": 0, "value": None}
    assert document["decision"]["recall_ok"] is None and "tau_b alone" in document["decision"]["decided_by"]
    assert document["decision"]["pass"] is agreement.kendall_tau_b(
        [i["annotator"]["score"] for i in document["items"]],
        [i["judge"]["score"] for i in document["items"]]).at_least(agreement.TAU_B_MIN)


# --- the CLI (AC3) ----------------------------------------------------------------------------------------------------

@pytest.fixture
def cli():
    spec = importlib.util.spec_from_file_location("eval_mqm_agreement_under_test",
                                                  config.ROOT / "eval" / "mqm_agreement.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(cli, directory, sha, paths, *extra):
    return cli.main([str(directory), "--expected-sha256", sha, *map(str, paths), *extra])


def snapshot(*directories):
    return {str(p): p.read_bytes() for d in directories if d.exists() for p in sorted(d.rglob("*")) if p.is_file()}


def test_the_cli_writes_the_agreement_file_next_to_the_sheet_and_prints_the_decision(cli, sample, capsys):
    directory, sha, paths = sample

    assert run_cli(cli, directory, sha, paths) == 0

    out = capsys.readouterr().out
    written = json.loads((directory / f"agreement-{MODEL}.json").read_text(encoding="utf-8"))
    assert written | {"computed_at": None} == json.loads(json.dumps(
        agreement.compute(directory, sha, paths) | {"computed_at": None}))
    assert f"sheet sha256 {sha}" in out and "tau_b " in out and "recall 66.7%" in out
    assert ("PASS" if written["decision"]["pass"] else "FAIL") in out and "not a person" in out
    assert f"Wrote {directory / f'agreement-{MODEL}.json'}" in out
    assert sorted(p.name for p in directory.iterdir()) == [f"agreement-{MODEL}.json", "key.json", "sheet.csv"]


def test_the_options_may_come_anywhere(cli, sample):
    directory, sha, paths = sample
    assert cli.main(["--expected-sha256", sha, str(directory), *map(str, paths)]) == 0
    assert cli.main([str(directory), *map(str, paths), "--expected-sha256", sha.upper()]) == 0


def test_a_second_run_with_the_same_result_keeps_the_first_file_and_its_time(cli, sample, capsys):
    directory, sha, paths = sample
    out = directory / f"agreement-{MODEL}.json"
    assert run_cli(cli, directory, sha, paths) == 0
    first = out.read_bytes()
    capsys.readouterr()

    assert run_cli(cli, directory, sha, paths) == 0

    assert out.read_bytes() == first
    assert "already holds this result; it was left as it is" in capsys.readouterr().out


def refusal_cases(tmp_path):
    """[(what is wrong, sample dir, expected sha256, MQM paths, text the message must hold)]."""
    cases = []

    def fresh(name, labels=LABELS, judged=JUDGED, **options):
        sub = tmp_path / name
        sub.mkdir()
        directory, sha = write_sample(sub, labels)
        return sub, directory, sha, write_mqm_files(sub, judged, **options)

    sub, d, sha, paths = fresh("hash")
    cases.append(("another sheet hash", d, "0" * 64, paths, f"the sheet's sha256 is {sha}, not the expected"))

    sub, d, sha, paths = fresh("relabelled")
    rows = (d / "sheet.csv").read_bytes().decode("utf-8-sig")
    changed = rows.replace("major,accuracy/mistranslation", "minor,accuracy/mistranslation", 1)
    assert changed != rows
    (d / "sheet.csv").write_bytes(changed.encode("utf-8-sig"))
    cases.append(("a sheet changed after its hash was recorded", d, sha, paths, "not the expected"))

    sub, d, sha, paths = fresh("incomplete", labels={**LABELS, 8: None})
    cases.append(("an incomplete sheet", d, sha, paths, "is not complete and valid"))

    for status in ("failed", "skipped"):
        sub, d, sha, paths = fresh(f"judge-{status}", judged={**JUDGED, 6: status})
        cases.append((f"an item whose segment the judge {status}", d, sha, paths,
                      f"item 6 ({item_place(6)[1]}/{item_place(6)[3]} idx {item_place(6)[4]}, {item_place(6)[0]}) has no judge "
                      f"score: its segment is {status}"))

    sub, d, sha, paths = fresh("judge-missing", judged={k: v for k, v in JUDGED.items() if k != 7})
    cases.append(("an item with no segment in its MQM file", d, sha, paths,
                  f"item 7 ({item_place(7)[1]}/{item_place(7)[3]} idx {item_place(7)[4]}, {item_place(7)[0]}) has no judge "
                  "score: "))

    sub, d, sha, paths = fresh("other-eval")
    other = write_mqm_files(sub / "other", JUDGED, runs=[("full_no_kb", A, "20261029T043724Z-full_no_kb")])
    cases.append(("an MQM file of an eval run the sample does not have", d, sha, [*paths, *other],
                  "20261029T043724Z-full_no_kb, which is not one of the sample's runs"))

    sub, d, sha, paths = fresh("twice")
    cases.append(("two files of one eval run", d, sha, [*paths, paths[0]], "are both MQM files of eval run"))

    sub, d, sha, paths = fresh("one-missing")
    cases.append(("a run of the sample without a file", d, sha, paths[:3], f"no MQM file is given for the sample's run "
                                                                          f"{RUNS[3][2]}"))

    sub, d, sha, paths = fresh("mixed-models")
    gpt51 = write_mqm_files(sub / "other", JUDGED, model="gpt-5.1-2025-11-13")
    cases.append(("files of two judge models", d, sha, [*paths[:3], gpt51[3]], "is by judge model 'gpt-5.1-2025-11-13'"))

    sub, d, sha, paths = fresh("other-segments")
    document = json.loads(paths[1].read_text(encoding="utf-8"))
    paths[1].write_text(json.dumps(document | {"segments_sha256": "another"}), encoding="utf-8")
    cases.append(("a file over another segments.jsonl", d, sha, paths, "scored another segments.jsonl than the one the "
                                                                      "sample was drawn from"))

    sub, d, sha, paths = fresh("other-run-id")
    document = json.loads(paths[2].read_text(encoding="utf-8"))
    for s in document["segments"]:
        s["run_id"] = 999
    paths[2].write_text(json.dumps(document), encoding="utf-8")
    cases.append(("a segment of another run than the key's", d, sha, paths, "judged run 999, the key's is run 102"))

    sub, d, sha, paths = fresh("other-weights")
    document = json.loads(paths[0].read_text(encoding="utf-8"))
    paths[0].write_text(json.dumps(document | {"weights": {"critical": 10, "major": 5, "minor": 1}}), encoding="utf-8")
    cases.append(("a file scored with other weights", d, sha, paths, "was scored with the weights"))

    sub, d, sha, paths = fresh("other-template")
    document = json.loads(paths[3].read_text(encoding="utf-8"))
    paths[3].write_text(json.dumps(document | {"gemba": {**document["gemba"], "template_sha256": "x"}}), encoding="utf-8")
    cases.append(("a file of another prompt", d, sha, paths, "has another prompt template"))

    sub, d, sha, paths = fresh("bad-score")
    document = json.loads(paths[0].read_text(encoding="utf-8"))
    document["segments"][1]["score"] = 0
    paths[0].write_text(json.dumps(document), encoding="utf-8")
    cases.append(("a score that is not its errors' score", d, sha, paths, "its score is not the MQM score of its errors"))

    sub, d, sha, paths = fresh("not-json")
    paths[0].write_text("{not json", encoding="utf-8")
    cases.append(("a file that is not JSON", d, sha, paths, "is not valid JSON"))

    sub, d, sha, paths = fresh("other-book")
    document = json.loads(paths[0].read_text(encoding="utf-8"))
    paths[0].write_text(json.dumps(document | {"book": A}), encoding="utf-8")
    cases.append(("a file of another book than its run", d, sha, paths, f"is of book '{A}'"))

    sub, d, sha, paths = fresh("existing-other-result")
    (d / f"agreement-{MODEL}.json").write_text(json.dumps({"tau_b": "another", "computed_at": "earlier"}),
                                               encoding="utf-8")
    cases.append(("an agreement file with another result", d, sha, paths, "is never overwritten"))
    return cases


def test_every_refusal_exits_2_and_writes_nothing(cli, tmp_path, capsys):
    cases = refusal_cases(tmp_path)
    assert len(cases) >= 17

    for what, directory, sha, paths, expected in cases:
        before = snapshot(directory, *{p.parent for p in paths})

        code = run_cli(cli, directory, sha, paths)

        err = capsys.readouterr().err
        assert code == 2, what
        assert err.startswith("Nothing was written: ") and expected in err, (what, err)
        assert snapshot(directory, *{p.parent for p in paths}) == before, what


def test_the_library_raises_refused_for_the_same_cases(tmp_path):
    for what, directory, sha, paths, expected in refusal_cases(tmp_path):
        if what == "an agreement file with another result":
            document = agreement.compute(directory, sha, paths)
            with pytest.raises(agreement.Refused, match="never overwritten"):
                agreement.write(document, directory / agreement.output_name(document["judge"]["model"]))
            continue
        with pytest.raises(agreement.Refused) as raised:
            agreement.compute(directory, sha, paths)
        assert expected in str(raised.value), what


def test_a_failing_write_leaves_no_file_and_no_temporary_file(sample, monkeypatch):
    directory, sha, paths = sample
    document = agreement.compute(directory, sha, paths)
    monkeypatch.setattr(agreement.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk on fire")))

    with pytest.raises(OSError, match="disk on fire"):
        agreement.write(document, directory / "agreement-x.json")

    assert sorted(p.name for p in directory.iterdir()) == ["key.json", "sheet.csv"]


def test_the_cli_exits_1_when_the_file_cannot_be_written(cli, sample, capsys, monkeypatch):
    directory, sha, paths = sample
    monkeypatch.setattr(agreement.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk on fire")))

    assert run_cli(cli, directory, sha, paths) == 1
    assert "Could not write the agreement file" in capsys.readouterr().err


# --- what the module is -----------------------------------------------------------------------------------------------

def test_the_statuses_are_the_scorers():
    from fukidashi.evaluation import mqm_scorer

    assert set(agreement.STATUSES) == {"scored", "skipped", "failed"}
    assert agreement.bare_model("openai/gpt-4o-mini") == mqm_scorer.bare_model("openai/gpt-4o-mini") == "gpt-4o-mini"
    assert agreement.output_name("openai/gpt-4o-mini-2024-07-18").removeprefix("agreement-") == \
        mqm_scorer.output_name("openai/gpt-4o-mini-2024-07-18").removeprefix("mqm-")


def test_the_check_reads_files_only():
    names = set()
    for path in (config.ROOT / "backend" / "fukidashi" / "evaluation" / "agreement.py",
                 config.ROOT / "eval" / "mqm_agreement.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(part for alias in node.names for part in alias.name.split("."))
            elif isinstance(node, ast.ImportFrom):
                names.update((node.module or "").split("."))
                names.update(alias.name for alias in node.names)
    assert not names & {"sqlite3", "store", "db", "runs", "llm", "openai", "httpx", "requests", "socket", "subprocess",
                        "mqm_scorer", "dotenv"}


def test_the_script_runs_as_a_program(sample):
    directory, sha, paths = sample
    script = str(config.ROOT / "eval" / "mqm_agreement.py")

    done = subprocess.run([sys.executable, script, str(directory), "--expected-sha256", sha, *map(str, paths)],
                          capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "decision:" in done.stdout

    refused = subprocess.run([sys.executable, script, str(directory), "--expected-sha256", "0" * 64, *map(str, paths)],
                             capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, timeout=60)
    assert refused.returncode == 2 and "Nothing was written" in refused.stderr


def test_the_pooled_report_reads_the_file_this_cli_writes(cli, sample):
    from fukidashi.evaluation import report

    directory, sha, paths = sample
    assert run_cli(cli, directory, sha, paths) == 0
    path = directory / f"agreement-{MODEL}.json"
    written = json.loads(path.read_text(encoding="utf-8"))

    quoted = report._agreement(path, f"openai/{MODEL}")

    assert quoted["pass"] is written["decision"]["pass"] and quoted["tau_b"] == written["tau_b"]["value"]
    assert quoted["recall"] == written["recall"]["value"] and quoted["annotator"] == written["annotator"]
    assert quoted["sheet_sha256"] == sha and quoted["judge_model"] == MODEL
    with pytest.raises(report.Refused, match="the agreement check is of judge"):
        report._agreement(path, "gpt-5.1-2025-11-13")
