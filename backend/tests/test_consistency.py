"""Term consistency (OUTLINE 4.7, v0.25): the rules in `evaluation/consistency.py` and the scorer eval/consistency.py, on
made-up text. The judge is the real LLMClient and openai SDK on an httpx.MockTransport; no network, and both ledgers live
in tmp_path. Nothing here needs data/, the database or a model.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

import httpx
import pytest

from fukidashi import config
from fukidashi.evaluation import consistency, glossary, terms
from fukidashi.evaluation.consistency import Extraction

BOOK = "boureisougi"  # a dev book; the glossaries here hold made-up terms


def entry(id, source, target, *, category="term", borderline=False, aliases=(), variants=()):
    return {"id": id, "category": category, "source": source, "target": target, "borderline": borderline,
            "source_aliases": list(aliases), "target_variants": list(variants)}


def glossary_document(*entries):
    return {"book_title": BOOK, "split": "dev", "opencc_config": "t2s", "drafted_by": "test",
            "reviewed_by_author": "2026-10-29", "entries": list(entries)}


def glossary_of(*entries):
    loaded = glossary.parse_glossary(glossary_document(*entries))
    assert loaded.problems == ()
    return loaded


def line(page, idx, ja, hyp, ref, **more):
    return {"book": BOOK, "page_index": page, "idx": idx, "run_id": 7, "ja": ja, "hyp": hyp, "ref_zh_hans": ref,
            "ref_zh_hant": f"HANT-{page}-{idx}", **more}


def occurrences_of(entries, lines):
    return consistency.find_occurrences(glossary_of(*entries), consistency.read_segments(lines))


def places(found):
    return [(o.entry.id, o.segment.page_index, o.segment.idx) for o in found]


def rendered(text):
    return Extraction("rendered", text, consistency.normalise_rendering(text), 1)


def other(status):
    return Extraction(status, attempts=1)


# --- the occurrences (AC1) ---------------------------------------------------------------------------------------------

ENTRIES = [
    entry("shiora", "シオラ", "希奥拉", category="character", variants=["西奥拉"]),
    entry("ryu", "リュウ", "龙", aliases=["ドラゴン"]),
    entry("resolve", "覚悟", "觉悟", borderline=True),  # an ordinary word: never scored
]
LINES = [
    line(1, 0, "シオラさん", "希奥拉小姐", "希奥拉小姐"),
    line(1, 1, "シオラ、覚悟しろ!", "西奥拉,觉悟吧!", "希奥拉!有觉悟吧!"),
    line(2, 0, "ドラゴンだ", "是飞龙", "是龙"),
    line(2, 1, "リュウが来た", "他来了", "那人来了"),  # the reference leaves the term out: still an occurrence
    line(3, 0, "覚悟", "觉悟", "觉悟"),  # the borderline entry only
    line(3, 1, "ｼｵﾗ", "希 奥拉", "希奥拉"),  # half-width katakana: NFKC makes it シオラ
]


def test_occurrences_are_the_segments_whose_japanese_holds_the_source_or_an_alias_and_never_a_borderline_entry():
    found = occurrences_of(ENTRIES, LINES)

    assert places(found) == [("shiora", 1, 0), ("shiora", 1, 1), ("shiora", 3, 1), ("ryu", 2, 0), ("ryu", 2, 1)]
    assert all(o.entry.id != "resolve" for o in found)  # 覚悟 is in 1/1 and 3/0, but the entry is borderline
    assert [o.form for o in found] == ["シオラ", "シオラ", "シオラ", "ドラゴン", "リュウ"]  # the glossary's own form
    assert [e.id for e in consistency.scored_entries(glossary_of(*ENTRIES))] == ["shiora", "ryu"]


def test_the_reference_and_the_translation_do_not_decide_an_occurrence():
    plain = places(occurrences_of(ENTRIES, LINES))
    scrambled = [{**l, "hyp": "没有", "ref_zh_hans": "什么都没有", "ref_zh_hant": "希奥拉龙"} for l in LINES]

    assert places(occurrences_of(ENTRIES, scrambled)) == plain


def test_the_occurrences_are_term_accuracys_occurrences_and_left_out_segments_together():
    """Term accuracy counts a segment when the Japanese holds the source and the reference the target, and lists the
    others as left out; consistency takes both, with the same NFKC matching."""
    accuracy = terms.score_terms(glossary_of(*ENTRIES), LINES)
    expected = sorted((e.id, p, i) for e in accuracy.entries if not e.borderline
                      for p, i in [*((o.page_index, o.idx) for o in e.occurrences), *e.left_out])

    assert sorted(places(occurrences_of(ENTRIES, LINES))) == expected
    assert any(e.left_out for e in accuracy.entries)  # the case is in the data: 2/1


def test_the_glossarys_forms_are_normalised_too():
    found = occurrences_of([entry("shiora", "ｼｵﾗ", "希奥拉")], [line(1, 0, "シオラ", "x", "y")])

    assert places(found) == [("shiora", 1, 0)]


@pytest.mark.parametrize("field", ["page_index", "idx", "ja", "hyp", "ref_zh_hans"])
def test_a_line_without_a_field_it_reads_is_an_error(field):
    broken = {k: v for k, v in LINES[0].items() if k != field}

    with pytest.raises(consistency.ConsistencyError, match=field):
        consistency.read_segments([broken])


def test_a_line_needs_no_other_field_and_a_repeated_segment_is_an_error():
    bare = [{k: l[k] for k in ("page_index", "idx", "ja", "hyp", "ref_zh_hans")} for l in LINES]
    assert places(occurrences_of(ENTRIES, bare)) == places(occurrences_of(ENTRIES, LINES))

    with pytest.raises(consistency.ConsistencyError, match="line 7 repeats page_index 1 idx 0, which line 1 has"):
        consistency.read_segments(LINES + [LINES[0]])


# --- normalisation and the answer ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("a, b", [
    ("希 奥拉", "希奥拉"), ("希\u3000奥拉", "希奥拉"), ("托蒂亚\u30fb家", "托蒂亚\u00b7家"), ("托蒂亚\u2027家", "托蒂亚\u2022家"),
    ("托蒂亚\uff65家", "托蒂亚\u00b7家"), ("ＡＢ型", "AB型"), ("希\n奥\u2028拉", "希奥拉")])
def test_renderings_are_compared_after_nfkc_without_whitespace_and_with_one_middle_dot(a, b):
    assert consistency.normalise_rendering(a) == consistency.normalise_rendering(b)


def test_different_renderings_stay_different():
    assert consistency.normalise_rendering("希奥拉") != consistency.normalise_rendering("西奥拉")
    assert consistency.normalise_rendering("托蒂亚-家") != consistency.normalise_rendering("托蒂亚\u00b7家")


def test_a_usable_answer_has_a_known_status_and_a_rendering_that_occurs_in_the_text():
    assert consistency.read_answer('{"status": "rendered", "rendering": "希奥拉"}', "希奥拉小姐!") == \
        consistency.Answer("rendered", "希奥拉", "希奥拉")
    # occurs after normalisation: the judge left out the space the text has
    assert consistency.read_answer('{"status": "rendered", "rendering": "希奥拉"}', "希 奥拉").normalised == "希奥拉"
    assert consistency.read_answer('{"status": "omitted", "rendering": "x"}', "t") == consistency.Answer("omitted")
    assert consistency.read_answer(' {"status": "not_this_term"} ', "t") == consistency.Answer("not_this_term")


@pytest.mark.parametrize("answer, why", [
    ("希奥拉", "not valid JSON"), ('```json\n{"status": "omitted"}\n```', "not valid JSON"),
    ('["rendered"]', "not a JSON object"), ('{"status": "unknown"}', "is not"), ('{"rendering": "希奥拉"}', "is not"),
    ('{"status": "invalid"}', "is not"),  # invalid is the scorer's, not an answer
    ('{"status": "rendered"}', "needs the rendering"), ('{"status": "rendered", "rendering": " "}', "needs the rendering"),
    ('{"status": "rendered", "rendering": 3}', "needs the rendering"),
    ('{"status": "rendered", "rendering": "西奥拉"}', "does not occur"),
])
def test_an_answer_that_cannot_be_used_says_why(answer, why):
    with pytest.raises(consistency.Unusable, match=why):
        consistency.read_answer(answer, "希奥拉小姐!")


# --- the scores (AC2) ---------------------------------------------------------------------------------------------------------

def test_an_entrys_consistency_is_its_most_common_renderings_share_of_the_rendered_occurrences():
    score = consistency.score_text([rendered("希奥拉"), rendered("西奥拉"), rendered("希 奥拉"), other("omitted"),
                                    other("not_this_term"), other("invalid"), rendered("希奥拉")])

    # 希奥拉 three times (one with a space), 西奥拉 once: 3 of 4; the other three are counted, not scored
    assert (score.rendered, score.most_common, score.most_common_count) == (4, "希奥拉", 3)
    assert score.consistency == 0.75 and score.pooled
    assert score.renderings == {"希奥拉": 3, "西奥拉": 1}
    assert score.statuses == {"rendered": 4, "omitted": 1, "not_this_term": 1, "invalid": 1}


def test_a_tie_keeps_the_first_rendering_and_the_share_is_the_same_either_way():
    score = consistency.score_text([rendered("乙"), rendered("甲"), rendered("甲"), rendered("乙")])

    assert (score.most_common, score.most_common_count, score.consistency) == ("乙", 2, 0.5)


def test_an_entry_with_one_rendered_occurrence_has_a_share_but_stays_out_of_the_pooled_score():
    one, none = consistency.score_text([rendered("甲"), other("omitted")]), consistency.score_text([other("invalid")])

    assert (one.rendered, one.consistency, one.pooled) == (1, 1.0, False)
    assert (none.rendered, none.most_common, none.consistency, none.pooled) == (0, None, None, False)


def test_the_pooled_score_sums_the_most_common_counts_over_the_entries_with_two_or_more_renderings():
    scores = [consistency.score_text([rendered("甲"), rendered("甲"), rendered("乙")]),  # 2 of 3
              consistency.score_text([rendered("丙"), other("omitted"), other("invalid")]),  # 1 rendered: left out
              consistency.score_text([rendered("丁")] * 3 + [rendered("戊")])]  # 3 of 4

    pooled = consistency.pool(scores)

    # (2 + 3) / (3 + 4); not the mean of the entries ((2/3 + 3/4) / 2), and not 6/8 with the left-out entry
    assert (pooled.entries, pooled.rendered, pooled.most_common_count) == (2, 7, 5)
    assert pooled.consistency == pytest.approx(5 / 7)
    assert consistency.pool([scores[1]]).consistency is None and consistency.pool([]).entries == 0


# --- the extractor's check (AC4) -------------------------------------------------------------------------------------------

SHIORA = glossary_of(entry("shiora", "シオラ", "希奥拉", variants=["西奥拉"])).entries[0]


def test_the_answer_is_known_only_where_the_reference_holds_the_target_or_a_variant():
    assert consistency.reference_has_target(SHIORA, "希奥拉小姐")
    assert consistency.reference_has_target(SHIORA, "西奥拉!")
    assert consistency.reference_has_target(SHIORA, "希 奥拉")  # normalised as the renderings are
    assert not consistency.reference_has_target(SHIORA, "她来了")


@pytest.mark.parametrize("extraction, correct", [
    (rendered("希奥拉"), True), (rendered("西奥拉"), True),
    (rendered("希奥拉小姐"), True),  # contains the target: an honorific at the edge
    (rendered("奥拉"), True),  # contained in it
    (rendered("希奥"), True), (rendered("蕾娜"), False), (rendered("希欧拉"), False),
    (other("omitted"), False), (other("not_this_term"), False), (other("invalid"), False)])
def test_a_correct_extraction_contains_or_is_contained_in_the_target_or_a_variant(extraction, correct):
    assert consistency.extraction_correct(SHIORA, extraction) is correct


@pytest.mark.parametrize("correct, checked, reportable", [
    (9, 10, True), (27, 30, True), (10, 10, True), (8, 9, False), (89, 100, False), (0, 0, False)])
def test_consistency_is_reportable_at_90_percent_and_not_below(correct, checked, reportable):
    check = consistency.check_extractor([(True, True)] * correct + [(True, False)] * (checked - correct)
                                        + [(False, False)] * 5)  # not checked: not counted either way

    assert (check.checked, check.correct, check.reportable) == (checked, correct, reportable)
    assert check.accuracy == (correct / checked if checked else None)


def test_the_book_scores_check_the_extractor_only_where_the_reference_holds_the_target():
    found = occurrences_of(ENTRIES, LINES)
    # translation renders everything as 希奥拉 / 龙; the reference's extractions: right, right, wrong, right, and the
    # one on 2/1 is wrong but its reference lacks the term, so it is not checked
    reference = {(1, 0): rendered("希奥拉"), (1, 1): rendered("西奥拉"), (3, 1): rendered("蕾娜"),
                 (2, 0): rendered("飞龙"), (2, 1): rendered("他")}
    results = [consistency.OccurrenceResult(o, {
        "translation": rendered("希奥拉" if o.entry.id == "shiora" else "龙"),
        "reference": reference[(o.segment.page_index, o.segment.idx)]}) for o in found]

    scores = consistency.score_book(glossary_of(*ENTRIES), results)

    assert (scores.extractor.checked, scores.extractor.correct, scores.extractor.reportable) == (4, 3, False)
    assert scores.extractor.accuracy == 0.75
    assert [r.correct for e in scores.entries for r in e.results] == [True, True, False, True, None]
    assert scores.left_out == ("resolve",) and [e.entry.id for e in scores.entries] == ["shiora", "ryu"]
    assert scores.pooled["translation"].consistency == 1.0  # 3/3 and 2/2
    # reference: shiora 希奥拉 / 西奥拉 / 蕾娜 (1 of 3), ryu 飞龙 / 他 (1 of 2): (1 + 1) / (3 + 2)
    assert scores.pooled["reference"].consistency == pytest.approx(2 / 5)


# --- the request (AC3) ------------------------------------------------------------------------------------------------------

def test_a_request_holds_the_japanese_the_form_the_kind_and_one_text_and_nothing_of_the_glossary():
    shiora, ryu = (occurrences_of(ENTRIES, LINES)[i] for i in (1, 3))

    for occurrence, which, text, kind in ((shiora, "translation", "西奥拉,觉悟吧!", "the name of a character"),
                                          (shiora, "reference", "希奥拉!有觉悟吧!", "the name of a character"),
                                          (ryu, "translation", "是飞龙", "a term of the story")):
        system, user = consistency.build_messages(occurrence, which)
        assert system == {"role": "system", "content": consistency.SYSTEM_PROMPT}
        assert user["role"] == "user" and f"```{text}```" in user["content"]
        assert f"```{occurrence.segment.ja}```" in user["content"] and f"「{occurrence.form}」" in user["content"]
        assert kind in user["content"] and "JSON" in system["content"] + user["content"]
        other_text = occurrence.segment.text("reference" if which == "translation" else "translation")
        assert other_text not in user["content"]
        assert "希奥拉" not in user["content"].replace(text, "")  # the target only where the text itself has it
        assert "image" not in json.dumps([system, user])


# --- eval/consistency.py ---------------------------------------------------------------------------------------------------

API_KEY = "sk-judge-test-secret"
JUDGE_ENV = {"JUDGE_BASE_URL": "http://judge.test/v1", "JUDGE_API_KEY": API_KEY, "JUDGE_MODEL": "openai/judge-model",
             "JUDGE_TEMPERATURE": "0", "JUDGE_PRICE_INPUT_PER_M": "0.15", "JUDGE_PRICE_OUTPUT_PER_M": "0.60"}
OUT = "consistency-judge-model.json"
EVAL_ID = "20261029T000000Z-full"
# Every Chinese text is distinctive, so a request can be told apart by the texts it holds
CLI_ENTRIES = [entry("shiora", "シオラ", "希奥拉", category="character", variants=["西奥拉"]),
               entry("gunman", "ガンマン", "枪手"), entry("resolve", "覚悟", "觉悟", borderline=True)]
CLI_LINES = [line(1, 0, "シオラさん", "希奥拉小姐好", "希奥拉小姐!"),
             line(1, 1, "ガンマンだ", "是枪手啊", "那是枪手"),
             line(2, 0, "覚悟", "觉悟一", "觉悟二"),
             line(2, 1, "シオラ、ガンマンか", "西奥拉是枪手吗", "希奥拉,你是枪手?"),
             line(3, 0, "シオラ!", "她呢", "希奥拉!")]
# (page, idx, entry) of the occurrences, in the order the scorer goes through them
CLI_OCCURRENCES = [(1, 0, "shiora"), (2, 1, "shiora"), (3, 0, "shiora"), (1, 1, "gunman"), (2, 1, "gunman")]
KNOWN = {"シオラ": ("希奥拉", "西奥拉"), "ガンマン": ("枪手",)}  # what the scripted judge looks for, by the form asked


def make_run(tmp_path, lines=CLI_LINES, entries=CLI_ENTRIES, *, manifest=None, name="run-1"):
    """(run dir, glossary dir), each its own directory in tmp_path."""
    run_dir, gloss_dir = tmp_path / name, tmp_path / f"glossaries-{name}"
    run_dir.mkdir()
    gloss_dir.mkdir()
    (gloss_dir / f"{BOOK}.json").write_text(json.dumps(glossary_document(*entries), ensure_ascii=False),
                                            encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({
        "eval_id": EVAL_ID, "book": BOOK, "config_name": "full", "finished_at": "2026-10-29T00:01:00+00:00",
        "requested_models": ["translation-model"], "response_models": ["translation-model-1"], **(manifest or {})}),
        encoding="utf-8")
    (run_dir / "segments.jsonl").write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines),
                                            encoding="utf-8")
    return run_dir, gloss_dir


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """eval/consistency.py with the judge's variables set, .env never read, and both ledgers in tmp_path."""
    for name in list(os.environ):
        if name.startswith(("LLM_", "JUDGE")):
            monkeypatch.delenv(name)
    for name, value in JUDGE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LLM_MODEL", "translation-model")
    monkeypatch.setattr(config, "LEDGER_PATH", tmp_path / "ledgers" / "usage.jsonl")
    monkeypatch.setattr(config, "JUDGE_LEDGER_PATH", tmp_path / "ledgers" / "judge_usage.jsonl")
    spec = importlib.util.spec_from_file_location("eval_consistency_under_test", config.ROOT / "eval" / "consistency.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_dotenv", lambda path: None)
    return module


def reply(text, *, finish="stop", usage=(300, 12), model="judge-model-2024"):
    body = {"id": "c", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}]}
    if usage is not None:
        body["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}
    return httpx.Response(200, json=body)


def chinese_text(body):
    """The one Chinese text of a request: the second fenced block of its first user message."""
    return re.findall(r"```(.*?)```", body["messages"][1]["content"], re.S)[1]


def sensible(body):
    """A judge that answers with the first rendering it knows for the form asked that is in the text, and says
    `omitted` when there is none."""
    text = chinese_text(body)
    form = re.search(r"「(.*?)」", body["messages"][1]["content"]).group(1)
    found = next((k for k in KNOWN.get(form, ()) if k in text), None)
    answer = {"status": "rendered", "rendering": found} if found else {"status": "omitted", "rendering": None}
    return json.dumps(answer, ensure_ascii=False)


class Judge:
    """A scripted provider: answers in order (an httpx.Response, a str, or None for `sensible`), then `sensible`."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        answer = self.answers.pop(0) if self.answers else None
        if answer is None:
            answer = sensible(body)
        return reply(answer) if isinstance(answer, str) else answer


def never_called(request):
    raise AssertionError("the judge was called")


def run(cli, run_dir, gloss_dir, judge, *args):
    return cli.main([str(run_dir), "--glossary-dir", str(gloss_dir), *args],
                    http_client=httpx.Client(transport=httpx.MockTransport(judge)))


def written(run_dir, name=OUT):
    return json.loads((run_dir / name).read_text(encoding="utf-8"))


def ledger(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l] if path.exists() else []


def test_the_scorer_writes_every_field(cli, tmp_path, capsys):
    run_dir, gloss_dir = make_run(tmp_path)

    assert run(cli, run_dir, gloss_dir, Judge()) == 0

    assert sorted(p.name for p in run_dir.iterdir()) == [OUT, "manifest.json", "segments.jsonl"]  # no .tmp left
    doc = written(run_dir)
    assert list(doc) == ["eval_id", "book", "segments_sha256", "glossary_sha256", "rules", "judge", "occurrences",
                         "entries_scored", "entries_left_out", "pooled", "statuses", "extractor", "entries", "usage",
                         "scored_at", "elapsed_s"]
    assert (doc["eval_id"], doc["book"], doc["rules"]) == (EVAL_ID, BOOK, consistency.RULES)
    assert doc["segments_sha256"] == hashlib.sha256((run_dir / "segments.jsonl").read_bytes()).hexdigest()
    assert doc["glossary_sha256"] == hashlib.sha256((gloss_dir / f"{BOOK}.json").read_bytes()).hexdigest()
    assert doc["judge"] == {"prefix": "JUDGE", "model": "openai/judge-model", "requested_model": "openai/judge-model",
                            "response_model": "judge-model-2024", "temperature": 0.0, "reasoning_effort": None,
                            "json_mode": "json_object", "max_tokens": 200}
    assert (doc["occurrences"], doc["entries_scored"], doc["entries_left_out"]) == (5, 2, ["resolve"])
    # translation: shiora 希奥拉, 西奥拉, (3/0 has none: omitted) -> 1 of 2; gunman 枪手 twice -> 2 of 2
    assert doc["pooled"]["translation"] == {"entries": 2, "rendered": 4, "most_common_count": 3, "consistency": 0.75}
    assert doc["pooled"]["reference"] == {"entries": 2, "rendered": 5, "most_common_count": 5, "consistency": 1.0}
    assert doc["statuses"]["translation"] == {"rendered": 4, "omitted": 1, "not_this_term": 0, "invalid": 0}
    assert doc["extractor"] == {"checked": 5, "correct": 5, "accuracy": 1.0, "threshold": 0.9, "reportable": True}
    shiora = doc["entries"][0]
    assert (shiora["id"], shiora["category"], shiora["occurrences"]) == ("shiora", "character", 3)
    assert shiora["translation"] == {"rendered": 2, "most_common": "希奥拉", "most_common_count": 1, "consistency": 0.5,
                                     "pooled": True, "renderings": {"希奥拉": 1, "西奥拉": 1},
                                     "statuses": {"rendered": 2, "omitted": 1, "not_this_term": 0, "invalid": 0}}
    assert shiora["extractor"] == {"checked": 3, "correct": 3, "accuracy": 1.0}
    first = shiora["segments"][0]
    assert first == {"page_index": 1, "idx": 0, "form": "シオラ",
                     "translation": {"status": "rendered", "rendering": "希奥拉", "normalised": "希奥拉", "attempts": 1,
                                     "problems": [], "answer": '{"status": "rendered", "rendering": "希奥拉"}'},
                     "reference": {"status": "rendered", "rendering": "希奥拉", "normalised": "希奥拉", "attempts": 1,
                                   "problems": [], "answer": '{"status": "rendered", "rendering": "希奥拉"}'},
                     "extractor": {"checked": True, "correct": True}}
    assert doc["usage"] == {"calls": 10, "unknown_usage_calls": 0, "input_tokens": 3000, "output_tokens": 120,
                            "cost_usd": round(10 * (300 * 0.15 + 12 * 0.60) / 1e6, 6)}
    assert doc["scored_at"].endswith("+00:00") and doc["elapsed_s"] >= 0
    out = capsys.readouterr().out
    assert "translation: 75.0% (3/4 over 2 entries with 2+ renderings)" in out
    assert "extractor on the reference: 100.0% (5/5" in out and "consistency is reportable" in out
    assert "calls: 10" in out and f"Wrote {run_dir / OUT}" in out


def test_two_calls_per_occurrence_one_on_each_text_and_neither_sees_the_other_text_or_another_occurrence(cli, tmp_path):
    run_dir, gloss_dir = make_run(tmp_path)
    judge = Judge()

    assert run(cli, run_dir, gloss_dir, judge) == 0

    by_place = {(l["page_index"], l["idx"]): l for l in CLI_LINES}
    every_text = {l[k] for l in CLI_LINES for k in ("hyp", "ref_zh_hans")}
    assert len(judge.bodies) == 2 * len(CLI_OCCURRENCES)
    for n, body in enumerate(judge.bodies):
        page, idx, entry_id = CLI_OCCURRENCES[n // 2]
        own = by_place[(page, idx)]["hyp" if n % 2 == 0 else "ref_zh_hans"]  # the Translation first, then the reference
        raw = json.dumps(body, ensure_ascii=False)
        assert [t for t in every_text if t in raw] == [own], (n, body)  # its own text and no other
        assert "HANT-" not in raw and "image" not in raw and "data:" not in raw
        occurrence = next(o for o in consistency.find_occurrences(
            glossary.load_glossary(gloss_dir / f"{BOOK}.json"), consistency.read_segments(CLI_LINES))
            if (o.segment.page_index, o.segment.idx, o.entry.id) == (page, idx, entry_id))
        assert body["messages"] == consistency.build_messages(occurrence, "translation" if n % 2 == 0 else "reference")
        assert body["model"] == "openai/judge-model" and body["max_tokens"] == 200 and body["temperature"] == 0
        assert body["response_format"] == {"type": "json_object"} and "tools" not in body


def test_the_judges_calls_go_to_its_own_ledger_and_never_to_the_translation_ledger(cli, tmp_path):
    run_dir, gloss_dir = make_run(tmp_path)

    assert run(cli, run_dir, gloss_dir, Judge()) == 0

    calls = ledger(config.JUDGE_LEDGER_PATH)
    assert len(calls) == 10 and {c["requested_model"] for c in calls} == {"openai/judge-model"}
    assert calls[0]["page_id"] == f"consistency:{EVAL_ID}:{BOOK}:1:0:shiora:translation"
    assert calls[1]["page_id"] == f"consistency:{EVAL_ID}:{BOOK}:1:0:shiora:reference"
    assert not config.LEDGER_PATH.exists()


def test_a_rendering_that_does_not_occur_gets_one_retry_that_shows_the_answer_and_why_then_it_is_invalid(cli, tmp_path):
    run_dir, gloss_dir = make_run(tmp_path, [line(1, 0, "シオラ", "希奥拉啊", "希奥拉呀"), line(2, 0, "シオラ", "希奥拉吧", "希奥拉")])
    wrong = '{"status": "rendered", "rendering": "西奥拉"}'
    judge = Judge(wrong, wrong)

    assert run(cli, run_dir, gloss_dir, judge) == 0

    first, retry = judge.bodies[0], judge.bodies[1]
    assert retry["messages"][:2] == first["messages"] and len(retry["messages"]) == 4
    assert retry["messages"][2] == {"role": "assistant", "content": wrong}
    assert retry["messages"][3]["role"] == "user" and "does not occur" in retry["messages"][3]["content"]
    assert "希奥拉呀" not in json.dumps(retry, ensure_ascii=False)  # still the Translation's call: no reference
    assert len(judge.bodies) == 2 + 3  # the other three extractions took one call each
    doc = written(run_dir)
    invalid = doc["entries"][0]["segments"][0]["translation"]
    assert (invalid["status"], invalid["rendering"], invalid["attempts"], invalid["answer"]) == ("invalid", None, 2, wrong)
    assert len(invalid["problems"]) == 2 and "attempt 2: the rendering" in invalid["problems"][1]
    # invalid is unknown: not a rendering, not counted in the consistency
    assert doc["statuses"]["translation"] == {"rendered": 1, "omitted": 0, "not_this_term": 0, "invalid": 1}
    assert doc["entries"][0]["translation"]["rendered"] == 1 and doc["pooled"]["translation"]["entries"] == 0
    assert doc["pooled"]["translation"]["consistency"] is None


@pytest.mark.parametrize("first", [
    "not json", '{"status": "maybe"}', '{"status": "rendered"}', reply('{"status": "omitted"}', finish="length")])
def test_an_unusable_answer_whose_retry_can_be_used_counts(cli, tmp_path, first):
    run_dir, gloss_dir = make_run(tmp_path, [line(1, 0, "シオラ", "希奥拉啊", "希奥拉呀")])

    assert run(cli, run_dir, gloss_dir, Judge(first, '{"status": "rendered", "rendering": "希奥拉"}')) == 0

    extraction = written(run_dir)["entries"][0]["segments"][0]["translation"]
    assert (extraction["status"], extraction["rendering"], extraction["attempts"]) == ("rendered", "希奥拉", 2)
    assert len(extraction["problems"]) == 1


def test_not_this_term_and_omitted_are_reported_not_scored(cli, tmp_path):
    lines = [line(p, 0, "シオラ", f"希奥拉{p}", f"希奥拉-{p}") for p in (1, 2, 3, 4)]
    run_dir, gloss_dir = make_run(tmp_path, lines)
    answers = ['{"status": "not_this_term", "rendering": null}', None, '{"status": "omitted"}', None, None, None,
               '{"status": "rendered", "rendering": "希奥拉"}', None]

    assert run(cli, run_dir, gloss_dir, Judge(*answers)) == 0

    doc = written(run_dir)
    assert doc["statuses"]["translation"] == {"rendered": 2, "omitted": 1, "not_this_term": 1, "invalid": 0}
    assert doc["pooled"]["translation"] == {"entries": 1, "rendered": 2, "most_common_count": 2, "consistency": 1.0}


@pytest.mark.parametrize("k", [1, 2, 5])
def test_a_provider_error_stops_with_exit_1_writes_nothing_and_leaves_its_calls_in_the_judges_ledger(cli, tmp_path, capsys,
                                                                                                    k):
    run_dir, gloss_dir = make_run(tmp_path)
    before = sorted(p.name for p in run_dir.iterdir())
    answers = [None] * (k - 1) + [httpx.Response(500, json={"error": {"message": "boom"}})]

    assert run(cli, run_dir, gloss_dir, Judge(*answers)) == 1

    assert sorted(p.name for p in run_dir.iterdir()) == before
    calls = ledger(config.JUDGE_LEDGER_PATH)
    assert len(calls) == k and calls[-1]["error"] and all(c["error"] is None for c in calls[:-1])
    err = capsys.readouterr().err
    assert "nothing written" in err and f"{k} judge call(s)" in err and API_KEY not in err
    assert not config.LEDGER_PATH.exists()


def test_a_spent_budget_stops_with_exit_1_and_writes_nothing(cli, tmp_path):
    run_dir, gloss_dir = make_run(tmp_path)
    huge = reply("not json", usage=(4500, 10))  # unusable, and over the 4,000 tokens: the retry is not sent

    assert run(cli, run_dir, gloss_dir, Judge(huge)) == 1

    assert not list(run_dir.glob("consistency-*")) and len(ledger(config.JUDGE_LEDGER_PATH)) == 1


def test_unknown_usage_stays_unknown(cli, tmp_path):
    run_dir, gloss_dir = make_run(tmp_path)
    no_usage = reply('{"status": "rendered", "rendering": "希奥拉"}', usage=None)

    assert run(cli, run_dir, gloss_dir, Judge(no_usage)) == 0

    usage = written(run_dir)["usage"]
    assert usage["cost_usd"] is None and usage["unknown_usage_calls"] == 1 and usage["calls"] == 10


def test_the_judge_prefix_chooses_the_configuration_and_names_the_file(cli, tmp_path, monkeypatch):
    for name, value in {"JUDGE2_BASE_URL": "http://judge2.test/v1", "JUDGE2_API_KEY": "sk-2",
                        "JUDGE2_MODEL": "fallback-model", "JUDGE2_JSON_MODE": "none"}.items():
        monkeypatch.setenv(name, value)
    run_dir, gloss_dir = make_run(tmp_path)
    judge = Judge()

    assert run(cli, run_dir, gloss_dir, judge, "--judge-prefix", "JUDGE2") == 0

    doc = written(run_dir, "consistency-fallback-model.json")
    assert (doc["judge"]["prefix"], doc["judge"]["model"], doc["judge"]["json_mode"]) == ("JUDGE2", "fallback-model", "none")
    assert "response_format" not in judge.bodies[0] and not (run_dir / OUT).exists()


def test_the_output_is_replaced_through_a_temporary_file(cli, tmp_path, monkeypatch):
    run_dir, gloss_dir = make_run(tmp_path)
    (run_dir / OUT).write_text("old", encoding="utf-8")
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: replaced.append((src, dst)) or real_replace(src, dst))

    assert run(cli, run_dir, gloss_dir, Judge()) == 0

    ((src, dst),) = replaced
    assert os.path.basename(src) == OUT + ".tmp" and os.path.basename(dst) == OUT
    assert written(run_dir)["occurrences"] == 5 and not (run_dir / (OUT + ".tmp")).exists()


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\u0085"], ids=["U+2028", "U+2029", "U+0085"])
def test_a_text_with_a_unicode_line_separator_is_read_as_one_line(cli, tmp_path, separator):
    run_dir, gloss_dir = make_run(tmp_path, [line(1, 0, "シオラ", f"希奥拉{separator}啊", "希奥拉"),
                                             line(1, 1, "シオラ", "希奥拉", "希奥拉")])
    assert separator in (run_dir / "segments.jsonl").read_text(encoding="utf-8")

    assert run(cli, run_dir, gloss_dir, Judge()) == 0

    assert written(run_dir)["occurrences"] == 2


def refused_cases(tmp_path, monkeypatch):
    """(what is wrong, run dir, glossary dir, a setup step, text expected in the message), each in its own directory."""
    cases = []
    run_dir, gloss_dir = make_run(tmp_path, name="nowhere-base")
    cases.append(("a directory that is not there", tmp_path / "nowhere", gloss_dir, None, "not a directory"))
    run_dir, gloss_dir = make_run(tmp_path, manifest={"finished_at": None}, name="unfinished")
    cases.append(("an unfinished eval", run_dir, gloss_dir, None, "not finished"))
    run_dir, gloss_dir = make_run(tmp_path, manifest={"book": "tojime_no_siora"}, name="no-glossary")
    cases.append(("no glossary for the book", run_dir, gloss_dir, None, "no glossary for book 'tojime_no_siora'"))
    run_dir, gloss_dir = make_run(tmp_path, name="unconfigured")
    cases.append(("the judge is not configured", run_dir, gloss_dir, lambda: monkeypatch.delenv("JUDGE_API_KEY"),
                  "JUDGE_API_KEY"))
    run_dir, gloss_dir = make_run(tmp_path, name="same-model")
    cases.append(("the judge's model is LLM_MODEL", run_dir, gloss_dir,
                  lambda: monkeypatch.setenv("LLM_MODEL", "judge-model"), "translation model (LLM_MODEL)"))
    # and the ones the shape of the files gives
    run_dir, gloss_dir = make_run(tmp_path, [], name="empty")
    cases.append(("an empty segments.jsonl", run_dir, gloss_dir, None, "no segments"))
    run_dir, gloss_dir = make_run(tmp_path, [{k: v for k, v in CLI_LINES[0].items() if k != "ref_zh_hans"}],
                                  name="no-ref")
    cases.append(("a line without ref_zh_hans", run_dir, gloss_dir, None, "`ref_zh_hans`"))
    run_dir, gloss_dir = make_run(tmp_path, CLI_LINES + [CLI_LINES[0]], name="repeated")
    cases.append(("a repeated segment", run_dir, gloss_dir, None, "repeats page_index 1 idx 0"))
    bad = [{**CLI_ENTRIES[0], "borderline": "no"}, *CLI_ENTRIES[1:]]
    run_dir, gloss_dir = make_run(tmp_path, entries=bad, name="bad-glossary")
    cases.append(("a glossary with problems", run_dir, gloss_dir, None, "has problems"))
    run_dir, gloss_dir = make_run(tmp_path, name="ran-with")
    cases.append(("the judge's model translated this run", run_dir, gloss_dir,
                  lambda: monkeypatch.setenv("JUDGE_MODEL", "translation-model-1"), "translated with"))
    return cases


def test_the_scorer_refuses_with_exit_2_sends_nothing_and_writes_nothing(cli, tmp_path, monkeypatch, capsys):
    cases = refused_cases(tmp_path, monkeypatch)
    for what, run_dir, gloss_dir, setup, message in cases:
        with monkeypatch.context() as m:
            for name, value in JUDGE_ENV.items():
                m.setenv(name, value)
            m.setenv("LLM_MODEL", "translation-model")
            if setup:
                setup()
            before = sorted(p.name for p in run_dir.iterdir()) if run_dir.exists() else None

            assert run(cli, run_dir, gloss_dir, never_called) == 2, what

            err = capsys.readouterr().err
            assert err.startswith(f"Cannot score {run_dir}: ") and message in err, (what, err)
            for value in (API_KEY, "http://judge.test/v1"):
                assert value not in err, (what, value)
            assert (sorted(p.name for p in run_dir.iterdir()) if run_dir.exists() else None) == before, what
            assert not config.JUDGE_LEDGER_PATH.exists() and not config.LEDGER_PATH.exists(), what


def test_the_scorer_never_loads_env_before_the_run_and_its_glossary_are_known_to_be_scorable(cli, tmp_path, monkeypatch):
    def refuse(path):
        raise AssertionError(".env was read")

    monkeypatch.setattr(cli, "load_dotenv", refuse)
    run_dir, gloss_dir = make_run(tmp_path, manifest={"book": "tojime_no_siora"})

    assert run(cli, tmp_path / "nowhere", gloss_dir, never_called) == 2
    assert run(cli, run_dir, gloss_dir, never_called) == 2


def test_the_script_runs_as_a_program_and_refuses_a_missing_directory(tmp_path):
    done = subprocess.run([sys.executable, str(config.ROOT / "eval" / "consistency.py"), str(tmp_path / "nowhere")],
                          capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, timeout=60)

    assert done.returncode == 2 and "Cannot score" in done.stderr and "not a directory" in done.stderr


def test_the_default_glossary_dir_is_the_projects(cli):
    assert cli.GLOSSARY_DIR == config.ROOT / "eval" / "gold_glossary"


# --- boundaries ------------------------------------------------------------------------------------------------------

LIBRARY = config.ROOT / "backend" / "fukidashi" / "evaluation" / "consistency.py"
SCRIPT = config.ROOT / "eval" / "consistency.py"


def imported_names(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(part for alias in node.names for part in alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            names.update((node.module or "").split("."))
            names.update(alias.name for alias in node.names)
    return names


def test_the_consistency_code_reaches_the_judge_only_through_llmclient_and_never_the_database():
    for path in (LIBRARY, SCRIPT):
        names = imported_names(path)
        assert not names & {"openai", "langchain", "langchain_core", "langgraph", "sqlite3", "store", "db", "runs"}, path
        assert "ref_zh_hant" not in path.read_text(encoding="utf-8"), path
        assert "mqm" not in names, path  # the GEMBA-MQM prompt and parser: the extraction is not GEMBA-MQM


def test_the_rules_make_no_model_call():
    assert not imported_names(LIBRARY) & {"llm", "httpx", "requests", "socket", "mqm_scorer"}
