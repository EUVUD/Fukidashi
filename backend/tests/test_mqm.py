# SPDX-License-Identifier: CC-BY-SA-4.0
# This file is licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), not under the
# repository's MIT License. It holds material copied and adapted from GEMBA
# (https://github.com/MicrosoftTranslator/GEMBA, Microsoft Translator: Tom Kocmi, Christian Federmann) at commit
# a7a7eff8e46998447c6cbf09d06affc8f1b99ab4, which is licensed under CC BY-SA 4.0: answers in GEMBA-MQM's format, the
# errors and scores of GEMBA's three example answers, and the hash of its prompt template. See LICENSE.

"""GEMBA-MQM (OUTLINE 4.4): the prompt, parser and score in `evaluation/mqm.py`, and the scorer (`evaluation/mqm_scorer.py`,
eval/mqm.py) on made-up eval runs. The judge is the real LLMClient and openai SDK on an httpx.MockTransport; no network,
and both ledgers live in tmp_path.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

import httpx
import pytest

from fukidashi import config
from fukidashi.evaluation import mqm, mqm_scorer

# GEMBA's TEMPLATE_GEMBA_MQM at commit a7a7eff8e46998447c6cbf09d06affc8f1b99ab4 (gemba/gemba_mqm_utils.py), rebuilt from
# that file with `ast` (its `few_shots` literal and the strings in `mqm_fewshot`, in mqm_fewshot's order) and hashed as
# sha256(json.dumps(template, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).
GEMBA_TEMPLATE_SHA256 = "d3b87df3827195db96ef55f0ebe126be3ed019cb1c1e0bca307b05d745bdb3c6"

NO_ERROR = "Critical:\nno-error\nMajor:\nno-error\nMinor:\nno-error\n"
MIXED = 'Critical:\nno-error\nMajor:\naccuracy/mistranslation - "他走了"\nMinor:\nfluency/punctuation - "。"\n'


# --- the prompt ------------------------------------------------------------------------------------------------------

def test_the_template_is_gembas_at_the_pinned_commit():
    assert mqm.template_sha256() == GEMBA_TEMPLATE_SHA256
    assert mqm.GEMBA["commit"] == "a7a7eff8e46998447c6cbf09d06affc8f1b99ab4"
    assert mqm.GEMBA["repository"] == "https://github.com/MicrosoftTranslator/GEMBA"
    assert mqm.GEMBA["licence"] == "CC BY-SA 4.0"


def test_one_segments_messages_are_the_template_filled_with_japanese_chinese_the_source_and_the_translation():
    messages = mqm.build_messages("おはよう{x}", "早上好")

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user", "assistant",
                                             "user"]
    assert messages[:-1] == mqm.template_messages()[:-1]  # the system prompt and GEMBA's three examples, unchanged
    assert messages[-1]["content"] == mqm.USER_TEMPLATE.format(
        source_lang="Japanese", source_seg="おはよう{x}", target_lang="Chinese", target_seg="早上好")
    assert messages[-1]["content"].startswith("Japanese source:\n```おはよう{x}```\nChinese translation:\n```早上好```\n\n"
                                              "Based on the source segment and machine translation")
    assert messages[1]["content"].startswith("English source:\n```I do apologise")
    assert messages[6]["content"].startswith("Critical:\naccuracy/addition")


# --- the parser and the score ------------------------------------------------------------------------------------------

def severities(answer):
    return [e.severity for e in mqm.parse_answer(answer)]


def test_the_weights_are_gembas():
    assert mqm.WEIGHTS == {"critical": 25, "major": 5, "minor": 1}
    assert (mqm.COUNTED_ERRORS, mqm.SCORE_CAP) == (5, 25)


def test_an_answer_with_no_error_scores_zero():
    assert mqm.parse_answer(NO_ERROR) == ()
    assert mqm.mqm_score([]) == 0
    assert mqm.parse_answer("no-error") == ()  # the prompt's own word for it, without headings


def test_mixed_severities_are_placed_under_their_headings_and_weighted():
    errors = mqm.parse_answer(MIXED)

    assert [e.to_dict() for e in errors] == [
        {"severity": "major", "category": "accuracy-mistranslation", "span": "他走了",
         "text": 'accuracy/mistranslation - "他走了"'},
        {"severity": "minor", "category": "fluency-punctuation", "span": "。", "text": 'fluency/punctuation - "。"'}]
    assert mqm.mqm_score(e.severity for e in errors) == -6


@pytest.mark.parametrize("shot, expected_score, expected", [
    (0, -12, [("major", "accuracy-mistranslation", "involvement"), ("major", "accuracy-omission", "the account holder"),
              ("minor", "fluency-grammar", "wäre"), ("minor", "fluency-register", "dir")]),
    (1, -11, [("major", "accuracy-addition", "ve Vídni"), ("major", "accuracy-omission", "the stop-start"),
              ("minor", "terminology-inappropriate", "partaje")]),
    (2, -25, [("critical", "accuracy-addition", "of high-speed rail"),
              ("major", "accuracy-mistranslation", "go to the reviews"), ("minor", "style", "etc.,")]),
])
def test_gembas_own_example_answers_parse_and_score_as_gemba_scores_them(shot, expected_score, expected):
    errors = mqm.parse_answer(mqm.FEW_SHOTS[shot]["answer"])

    assert [(e.severity, e.category, e.span) for e in errors] == expected
    assert mqm.mqm_score(e.severity for e in errors) == expected_score  # 25 + 5 + 1 is capped at 25


@pytest.mark.parametrize("found, score", [
    (["minor"], -1), (["major"], -5), (["critical"], -25), (["major", "minor", "minor"], -7),
    (["minor"] * 7, -5),  # only the first five errors count
    (["major"] + ["minor"] * 6, -9),  # the major first, then four minors
    (["minor", "minor", "major", "major", "major", "major"], -21),  # majors are counted before minors
    (["major"] * 6, -25),  # 6 x 5 = 30 would be more, but only five count
])
def test_the_score_counts_the_first_five_errors_by_severity_and_caps_at_25(found, score):
    assert mqm.mqm_score(found) == score


def test_a_non_translation_is_critical_wherever_it_is_listed():
    errors = mqm.parse_answer('Critical:\nno-error\nMajor:\nno-error\nMinor:\nnon-translation - "x"\n')

    assert [(e.severity, e.category) for e in errors] == [("critical", "non-translation")]


def test_headings_are_read_whatever_their_case_and_lines_that_say_no_error_are_skipped():
    answer = 'CRITICAL:\n  no error  \nMajor:\nAccuracy/Omission - "x"\n\nminor:\n'

    assert [(e.severity, e.category) for e in mqm.parse_answer(answer)] == [("major", "accuracy-omission")]


def test_the_class_is_read_before_the_span():
    (error,) = mqm.parse_answer('Major:\nstyle/awkward - "accuracy"\n')

    assert (error.category, error.span) == ("style", "accuracy")


@pytest.mark.parametrize("answer", [
    "", "  \n ",
    "The translation is accurate and fluent.",  # GEMBA would drop the line and score 0
    'Major errors: accuracy/mistranslation - "x"',  # no heading on a line of its own
    'accuracy/mistranslation - "x"\nCritical:\nno-error',  # an error before any heading
    '{"improved translation": "x", "errors": {}}',  # GEMBA's JSON branch serves another prompt
])
def test_an_answer_the_parser_cannot_place_is_unreadable(answer):
    with pytest.raises(mqm.Unreadable):
        mqm.parse_answer(answer)


# GPT-5.1's shape for a correct translation: a no-error line, then lines that explain it, and no severity heading
NO_ERROR_EXPLAINED = ("No-error:  \nThe translation conveys the meaning and the tone of the source. No accuracy, fluency, "
                      "style, or terminology errors.")


@pytest.mark.parametrize("answer", [
    NO_ERROR_EXPLAINED,
    "No-error:\n- The translation is accurate and fluent.\n- The register fits.",
    "\n  \nNo-error:\nThe translation is accurate.",  # the first non-empty line counts
    "No error.\nThe translation is accurate.",  # "no error" is a no-error line too
    'No-error\naccuracy/mistranslation - "x"',  # a line that looks like an error, with no heading above it
    'No-error:\nThe translation is not a non-translation.',  # no heading: GEMBA drops even this line
    "No-error:\nMajor: none found",  # not a heading: a heading is a line of its own
])
def test_an_answer_that_starts_with_no_error_and_has_no_heading_has_no_error(answer):
    assert mqm.parse_answer(answer) == ()
    assert mqm.mqm_score(e.severity for e in mqm.parse_answer(answer)) == 0


def test_a_no_error_first_line_does_not_excuse_an_answer_with_a_heading():
    with pytest.raises(mqm.Unreadable, match="line 2 comes before any severity heading"):
        mqm.parse_answer('No-error:\nThe translation is fine.\nMajor:\naccuracy/mistranslation - "x"\n')
    with pytest.raises(mqm.Unreadable, match="line 2 comes before any severity heading"):
        mqm.parse_answer("No-error:\nThe translation is fine.\nminor:\n")  # an empty section is still a heading

    errors = mqm.parse_answer('No-error\nMajor:\naccuracy/mistranslation - "x"\n')  # read under its heading, as before

    assert [(e.severity, e.category, e.span) for e in errors] == [("major", "accuracy-mistranslation", "x")]


@pytest.mark.parametrize("answer", [
    "The translation is accurate.\nNo-error",  # the no-error line is not the first
    "The translation is accurate.\n\nNo-error:\nNothing to fix.",
    "Looks good.\nNothing to fix.",  # no no-error line at all
    "", "\n\n", "   ",  # empty
])
def test_a_no_error_line_that_is_not_first_or_none_at_all_is_still_unreadable(answer):
    with pytest.raises(mqm.Unreadable):
        mqm.parse_answer(answer)


def test_an_unknown_severity_cannot_be_scored():
    with pytest.raises(ValueError):
        mqm.mqm_score(["serious"])


def test_the_breakdown_counts_errors_by_category_and_severity_most_frequent_first():
    segments = [mqm.parse_answer(MIXED), (), mqm.parse_answer(mqm.FEW_SHOTS[0]["answer"])]

    b = mqm.breakdown(segments)

    assert b["errors"] == 6 and b["segments_with_errors"] == 2
    assert b["by_severity"] == {"critical": 0, "major": 3, "minor": 3}
    assert list(b["by_category"].items())[0] == ("accuracy-mistranslation", 2)
    assert b["by_category_and_severity"]["fluency-register"] == {"critical": 0, "major": 0, "minor": 1}


# --- the scorer ------------------------------------------------------------------------------------------------------

API_KEY = "sk-judge-test-secret"
JUDGE_ENV = {"JUDGE_BASE_URL": "http://judge.test/v1", "JUDGE_API_KEY": API_KEY, "JUDGE_MODEL": "openai/judge-model",
             "JUDGE_TEMPERATURE": "0", "JUDGE_PRICE_INPUT_PER_M": "0.15", "JUDGE_PRICE_OUTPUT_PER_M": "0.60"}
OUT = "mqm-judge-model.json"
BOOK = "book_a"


def line(page, idx, ja, hyp, run_id=7):
    return {"book": BOOK, "page_index": page, "idx": idx, "run_id": run_id, "ja": ja, "hyp": hyp,
            "ref_zh_hans": f"REF-HANS-{page}-{idx}", "ref_zh_hant": f"REF-HANT-{page}-{idx}"}


LINES = [line(1, 0, "おはよう", "早上好"), line(1, 1, "", "……"), line(2, 0, "行くぞ 今だ", "走吧 就是现在"),
         line(2, 1, "ありがとう", "谢谢")]


def make_run(tmp_path, lines=LINES, *, manifest=None, name="run-1"):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({
        "eval_id": "20261028T000000Z-full", "book": BOOK, "config_name": "full",
        "finished_at": "2026-10-28T00:01:00+00:00", "requested_models": ["translation-model"],
        "response_models": ["translation-model-1"], **(manifest or {})}), encoding="utf-8")
    (run_dir / "segments.jsonl").write_text(
        "".join(json.dumps(l, ensure_ascii=False) + "\n" for l in lines), encoding="utf-8")
    return run_dir


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """eval/mqm.py with the judge's variables set, .env never read, and both ledgers in tmp_path."""
    for name in list(os.environ):
        if name.startswith(("LLM_", "JUDGE")):
            monkeypatch.delenv(name)
    for name, value in JUDGE_ENV.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LLM_MODEL", "translation-model")
    monkeypatch.setattr(config, "LEDGER_PATH", tmp_path / "ledgers" / "usage.jsonl")
    monkeypatch.setattr(config, "JUDGE_LEDGER_PATH", tmp_path / "ledgers" / "judge_usage.jsonl")
    spec = importlib.util.spec_from_file_location("eval_mqm_under_test", config.ROOT / "eval" / "mqm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "load_dotenv", lambda path: None)
    return module


def reply(text=NO_ERROR, *, finish="stop", usage=(1000, 30), model="judge-model-2024"):
    body = {"id": "c", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}]}
    if usage is not None:
        body["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1], "total_tokens": sum(usage)}
    return httpx.Response(200, json=body)


class Judge:
    """A scripted provider: answers in order (an httpx.Response, or a str for a normal reply) and keeps the bodies."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        answer = self.answers.pop(0) if self.answers else NO_ERROR
        return reply(answer) if isinstance(answer, str) else answer

    def client(self):
        return httpx.Client(transport=httpx.MockTransport(self))


def never_called(request):
    raise AssertionError("the judge was called")


def run(cli, run_dir, judge, *args):
    return cli.main([str(run_dir), *args], http_client=httpx.Client(transport=httpx.MockTransport(judge)))


def written(run_dir, name=OUT):
    return json.loads((run_dir / name).read_text(encoding="utf-8"))


def ledger(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l] if path.exists() else []


def test_the_scorer_writes_every_field(cli, tmp_path, capsys):
    run_dir = make_run(tmp_path)
    judge = Judge(MIXED, NO_ERROR, NO_ERROR)

    assert run(cli, run_dir, judge) == 0

    assert sorted(p.name for p in run_dir.iterdir()) == ["manifest.json", OUT, "segments.jsonl"]  # no .tmp left
    doc = written(run_dir)
    assert list(doc) == ["eval_id", "book", "segments_sha256", "judge", "gemba", "weights", "score_rule",
                         "n_segments", "scored", "skipped", "failed", "system_score", "breakdown", "segments", "usage",
                         "scored_at", "elapsed_s"]
    assert doc["eval_id"] == "20261028T000000Z-full"
    assert doc["segments_sha256"] == hashlib.sha256((run_dir / "segments.jsonl").read_bytes()).hexdigest()
    assert doc["judge"] == {"prefix": "JUDGE", "model": "openai/judge-model", "response_model": "judge-model-2024",
                            "temperature": 0.0, "reasoning_effort": None, "max_tokens": 500}
    assert doc["gemba"]["commit"] == mqm.GEMBA["commit"] and doc["gemba"]["licence"] == "CC BY-SA 4.0"
    assert doc["gemba"]["template_sha256"] == GEMBA_TEMPLATE_SHA256
    assert doc["gemba"]["repository"] == "https://github.com/MicrosoftTranslator/GEMBA"
    assert doc["weights"] == {"critical": 25, "major": 5, "minor": 1}
    assert (doc["n_segments"], doc["scored"]) == (4, 3)
    assert doc["skipped"] == {"count": 1, "segments": [{"book": BOOK, "page_index": 1, "idx": 1}]}
    assert doc["failed"] == {"count": 0, "segments": []}
    assert doc["system_score"] == pytest.approx(-2.0)  # (-6 + 0 + 0) / 3
    assert doc["breakdown"]["segments"] == {"scored": 3, "failed": 0}
    assert doc["breakdown"]["by_severity"] == {"critical": 0, "major": 1, "minor": 1}
    first, skipped = doc["segments"][0], doc["segments"][1]
    assert first == {"book": BOOK, "page_index": 1, "idx": 0, "run_id": 7, "status": "scored",
                     "errors": [e.to_dict() for e in mqm.parse_answer(MIXED)], "score": -6, "answer": MIXED,
                     "attempts": 1, "problems": []}
    assert skipped["status"] == "skipped" and skipped["score"] is None and skipped["attempts"] == 0
    assert doc["usage"] == {"calls": 3, "unknown_usage_calls": 0, "input_tokens": 3000, "output_tokens": 90,
                            "cost_usd": round(3 * (1000 * 0.15 + 30 * 0.60) / 1e6, 6)}
    assert doc["scored_at"].endswith("+00:00") and doc["elapsed_s"] >= 0
    out = capsys.readouterr().out
    assert "segments: 4 (scored 3, skipped 1, failed 0)" in out and "system score: -2.000" in out
    assert "accuracy-mistranslation 1" in out and "calls: 3" in out and f"Wrote {run_dir / OUT}" in out


def test_one_call_per_segment_with_text_in_file_order_with_gembas_prompt_and_no_reference_or_image(cli, tmp_path):
    run_dir = make_run(tmp_path)
    judge = Judge()

    assert run(cli, run_dir, judge) == 0

    texts = [l for l in LINES if l["ja"]]
    assert [b["messages"] for b in judge.bodies] == [mqm.build_messages(l["ja"], l["hyp"]) for l in texts]
    for body in judge.bodies:
        assert body["model"] == "openai/judge-model" and body["temperature"] == 0 and body["max_tokens"] == 500
        assert set(body) == {"model", "messages", "max_tokens", "temperature"}  # no tools, no response_format
        raw = json.dumps(body, ensure_ascii=False)
        assert "REF-" not in raw and "image_url" not in raw and "data:image" not in raw


def test_the_judges_calls_go_to_its_own_ledger_and_never_to_the_translation_ledger(cli, tmp_path):
    run_dir = make_run(tmp_path)

    assert run(cli, run_dir, Judge()) == 0

    calls = ledger(config.JUDGE_LEDGER_PATH)
    assert len(calls) == 3 and {c["requested_model"] for c in calls} == {"openai/judge-model"}
    assert [c["page_id"] for c in calls] == ["mqm:20261028T000000Z-full:book_a:1:0",
                                             "mqm:20261028T000000Z-full:book_a:2:0",
                                             "mqm:20261028T000000Z-full:book_a:2:1"]
    assert not config.LEDGER_PATH.exists()


def test_the_judge_prefix_chooses_the_configuration_and_the_output_records_it(cli, tmp_path, monkeypatch):
    for name, value in {"JUDGE2_BASE_URL": "http://judge2.test/v1", "JUDGE2_API_KEY": "sk-2",
                        "JUDGE2_MODEL": "fallback-model", "JUDGE2_TOKEN_PARAM": "max_completion_tokens",
                        "JUDGE2_REASONING_EFFORT": "none"}.items():
        monkeypatch.setenv(name, value)
    run_dir = make_run(tmp_path)
    judge = Judge()

    assert run(cli, run_dir, judge, "--judge-prefix", "JUDGE2") == 0

    doc = written(run_dir, "mqm-fallback-model.json")
    assert doc["judge"]["prefix"] == "JUDGE2" and doc["judge"]["model"] == "fallback-model"
    assert (doc["judge"]["reasoning_effort"], doc["judge"]["temperature"]) == ("none", None)
    assert doc["usage"]["cost_usd"] is None  # JUDGE2_PRICE_* unset: unknown, not 0
    body = judge.bodies[0]
    assert body["model"] == "fallback-model" and body["reasoning_effort"] == "none"
    assert body["max_completion_tokens"] == 500 and "temperature" not in body
    assert not (run_dir / OUT).exists()


def test_an_unreadable_answer_gets_exactly_one_retry_and_then_the_segment_is_failed(cli, tmp_path):
    run_dir = make_run(tmp_path, [line(1, 0, "おはよう", "早上好"), line(1, 1, "ありがとう", "谢谢")])
    judge = Judge("Looks fine to me.", "Still fine.", MIXED)

    assert run(cli, run_dir, judge) == 0

    assert len(judge.bodies) == 3 and judge.bodies[0] == judge.bodies[1]  # the retry is the same request
    doc = written(run_dir)
    failed, scored = doc["segments"]
    assert failed["status"] == "failed" and failed["score"] is None and failed["errors"] == []
    assert failed["attempts"] == 2 and failed["answer"] == "Still fine." and len(failed["problems"]) == 2
    assert doc["failed"] == {"count": 1, "segments": [{"book": BOOK, "page_index": 1, "idx": 0}]}
    assert doc["scored"] == 1 and doc["system_score"] == -6  # the failed segment is not a 0 in the mean
    assert doc["breakdown"]["segments"] == {"scored": 1, "failed": 1}
    assert len(ledger(config.JUDGE_LEDGER_PATH)) == 3


def test_a_retry_that_reads_scores_the_segment(cli, tmp_path):
    run_dir = make_run(tmp_path, [line(1, 0, "おはよう", "早上好")])

    assert run(cli, run_dir, Judge(reply(MIXED, finish="length"), MIXED)) == 0  # a cut-off answer is not read

    (segment,) = written(run_dir)["segments"]
    assert (segment["status"], segment["attempts"], segment["score"]) == ("scored", 2, -6)
    assert "cut off" in segment["problems"][0]


def test_a_no_error_answer_without_headings_scores_zero_with_no_retry(cli, tmp_path):
    run_dir = make_run(tmp_path, [line(1, 0, "おはよう", "早上好")])
    judge = Judge(NO_ERROR_EXPLAINED)

    assert run(cli, run_dir, judge) == 0

    (segment,) = written(run_dir)["segments"]
    assert (segment["status"], segment["score"], segment["errors"], segment["attempts"]) == ("scored", 0, [], 1)
    assert len(judge.bodies) == 1


def test_no_segment_scored_gives_no_system_score(cli, tmp_path):
    run_dir = make_run(tmp_path, [line(1, 0, "おはよう", "早上好")])

    assert run(cli, run_dir, Judge("?", "?")) == 0

    doc = written(run_dir)
    assert doc["system_score"] is None and doc["scored"] == 0 and doc["failed"]["count"] == 1


@pytest.mark.parametrize("k", [1, 2, 3])
def test_a_provider_error_on_the_kth_segment_stops_with_exit_1_and_writes_nothing(cli, tmp_path, capsys, k):
    run_dir = make_run(tmp_path)
    answers = [NO_ERROR] * (k - 1) + [httpx.Response(500, json={"error": {"message": "boom"}})]

    assert run(cli, run_dir, Judge(*answers)) == 1

    assert not list(run_dir.glob("mqm-*"))
    calls = ledger(config.JUDGE_LEDGER_PATH)
    assert len(calls) == k and calls[-1]["error"] and all(c["error"] is None for c in calls[:-1])
    err = capsys.readouterr().err
    assert "nothing written" in err and API_KEY not in err
    assert not config.LEDGER_PATH.exists()


def test_a_spent_segment_budget_stops_with_exit_1_and_writes_nothing(cli, tmp_path):
    run_dir = make_run(tmp_path)
    huge = reply("not an answer", usage=(9000, 10))  # unreadable, and over the 8,000 tokens: no retry is sent

    assert run(cli, run_dir, Judge(huge)) == 1

    assert not list(run_dir.glob("mqm-*")) and len(ledger(config.JUDGE_LEDGER_PATH)) == 1


def test_unknown_usage_stays_unknown(cli, tmp_path):
    run_dir = make_run(tmp_path)

    assert run(cli, run_dir, Judge(reply(NO_ERROR, usage=None), NO_ERROR, NO_ERROR)) == 0

    usage = written(run_dir)["usage"]
    assert usage["cost_usd"] is None and usage["unknown_usage_calls"] == 1 and usage["calls"] == 3
    assert "cost unknown" in mqm_scorer.format_summary(written(run_dir))


def test_the_output_is_replaced_through_a_temporary_file(cli, tmp_path, monkeypatch):
    run_dir = make_run(tmp_path)
    (run_dir / OUT).write_text("old", encoding="utf-8")
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: replaced.append((src, dst)) or real_replace(src, dst))

    assert run(cli, run_dir, Judge()) == 0

    ((src, dst),) = replaced
    assert os.path.basename(src) == OUT + ".tmp" and os.path.basename(dst) == OUT
    assert written(run_dir)["scored"] == 3 and not (run_dir / (OUT + ".tmp")).exists()


def test_a_failing_write_keeps_the_earlier_file_intact(cli, tmp_path, monkeypatch, capsys):
    run_dir = make_run(tmp_path)
    (run_dir / OUT).write_text("earlier", encoding="utf-8")

    def broken(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)

    assert run(cli, run_dir, Judge()) == 1

    assert (run_dir / OUT).read_text(encoding="utf-8") == "earlier"
    assert not (run_dir / (OUT + ".tmp")).exists()
    assert "disk full" in capsys.readouterr().err


def refused_cases(tmp_path, monkeypatch):
    """(what is wrong, run dir, a setup step, text expected in the message), each in a directory of its own."""
    cases = [("a directory that is not there", tmp_path / "nowhere", None, "not a directory")]
    cases.append(("an unfinished eval", make_run(tmp_path, manifest={"finished_at": None}, name="unfinished"), None,
                  "not finished"))
    cases.append(("an empty segments.jsonl", make_run(tmp_path, [], name="empty"), None, "no segments"))
    no_hyp = {k: v for k, v in LINES[0].items() if k != "hyp"}
    cases.append(("a line without hyp", make_run(tmp_path, [no_hyp], name="no-hyp"), None, "`hyp`"))
    no_ja = {k: v for k, v in LINES[0].items() if k != "ja"}
    cases.append(("a line without ja", make_run(tmp_path, [no_ja], name="no-ja"), None, "`ja`"))
    cases.append(("the judge is not configured", make_run(tmp_path, name="unconfigured"),
                  lambda: monkeypatch.delenv("JUDGE_MODEL"), "JUDGE_MODEL"))
    cases.append(("the judge's model is LLM_MODEL", make_run(tmp_path, name="same-model"),
                  lambda: monkeypatch.setenv("LLM_MODEL", "judge-model"), "translation model (LLM_MODEL)"))
    cases.append(("the judge's model translated this run",
                  make_run(tmp_path, manifest={"requested_models": ["judge-model"]}, name="ran-with"), None,
                  "translated with"))
    cases.append(("a price that is not a number", make_run(tmp_path, name="bad-price"),
                  lambda: monkeypatch.setenv("JUDGE_PRICE_INPUT_PER_M", "cheap"), "is not a number"))
    cases.append(("the translation system's own prefix", make_run(tmp_path, name="llm-prefix"), None, "LLM_*"))
    return cases


def test_the_scorer_refuses_with_exit_2_sends_nothing_and_writes_nothing(cli, tmp_path, monkeypatch, capsys):
    for what, run_dir, setup, message in refused_cases(tmp_path, monkeypatch):
        with monkeypatch.context() as m:
            for name, value in JUDGE_ENV.items():
                m.setenv(name, value)
            m.setenv("LLM_MODEL", "translation-model")
            if setup:
                setup()
            before = sorted(p.name for p in run_dir.iterdir()) if run_dir.exists() else None
            args = ["--judge-prefix", "LLM"] if what == "the translation system's own prefix" else []

            assert run(cli, run_dir, never_called, *args) == 2, what

            err = capsys.readouterr().err
            assert message in err, (what, err)
            for value in (API_KEY, "openai/judge-model", "http://judge.test/v1", "cheap"):
                assert value not in err, (what, value)
            assert (sorted(p.name for p in run_dir.iterdir()) if run_dir.exists() else None) == before, what
            assert not config.JUDGE_LEDGER_PATH.exists() and not config.LEDGER_PATH.exists(), what


def test_a_repeated_segment_is_refused(cli, tmp_path, capsys):
    run_dir = make_run(tmp_path, LINES + [LINES[0]])

    assert run(cli, run_dir, never_called) == 2
    assert "repeats page_index 1 idx 0" in capsys.readouterr().err


def test_the_scorer_never_loads_env_before_the_run_is_known_to_be_scorable(cli, tmp_path, monkeypatch):
    def refuse(path):
        raise AssertionError(".env was read")

    monkeypatch.setattr(cli, "load_dotenv", refuse)

    assert run(cli, tmp_path / "nowhere", never_called) == 2


def test_the_script_runs_as_a_program_and_refuses_a_missing_directory(tmp_path):
    done = subprocess.run([sys.executable, str(config.ROOT / "eval" / "mqm.py"), str(tmp_path / "nowhere")],
                          capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, timeout=60)

    assert done.returncode == 2 and "Cannot score" in done.stderr and "not a directory" in done.stderr


# --- the re-parse (eval/mqm.py --reparse) ------------------------------------------------------------------------------

real_parse_answer = mqm.parse_answer


def parse_as_023(answer):
    """The parser before the no-error rule: in an answer without a severity heading, every line that is neither empty
    nor a no-error line is unreadable. With a heading, the two parsers are the same."""
    lowered = [l.strip().lower() for l in answer.split("\n")]
    if not answer.strip() or any(l in mqm._HEADINGS for l in lowered):
        return real_parse_answer(answer)
    for number, l in enumerate(lowered, start=1):
        if l and not mqm._says_no_error(l):
            raise mqm.Unreadable(f"line {number} comes before any severity heading: {l!r}")
    return ()


REPARSE_LINES = [line(1, 0, "おはよう", "早上好"), line(1, 1, "", "……"), line(2, 0, "行くぞ 今だ", "走吧 就是现在"),
                 line(2, 1, "ありがとう", "谢谢"), line(3, 0, "はい", "是"), line(3, 1, "いいえ", "不"),
                 line(4, 0, "さあ", "来吧")]
# (1,0) failed under 023 with GPT-5.1's no-error shape; (1,1) skipped; (2,0) scored; (2,1) unreadable both times;
# (3,0) cut off both times; (3,1) unreadable, then cut off; (4,0) cut off, then the no-error shape


def reparse_answers():
    cut = lambda: reply(NO_ERROR_EXPLAINED, finish="length")  # noqa: E731
    return [NO_ERROR_EXPLAINED, NO_ERROR_EXPLAINED, MIXED, "Looks fine to me.", "Still fine.", cut(), cut(),
            "Looks fine.", cut(), cut(), NO_ERROR_EXPLAINED]


NOW_SCORED = [(1, 0), (4, 0)]


@pytest.fixture
def scored_by_023(cli, tmp_path, monkeypatch):
    """A run scored when the parser was 023's: (run directory, its MQM file, the file's bytes)."""
    run_dir = make_run(tmp_path, REPARSE_LINES)
    with monkeypatch.context() as m:
        m.setattr(mqm, "parse_answer", parse_as_023)
        assert run(cli, run_dir, Judge(*reparse_answers())) == 0
    doc = written(run_dir)
    assert [(s["page_index"], s["idx"]) for s in doc["segments"] if s["status"] == "failed"] == [
        (1, 0), (2, 1), (3, 0), (3, 1), (4, 0)]
    return run_dir, run_dir / OUT, (run_dir / OUT).read_bytes()


@pytest.fixture
def offline(cli, scored_by_023, monkeypatch):
    """eval/mqm.py once the run is scored: from now on nothing may read .env or make a client."""
    def refuse(*args, **kwargs):
        raise AssertionError("the re-parse reached for the judge")

    monkeypatch.setattr(cli, "load_dotenv", refuse)
    monkeypatch.setattr(mqm_scorer, "make_client", refuse)
    monkeypatch.setattr(mqm_scorer, "LLMClient", refuse)
    return cli


def test_the_reparse_scores_only_failed_segments_the_parser_could_not_read(offline, scored_by_023, tmp_path, monkeypatch,
                                                                           capsys):
    run_dir, path, old_bytes = scored_by_023
    old = json.loads(old_bytes)
    calls_before = len(ledger(config.JUDGE_LEDGER_PATH))
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: replaced.append((src, dst)) or real_replace(src, dst))

    assert offline.main(["--reparse", str(path)]) == 0

    new = written(run_dir)
    for before, after in zip(old["segments"], new["segments"], strict=True):
        if (before["page_index"], before["idx"]) in NOW_SCORED:
            assert after == {**before, "status": "scored", "errors": [], "score": 0}  # attempts, answer, problems kept
        else:
            assert after == before  # scored, skipped, cut off, and unreadable by the current parser too
    assert (new["scored"], new["failed"]["count"]) == (old["scored"] + 2, old["failed"]["count"] - 2)
    assert [(s["page_index"], s["idx"]) for s in new["failed"]["segments"]] == [(2, 1), (3, 0), (3, 1)]
    assert new["system_score"] == pytest.approx(-2.0)  # (0 - 6 + 0) / 3; it was -6 over one scored segment
    assert new["breakdown"] == {"segments": {"scored": 3, "failed": 3}, **mqm.breakdown(
        [(), mqm.parse_answer(MIXED), ()])}
    assert list(new) == [*old, "reparsed"]  # every other field, the spend and scored_at among them, is as it was
    assert {k: v for k, v in new.items() if k not in ("scored", "failed", "system_score", "breakdown", "segments",
                                                      "reparsed")} == \
           {k: v for k, v in old.items() if k not in ("scored", "failed", "system_score", "breakdown", "segments")}
    (record,) = new["reparsed"]
    assert record["at"].endswith("+00:00") and set(record) == {"at", "commit", "dirty", "parser", "segments",
                                                               "still_failed", "not_reread"}
    assert record["commit"] is None or len(record["commit"]) == 40
    assert [(s["page_index"], s["idx"], s["score"]) for s in record["segments"]] == [(1, 0, 0), (4, 0, 0)]
    assert [(s["page_index"], s["idx"]) for s in record["still_failed"]] == [(2, 1)]
    assert [(s["page_index"], s["idx"]) for s in record["not_reread"]] == [(3, 0), (3, 1)]
    ((src, dst),) = replaced
    assert os.path.basename(src) == OUT + ".tmp" and os.path.basename(dst) == OUT
    assert len(ledger(config.JUDGE_LEDGER_PATH)) == calls_before and not config.LEDGER_PATH.exists()
    out = capsys.readouterr().out
    assert "book_a/1 idx 0: failed -> scored, score 0" in out and "book_a/2 idx 1: still failed" in out


def test_the_reparsed_totals_are_those_of_scoring_with_the_current_parser(scored_by_023, cli, tmp_path):
    run_dir, path, _ = scored_by_023
    fresh_dir = make_run(tmp_path, REPARSE_LINES, name="fresh")
    assert run(cli, fresh_dir, Judge(*reparse_answers()[1:])) == 0  # (1,0) is read on its first attempt now

    assert cli.main(["--reparse", str(path)]) == 0

    reparsed, fresh = written(run_dir), written(fresh_dir)
    for key in ("scored", "failed", "system_score", "breakdown"):
        assert reparsed[key] == fresh[key], key


def test_a_dry_run_says_what_would_change_and_writes_nothing(offline, scored_by_023, capsys):
    run_dir, path, old_bytes = scored_by_023

    assert offline.main(["--reparse", str(path), "--dry-run"]) == 0

    assert path.read_bytes() == old_bytes and sorted(p.name for p in run_dir.iterdir()) == [
        "manifest.json", OUT, "segments.jsonl"]
    out = capsys.readouterr().out
    assert "2 failed segment(s) read again (dry run, nothing written)" in out
    assert "book_a/4 idx 0: failed -> scored, score 0 (0 error(s))" in out
    assert "scored 1 -> 3, failed 5 -> 3, system score -6.000 -> -2.000" in out


def test_a_file_with_nothing_to_reparse_is_left_as_it_is(offline, scored_by_023, capsys):
    run_dir, path, _ = scored_by_023
    assert offline.main(["--reparse", str(path)]) == 0
    once = path.read_bytes()

    assert offline.main(["--reparse", str(path)]) == 0

    assert path.read_bytes() == once and "Nothing to change" in capsys.readouterr().out


def test_the_reparse_refuses_a_file_that_is_not_its_runs(offline, scored_by_023, capsys):
    run_dir, path, old_bytes = scored_by_023

    def tamper(change):
        def apply():
            doc = json.loads(old_bytes)
            change(doc)
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        return apply

    def other_segments():
        (run_dir / "segments.jsonl").write_text(
            "".join(json.dumps(l, ensure_ascii=False) + "\n" for l in REPARSE_LINES[:-1]), encoding="utf-8")

    cases = [
        ("another segments.jsonl", other_segments, "segments_sha256"),
        ("another eval id", tamper(lambda d: d.update(eval_id="other")), "eval run 'other'"),
        ("totals that are not its segments'", tamper(lambda d: d.update(system_score=-1.0)), "not its segments' totals"),
        ("another template", tamper(lambda d: d["gemba"].update(template_sha256="0" * 64)), "prompt template"),
        ("other weights", tamper(lambda d: d.update(weights={"critical": 10, "major": 5, "minor": 1})), "weights"),
        ("another judge's name", tamper(lambda d: d["judge"].update(model="someone-else")), "judge model"),
        ("not JSON", lambda: path.write_text("{", encoding="utf-8"), "not valid JSON"),
    ]
    segments_bytes = (run_dir / "segments.jsonl").read_bytes()
    for what, setup, message in cases:
        path.write_bytes(old_bytes)
        (run_dir / "segments.jsonl").write_bytes(segments_bytes)
        setup()
        before = path.read_bytes()

        assert offline.main(["--reparse", str(path)]) == 2, what

        assert message in capsys.readouterr().err, what
        assert path.read_bytes() == before and not (run_dir / (OUT + ".tmp")).exists(), what


@pytest.mark.parametrize("change, rereadable", [
    ({}, True),
    ({"problems": ["attempt 1: the answer is empty", "attempt 2: the answer is empty"]}, True),
    ({"problems": ["attempt 1: line 2 comes before any severity heading: 'x'"]}, False),  # not the last attempt's
    ({"problems": ["attempt 1: x", "attempt 2: the answer was cut off (finish_reason 'length')"]}, False),
    ({"problems": []}, False), ({"answer": None}, False), ({"status": "scored"}, False), ({"status": "skipped"}, False),
])
def test_only_a_failed_segment_whose_last_answer_the_parser_could_not_read_is_read_again(cli, change, rereadable):
    segment = {"book": BOOK, "page_index": 1, "idx": 0, "run_id": 7, "status": "failed", "errors": [], "score": None,
               "answer": NO_ERROR_EXPLAINED, "attempts": 2,
               "problems": ["attempt 1: line 2 comes before any severity heading: 'x'",
                            "attempt 2: line 2 comes before any severity heading: 'y'"]}

    assert cli.rereadable({**segment, **change}) is rereadable


@pytest.mark.parametrize("args", [[], ["run-dir", "--reparse", "mqm-x.json"], ["run-dir", "--dry-run"],
                                  ["--reparse", "mqm-x.json", "--judge-prefix", "JUDGE2"]])
def test_the_reparse_options_go_together_only_one_way(offline, args):
    with pytest.raises(SystemExit) as stopped:
        offline.main(args)
    assert stopped.value.code == 2


# --- boundaries ------------------------------------------------------------------------------------------------------

MQM_CODE = [config.ROOT / "backend" / "fukidashi" / "evaluation" / name
            for name in ("mqm.py", "mqm_scorer.py", "mqm_sample.py")] + \
           [config.ROOT / "eval" / name for name in ("mqm.py", "mqm_sample.py")]


def imported_names(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(part for alias in node.names for part in alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            names.update((node.module or "").split("."))
            names.update(alias.name for alias in node.names)
    return names


def test_the_mqm_code_reaches_the_judge_only_through_llmclient_and_never_the_database_or_a_reference():
    for path in MQM_CODE:
        names = imported_names(path)
        assert not names & {"openai", "langchain", "langchain_core", "langgraph", "sqlite3", "store", "db", "runs"}, path
        assert "ref_zh" not in path.read_text(encoding="utf-8"), path


def test_the_gemba_library_and_the_sample_make_no_model_call():
    for name in ("mqm.py", "mqm_sample.py"):
        path = config.ROOT / "backend" / "fukidashi" / "evaluation" / name
        assert not imported_names(path) & {"llm", "httpx", "requests", "socket"}, path
