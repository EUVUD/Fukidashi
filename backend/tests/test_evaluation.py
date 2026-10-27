"""The eval harness, driven with FakeClient: no network, no real model."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field

import pytest
from conftest import FakeClient
from sacrebleu.metrics import BLEU, CHRF

from fukidashi import config
from fukidashi.datasets.importer import import_openmantra
from fukidashi.evaluation import harness, metrics
from fukidashi.evaluation.manifest import git_state
from fukidashi.evaluation.references import to_zh_hans
from fukidashi.llm.client import LLMNotConfigured, LLMResult
from fukidashi.llm.metering import CallRecord, Usage

# Page 1 and page 3 have text; page 2 has none. The references are Traditional, as in OpenMantra.
PAGES = {
    1: [("こんにちは", "這個測試"), ("さようなら", "再見了")],
    2: [],
    3: [("ありがとう", "謝謝")],
}
HYP = {1: ["这是测试", "拜拜"], 3: ["谢谢你"]}


def reply(page_index):
    return json.dumps({"translations": [{"id": i, "zh": zh} for i, zh in enumerate(HYP[page_index])]},
                      ensure_ascii=False)


@pytest.fixture
def root(tmp_path):
    root = tmp_path / "open-mantra-dataset"
    (root / "images" / "book_e" / "ja").mkdir(parents=True)
    pages = []
    for page_index, segments in PAGES.items():
        image = f"images/book_e/ja/{page_index - 1:03d}.jpg"
        (root / image).write_bytes(b"\xff\xd8fake-jpeg")
        text = [{"x": 1, "y": 2, "w": 3, "h": 4, "text_ja": ja, "text_en": "EN", "text_zh": zh} for ja, zh in segments]
        pages.append({"page_index": page_index, "image_paths": {"ja": image}, "frame": [], "text": text})
    (root / "annotation.json").write_text(json.dumps([{"book_title": "book_e", "pages": pages}]), encoding="utf-8")
    return root


@pytest.fixture
def imported(conn, root):
    import_openmantra(conn, root)
    return conn


@dataclass
class UsageClient(FakeClient):
    """Like FakeClient, but each call reports the usage listed in `usages` (None: the provider sent none)."""

    usages: list[Usage | None] = field(default_factory=list)

    def chat(self, messages, *, budget, max_tokens=4096, json_output=False):
        usage = self.usages.pop(0)
        rec = CallRecord(budget.page_id, "fake-model", "fake-model-1", usage, 0.01,
                         None if usage is None else 0.0001)
        self.meter.record(rec)
        budget.charge(usage)
        return LLMResult(text=self.replies.pop(0), finish_reason="stop", record=rec)


class ExplodingClient(FakeClient):
    def chat(self, messages, **kwargs):
        raise RuntimeError("provider exploded")


def evaluate(conn, root, tmp_path, clients, pages, log=None):
    queue = list(clients)

    def make_client():
        return queue.pop(0)

    return harness.run_eval(
        conn, config="baseline", book="book_e", page_indexes=pages, make_client=make_client,
        dataset_root=root, out_root=tmp_path / "out", repo_root=config.ROOT, **({"log": log} if log else {}),
    )


def sacrebleu_on(hyps, refs):
    """sacreBLEU's own result on the same lists: (chrF, chrF signature, BLEU, BLEU signature)."""
    chrf, bleu = CHRF(), BLEU(tokenize="zh")
    scores = chrf.corpus_score(hyps, [refs]).score, bleu.corpus_score(hyps, [refs]).score
    return scores[0], str(chrf.get_signature()), scores[1], str(bleu.get_signature())


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_outputs_manifest_segments_and_scores(imported, root, tmp_path):
    result = evaluate(imported, root, tmp_path, [FakeClient([reply(1)]), FakeClient([reply(3)])], [1, 3])

    assert sorted(p.name for p in result.out_dir.iterdir()) == ["manifest.json", "scores.json", "segments.jsonl"]
    assert result.exit_code == 0
    assert result.out_dir.parent == tmp_path / "out"

    manifest = json.loads((result.out_dir / "manifest.json").read_text(encoding="utf-8"))
    runs = imported.execute("SELECT id, config, status, model FROM runs ORDER BY id").fetchall()
    assert [(r["config"], r["status"]) for r in runs] == [("baseline", "succeeded")] * 2
    assert [(p["page_index"], p["status"], p["run_id"]) for p in manifest["pages"]] == [
        (1, "scored", runs[0]["id"]), (3, "scored", runs[1]["id"])]
    assert manifest["config_name"] == "baseline"
    assert manifest["config_params"] == {"page_budget": 40_000, "max_tokens": 16_000}
    assert manifest["requested_models"] == ["fake-model"]
    assert manifest["response_models"] == ["fake-model-1"]
    assert manifest["opencc_config"] == "t2s"
    _, chrf_signature, _, bleu_signature = sacrebleu_on(["这是测试"], ["这个测试"])
    assert manifest["sacrebleu_signatures"] == {"chrf": chrf_signature, "bleu": bleu_signature}
    assert "tok:zh" in manifest["sacrebleu_signatures"]["bleu"]
    assert len(manifest["annotation_sha256"]) == 64
    assert manifest["started_at"] <= manifest["finished_at"]
    assert len(manifest["git_commit"]) == 40 and isinstance(manifest["git_dirty"], bool)
    assert manifest["eval_id"] == result.out_dir.name


def test_segments_pair_the_translation_with_the_converted_reference(imported, root, tmp_path):
    result = evaluate(imported, root, tmp_path, [FakeClient([reply(1)]), FakeClient([reply(3)])], [1, 3])

    lines = read_lines(result.out_dir / "segments.jsonl")
    assert [(s["page_index"], s["idx"]) for s in lines] == [(1, 0), (1, 1), (3, 0)]
    assert lines[0]["ref_zh_hant"] == "這個測試"
    assert lines[0]["ref_zh_hans"] == "这个测试"
    assert lines[0]["hyp"] == "这是测试"  # the Run's translation, not the reference
    assert all(s["ref_zh_hans"] == to_zh_hans(s["ref_zh_hant"]) for s in lines)
    assert [s["ja"] for s in lines] == ["こんにちは", "さようなら", "ありがとう"]
    assert set(lines[0]) >= {"book", "page_index", "idx", "ja", "hyp", "ref_zh_hans", "ref_zh_hant"}


def test_scores_equal_sacrebleu_on_the_same_lists(imported, root, tmp_path):
    result = evaluate(imported, root, tmp_path, [FakeClient([reply(1)]), FakeClient([reply(3)])], [1, 3])

    hyps = ["这是测试", "拜拜", "谢谢你"]
    refs = ["这个测试", "再见了", "谢谢"]
    scores = json.loads((result.out_dir / "scores.json").read_text(encoding="utf-8"))
    chrf, chrf_signature, bleu, bleu_signature = sacrebleu_on(hyps, refs)
    assert (scores["chrf"]["score"], scores["chrf"]["signature"]) == (chrf, chrf_signature)
    assert (scores["bleu"]["score"], scores["bleu"]["signature"]) == (bleu, bleu_signature)
    assert (scores["pages_scored"], scores["segments_scored"]) == (2, 3)
    assert scores["cost"] == {"total_usd": 0.0002, "per_page_usd": 0.0001, "unknown_cost_runs": 0,
                              "unknown_usage_calls": 0}
    assert scores["latency"]["n"] == 2 and scores["latency"]["p50_s"] <= scores["latency"]["p95_s"]
    assert [p["cost_usd"] for p in scores["pages"]] == [0.0001, 0.0001]


def test_prompts_carry_no_reference(imported, root, tmp_path):
    client = FakeClient([reply(1)])
    evaluate(imported, root, tmp_path, [client], [1])

    sent = json.dumps(client.requests, ensure_ascii=False)
    assert "こんにちは" in sent
    assert "這個測試" not in sent and "这个测试" not in sent


def test_a_page_without_text_is_skipped_without_a_model_call(imported, root, tmp_path):
    clients = [FakeClient([reply(1)])]
    result = evaluate(imported, root, tmp_path, clients, [2, 1])

    assert result.exit_code == 0
    assert [(p["page_index"], p["status"], p["run_id"]) for p in result.manifest["pages"]][0] == (2, "skipped", None)
    assert [p["page_index"] for p in result.scores["skipped_pages"]] == [2]
    assert imported.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert clients[0].requests  # the one client built was used for page 1, none was built for page 2


def test_a_failed_run_is_listed_and_left_out_of_the_scores(imported, root, tmp_path):
    garbage = FakeClient(["not json", "still not json"])
    result = evaluate(imported, root, tmp_path, [garbage, FakeClient([reply(3)])], [1, 3])

    assert result.exit_code == 1
    assert [(p["book"], p["page_index"]) for p in result.scores["failed_pages"]] == [("book_e", 1)]
    failed = result.scores["failed_pages"][0]
    assert "Could not parse" in failed["error"]
    assert imported.execute("SELECT status FROM runs WHERE id = ?", (failed["run_id"],)).fetchone()[0] == "failed"
    assert failed["cost_usd"] == 0.0002  # the failed Run still cost money; it is reported, not scored
    assert [s["page_index"] for s in read_lines(result.out_dir / "segments.jsonl")] == [3]
    assert (result.scores["pages_scored"], result.scores["segments_scored"]) == (1, 1)
    assert result.scores["cost"]["total_usd"] == 0.0001  # scored Runs only
    assert [p["status"] for p in result.manifest["pages"]] == ["failed", "scored"]


def test_an_unexpected_error_fails_that_page_and_the_rest_still_run(imported, root, tmp_path):
    result = evaluate(imported, root, tmp_path, [ExplodingClient([]), FakeClient([reply(3)])], [1, 3])

    assert result.exit_code == 1
    assert result.scores["failed_pages"][0]["error"] == "RuntimeError: provider exploded"
    assert result.scores["failed_pages"][0]["run_id"] is None
    assert result.scores["pages_scored"] == 1
    assert imported.execute("SELECT status FROM runs ORDER BY id").fetchall()[0][0] == "failed"


def test_an_unknown_page_or_missing_image_stops_before_any_model_call(imported, root, tmp_path):
    (root / "images" / "book_e" / "ja" / "002.jpg").unlink()  # page 3's image

    def no_client():
        raise AssertionError("make_client must not be called")

    with pytest.raises(harness.EvalAborted) as stopped:
        harness.run_eval(imported, config="baseline", book="book_e", page_indexes=[1, 99, 3], make_client=no_client,
                         dataset_root=root, out_root=tmp_path / "out", repo_root=config.ROOT)

    assert len(stopped.value.problems) == 2  # both are reported, not just the first
    assert "book_e/99" in stopped.value.problems[0] and "book_e/3" in stopped.value.problems[1]
    assert imported.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert not (tmp_path / "out").exists()


def test_a_client_that_cannot_be_built_leaves_no_run_and_no_output(imported, root, tmp_path):
    def unconfigured():
        raise LLMNotConfigured("Set LLM_API_KEY in .env")

    with pytest.raises(LLMNotConfigured):
        harness.run_eval(imported, config="baseline", book="book_e", page_indexes=[1], make_client=unconfigured,
                         dataset_root=root, out_root=tmp_path / "out", repo_root=config.ROOT)

    assert imported.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0
    assert not (tmp_path / "out").exists()


def test_only_known_configs_and_distinct_pages_are_accepted(imported, root, tmp_path):
    for kwargs in ({"config": "full", "page_indexes": [1]}, {"config": "baseline", "page_indexes": [1, 1]}):
        with pytest.raises(ValueError):
            harness.run_eval(imported, book="book_e", make_client=lambda: None, dataset_root=root,
                             out_root=tmp_path / "out", repo_root=config.ROOT, **kwargs)
    assert imported.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_cost_is_null_when_any_scored_run_has_unknown_cost(imported, root, tmp_path):
    ok = UsageClient([reply(1)], usages=[Usage(100, 20)])
    unknown = UsageClient([reply(3)], usages=[None])
    result = evaluate(imported, root, tmp_path, [ok, unknown], [1, 3])

    cost = result.scores["cost"]
    assert cost["total_usd"] is None and cost["per_page_usd"] is None  # not the partial 0.0001
    assert (cost["unknown_cost_runs"], cost["unknown_usage_calls"]) == (1, 1)
    assert [p["cost_usd"] for p in result.scores["pages"]] == [0.0001, None]
    assert result.scores["pages_scored"] == 2


def test_a_run_with_one_unknown_call_is_not_trusted_even_if_the_meter_summed_the_rest(imported, root, tmp_path):
    # First call has no usage and unparseable output; the retry is priced. Meter.summary() reports 0.0001
    # for the Run (a partial sum); the harness must not add it up as if it were complete.
    client = UsageClient(["not json", reply(1)], usages=[None, Usage(100, 20)])
    result = evaluate(imported, root, tmp_path, [client], [1])

    assert imported.execute("SELECT cost_usd, unknown_usage_calls FROM runs").fetchone()[:] == (0.0001, 1)
    assert result.scores["cost"]["total_usd"] is None
    assert result.scores["pages"][0]["cost_usd"] is None
    assert result.scores["cost"]["unknown_usage_calls"] == 1


def test_a_second_eval_in_the_same_second_gets_its_own_directory(imported, root, tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "_now", lambda: "2026-10-26T12:00:00+00:00")
    first = evaluate(imported, root, tmp_path, [FakeClient([reply(1)])], [1])
    second = evaluate(imported, root, tmp_path, [FakeClient([reply(1)])], [1])

    assert first.out_dir.name == "20261026T120000Z-baseline"
    assert second.out_dir.name == "20261026T120000Z-baseline-2"
    assert (first.out_dir / "manifest.json").is_file()


def test_log_reports_progress(imported, root, tmp_path):
    lines: list[str] = []
    evaluate(imported, root, tmp_path, [FakeClient([reply(1)])], [1], log=lines.append)

    assert lines[0] == "translating book_e/1 ..."


def test_summary_prints_scores_signatures_and_failures(imported, root, tmp_path):
    result = evaluate(imported, root, tmp_path, [FakeClient(["x", "y"]), FakeClient([reply(3)])], [1, 3])

    text = harness.format_summary(result)
    assert "chrF" in text and "BLEU" in text and "tok:zh" in text and "nc:6" in text
    assert "FAILED  book_e/1" in text


def test_nothing_scored_gives_null_scores_not_a_crash(imported, root, tmp_path):
    result = evaluate(imported, root, tmp_path, [FakeClient(["x", "y"])], [1])

    assert result.scores["chrf"] == {"score": None, "signature": None}
    assert result.scores["bleu"] == {"score": None, "signature": None}
    assert result.manifest["sacrebleu_signatures"] == {"chrf": None, "bleu": None}
    assert result.scores["cost"]["total_usd"] is None
    assert "n/a" in harness.format_summary(result)


def test_t2s_converts_characters_and_keeps_the_wording():
    assert to_zh_hans("這個測試") == "这个测试"
    assert to_zh_hans("軟體") == "软体"  # tw2sp would rewrite this to 软件


def test_percentile_interpolates_like_numpy():
    assert metrics.percentile([], 50) is None
    assert metrics.percentile([4.0], 95) == 4.0
    assert metrics.percentile([1.0, 2.0, 3.0], 50) == 2.0
    assert metrics.percentile([3.0, 1.0, 2.0, 4.0], 50) == 2.5
    assert metrics.percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95) == pytest.approx(4.8)
    assert metrics.latency_summary([1.0, 2.0, 3.0]) == {"n": 3, "p50_s": 2.0, "p95_s": 2.9}
    assert metrics.latency_summary([]) == {"n": 0, "p50_s": None, "p95_s": None}


def test_score_corpus_refuses_mismatched_lists():
    with pytest.raises(ValueError):
        metrics.score_corpus(["a"], ["a", "b"])


def test_git_state_is_unknown_outside_a_repository(tmp_path):
    assert git_state(tmp_path) == {"commit": None, "dirty": None}


def run_cli(tmp_path, *args):
    env = {"PATH": "/usr/bin:/bin", "FUKIDASHI_DB": str(tmp_path / "cli.db")}
    return subprocess.run([sys.executable, str(config.ROOT / "eval" / "run.py"), *args],
                          capture_output=True, text=True, env=env, timeout=60)


def test_cli_lists_its_options_and_accepts_only_baseline(tmp_path):
    helped = run_cli(tmp_path, "--help")
    assert helped.returncode == 0
    assert all(option in helped.stdout for option in ("--config", "--book", "--pages", "baseline"))

    rejected = run_cli(tmp_path, "--config", "full", "--book", "book_e", "--pages", "1")
    assert rejected.returncode == 2 and "invalid choice" in rejected.stderr
    assert not (tmp_path / "cli.db").exists()  # it stopped before touching the database

    bad_pages = run_cli(tmp_path, "--config", "baseline", "--book", "book_e", "--pages", "1,x")
    assert bad_pages.returncode == 2 and "comma-separated" in bad_pages.stderr
