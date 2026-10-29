"""Term consistency of a finished eval run, with renderings extracted by the judge (OUTLINE 4.7, 8).

Usage:
  uv run python eval/consistency.py runs/eval/<id>                       # the judge: JUDGE_* in .env
  uv run python eval/consistency.py runs/eval/<id> --judge-prefix JUDGE2  # another judge configuration

This makes real model calls, to the judge only: for every occurrence of a gold glossary term that is not borderline
(a segment whose Japanese text contains the term's source or an alias), one call on the Translation and one on the
converted reference, each with one retry when the answer cannot be used. A call sees the Japanese text, the term's form
in it, whether it is a name or a term, and that one Chinese text: never the other text, another occurrence, the
glossary's target or the page image. The judge is the client GEMBA-MQM uses (evaluation/mqm_scorer.py): <prefix>_* in
.env, a Meter of its own and its own ledger, runs/judge_usage.jsonl, so its calls never reach runs/usage.jsonl. The
extraction is not GEMBA-MQM and holds no MQM annotation. The rules are in evaluation/consistency.py.

Writes consistency-<judge model>.json into the run's directory (replaced through a temporary file): every occurrence's
status and rendering in both texts, each entry's consistency, the pooled consistency of the Translations and of the
reference, the counts by status, the extractor's accuracy against the glossary and whether consistency is reportable,
the judge and its spend. Prints a summary.
Exit code: 0 when the file is written; 2 (with a message, nothing sent and nothing written) when the directory is
missing, the manifest is missing or unfinished, segments.jsonl is missing, empty or has a line without its fields, the
book has no glossary or the glossary has problems, the judge is not configured, or the judge's model is the translation
model; 1 when a provider error or the per-call token budget stops it (no file is written, and the calls already made
stay in the judge's ledger) or the file cannot be written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv

from fukidashi import config
from fukidashi.evaluation import consistency, glossary, mqm_scorer
from fukidashi.evaluation.manifest import file_sha256
from fukidashi.evaluation.mqm_scorer import Refused
from fukidashi.llm.errors import ProviderError
from fukidashi.llm.metering import BudgetExceeded, PageBudget

GLOSSARY_DIR = config.ROOT / "eval" / "gold_glossary"
CALL_BUDGET_TOKENS = 4_000  # per occurrence and text, the retry included
MAX_TOKENS = 200  # an answer is one short JSON object


class Run(NamedTuple):  # not a dataclass: tests load this script without registering it in sys.modules
    """A finished eval run, as this scorer reads it."""

    directory: Path
    manifest: dict  # mqm_scorer.judge_config reads the models the run translated with from here
    segments: tuple[consistency.Segment, ...]
    segments_sha256: str  # of the bytes the segments were read from

    @property
    def eval_id(self) -> str:
        return self.manifest["eval_id"]

    @property
    def book(self) -> str:
        return self.manifest["book"]


def main(argv: list[str] | None = None, *, http_client=None) -> int:
    """`http_client` is for tests: the judge's requests go through it (an httpx.MockTransport), not the network."""
    ap = argparse.ArgumentParser(description="Score a finished eval run's term consistency, with the judge.")
    ap.add_argument("run_dir", type=Path, help="an eval run's directory, e.g. runs/eval/<id>")
    ap.add_argument("--glossary-dir", type=Path, default=GLOSSARY_DIR, help="where the glossary files are")
    ap.add_argument("--judge-prefix", default="JUDGE", help="the judge's variables in .env (default JUDGE)")
    args = ap.parse_args(argv)
    try:
        run = read_run(args.run_dir)  # before .env: a run that cannot be scored needs no configuration
        loaded, glossary_path = read_glossary(args.glossary_dir, run.book)
        load_dotenv(config.ROOT / ".env")
        judge = mqm_scorer.judge_config(args.judge_prefix, run)
    except Refused as e:
        print(f"Cannot score {args.run_dir}: {e}", file=sys.stderr)
        return 2

    client = mqm_scorer.make_client(judge, config.JUDGE_LEDGER_PATH, http_client)
    try:
        document = score_run(run, loaded, glossary_path, client, prefix=args.judge_prefix)
    except (ProviderError, BudgetExceeded) as e:
        usage = client.meter.summary()
        print(f"Stopped, nothing written: {e}\n{usage['calls']} judge call(s) were made; they are in "
              f"{config.JUDGE_LEDGER_PATH}.", file=sys.stderr)
        return 1

    out = args.run_dir / output_name(judge.model)
    try:
        mqm_scorer.write_json(out, document)
    except OSError as e:
        print(f"Could not write {out}: {e}", file=sys.stderr)
        return 1
    print(format_summary(document))
    print(f"Wrote {out}")
    return 0


# --- reading ----------------------------------------------------------------------------------------------------------

def read_run(run_dir: Path) -> Run:
    """The finished eval run in `run_dir`; raises Refused, having sent and written nothing."""
    if not run_dir.is_dir():
        raise Refused("it is not a directory")
    path = run_dir / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise Refused(f"cannot read {path.name}: {e}")
    except ValueError as e:
        raise Refused(f"{path.name} is not valid JSON: {e}")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("eval_id"), str) \
            or not isinstance(manifest.get("book"), str):
        raise Refused(f"{path.name} has no `eval_id` and `book`")
    if manifest.get("finished_at") is None:
        raise Refused("the eval is not finished (manifest.json has `finished_at` null)")

    path = run_dir / "segments.jsonl"
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise Refused(f"cannot read {path.name}: {e}")
    lines = []
    # "\n" only: `splitlines()` also cuts at U+2028, U+2029 and U+0085, which the harness writes unescaped
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except ValueError as e:
            raise Refused(f"{path.name} line {number} is not valid JSON: {e}")
    if not lines:
        raise Refused(f"{path.name} has no segments")
    try:
        segments = consistency.read_segments(lines)
    except consistency.ConsistencyError as e:
        raise Refused(f"{path.name}: {e}")
    return Run(run_dir, manifest, tuple(segments), hashlib.sha256(data).hexdigest())


def read_glossary(glossary_dir: Path, book: str) -> tuple[glossary.Glossary, Path]:
    path = glossary_dir / f"{book}.json"
    if not path.is_file():
        raise Refused(f"there is no glossary for book {book!r} at {path}")
    loaded = glossary.load_glossary(path)
    if loaded.problems:
        raise Refused(f"{path} has problems:" + "".join(f"\n  {p}" for p in loaded.problems))
    if loaded.book_title != book:
        raise Refused(f"{path} is the glossary of {loaded.book_title!r}, the run is of {book!r}")
    return loaded, path


def output_name(model: str) -> str:
    """consistency-<the model without an organisation prefix>.json; `judge_config` has already refused a model name
    that cannot name a file."""
    return f"consistency-{mqm_scorer.bare_model(model)}.json"


# --- the calls --------------------------------------------------------------------------------------------------------

def extract(client, occurrence: consistency.Occurrence, which: str, *, eval_id: str,
            book: str) -> consistency.Extraction:
    """One occurrence in one text: up to `consistency.ATTEMPTS` calls under one budget. ProviderError and BudgetExceeded
    leave unchanged."""
    segment = occurrence.segment
    page_id = f"consistency:{eval_id}:{book}:{segment.page_index}:{segment.idx}:{occurrence.entry.id}:{which}"
    budget = PageBudget(page_id, CALL_BUDGET_TOKENS)
    text = segment.text(which)
    messages = consistency.build_messages(occurrence, which)
    result = consistency.Extraction()
    for attempt in range(1, consistency.ATTEMPTS + 1):
        reply = client.chat(messages, budget=budget, max_tokens=MAX_TOKENS, json_output=True)
        result.attempts, result.answer = attempt, reply.text
        try:
            if reply.finish_reason != "stop":
                raise consistency.Unusable(f"the answer was cut off (finish_reason {reply.finish_reason!r})")
            answer = consistency.read_answer(reply.text, text)
        except consistency.Unusable as e:
            result.problems.append(f"attempt {attempt}: {e}")
            messages = consistency.retry_messages(messages, reply.text, str(e))
            continue
        result.status, result.rendering, result.normalised = answer.status, answer.rendering, answer.normalised
        return result
    return result


def score_run(run: Run, loaded: glossary.Glossary, glossary_path: Path, client, *, prefix: str) -> dict:
    """The content of consistency-<model>.json: two extractions per occurrence, in the order of the occurrences."""
    started = time.monotonic()
    results = []
    for occurrence in consistency.find_occurrences(loaded, run.segments):
        extractions = {which: extract(client, occurrence, which, eval_id=run.eval_id, book=run.book)
                       for which in consistency.TEXTS}
        results.append(consistency.OccurrenceResult(occurrence, extractions))
    scores = consistency.score_book(loaded, results)
    records = client.meter.records
    config_ = client.config
    return {
        "eval_id": run.eval_id,
        "book": run.book,
        "segments_sha256": run.segments_sha256,
        "glossary_sha256": file_sha256(glossary_path),
        "rules": consistency.RULES,
        "judge": {
            "prefix": prefix,
            "model": config_.model,
            # one string when every call agrees; a list when they differ; None when no call was made or none said
            "requested_model": _one_or_list(rec.requested_model for rec in records),
            "response_model": _one_or_list(rec.response_model for rec in records),
            "temperature": config_.temperature,
            "reasoning_effort": config_.reasoning_effort,
            "json_mode": config_.json_mode,
            "max_tokens": MAX_TOKENS,
        },
        **scores.to_dict(),
        "usage": client.meter.summary(),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def _one_or_list(values) -> str | list[str] | None:
    found = sorted({v for v in values if v})
    return found[0] if len(found) == 1 else found or None


# --- the summary ------------------------------------------------------------------------------------------------------

def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_summary(document: dict) -> str:
    judge, usage, extractor = document["judge"], document["usage"], document["extractor"]
    out = [f"{document['eval_id']}: term consistency on {document['book']} ({document['rules']}), renderings by "
           f"{judge['model']} ({judge['prefix']}_*), responses from {judge['response_model'] or 'unknown'}"]
    out.append(f"  {document['occurrences']} occurrence(s) of {document['entries_scored']} entries; "
               f"{len(document['entries_left_out'])} borderline entries left out")
    for which in consistency.TEXTS:
        p, s = document["pooled"][which], document["statuses"][which]
        statuses = ", ".join(f"{status} {s[status]}" for status in consistency.STATUSES)
        out.append(f"  {which}: {percent(p['consistency'])} ({p['most_common_count']}/{p['rendered']} over "
                   f"{p['entries']} entries with {consistency.MIN_RENDERED}+ renderings); {statuses}")
    verdict = "reportable" if extractor["reportable"] else "for reference only"
    out.append(f"  extractor on the reference: {percent(extractor['accuracy'])} ({extractor['correct']}/"
               f"{extractor['checked']} where the reference holds the target), threshold "
               f"{extractor['threshold']:.0%}: consistency is {verdict}")
    cost = usage["cost_usd"]
    cost_text = (f"${cost:.6f}" if cost is not None else
                 f"unknown ({usage['unknown_usage_calls']} call(s) without usage, or no {judge['prefix']}_PRICE_*)")
    out.append(f"  calls: {usage['calls']}, tokens {usage['input_tokens']} in / {usage['output_tokens']} out, "
               f"cost {cost_text}")
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
