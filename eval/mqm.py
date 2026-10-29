"""GEMBA-MQM of a finished eval run, by the judge (OUTLINE 4.4, 4.5, 8).

Usage:
  uv run python eval/mqm.py runs/eval/<id>                       # the judge: JUDGE_* in .env
  uv run python eval/mqm.py runs/eval/<id> --judge-prefix JUDGE2  # its fallback, only after the agreement check
  uv run python eval/mqm.py --reparse runs/eval/<id>/mqm-<model>.json            # read failed answers again
  uv run python eval/mqm.py --reparse runs/eval/<id>/mqm-<model>.json --dry-run  # say what would change, write nothing

This makes real model calls, to the judge only: one per segment whose Japanese text is not empty (two when the first
answer cannot be read). Each call sees the Japanese source and the translation and nothing else: no image, no page
context, no reference. The judge is a second LLMClient read from <prefix>_* in .env, with its own ledger,
runs/judge_usage.jsonl: its calls are the evaluation's cost and never reach runs/usage.jsonl.

Writes mqm-<judge model>.json into the run's directory (replaced through a temporary file): every segment's errors,
MQM score, raw answer and attempts, the system score, the error breakdown, the judge, the GEMBA commit and the judge's
spend. Prints a summary.
Exit code: 0 when the file is written; 2 (with a message, nothing sent and nothing written) when the directory is
missing, the manifest is missing or unfinished, segments.jsonl is missing, empty or has a line without its fields, the
judge is not configured, or the judge's model is the translation model; 1 when a provider error or the per-segment
token budget stops it (no file is written, and the calls already made stay in the judge's ledger) or the file cannot be
written.

**The re-parse** (`--reparse`) reads an MQM file's `failed` segments again with the current parser
(evaluation/mqm.py `parse_answer`), from the last answer the file stored for each. It reads .env, makes a client and
calls a model never, so it adds no ledger line. It re-reads only a segment whose last attempt failed because the parser
could not read the answer (its last problem is the parser's); a segment whose last answer was cut off, and every scored
or skipped segment, stays exactly as it is. A segment the current parser reads becomes `scored`; the file's totals
(`scored`, `failed`, `system_score`, `breakdown`) are recomputed from its segments, and `reparsed` records when, at which
commit and which segments; its spend, `scored_at` and everything else stay. The file is replaced through a temporary
file, and only when a segment changes. `--dry-run` prints the same and writes nothing.
Exit code: 0 when the file is re-parsed, has nothing to change, or with --dry-run; 2 (nothing written) when the file
cannot be read or is not the run's (another eval id or segments.jsonl, another prompt template or weights, totals that
are not its segments'); 1 when it cannot be written.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from fukidashi import config
from fukidashi.evaluation import mqm, mqm_scorer
from fukidashi.evaluation.manifest import git_state
from fukidashi.llm.errors import ProviderError
from fukidashi.llm.metering import BudgetExceeded

# what judge_segment records for an attempt the parser could not read (mqm.parse_answer's two messages); a cut-off
# answer's problem says "the answer was cut off" and is never re-read
_PARSER_PROBLEM = re.compile(r"attempt (\d+): (?:the answer is empty|line \d+ comes before any severity heading: )")


def main(argv: list[str] | None = None, *, http_client=None) -> int:
    """`http_client` is for tests: the judge's requests go through it (an httpx.MockTransport) instead of the network."""
    ap = argparse.ArgumentParser(description="Score a finished eval run with GEMBA-MQM through the judge.")
    ap.add_argument("run_dir", type=Path, nargs="?", help="an eval run's directory, e.g. runs/eval/<id>")
    ap.add_argument("--judge-prefix", default="JUDGE",
                    help="the judge's variables in .env: JUDGE (default, GPT-4o mini) or JUDGE2 (its fallback)")
    ap.add_argument("--reparse", type=Path, metavar="MQM_FILE",
                    help="read the failed segments of this mqm-<model>.json again with the current parser; no call")
    ap.add_argument("--dry-run", action="store_true", help="with --reparse: print what would change, write nothing")
    args = ap.parse_args(argv)
    if (args.run_dir is None) == (args.reparse is None):
        ap.error("give an eval run's directory to score, or --reparse with an MQM file, not both")
    if args.reparse is not None:
        if args.judge_prefix != "JUDGE":
            ap.error("--reparse calls no judge: --judge-prefix does not apply")
        return reparse(args.reparse, dry_run=args.dry_run)
    if args.dry_run:
        ap.error("--dry-run applies to --reparse only")
    try:
        run = mqm_scorer.read_run(args.run_dir)  # before .env: a run that cannot be scored needs no configuration
        load_dotenv(config.ROOT / ".env")
        judge = mqm_scorer.judge_config(args.judge_prefix, run)
    except mqm_scorer.Refused as e:
        print(f"Cannot score {args.run_dir}: {e}", file=sys.stderr)
        return 2

    client = mqm_scorer.make_client(judge, config.JUDGE_LEDGER_PATH, http_client)
    try:
        document = mqm_scorer.score_run(run, client, prefix=args.judge_prefix)
    except (ProviderError, BudgetExceeded) as e:
        usage = client.meter.summary()
        print(f"Stopped, nothing written: {e}\n{usage['calls']} judge call(s) were made; they are in "
              f"{config.JUDGE_LEDGER_PATH}.", file=sys.stderr)
        return 1

    out = args.run_dir / mqm_scorer.output_name(judge.model)
    try:
        mqm_scorer.write_json(out, document)
    except OSError as e:
        print(f"Could not write {out}: {e}", file=sys.stderr)
        return 1
    print(mqm_scorer.format_summary(document))
    print(f"Wrote {out}")
    return 0


# --- the re-parse -------------------------------------------------------------------------------------------------------

def _key(segment: dict) -> dict:
    return {"book": segment["book"], "page_index": segment["page_index"], "idx": segment["idx"]}


def _where(segment: dict) -> str:
    return f"{segment['book']}/{segment['page_index']} idx {segment['idx']}"


def totals(segments: list[dict]) -> dict:
    """The fields of an MQM file that follow from its segments, computed as mqm_scorer.build_document computes them."""
    scored = [s for s in segments if s["status"] == "scored"]
    failed = [_key(s) for s in segments if s["status"] == "failed"]
    scores = [s["score"] for s in scored]
    errors = [[mqm.MQMError(**e) for e in s["errors"]] for s in scored]
    return {
        "scored": len(scored),
        "failed": {"count": len(failed), "segments": failed},
        "system_score": sum(scores) / len(scores) if scores else None,
        "breakdown": {"segments": {"scored": len(scored), "failed": len(failed)}, **mqm.breakdown(errors)},
    }


def rereadable(segment: dict) -> bool:
    """A failed segment whose stored answer is its last attempt's and failed only because the parser could not read it."""
    problems, attempts = segment.get("problems"), segment.get("attempts")
    if segment.get("status") != "failed" or not isinstance(segment.get("answer"), str) or not problems \
            or not isinstance(problems, list) or not isinstance(problems[-1], str):
        return False
    matched = _PARSER_PROBLEM.match(problems[-1])
    return matched is not None and int(matched.group(1)) == attempts


def reparse_document(document: dict, run: mqm_scorer.EvalRun) -> tuple[list[dict], dict]:
    """(the new segments, what happened): each re-readable failed segment read again with mqm.parse_answer. Raises
    Refused when the file is not the run's or its totals are not its segments'."""
    if document.get("eval_id") != run.eval_id:
        raise mqm_scorer.Refused(f"it is of eval run {document.get('eval_id')!r}; the directory's is {run.eval_id!r}")
    if document.get("segments_sha256") != run.segments_sha256:
        raise mqm_scorer.Refused("it scored another segments.jsonl than the one next to it (segments_sha256 differs)")
    if (document.get("gemba") or {}).get("template_sha256") != mqm.template_sha256():
        raise mqm_scorer.Refused("it was scored with another prompt template than the current parser's")
    if document.get("weights") != mqm.WEIGHTS:
        raise mqm_scorer.Refused(f"it was scored with the weights {document.get('weights')!r}, not {mqm.WEIGHTS!r}")
    segments = document.get("segments")
    try:
        if not isinstance(segments, list) or not all(isinstance(s, dict) for s in segments):
            raise TypeError("`segments` is not a list of objects")
        stored = {key: document.get(key) for key in ("scored", "failed", "system_score", "breakdown")}
        if totals(segments) != stored:
            raise mqm_scorer.Refused("its totals (scored, failed, system_score, breakdown) are not its segments' totals")
    except (KeyError, TypeError, ValueError) as e:
        raise mqm_scorer.Refused(f"it is not shaped as eval/mqm.py writes it ({type(e).__name__}: {e})")

    changed, still_failed, not_reread = [], [], []
    new_segments = []
    for segment in segments:
        if segment["status"] != "failed":
            new_segments.append(segment)
            continue
        if not rereadable(segment):
            not_reread.append({**_key(segment), "problems": segment.get("problems")})
            new_segments.append(segment)
            continue
        try:
            errors = mqm.parse_answer(segment["answer"])
        except mqm.Unreadable as e:
            still_failed.append({**_key(segment), "problem": str(e)})
            new_segments.append(segment)
            continue
        score = mqm.mqm_score(e.severity for e in errors)
        new_segments.append({**segment, "status": "scored", "errors": [e.to_dict() for e in errors], "score": score})
        changed.append({**_key(segment), "score": score, "errors": len(errors)})
    return new_segments, {"changed": changed, "still_failed": still_failed, "not_reread": not_reread}


def reparse(path: Path, *, dry_run: bool, now: datetime | None = None) -> int:
    try:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except OSError as e:
            raise mqm_scorer.Refused(f"cannot read it: {e}")
        except ValueError as e:
            raise mqm_scorer.Refused(f"it is not valid JSON: {e}")
        if not isinstance(document, dict):
            raise mqm_scorer.Refused("it is not a JSON object")
        judge = document.get("judge")
        if not isinstance(judge, dict) or not isinstance(judge.get("model"), str) \
                or path.name != mqm_scorer.output_name(judge["model"]):
            raise mqm_scorer.Refused("it is not an mqm-<judge model>.json of its judge model")
        run = mqm_scorer.read_run(path.parent)
        segments, outcome = reparse_document(document, run)
    except mqm_scorer.Refused as e:
        print(f"Cannot re-parse {path}: {e}", file=sys.stderr)
        return 2

    before = {key: document.get(key) for key in ("scored", "failed", "system_score")}
    after = totals(segments)
    lines = [f"{path}: {len(outcome['changed'])} failed segment(s) read again"
             f"{' (dry run, nothing written)' if dry_run else ''}"]
    lines += [f"  {_where(c)}: failed -> scored, score {c['score']} ({c['errors']} error(s))" for c in outcome["changed"]]
    lines += [f"  {_where(s)}: still failed: {s['problem']}" for s in outcome["still_failed"]]
    lines += [f"  {_where(s)}: not read again (its last answer was not unreadable by the parser): {s['problems']}"
              for s in outcome["not_reread"]]
    score = (lambda v: "n/a" if v is None else f"{v:.3f}")
    lines.append(f"  scored {before['scored']} -> {after['scored']}, failed {before['failed']['count']} -> "
                 f"{after['failed']['count']}, system score {score(before['system_score'])} -> "
                 f"{score(after['system_score'])}")
    print("\n".join(lines))
    if dry_run:
        return 0
    if not outcome["changed"]:
        print("Nothing to change; the file was left as it is.")
        return 0

    record = {"at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"), **git_state(config.ROOT),
              "parser": "evaluation/mqm.py parse_answer", "segments": outcome["changed"],
              "still_failed": outcome["still_failed"], "not_reread": [_key(s) for s in outcome["not_reread"]]}
    earlier = document.get("reparsed")
    new = {**document, **after, "segments": segments,
           "reparsed": [*(earlier if isinstance(earlier, list) else []), record]}
    try:
        mqm_scorer.write_json(path, new)
    except OSError as e:
        print(f"Could not write {path}: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
