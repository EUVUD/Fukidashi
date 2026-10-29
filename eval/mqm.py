"""GEMBA-MQM of a finished eval run, by the judge (OUTLINE 4.4, 4.5, 8).

Usage:
  uv run python eval/mqm.py runs/eval/<id>                       # the judge: JUDGE_* in .env
  uv run python eval/mqm.py runs/eval/<id> --judge-prefix JUDGE2  # its fallback, only after the agreement check

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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from fukidashi import config
from fukidashi.evaluation import mqm_scorer
from fukidashi.llm.errors import ProviderError
from fukidashi.llm.metering import BudgetExceeded


def main(argv: list[str] | None = None, *, http_client=None) -> int:
    """`http_client` is for tests: the judge's requests go through it (an httpx.MockTransport) instead of the network."""
    ap = argparse.ArgumentParser(description="Score a finished eval run with GEMBA-MQM through the judge.")
    ap.add_argument("run_dir", type=Path, help="an eval run's directory, e.g. runs/eval/<id>")
    ap.add_argument("--judge-prefix", default="JUDGE",
                    help="the judge's variables in .env: JUDGE (default, GPT-4o mini) or JUDGE2 (its fallback)")
    args = ap.parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
