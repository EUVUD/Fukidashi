"""Cost and latency of finished eval runs, from the translation ledger (eval/PROTOCOL.md, "Statistics").

Usage:
  uv run python eval/costs.py runs/eval/<id> [runs/eval/<id> ...] --ledger-range <eval id>=<first>-<last> [...] \
      [--ledger runs/usage.jsonl] [--mqm-judge <judge model>] [--cached-input-price-per-m <dollars>] [--out-root DIR]

Takes the eval runs that eval/report.py pools (it runs the report's own checks first, so the same inputs are refused
with the same messages) and, for each run, the first and last line (counted from 1, both included) of the ledger
(`runs/usage.jsonl`) that its calls wrote: the ledger has one line per call and no run id. For each config, pooled over
the books and for each book, on the common pages (as the report has them) and on all attempted pages (the failed ones
included, so their calls count), it reports calls per page; tokens per page, split into uncached input, cached input and
output; the cached share of input; cost per page as a range, from cached input at $0 up to the ledger's cost, which
prices every input token at the input price (with --cached-input-price-per-m, also a value at that price); and latency
p50, p95 and mean. The failed pages and the ledger lines without usage or cost are listed, never counted as 0.

The prices are recovered from the ledger lines of the ranges, not read from .env, and must fit exactly. A range that is
not the run's calls is refused: it must split into one run of one `page_id` per attempted page, in the manifest's order,
as long as `calls` for a scored page, made with the model the run asked for, and no two ranges may overlap.

Writes runs/analysis/<UTC time>-costs/costs.json (a new directory each time, the file replaced through a temporary file)
and prints a plain-text table. It reads only files: nothing here calls a model, opens a network connection or reads the
database, and it never writes to the ledger.
Exit code: 0 when costs.json is written; 2 (with a message naming the directory and the cause, and nothing written) for
anything the report refuses, a ledger range that is missing, malformed or not the run's calls, or costs that do not fit
one pair of prices; 1 when the file cannot be written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fukidashi import config
from fukidashi.evaluation import costs

GLOSSARY_DIR = config.ROOT / "eval" / "gold_glossary"
LEDGER = config.LEDGER_PATH
ANALYSIS_DIR = config.ROOT / "runs" / "analysis"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cost and latency of finished eval runs, from the translation ledger.")
    ap.add_argument("run_dirs", nargs="+", type=Path, help="eval runs' directories, e.g. runs/eval/<id>")
    ap.add_argument("--ledger-range", action="append", default=[], metavar="EVAL_ID=FIRST-LAST",
                    help="the ledger lines (counted from 1, both included) that an eval run's calls wrote; one for "
                         "each run")
    ap.add_argument("--ledger", type=Path, default=LEDGER, help="the translation ledger")
    ap.add_argument("--glossary-dir", type=Path, default=GLOSSARY_DIR, help="where the glossary files are")
    ap.add_argument("--out-root", type=Path, default=ANALYSIS_DIR, help="where the costs' own directory is made")
    ap.add_argument("--mqm-judge", help="the judge model whose MQM files the report pools, when a run has more than one")
    ap.add_argument("--cached-input-price-per-m", type=float, metavar="DOLLARS",
                    help="also give the cost at this price per million cached input tokens (between 0 and the input "
                         "price)")
    args = ap.parse_args(argv)
    try:
        document = costs.build_costs(args.run_dirs, args.glossary_dir, args.ledger, costs.parse_ranges(args.ledger_range),
                                     mqm_judge=args.mqm_judge, cached_input_price_per_m=args.cached_input_price_per_m)
    except costs.Refused as e:
        print(f"Nothing was written: {e}", file=sys.stderr)
        return 2
    try:
        out_dir = costs.write_costs(document, args.out_root)
    except OSError as e:
        print(f"Nothing was written: could not write the costs under {args.out_root}: {e}", file=sys.stderr)
        return 1
    print(costs.render(document))
    print(f"Wrote {out_dir / 'costs.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
