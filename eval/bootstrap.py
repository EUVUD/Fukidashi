"""The paired bootstrap by page of the two comparisons (OUTLINE 4.3, 8; eval/PROTOCOL.md, "Statistics").

Usage:
  uv run python eval/bootstrap.py runs/eval/<id> [runs/eval/<id> ...] [--mqm-judge <judge model>]

Takes the eval runs that eval/report.py pools: one finished, scored eval run for each of `baseline`, `full_no_kb` and
`full` and each book, every run with its `terms.json` (eval/terms.py), `comet-wmt22-comet-da.json` (eval/comet.py),
consistency file (eval/consistency.py) and MQM file (eval/mqm.py). It reads and refuses them with the report's own code,
so the same inputs are refused with the same messages; --mqm-judge names the judge whose MQM files are used, which is
needed when a run has files of more than one judge.

For each comparison, over the pages that both of its configs scored (per book, pooled over the books), it resamples the
pages with replacement, 10,000 times with seed 2027, recomputes every metric on each resample and reports the 95%
percentile interval of the difference, `full` minus the other config:
  full vs baseline:    COMET (the primary metric), chrF, BLEU, term accuracy, term consistency and GEMBA-MQM;
  full vs full_no_kb:  term accuracy (the primary metric) and term consistency.
Cost and latency get no interval. The GEMBA-MQM interval is for reference only (neither judge passed the agreement
check, OUTLINE 4.5), and so is the term consistency interval of a comparison when the extractor's accuracy on the
references is under the threshold in any of its runs. The details (the pages, the draws, a page drawn twice, a resample
on which a metric is undefined) are in evaluation/bootstrap.py and are recorded in the output.

Writes runs/bootstrap/<UTC time>-bootstrap/bootstrap.json (a new directory each time, the file replaced through a
temporary file) and prints a short table. It reads only files: nothing here calls a model, opens a network connection or
reads the database. --resamples is for tests only; the protocol's number is 10,000, and the output records the number
used.
Exit code: 0 when bootstrap.json is written; 2 (with a message naming the directory and the cause, and nothing written)
for anything the report refuses, and when the runs are not exactly the three configs, or a run has no consistency or MQM
file; 1 when the file cannot be written.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from fukidashi import config
from fukidashi.evaluation import bootstrap

GLOSSARY_DIR = config.ROOT / "eval" / "gold_glossary"
BOOTSTRAP_DIR = config.ROOT / "runs" / "bootstrap"


def _positive(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number")
    if value < 1:
        raise argparse.ArgumentTypeError("it must be at least 1")
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paired bootstrap by page of full vs baseline and full vs full_no_kb.")
    ap.add_argument("run_dirs", nargs="+", type=Path, help="eval runs' directories, e.g. runs/eval/<id>")
    ap.add_argument("--glossary-dir", type=Path, default=GLOSSARY_DIR, help="where the glossary files are")
    ap.add_argument("--out-root", type=Path, default=BOOTSTRAP_DIR, help="where the bootstrap's own directory is made")
    ap.add_argument("--mqm-judge", help="the judge model whose MQM files are used, when a run has more than one")
    ap.add_argument("--resamples", type=_positive, default=bootstrap.RESAMPLES,
                    help=f"for tests only: the protocol's number is {bootstrap.RESAMPLES:,}")
    args = ap.parse_args(argv)
    started = time.monotonic()
    try:
        document = bootstrap.build_bootstrap(args.run_dirs, args.glossary_dir, mqm_judge=args.mqm_judge,
                                             resamples=args.resamples)
    except bootstrap.Refused as e:
        print(f"Nothing was written: {e}", file=sys.stderr)
        return 2
    elapsed = time.monotonic() - started
    try:
        out_dir = bootstrap.write_bootstrap(document, args.out_root)
    except OSError as e:
        print(f"Nothing was written: could not write the bootstrap under {args.out_root}: {e}", file=sys.stderr)
        return 1
    if args.resamples != bootstrap.RESAMPLES:
        print(f"note: {args.resamples:,} resamples, not the protocol's {bootstrap.RESAMPLES:,} "
              "(--resamples is for tests)", file=sys.stderr)
    print(bootstrap.render(document))
    print(f"Computed in {elapsed:.1f} s")
    print(f"Wrote {out_dir / 'bootstrap.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
