"""Pool finished eval runs into one report (OUTLINE 4.3, 4.4).

Usage:
  uv run python eval/report.py runs/eval/<id> [runs/eval/<id> ...]

Takes one finished eval run for each config and book (the same books for every config), each with its `terms.json`
(eval/terms.py) and `comet-wmt22-comet-da.json` (eval/comet.py). Per config, pooled over the books and per book, it
reports chrF / BLEU, COMET and term accuracy over the pages that every config scored, then cost per page and latency
p50 / p95 over the same pages, and the fallback pages, the review loop, the Curator and the KB's reach where the config
has them. The pages left out, and what each config's Runs cost altogether, are listed. Provenance that does not make the
numbers wrong (a dirty tree, different commits, different models) is a warning. There are no confidence intervals: the
paired bootstrap is M8.

Writes runs/report/<UTC time>-report/report.json (a new directory each time, the file replaced through a temporary
file) and prints a plain-text summary. It reads only files: nothing here calls a model, opens a network connection or
reads the database.
Exit code: 0 when report.json is written; 2 (with a message naming the directory and the cause, and nothing written) when
a directory is missing or unfinished, a file is missing or does not match the others, or the runs are not one for each
config and book over the same books; 1 when the report cannot be written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fukidashi import config
from fukidashi.evaluation import report

GLOSSARY_DIR = config.ROOT / "eval" / "gold_glossary"
REPORT_DIR = config.ROOT / "runs" / "report"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pool finished eval runs (one per config and book) into one report.")
    ap.add_argument("run_dirs", nargs="+", type=Path, help="eval runs' directories, e.g. runs/eval/<id>")
    ap.add_argument("--glossary-dir", type=Path, default=GLOSSARY_DIR, help="where the glossary files are")
    ap.add_argument("--out-root", type=Path, default=REPORT_DIR, help="where the report's own directory is made")
    args = ap.parse_args(argv)
    try:
        document = report.build_report(args.run_dirs, args.glossary_dir)
    except report.Refused as e:
        print(f"Nothing was written: {e}", file=sys.stderr)
        return 2
    try:
        out_dir = report.write_report(document, args.out_root)
    except OSError as e:
        print(f"Nothing was written: could not write the report under {args.out_root}: {e}", file=sys.stderr)
        return 1
    print(report.render(document))
    print(f"Wrote {out_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
