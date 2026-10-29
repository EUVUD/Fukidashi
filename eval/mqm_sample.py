"""The human sample for GEMBA-MQM's agreement check (OUTLINE 4.5): draw it, then check the filled sheet.

Usage:
  uv run python eval/mqm_sample.py draw --baseline runs/eval/<id> runs/eval/<id> --full runs/eval/<id> runs/eval/<id> \\
      --seed 0 [--out runs/mqm-human]
  uv run python eval/mqm_sample.py check runs/mqm-human/<time>-sample/sheet.csv

`draw` reads only the eval runs' manifest.json and segments.jsonl (finished runs, one baseline and one full run per
book). It picks 15 source segments whose Japanese text is not empty and whose baseline and full translations differ, in
proportion to the books' segment counts, and writes a new directory <out>/<UTC time>-sample/ with sheet.csv (30 items,
shuffled, with only the Japanese source and the translation), key.json (each item's book, page, idx, config, eval id and
run id: not to be opened before the annotation is done) and README.txt. An existing directory is never written into.
The sheet quotes the dataset and model output: keep it under runs/, which git ignores. The guide is
docs/mqm-annotation.md.

`check` validates a filled sheet against the item numbers in the key.json beside it: every item labelled, every severity
minor, major, critical or no-error, every category in the typology, no item both no-error and with an error. It prints
each item's MQM score (GEMBA's weights) and the sheet's sha256.

Nothing here calls a model, opens a network connection or reads the database.
Exit codes: draw: 0 written, 2 refused (nothing written), 1 the files could not be written. check: 0 the sheet is
complete and valid, 1 it has problems (listed), 2 it cannot be read.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fukidashi import config
from fukidashi.evaluation import mqm, mqm_sample

OUT_ROOT = config.ROOT / "runs" / "mqm-human"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Draw the GEMBA-MQM human sample, or check a filled sheet.")
    commands = ap.add_subparsers(dest="command", required=True)
    d = commands.add_parser("draw", help="draw the sample from finished baseline and full eval runs")
    d.add_argument("--baseline", nargs="+", type=Path, required=True, help="the baseline eval runs, one per book")
    d.add_argument("--full", nargs="+", type=Path, required=True, help="the full eval runs, one per book")
    d.add_argument("--seed", type=int, required=True, help="the random seed, e.g. 0")
    d.add_argument("--out", type=Path, default=OUT_ROOT, help="where the sample's own directory is made")
    c = commands.add_parser("check", help="validate a filled sheet.csv")
    c.add_argument("sheet", type=Path, help="the sheet, e.g. runs/mqm-human/<time>-sample/sheet.csv")
    args = ap.parse_args(argv)
    return draw(args) if args.command == "draw" else check(args)


def draw(args) -> int:
    try:
        sample = mqm_sample.draw(args.baseline, args.full, args.seed)
    except mqm_sample.Refused as e:
        print(f"Nothing was written: {e}", file=sys.stderr)
        return 2
    try:
        out_dir = mqm_sample.write_sample(sample, args.out)
    except OSError as e:
        print(f"Could not write the sample under {args.out}: {e}", file=sys.stderr)
        return 1
    per_book = ", ".join(f"{book} {n}" for book, n in sample.key["allocation"].items())
    print(f"Drew {len(sample.rows)} items from {sum(sample.key['allocation'].values())} source segments "
          f"({per_book}), seed {args.seed}.")
    print(f"Wrote {out_dir}/: sheet.csv, key.json (do not open before annotating), README.txt")
    return 0


def check(args) -> int:
    try:
        result = mqm_sample.check_sheet(args.sheet)
    except OSError as e:
        print(f"Cannot read {args.sheet}: {e}", file=sys.stderr)
        return 2
    if not result.ok:
        for problem in result.problems:
            print(problem, file=sys.stderr)
        if result.unlabelled:
            print(f"{len(result.unlabelled)} of {len(result.items)} items have no label: "
                  + ", ".join(str(i) for i in result.unlabelled), file=sys.stderr)
        print(f"sha256 {result.sha256}", file=sys.stderr)
        return 1
    for item in result.items:
        errors = result.errors.get(item, [])
        counts = ", ".join(f"{s} {n}" for s in mqm.SEVERITIES if (n := sum(e[0] == s for e in errors))) or "no-error"
        print(f"item {item}: {result.score(item)} ({counts})")
    scores = [result.score(item) for item in result.items]
    mean = f"{sum(scores) / len(scores):.3f}" if scores else "n/a"
    print(f"{len(scores)} items labelled; mean MQM score {mean} "
          f"(weights {mqm.WEIGHTS}; at most {mqm.COUNTED_ERRORS} errors count, capped at -{mqm.SCORE_CAP})")
    print(f"sha256 {result.sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
