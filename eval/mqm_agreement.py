"""GEMBA-MQM's human agreement check (OUTLINE 4.5): compare the annotator's labels with the judge's, and decide.

Usage:
  uv run python eval/mqm_agreement.py runs/mqm-human/<time>-sample --expected-sha256 <sha256 in OUTLINE 4.5> \\
      runs/eval/<id>/mqm-<judge model>.json [runs/eval/<id>/mqm-<judge model>.json ...]

Reads the sample's filled sheet.csv and its key.json, and one MQM file (eval/mqm.py) for each of the sample's eval runs,
all from one judge model. Per item it takes the annotator's MQM score (GEMBA's weights, as eval/mqm_sample.py check
prints it) and the judge's score of the same segment; over the items it computes Kendall's tau_b and the recall of the
items where the annotator marked a major or critical error (a hit is an item where the judge marked one too). The judge
passes when tau_b >= 0.30 and the recall >= 60%; when the annotator marked no major or critical error, tau_b decides
alone (OUTLINE 4.5).

Writes agreement-<judge model>.json into the sample's directory: per item the annotator's and the judge's scores and
whether each marked a major or critical error; tau_b with its pair counts, the recall, the thresholds, the decision, the
sheet's sha256, the annotator and the MQM files it read (each with its sha256 and scored_at). An existing file with the
same result is left as it is (so a second run keeps the first one's computed_at); one with another result is never
overwritten. --annotator says who labelled the sheet: llm (the default; the recorded sample was labelled by an LLM, not
a person, OUTLINE 4.5) or person. Prints a summary.

Nothing here calls a model, opens a network connection or reads the database.
Exit code: 0 when the file is written or already holds this result; 2 (with a message, nothing written) when the sheet's
sha256 is not the expected one, the sheet is incomplete or invalid, an item has no judge score (its segment failed or
was skipped), or the inputs do not belong together (an MQM file of another eval run, two files of one run, files of
different judge models, prompts or weights, a run of the sample without a file, a file over another segments.jsonl, a
segment of another run), or an agreement file with another result is there; 1 when the file cannot be written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fukidashi.evaluation import agreement


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare the annotator's MQM labels with the judge's (OUTLINE 4.5).")
    ap.add_argument("sample_dir", type=Path, help="the sample's directory, e.g. runs/mqm-human/<time>-sample")
    ap.add_argument("mqm_files", nargs="+", type=Path, help="one MQM file (eval/mqm.py) per eval run of the sample")
    ap.add_argument("--expected-sha256", required=True, help="the sheet's sha256 as OUTLINE 4.5 records it")
    ap.add_argument("--annotator", choices=sorted(agreement.ANNOTATORS), default="llm",
                    help="who labelled the sheet (default llm: the recorded sample's labels are an LLM's)")
    args = ap.parse_intermixed_args(argv)
    try:
        document = agreement.compute(args.sample_dir, args.expected_sha256, args.mqm_files, annotator=args.annotator)
        out = args.sample_dir / agreement.output_name(document["judge"]["model"])
        written = agreement.write(document, out)
    except agreement.Refused as e:
        print(f"Nothing was written: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"Could not write the agreement file in {args.sample_dir}: {e}", file=sys.stderr)
        return 1
    print(agreement.format_summary(document))
    print(f"Wrote {out}" if written else f"{out} already holds this result; it was left as it is")
    return 0


if __name__ == "__main__":
    sys.exit(main())
