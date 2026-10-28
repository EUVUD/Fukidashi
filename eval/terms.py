"""Term accuracy of a finished eval run, against the gold glossary (OUTLINE 4.7).

Usage:
  uv run python eval/terms.py runs/eval/<id>

Reads the run's manifest.json and segments.jsonl and eval/gold_glossary/<book>.json, and writes terms.json into the
run's directory (replaced through a temporary file): the primary score (entries that are not borderline), the score over
all entries, the mean over entries, and every entry's occurrences, hits and left-out segments. Nothing here calls a
model, opens a network connection or reads the database: a change of rules can score old runs again for free.
Exit code: 0 when terms.json is written, 2 (with a message, and nothing written) when the directory is missing, the
manifest is missing or unfinished, segments.jsonl is missing, empty or unreadable, the book has no glossary, or the
glossary has problems.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fukidashi import config
from fukidashi.evaluation import glossary, terms
from fukidashi.evaluation.manifest import file_sha256

GLOSSARY_DIR = config.ROOT / "eval" / "gold_glossary"


class Refused(Exception):
    """The run or the glossary cannot be scored; the message says why."""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score a finished eval run's term accuracy against the gold glossary.")
    ap.add_argument("run_dir", type=Path, help="an eval run's directory, e.g. runs/eval/<id>")
    ap.add_argument("--glossary-dir", type=Path, default=GLOSSARY_DIR, help="where the glossary files are")
    args = ap.parse_args(argv)
    try:
        document = score_run(args.run_dir, args.glossary_dir)
    except Refused as e:
        print(f"Cannot score {args.run_dir}: {e}", file=sys.stderr)
        return 2
    out = args.run_dir / "terms.json"
    write_json(out, document)
    print(format_summary(document))
    print(f"Wrote {out}")
    return 0


def score_run(run_dir: Path, glossary_dir: Path) -> dict:
    """The content of terms.json for the run in `run_dir`; raises Refused, having written nothing."""
    if not run_dir.is_dir():
        raise Refused("it is not a directory")
    manifest = read_manifest(run_dir / "manifest.json")
    book = manifest["book"]
    lines = read_lines(run_dir / "segments.jsonl")

    path = glossary_dir / f"{book}.json"
    if not path.is_file():
        raise Refused(f"there is no glossary for book {book!r} at {path}")
    loaded = glossary.load_glossary(path)
    if loaded.problems:
        raise Refused(f"{path} has problems:" + "".join(f"\n  {p}" for p in loaded.problems))
    if loaded.book_title != book:
        raise Refused(f"{path} is the glossary of {loaded.book_title!r}, the run is of {book!r}")

    try:
        scores = terms.score_terms(loaded, lines)
    except terms.TermsError as e:
        raise Refused(f"segments.jsonl: {e}")
    return {
        "eval_id": manifest["eval_id"],
        "book": book,
        "glossary_sha256": file_sha256(path),
        "rules": terms.RULES,
        "pages_scored": sorted({line["page_index"] for line in lines}),
        "pages_not_scored": [{"page_index": p.get("page_index"), "status": p.get("status")}
                             for p in manifest.get("pages", []) if isinstance(p, dict) and p.get("status") != "scored"],
        **scores.to_dict(),
    }


def read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise Refused(f"cannot read {path.name}: {e}")
    except ValueError as e:
        raise Refused(f"{path.name} is not valid JSON: {e}")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("book"), str) \
            or not isinstance(manifest.get("eval_id"), str):
        raise Refused(f"{path.name} has no `eval_id` and `book`")
    if manifest.get("finished_at") is None:
        raise Refused("the eval is not finished (manifest.json has `finished_at` null)")
    return manifest


def read_lines(path: Path) -> list[dict]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise Refused(f"cannot read {path.name}: {e}")
    lines = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            lines.append(json.loads(line))
        except ValueError as e:
            raise Refused(f"{path.name} line {number} is not valid JSON: {e}")
    if not lines:
        raise Refused(f"{path.name} has no segments")
    return lines


def write_json(path: Path, data: dict) -> None:
    """Replace `path` in one step: write a temporary file next to it, then rename it over."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def format_summary(document: dict) -> str:
    out = [f"{document['eval_id']}: term accuracy on {document['book']} ({document['rules']})"]
    for name, label in (("primary", "primary (not borderline)"), ("all_entries", "all entries")):
        s = document["scores"][name]
        out.append(f"  {label}: {s['hits']}/{s['occurrences']} = {percent(s['accuracy'])} pooled; "
                   f"mean over {s['entries_with_occurrences']} of {s['entries']} entries "
                   f"{percent(s['mean_over_entries'])}")
    left_out = sum(len(e["left_out"]) for e in document["entries"])
    out.append(f"  {left_out} segment(s) left out because the reference lacks the term")
    return "\n".join(out)


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


if __name__ == "__main__":
    sys.exit(main())
