"""Gold glossary tools (OUTLINE 4.7): candidate lists and review sheets for the author, and the validator.

Usage:
  uv run python eval/glossary.py candidates --book tojime_no_siora   # runs/glossary/<book>-candidates.md
  uv run python eval/glossary.py sheet --book tojime_no_siora        # runs/glossary/<book>-review.md
  uv run python eval/glossary.py check [paths]                       # default: every file in eval/gold_glossary/

Nothing here calls a model, opens a network connection or reads the database or the eval outputs: the only inputs are
the dataset's annotation.json and the glossary files. `candidates` and `sheet` quote the dataset (CC BY-NC), so they
write under the gitignored runs/glossary/. Exit code of `check`: 0 when every glossary is valid, 1 when a problem was
found, 2 when it could not start (no dataset, no glossary files).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fukidashi import config
from fukidashi.datasets.openmantra import Page, load_pages
from fukidashi.evaluation import glossary

GLOSSARY_DIR = config.ROOT / "eval" / "gold_glossary"
OUT_DIR = config.ROOT / "runs" / "glossary"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Candidates, review sheets and validation for the gold glossary.")
    ap.add_argument("--dataset", type=Path, default=config.DATASET_ROOT, help="the OpenMantra directory")
    ap.add_argument("--glossary-dir", type=Path, default=GLOSSARY_DIR, help="where the glossary files are")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="where candidates and sheets are written")
    sub = ap.add_subparsers(dest="command", required=True)
    for name, text in (("candidates", "list possible terms found in a book's Japanese text"),
                       ("sheet", "write the author's review sheet for a book's glossary")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--book", required=True, help="book_title, e.g. tojime_no_siora")
    p = sub.add_parser("check", help="validate glossary files against the dataset")
    p.add_argument("paths", nargs="*", type=Path, help="glossary files (default: all of --glossary-dir)")
    args = ap.parse_args(argv)

    try:
        pages = load_pages(args.dataset)
    except (OSError, ValueError, KeyError) as e:
        print(f"Cannot read the dataset at {args.dataset}: {e}", file=sys.stderr)
        return 2

    if args.command == "check":
        return check(args.paths or sorted(args.glossary_dir.glob("*.json")), pages)
    if not glossary.book_pages(pages, args.book):
        print(f"The dataset has no book {args.book!r}.", file=sys.stderr)
        return 2
    if args.command == "candidates":
        return write(args.out / f"{args.book}-candidates.md", glossary.render_candidates(args.book, pages))
    path = args.glossary_dir / f"{args.book}.json"
    loaded = glossary.load_glossary(path)
    if loaded.problems or loaded.book_title != args.book:
        print(f"{path} cannot be used for a sheet:", *map(str, loaded.problems), sep="\n  ", file=sys.stderr)
        return 2
    return write(args.out / f"{args.book}-review.md", glossary.render_sheet(loaded, pages))


def write(path: Path, text: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"Wrote {path}")
    return 0


def check(paths: list[Path], pages: list[Page]) -> int:
    if not paths:
        print("No glossary files to check.", file=sys.stderr)
        return 2
    found = 0
    for path in paths:
        loaded = glossary.load_glossary(path)
        problems = glossary.check(loaded, pages)
        found += len(problems)
        print(shown(path))
        if loaded.book_title:
            stats = glossary.summarize(loaded, pages)
            categories = ", ".join(f"{name} {n}" for name, n in sorted(stats.by_category.items())) or "none"
            print(f"  book {loaded.book_title} ({loaded.split}): {stats.entries} entries ({categories}); "
                  f"{stats.multi_page} occur on 2 or more pages")
            print(f"  coverage: mean {percent(stats.mean_coverage)}, pooled {percent(stats.pooled_coverage)}")
            print(f"  reviewed by author: {loaded.reviewed_by_author or 'not yet'}")
        print(f"  problems: {len(problems) or 'none'}")
        for problem in problems:
            print(f"    {problem}")
    return 1 if found else 0


def percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
