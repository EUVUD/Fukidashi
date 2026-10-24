"""Import OpenMantra (annotation.json) into the SQLite database. Safe to run again.

Usage:
  uv run python scripts/import_openmantra.py
  uv run python scripts/import_openmantra.py --db /tmp/test.db --root data/open-mantra-dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fukidashi.config import DATASET_ROOT, DB_PATH
from fukidashi.datasets.importer import import_openmantra
from fukidashi.db import connect, init_db


def main() -> None:
    ap = argparse.ArgumentParser(description="Import OpenMantra into SQLite.")
    ap.add_argument("--db", type=Path, default=DB_PATH, help="SQLite file (default: runs/fukidashi.db)")
    ap.add_argument("--root", type=Path, default=DATASET_ROOT, help="OpenMantra checkout")
    args = ap.parse_args()

    if not (args.root / "annotation.json").is_file():
        sys.exit(f"No annotation.json under {args.root}. Clone the dataset first (see README Setup).")
    conn = connect(args.db)
    init_db(conn)
    counts = import_openmantra(conn, args.root)
    print(f"{args.db}: {counts['books']} books, {counts['pages']} pages, {counts['segments']} segments")


if __name__ == "__main__":
    main()
