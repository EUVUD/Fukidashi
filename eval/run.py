"""Eval harness entry point: translate pages the way the editor does and score them with sacreBLEU.

Usage:
  uv run python eval/run.py --config baseline --book tojime_no_siora --pages 2,3,4
  uv run python eval/run.py --config translator --book tojime_no_siora --pages 2,3,4
  uv run python eval/run.py --config full_no_kb --book tojime_no_siora --pages 2,3,4

This makes real model calls: one per page for baseline, several for translator (the agent's steps, a repair,
and the fallback to baseline when it comes to that), and for full_no_kb those plus a review, and up to two
more rounds of revision and review per page. Outputs go to runs/eval/<id>/: manifest.json, segments.jsonl,
scores.json. The logic lives in fukidashi.evaluation; the exit code is 0 when every page was scored or
skipped, 1 when a page failed, 2 when the harness could not start.

With FUKIDASHI_TRACING=phoenix (shell or .env) and a Phoenix running on this machine, every page leaves one trace there;
its ID is in the outputs.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from fukidashi import config, tracing
from fukidashi.agents.loop import TracingEnabled
from fukidashi.api.deps import get_client_factory
from fukidashi.db import connect, init_db
from fukidashi.evaluation.harness import CONFIGS, EvalAborted, format_summary, run_eval
from fukidashi.llm.client import LLMNotConfigured


def parse_pages(text: str) -> list[int]:
    try:
        pages = [int(p) for p in text.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated page numbers like 2,3,4, got {text!r}")
    if len(set(pages)) != len(pages):
        raise argparse.ArgumentTypeError(f"duplicate pages in {text!r}")
    return pages


def main() -> int:
    ap = argparse.ArgumentParser(description="Translate pages with a config and score them against the references.")
    ap.add_argument("--config", required=True, choices=CONFIGS, help="configuration to run")
    ap.add_argument("--book", required=True, help="book_title, e.g. tojime_no_siora")
    ap.add_argument("--pages", required=True, type=parse_pages, help="page_index values, e.g. 2,3,4")
    args = ap.parse_args()

    load_dotenv(config.ROOT / ".env")
    try:
        tracing.setup()
    except tracing.TracingError as e:
        print(f"Stopped before any model call: {e}", file=sys.stderr)
        return 2
    if (traced := tracing.describe())["enabled"]:
        print("Tracing on: " + ", ".join(f"{k} {v}" for k, v in traced.items() if k != "enabled"), flush=True)

    conn = connect(config.DB_PATH)
    init_db(conn)
    try:
        result = run_eval(
            conn, config=args.config, book=args.book, page_indexes=args.pages,
            make_client=get_client_factory(), dataset_root=config.DATASET_ROOT,
            out_root=config.ROOT / "runs" / "eval", repo_root=config.ROOT,
            log=lambda msg: print(msg, flush=True),
        )
    except EvalAborted as e:
        print("Stopped before any model call:", *e.problems, sep="\n  ", file=sys.stderr)
        print("(If pages are missing, import the dataset: uv run python scripts/import_openmantra.py)", file=sys.stderr)
        return 2
    except (LLMNotConfigured, TracingEnabled) as e:
        print(f"Stopped before any model call: {e}", file=sys.stderr)
        return 2
    finally:
        conn.close()
        tracing.shutdown()  # sends the last spans; with no Phoenix listening this waits for the exporter's timeout

    print(format_summary(result))
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
