"""Eval harness entry point: translate pages the way the editor does and score them with sacreBLEU.

Usage:
  uv run python eval/run.py --config baseline --book tojime_no_siora            # every page of the book, in page order
  uv run python eval/run.py --config baseline --book tojime_no_siora --pages 2,3,4
  uv run python eval/run.py --config translator --book tojime_no_siora --pages 2,3,4
  uv run python eval/run.py --config full_no_kb --book tojime_no_siora --pages 2,3,4
  uv run python eval/run.py --config full --book tojime_no_siora --pages 2,3,4

This makes real model calls: one per page for baseline, several for translator (the agent's steps, a repair,
and the fallback to baseline when it comes to that), and for full_no_kb those plus a review, and up to two
more rounds of revision and review per page. full is full_no_kb with the Knowledge Base: the same calls, a
few more tool calls for the KB, and the Curator's one call per page that proposed a term. It makes a new KB
for the eval run (never shared with another run) and the pages go through it in the order given, so a page
reads what the earlier ones wrote; list the pages in book order. Without --pages, the eval covers every page of
the book that is in the database, in page order (a page without text is skipped, with no call).

Outputs go to runs/eval/<id>/: manifest.json, segments.jsonl, scores.json. They are written page by page: after
each page the three files cover the pages done so far, with the scores of those pages. So an eval that is stopped
(Ctrl-C, a crash, a killed process) leaves the outputs of the pages it finished: its manifest has "finished_at"
null and the pages it did not reach "pending". It is not continued; to finish it, run the whole eval again. A
missing LLM configuration stops the eval before the KB and the directory are made. The logic lives in
fukidashi.evaluation; the exit code is 0 when every page was scored or skipped, 1 when a page failed, 2 when the
harness could not start (including a book with no pages in the database).

With FUKIDASHI_TRACING=phoenix (shell or .env) and a Phoenix running on this machine, every page leaves one trace there;
its ID is in the outputs.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from fukidashi import config, store, tracing
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
    ap.add_argument("--pages", type=parse_pages,
                    help="page_index values, e.g. 2,3,4 (default: every page of the book, in page order)")
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
        pages = args.pages if args.pages is not None else store.book_page_indexes(conn, args.book)
        if not pages:
            print(f"Stopped before any model call: the database has no pages of book {args.book!r}.\n"
                  "(Import the dataset: uv run python scripts/import_openmantra.py)", file=sys.stderr)
            return 2
        result = run_eval(
            conn, config=args.config, book=args.book, page_indexes=pages,
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
