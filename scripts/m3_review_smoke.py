"""M3 review smoke test: the flagging Translator, then the Reviewer once, on one OpenMantra page.

The config that runs Translator -> Reviewer is a later task's; this is how the Reviewer is seen against the real
provider before that. It reads the earlier pages' `translator` Runs from the database (read only), writes no Run, and
saves what it did under runs/smoke/.

Usage:
  uv run python scripts/m3_review_smoke.py --book tojime_no_siora --page 3 --dry-run   # print both first requests, send nothing
  uv run python scripts/m3_review_smoke.py --book tojime_no_siora --page 3             # real calls, configured by .env
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from fukidashi import config, runs, tracing
from fukidashi.agents import reviewer, translator
from fukidashi.agents.loop import DEFAULT_MAX_STEPS
from fukidashi.datasets.openmantra import get_page
from fukidashi.db import connect
from fukidashi.llm.client import LLMClient, LLMConfig
from fukidashi.llm.metering import CallRecord, Meter
from fukidashi.pipeline import baseline

ROOT = Path(__file__).resolve().parents[1]


class FirstRequest(Exception):
    """Raised by the dry-run client when an agent makes its first model call."""

    def __init__(self, messages: list[dict], tools: list[dict] | None, options: dict):
        super().__init__("first request")
        self.messages, self.tools, self.options = messages, tools, options


class DryRunClient:
    """Stands in for LLMClient: stops at the first call, so the request printed is the one that would go out."""

    def chat(self, messages, *, tools=None, **options):
        raise FirstRequest(messages, tools, options)


def earlier_pages(book: str, page_index: int) -> list[translator.PreviousPage]:
    """The pages before this one with their latest succeeded `translator` Runs, as the Translator's tool gets them.

    Reads the database and nothing else; with no database there are none.
    """
    if not config.DB_PATH.is_file():
        return []
    conn = connect(config.DB_PATH)
    try:
        return runs.previous_pages(conn, book, page_index, translator.CONFIG_NAME)
    finally:
        conn.close()


def show_request(first: FirstRequest, page) -> dict:
    """The request as it would go out, with the page image replaced: the base64 is the one thing nobody reads."""
    messages = copy.deepcopy(first.messages)
    placeholder = f"(page image {page.image_path.name}, {page.image_path.stat().st_size / 1024:.0f} KB, base64 omitted)"
    for message in messages:
        if isinstance(message["content"], list):
            for part in message["content"]:
                if part["type"] == "image_url":
                    part["image_url"]["url"] = placeholder
    sent = {k: v for k, v in first.options.items() if k != "budget"}  # the budget is ours, not the request's
    return {"messages": messages, "tools": first.tools, **sent}


def dry_run(page, previous: list[translator.PreviousPage], args) -> None:
    """Print the first request of each stage and send nothing. The Reviewer's request carries a placeholder
    translation and a placeholder flag: the real ones exist only after the Translator has run."""
    try:
        translator.run_translator(page, DryRunClient(), previous=previous, flagging=True,
                                  agent_budget=args.agent_budget, max_tokens=args.max_tokens, max_steps=args.max_steps)
    except FirstRequest as first:
        translator_request = show_request(first, page)
    placeholder = {s.index: f"(placeholder translation of region {s.index})" for s in page.segments}
    flags = [{"id": page.segments[0].index, "reason": "(placeholder flag)"}]
    try:
        reviewer.run_reviewer(page, placeholder, flags, DryRunClient(), review_budget=args.review_budget,
                              max_tokens=args.max_tokens, max_steps=args.max_steps)
    except FirstRequest as first:
        reviewer_request = show_request(first, page)
    print(json.dumps({"translator_first_request": translator_request, "reviewer_first_request": reviewer_request},
                     ensure_ascii=False, indent=2))
    print("\nThe Reviewer's translation and flag are placeholders. Page images are replaced. "
          "LLM_MODEL and the key are not needed for a dry run.")
    print("\nDry run: nothing was sent.")


def stage_numbers(records: list[CallRecord], stop_reason: str, steps: list[dict]) -> dict:
    """What a stage cost: `steps` are the agent's model calls, `calls` every request it sent (a repair or the
    Translator's fallback included). Latency is the sum over the calls."""
    summary = Meter(records=list(records)).summary()
    return {"stop_reason": stop_reason, "steps": len(steps), **summary,
            "latency_s": round(sum(r.latency_s for r in records), 3)}


def cost_text(cost: float | None) -> str:
    return f"${cost:.4f}" if cost is not None else "unknown"


def describe_stage(name: str, numbers: dict) -> str:
    return (f"{name}: stop reason {numbers['stop_reason']}, {numbers['steps']} steps, {numbers['calls']} calls, "
            f"{numbers['input_tokens']} input / {numbers['output_tokens']} output tokens, "
            f"cost {cost_text(numbers['cost_usd'])}, latency {numbers['latency_s']:.1f} s"
            + (f", {numbers['unknown_usage_calls']} calls without usage" if numbers["unknown_usage_calls"] else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description="M3 smoke test: the flagging Translator, then the Reviewer, on one page.")
    ap.add_argument("--book", required=True, help="book_title slug")
    ap.add_argument("--page", type=int, required=True, help="page_index (1-based)")
    ap.add_argument("--agent-budget", type=int, default=translator.DEFAULT_PAGE_BUDGET,
                    help="token budget of the Translator (and its repair)")
    ap.add_argument("--review-budget", type=int, default=reviewer.DEFAULT_REVIEW_BUDGET,
                    help="token budget of the Reviewer (and its repair)")
    ap.add_argument("--max-tokens", type=int, default=baseline.DEFAULT_MAX_TOKENS,
                    help="output cap per call; leave room for reasoning tokens when thinking is on")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="model calls per agent run")
    ap.add_argument("--dry-run", action="store_true", help="print both stages' first requests and send nothing")
    args = ap.parse_args()

    page = get_page(args.book, args.page)
    previous = earlier_pages(args.book, args.page)
    if args.dry_run:
        dry_run(page, previous, args)
        return

    load_dotenv(ROOT / ".env")
    try:
        llm_config = LLMConfig.from_env()
    except RuntimeError as e:
        sys.exit(str(e))
    try:
        tracing.setup()
    except tracing.TracingError as e:
        sys.exit(str(e))
    meter = Meter(ledger_path=ROOT / "runs" / "usage.jsonl")
    client = LLMClient(llm_config, meter)
    review = None
    with tracing.page_span(book=args.book, page_index=args.page, config="review-smoke"):  # one trace for both stages
        made = translator.run_translator(page, client, previous=previous, flagging=True,
                                         agent_budget=args.agent_budget, max_tokens=args.max_tokens,
                                         max_steps=args.max_steps)
        translator_calls = len(meter.records)
        if made.ok:
            review = reviewer.run_reviewer(page, made.translations, made.flags, client,
                                           review_budget=args.review_budget, max_tokens=args.max_tokens,
                                           max_steps=args.max_steps)

    translator_numbers = stage_numbers(meter.records[:translator_calls], made.attempts["agent"]["stop_reason"],
                                       made.attempts["agent"]["steps"])
    reviewer_numbers = (None if review is None else stage_numbers(
        meter.records[translator_calls:], review.attempts["agent"]["stop_reason"], review.attempts["agent"]["steps"]))
    models = sorted({r.response_model for r in meter.records if r.response_model})
    flagged = {}
    for f in made.flags:
        flagged.setdefault(f["id"], []).append(f["reason"])
    out = {
        "milestone": "M3",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "page_id": page.page_id,
        "provider_host": urlparse(llm_config.base_url).hostname,
        "requested_model": llm_config.model,
        "response_models": models,
        "previous_pages": [{"page_index": p.page_index, "translated": p.translated} for p in previous],
        "regions": [{"id": s.index, "ja": s.ja, "zh": (made.translations or {}).get(s.index),
                     "flags": flagged.get(s.index, [])} for s in page.segments],
        "translator": {"ok": made.ok, "error": made.error, "fallback_reason": made.fallback_reason,
                       "flags": made.flags, **(translator_numbers), "attempts": made.attempts},
        "reviewer": None if review is None else {
            "status": review.status, "verdict": review.verdict, "issues": review.issues, "error": review.error,
            **reviewer_numbers, "attempts": review.attempts},
        "meter": meter.summary(),
    }
    run_dir = ROOT / "runs" / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{datetime.now():%Y%m%d-%H%M%S}-review-{args.book}-{args.page:03d}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model (as returned by the API): {', '.join(models) or '-'}")
    print(f"Earlier pages given to get_previous_context: "
          f"{', '.join(f'{p.page_index}' + ('' if p.translated else ' (not translated)') for p in previous) or 'none'}")
    print()
    print(describe_stage("Translator", translator_numbers))
    if made.fallback_reason:
        print(f"  fell back to baseline: {made.fallback_reason}")
    if made.error:
        print(f"  failed: {made.error}")
    for r in out["regions"]:
        flags = "".join(f"\n      flagged: {reason}" for reason in r["flags"])
        print(f"  [{r['id']}] {r['ja']}\n      {r['zh']}{flags}")
    print(f"  flags: {len(made.flags)}")
    print()
    if review is None:
        print("Reviewer: not run, because the Translator gave no translations")
    else:
        print(describe_stage("Reviewer", reviewer_numbers))
        if review.ok:
            print(f"  verdict: {review.verdict}, {len(review.issues)} issue(s)")
            for i in review.issues:
                print(f"  [{i['id']}] {i['comment']}" + (f"\n      suggestion: {i['suggestion']}" if i["suggestion"] else ""))
        else:
            print(f"  review {review.status}: {review.error}")
    total = meter.summary()
    print(f"\nTotal: {total['calls']} calls, {total['input_tokens']} input / {total['output_tokens']} output tokens, "
          f"cost {cost_text(total['cost_usd'])}")
    print(f"Saved {out_path.relative_to(ROOT)}")
    if not made.ok or review is None or not review.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
