"""M0 first call: one OpenMantra page (image + Japanese text) -> real model -> JSON translations in zh-Hans.

Usage:
  uv run python scripts/m0_first_call.py --dry-run   # build the request, send nothing
  uv run python scripts/m0_first_call.py             # real call, configured by .env
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from fukidashi.datasets.openmantra import get_page
from fukidashi.llm.client import LLMClient, LLMConfig
from fukidashi.llm.metering import Meter, PageBudget
from fukidashi.pipeline.baseline import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_PAGE_BUDGET,
    SYSTEM_PROMPT,
    build_messages,
    run_baseline,
)

ROOT = Path(__file__).resolve().parents[1]


def cost_line(summary: dict, config: LLMConfig) -> str:
    """`summary` is `Meter.summary()`. The cost is None for two reasons that can both apply: say which."""
    if summary["cost_usd"] is not None:
        return f"Cost: ${summary['cost_usd']:.4f}"
    reasons = []
    if n := summary["unknown_usage_calls"]:
        reasons.append(f"unknown ({n} call{'s' if n != 1 else ''} returned no usage)")
    if config.price_input_per_m is None or config.price_output_per_m is None:
        reasons.append("not computed (set LLM_PRICE_INPUT_PER_M / LLM_PRICE_OUTPUT_PER_M in .env)")
    return "Cost: " + ("; ".join(reasons) or "unknown")


def main() -> None:
    ap = argparse.ArgumentParser(description="M0 first real call on one OpenMantra page.")
    ap.add_argument("--book", default="tojime_no_siora", help="book_title slug")
    ap.add_argument("--page", type=int, default=2, help="page_index (1-based)")
    ap.add_argument("--page-budget", type=int, default=DEFAULT_PAGE_BUDGET, help="token budget for this page")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="output cap per call; leave room for reasoning tokens when thinking is on")
    ap.add_argument("--dry-run", action="store_true", help="build the request and print it without sending")
    args = ap.parse_args()

    page = get_page(args.book, args.page)

    if args.dry_run:
        size_kb = page.image_path.stat().st_size / 1024
        print(f"{page.page_id}: {len(page.segments)} segments, image {page.image_path.name} ({size_kb:.0f} KB)\n")
        print(SYSTEM_PROMPT + "\n")
        print(build_messages(page)[1]["content"][1]["text"])
        return

    load_dotenv(ROOT / ".env")
    try:
        config = LLMConfig.from_env()
    except RuntimeError as e:
        sys.exit(str(e))
    meter = Meter(ledger_path=ROOT / "runs" / "usage.jsonl")
    client = LLMClient(config, meter)
    budget = PageBudget(page.page_id, args.page_budget)

    result = run_baseline(page, client, budget, max_tokens=args.max_tokens)
    for a in result.attempts:
        if "parse_error" in a:
            print(f"Attempt {a['attempt']}: could not parse the response ({a['parse_error']})")
    if result.error and not result.ok:
        print(f"Failed: {result.error}")
    attempts, translations = result.attempts, result.translations

    out = {
        "milestone": "M0",
        "page_id": page.page_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider_host": urlparse(config.base_url).hostname,
        "requested_model": config.model,
        "response_models": sorted({a["record"]["response_model"] or "?" for a in attempts}),
        "ok": translations is not None,
        "meter": meter.summary(),
        "page_budget": {"limit": budget.limit_tokens, "used": budget.used_tokens, "verified": budget.verified},
        "segments": [
            {"id": s.index, "ja": s.ja, "zh_hans_output": (translations or {}).get(s.index),
             "zh_hant_reference": s.zh_hant, "en_reference": s.en}
            for s in page.segments
        ],
        "attempts": attempts,
    }
    run_dir = ROOT / "runs" / "m0"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{args.book}-{args.page:03d}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    m = out["meter"]
    print(f"Model (as returned by the API): {', '.join(out['response_models'])}")
    print(f"Calls: {m['calls']}  input tokens: {m['input_tokens']}  output tokens: {m['output_tokens']}"
          f"  unknown-usage calls: {m['unknown_usage_calls']}")
    print(cost_line(m, config))
    for seg in out["segments"]:
        print(f"\n[{seg['id']}] ja : {seg['ja']}\n    out: {seg['zh_hans_output']}\n    ref: {seg['zh_hant_reference']}")
    print(f"\nSaved {out_path.relative_to(ROOT)}")
    if not out["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
