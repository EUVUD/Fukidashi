"""M0 first call: one OpenMantra page (image + Japanese text) -> real model -> JSON translations in zh-Hans.

Usage:
  uv run python scripts/m0_first_call.py --dry-run   # build the request, send nothing
  uv run python scripts/m0_first_call.py             # real call, configured by .env
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))  # no package layout yet; M1 sets one up

import openai  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from app.datasets.openmantra import Page, get_page  # noqa: E402
from app.llm.client import LLMClient, LLMConfig  # noqa: E402
from app.llm.metering import BudgetExceeded, Meter, PageBudget  # noqa: E402

SYSTEM_PROMPT = """You translate Japanese manga dialogue into Simplified Chinese (zh-Hans).
You get one page image and its text regions. Each region has an id, a bounding box [x, y, w, h] in pixels, and the Japanese text.
Use the image for context: who is speaking, their tone, and what is happening. Translate every region into natural spoken Chinese that fits a speech bubble.
Return only JSON in this shape, with one entry per region id:
{"translations": [{"id": 0, "zh": "..."}]}"""


def build_messages(page: Page) -> list[dict]:
    regions = [{"id": s.index, "box": [s.x, s.y, s.w, s.h], "ja": s.ja} for s in page.segments]
    image_b64 = base64.b64encode(page.image_path.read_bytes()).decode("ascii")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": "Text regions, one JSON object per line:\n"
                 + "\n".join(json.dumps(r, ensure_ascii=False) for r in regions)},
            ],
        },
    ]


def parse_translations(text: str, ids: list[int]) -> dict[int, str]:
    # tolerate a ```json fence when the provider has no JSON mode
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the response")
    data = json.loads(text[start : end + 1])
    got = {int(t["id"]): str(t["zh"]) for t in data["translations"]}
    missing = sorted(set(ids) - set(got))
    if missing:
        raise ValueError(f"missing region ids {missing}")
    return got


def call_once(client: LLMClient, messages: list[dict], budget: PageBudget, max_tokens: int):
    try:
        return client.chat(messages, budget=budget, max_tokens=max_tokens, json_output=True)
    except openai.AuthenticationError as e:
        sys.exit(f"Authentication failed ({e.status_code}). Check LLM_API_KEY in .env.")
    except openai.NotFoundError as e:
        sys.exit(f"Not found ({e.status_code}). Check LLM_BASE_URL and LLM_MODEL in .env.\n{e.message}")
    except openai.BadRequestError as e:
        sys.exit(
            f"Bad request: {e.message}\n"
            "If it mentions response_format, set LLM_JSON_MODE=none. "
            "If it mentions max_completion_tokens, set LLM_TOKEN_PARAM=max_completion_tokens. "
            "If it mentions images, this model has no vision support."
        )
    except openai.RateLimitError as e:
        sys.exit(f"Rate limited: {e.message}")
    except openai.APIStatusError as e:
        sys.exit(f"API error {e.status_code}: {e.message}")
    except openai.APIConnectionError as e:
        sys.exit(f"Cannot reach {client.config.base_url}: {e}")


def main() -> None:
    ap = argparse.ArgumentParser(description="M0 first real call on one OpenMantra page.")
    ap.add_argument("--book", default="tojime_no_siora", help="book_title slug")
    ap.add_argument("--page", type=int, default=2, help="page_index (1-based)")
    ap.add_argument("--page-budget", type=int, default=40_000, help="token budget for this page")
    ap.add_argument("--max-tokens", type=int, default=16_000,
                    help="output cap per call; leave room for reasoning tokens when thinking is on")
    ap.add_argument("--dry-run", action="store_true", help="build the request and print it without sending")
    args = ap.parse_args()

    page = get_page(args.book, args.page)
    messages = build_messages(page)
    ids = [s.index for s in page.segments]

    if args.dry_run:
        size_kb = page.image_path.stat().st_size / 1024
        print(f"{page.page_id}: {len(ids)} segments, image {page.image_path.name} ({size_kb:.0f} KB)\n")
        print(SYSTEM_PROMPT + "\n")
        print(messages[1]["content"][1]["text"])
        return

    load_dotenv(ROOT / ".env")
    try:
        config = LLMConfig.from_env()
    except RuntimeError as e:
        sys.exit(str(e))
    meter = Meter(ledger_path=ROOT / "runs" / "usage.jsonl")
    client = LLMClient(config, meter)
    budget = PageBudget(page.page_id, args.page_budget)

    attempts, translations = [], None
    for attempt in (1, 2):  # structured-output failure: retry once, then fail visibly
        try:
            result = call_once(client, messages, budget, args.max_tokens)
        except BudgetExceeded as e:
            print(f"Stopped: {e}")
            break
        attempts.append({"attempt": attempt, "finish_reason": result.finish_reason, "raw": result.text,
                         "record": asdict(result.record)})
        try:
            translations = parse_translations(result.text, ids)
            break
        except (ValueError, KeyError, TypeError) as e:
            attempts[-1]["parse_error"] = str(e)
            print(f"Attempt {attempt}: could not parse the response ({e})")

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
    print("Cost: " + (f"${m['cost_usd']:.4f}" if m["cost_usd"] is not None
                      else "not computed (set LLM_PRICE_INPUT_PER_M / LLM_PRICE_OUTPUT_PER_M in .env)"))
    for seg in out["segments"]:
        print(f"\n[{seg['id']}] ja : {seg['ja']}\n    out: {seg['zh_hans_output']}\n    ref: {seg['zh_hant_reference']}")
    print(f"\nSaved {out_path.relative_to(ROOT)}")
    if not out["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
