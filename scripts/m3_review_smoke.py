"""M3 smoke test: the flagging Translator, then the Reviewer (or one Revision) once, on one OpenMantra page.

`eval/run.py --config full_no_kb` runs this flow as Runs; this script runs the agents one at a time on a single page,
with no Run, so each stage can be seen against the real provider on its own: its numbers, its tool calls and what it
answered. It reads the earlier pages' `translator` Runs from the database (read only), writes no Run and nothing else to
the database, and saves what it did under runs/smoke/.

  --kb             the Translator and the Reviewer run with the Knowledge Base on, over an empty snapshot that is built
                   in memory, for a KB that is not in the database. The script prints the proposals made and the
                   lookups called.
  --revise-check   after the Translator, instead of a review, one Revision with a fixed, made-up Issue on the page's
                   first region (a generic request in Chinese, not taken from the dataset). It shows whether the
                   provider accepts the Revision's user message (the page, the text, and the current translation with
                   the Issues) and returns a usable translation, and which regions it changed. With --kb the Revision
                   has the KB on too, and the Reviewer does not run.
  --dry-run        with either option too: print the first request of each stage and send nothing.

Usage:
  uv run python scripts/m3_review_smoke.py --book tojime_no_siora --page 3 --dry-run   # print both first requests, send nothing
  uv run python scripts/m3_review_smoke.py --book tojime_no_siora --page 3             # real calls, configured by .env
  uv run python scripts/m3_review_smoke.py --book tojime_no_siora --page 3 --kb --dry-run
  uv run python scripts/m3_review_smoke.py --book tojime_no_siora --page 3 --revise-check
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from fukidashi import config, runs, tracing
from fukidashi.agents import reviewer, translator
from fukidashi.agents.loop import DEFAULT_MAX_STEPS
from fukidashi.datasets.openmantra import get_page
from fukidashi.db import connect
from fukidashi.kb import KBSnapshot, ProposalCollector
from fukidashi.kb.entities import KEY_FIELDS
from fukidashi.llm.client import LLMClient, LLMConfig
from fukidashi.llm.metering import CallRecord, Meter
from fukidashi.orchestrator.flow import proposals_in_order
from fukidashi.pipeline import baseline

ROOT = Path(__file__).resolve().parents[1]

SMOKE_KB_ID = 0  # the id of the in-memory KB: the database's ids start at 1, so no KB there has it
# The Issue of --revise-check: made up, generic, and not from the dataset ("please check whether this line's tone fits
# the speaker and the scene").
REVISE_COMMENT = "请检查这一句台词的语气是否符合说话人的身份和当时的场景。"
KB_TOOLS = ("lookup_term", "lookup_character", translator.PROPOSE_ENTRY)


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


def smoke_kb(args) -> tuple[KBSnapshot | None, ProposalCollector | None]:
    """With --kb, an empty snapshot made in memory and the page's collector; the database is not involved."""
    if not args.kb:
        return None, None
    return KBSnapshot(SMOKE_KB_ID, args.book, (), ()), ProposalCollector(SMOKE_KB_ID, args.page)


def fixed_issue(page) -> dict:
    """The one Issue of --revise-check: on the page's first region, with no suggestion."""
    return {"id": page.segments[0].index, "comment": REVISE_COMMENT, "suggestion": None}


def dry_run(page, previous: list[translator.PreviousPage], args) -> None:
    """Print the first request of each stage and send nothing. The second stage's request carries a placeholder
    translation (and, for the Reviewer, a placeholder flag): the real ones exist only after the Translator has run."""
    kb, collector = smoke_kb(args)
    try:
        translator.run_translator(page, DryRunClient(), previous=previous, flagging=True, kb=kb, collector=collector,
                                  agent_budget=args.agent_budget, max_tokens=args.max_tokens, max_steps=args.max_steps)
    except FirstRequest as first:
        translator_request = show_request(first, page)
    placeholder = {s.index: f"(placeholder translation of region {s.index})" for s in page.segments}
    shown = {"translator_first_request": translator_request}
    if args.revise_check:
        try:
            translator.run_translator(
                page, DryRunClient(), previous=previous, flagging=True, kb=kb, collector=collector,
                revision=translator.Revision(placeholder, [fixed_issue(page)]), agent_budget=args.agent_budget,
                max_tokens=args.max_tokens, max_steps=args.max_steps)
        except FirstRequest as first:
            shown["revision_first_request"] = show_request(first, page)
        placeholders = "The Revision's current translation is a placeholder"
    else:
        flags = [{"id": page.segments[0].index, "reason": "(placeholder flag)"}]
        try:
            reviewer.run_reviewer(page, placeholder, flags, DryRunClient(), review_budget=args.review_budget,
                                  max_tokens=args.max_tokens, max_steps=args.max_steps, kb=kb)
        except FirstRequest as first:
            shown["reviewer_first_request"] = show_request(first, page)
        placeholders = "The Reviewer's translation and flag are placeholders"
    print(json.dumps(shown, ensure_ascii=False, indent=2))
    print(f"\n{placeholders}. Page images are replaced. LLM_MODEL and the key are not needed for a dry run.")
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


def tool_calls_of(attempts: dict) -> list[dict]:
    """Every tool call of an agent run, in the order of its steps, as the steps record them."""
    return [c for step in attempts["agent"]["steps"] for c in step["tool_calls"]]


def tools_text(attempts: dict) -> str:
    counts = Counter(c["name"] for c in tool_calls_of(attempts))
    return ", ".join(f"{name} x{n}" for name, n in counts.items()) or "none"


def changed_regions(before: dict[int, str], after: dict[int, str]) -> list[int]:
    return [i for i in sorted(before) if before[i] != after.get(i)]


def kb_report(kb: KBSnapshot, collector: ProposalCollector, stages: list[tuple[str, object]]) -> dict:
    """What --kb found out: the page's proposals, in the order of the calls, and every KB tool call of each stage.
    `stages` are (name, result) pairs of the stages that ran; a Reviewer's result has no proposals."""
    order = [i for _, result in stages for i in getattr(result, "proposal_ids", [])]
    proposals = [{"kind": p.kind, **dict(zip(("source", "target"), (p.payload[f] for f in KEY_FIELDS[p.kind]))),
                  "payload": p.payload, "proposal_id": p.proposal_id}
                 for p in proposals_in_order(collector, order)]
    calls = [{"stage": name, "tool": c["name"], "arguments": c["arguments"], "result": c["result"], "error": c["error"],
              "ran": c["ran"], "duplicate": c["duplicate"]}
             for name, result in stages for c in tool_calls_of(result.attempts) if c["name"] in KB_TOOLS]
    return {"kb_id": kb.kb_id, "entries": len(kb.entries), "profiles": len(kb.profiles), "proposals": proposals,
            "calls": calls}


def main() -> None:
    ap = argparse.ArgumentParser(description="M3 smoke test: the flagging Translator, then the Reviewer (or one "
                                             "Revision), on one page.")
    ap.add_argument("--book", required=True, help="book_title slug")
    ap.add_argument("--page", type=int, required=True, help="page_index (1-based)")
    ap.add_argument("--agent-budget", type=int, default=translator.DEFAULT_PAGE_BUDGET,
                    help="token budget of the Translator (and its repair)")
    ap.add_argument("--review-budget", type=int, default=reviewer.DEFAULT_REVIEW_BUDGET,
                    help="token budget of the Reviewer (and its repair)")
    ap.add_argument("--max-tokens", type=int, default=baseline.DEFAULT_MAX_TOKENS,
                    help="output cap per call; leave room for reasoning tokens when thinking is on")
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="model calls per agent run")
    ap.add_argument("--kb", action="store_true",
                    help="run the agents with the KB on, over an empty snapshot made in memory (nothing is written to "
                         "the database), and show the proposals and the lookups")
    ap.add_argument("--revise-check", action="store_true",
                    help="instead of a review, run one Revision with a fixed, made-up Issue on the first region")
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
    kb, collector = smoke_kb(args)
    review = revision = None
    with tracing.page_span(book=args.book, page_index=args.page, config="review-smoke"):  # one trace for both stages
        made = translator.run_translator(page, client, previous=previous, flagging=True, kb=kb, collector=collector,
                                         agent_budget=args.agent_budget, max_tokens=args.max_tokens,
                                         max_steps=args.max_steps)
        translator_calls = len(meter.records)
        if made.ok and args.revise_check:
            revision = translator.run_translator(
                page, client, previous=previous, flagging=True, kb=kb, collector=collector,
                revision=translator.Revision(made.translations, [fixed_issue(page)]), agent_budget=args.agent_budget,
                max_tokens=args.max_tokens, max_steps=args.max_steps)
        elif made.ok:
            review = reviewer.run_reviewer(page, made.translations, made.flags, client,
                                           review_budget=args.review_budget, max_tokens=args.max_tokens,
                                           max_steps=args.max_steps, kb=kb)

    second = revision if revision is not None else review  # whichever stage followed the Translator
    translator_numbers = stage_numbers(meter.records[:translator_calls], made.attempts["agent"]["stop_reason"],
                                       made.attempts["agent"]["steps"])
    second_numbers = (None if second is None else stage_numbers(
        meter.records[translator_calls:], second.attempts["agent"]["stop_reason"], second.attempts["agent"]["steps"]))
    kb_stages = [("translator", made), *([("revision", revision)] if revision is not None else []),
                 *([("reviewer", review)] if review is not None else [])]
    knowledge_base = None if kb is None else kb_report(kb, collector, kb_stages)
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
            **second_numbers, "attempts": review.attempts},
        "revision": None if revision is None else {
            "ok": revision.ok, "unusable": revision.unusable, "error": revision.error, "issue": fixed_issue(page),
            "translations": revision.translations,
            "changed_regions": changed_regions(made.translations, revision.translations) if revision.ok else None,
            "flags": revision.flags, **second_numbers, "attempts": revision.attempts},
        "knowledge_base": knowledge_base,
        "meter": meter.summary(),
    }
    run_dir = ROOT / "runs" / "smoke"
    run_dir.mkdir(parents=True, exist_ok=True)
    tag = ("revise" if args.revise_check else "review") + ("-kb" if args.kb else "")
    out_path = run_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{tag}-{args.book}-{args.page:03d}.json"
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
    print(f"  tools called: {tools_text(made.attempts)}")
    print()
    if args.revise_check:
        if revision is None:
            print("Revision: not run, because the Translator gave no translations")
        else:
            print(describe_stage("Revision", second_numbers))
            if revision.ok:
                changed = out["revision"]["changed_regions"]
                print(f"  usable: yes; the Issue was on region {out['revision']['issue']['id']}; "
                      f"regions changed: {changed or 'none'}")
                for i in changed:
                    print(f"  [{i}] {made.translations[i]}\n      -> {revision.translations[i]}")
            else:
                print(f"  usable: no ({'unusable' if revision.unusable else 'failed'}): {revision.error}")
            print(f"  tools called: {tools_text(revision.attempts)}")
    elif review is None:
        print("Reviewer: not run, because the Translator gave no translations")
    else:
        print(describe_stage("Reviewer", second_numbers))
        if review.ok:
            print(f"  verdict: {review.verdict}, {len(review.issues)} issue(s)")
            for i in review.issues:
                print(f"  [{i['id']}] {i['comment']}" + (f"\n      suggestion: {i['suggestion']}" if i["suggestion"] else ""))
        else:
            print(f"  review {review.status}: {review.error}")
        print(f"  tools called: {tools_text(review.attempts)}")
    if knowledge_base is not None:
        print(f"\nKnowledge base: in memory, KB {kb.kb_id}, {knowledge_base['entries']} entries and "
              f"{knowledge_base['profiles']} profiles; nothing was written to the database")
        print(f"  proposals made: {len(knowledge_base['proposals'])}")
        for proposal in knowledge_base["proposals"]:
            print(f"  {proposal['kind']}: {proposal['source']} -> {proposal['target']}")
        print(f"  KB tool calls: {len(knowledge_base['calls'])}")
        for c in knowledge_base["calls"]:
            outcome = c["error"] if c["error"] is not None else c["result"] if c["ran"] else "(not run: a duplicate)"
            print(f"  {c['stage']}: {c['tool']}({c['arguments']}) -> {outcome}")
    total = meter.summary()
    print(f"\nTotal: {total['calls']} calls, {total['input_tokens']} input / {total['output_tokens']} output tokens, "
          f"cost {cost_text(total['cost_usd'])}")
    print(f"Saved {out_path.relative_to(ROOT)}")
    if not made.ok or second is None or not second.ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
