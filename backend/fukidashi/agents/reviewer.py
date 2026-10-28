"""The Reviewer Agent: reads one page's translation in a context of its own and returns a validated verdict.

The context is independent (OUTLINE §3.2): a fresh conversation with its own system prompt and one user message, built
here from plain values (the page, the translations by region id, the flags). It never holds the Translator's messages,
reasoning, tool calls or tool results, and `run_reviewer` has no way to reach them. The page's reference translations
are not read.

It runs through the agent loop with no tools for now (the KB's read-only lookups arrive with the KB), so it already has
the loop's guardrails. The verdict must pass `parse_verdict`; a bad one gets one repair call, as the Translator's
answer does. A review that gives no verdict is returned, never raised: `unusable` when nothing more was sent for a
reason that leaves the translation as it is (a Repair that failed, the Step limit, a cut-off reply), `failed` after a
provider error or a spent budget, where nothing more is sent either (OUTLINE §3.6).

This module imports no LangChain and no database.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from ..datasets.openmantra import Page
from ..llm.client import LLMClient
from ..llm.errors import ProviderError
from ..llm.metering import BudgetExceeded, PageBudget
from ..pipeline import baseline
from .loop import BUDGET, DEFAULT_MAX_STEPS, FINAL, PROVIDER_ERROR, run_agent

DEFAULT_REVIEW_BUDGET = 30_000  # tokens; the Reviewer's own, so a long Translator run cannot leave it without

APPROVE = "approve"
REVISE = "revise"

OK = "ok"  # a valid verdict
UNUSABLE = "unusable"  # no verdict, and the translation stands as it is
FAILED = "failed"  # no verdict, because of a provider error or a spent budget

SYSTEM_PROMPT = """You review the Simplified Chinese (zh-Hans) translation of one page of Japanese manga dialogue. You did not write it, and you do not know how it was made.
You get the page image and its text regions. Each region has an id, a bounding box [x, y, w, h] in pixels, the Japanese text, and its Chinese translation. The translator may have flagged some regions as uncertain, with a reason.

Check every region against the Japanese and the image:
- meaning: nothing is mistranslated;
- omissions and additions: nothing is left out or invented;
- names and terms: they are rendered the same way throughout the page;
- tone: it fits the speaker and the scene;
- natural spoken Chinese that fits a speech bubble: not stiff, and not far longer than the original.
Look closely at the flagged regions: confirm the concern or clear it.

Answer "approve" when the translation can stand as it is, even if you would have chosen other words. Answer "revise" only when at least one region needs a change; then give one issue for each region that needs it, with a comment that says what is wrong and a suggestion with the Chinese you would use. An "approve" may carry issues as notes: the comment says what you noticed, the suggestion is null, and nothing needs to change. Give at most one issue per region.
Write every comment and every suggestion in Simplified Chinese (zh-Hans), not in English or Japanese. Quote the Japanese text only where you must point at it.

Answer with only JSON in this shape:
{"verdict": "approve" or "revise", "issues": [{"id": 0, "comment": "...", "suggestion": "..." or null}]}"""


@dataclass
class ReviewResult:
    verdict: str | None  # APPROVE or REVISE; None unless the review is ok
    issues: list[dict]  # {"id", "comment", "suggestion"}, at most one per region; empty unless the review is ok
    status: str  # OK, UNUSABLE or FAILED
    attempts: dict  # see run_reviewer
    error: str | None = None  # why there is no verdict

    @property
    def ok(self) -> bool:
        return self.status == OK

    @property
    def unusable(self) -> bool:
        return self.status == UNUSABLE

    @property
    def failed(self) -> bool:
        return self.status == FAILED


def run_reviewer(
    page: Page,
    translations: Mapping[int, str],
    flags: Sequence[Mapping],
    client: LLMClient,
    *,
    review_budget: int = DEFAULT_REVIEW_BUDGET,
    max_tokens: int = baseline.DEFAULT_MAX_TOKENS,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> ReviewResult:
    """Review the translations of one page. Never raises for provider errors or bad output: they end up in the result.

    `translations` is region id -> zh-Hans for every region of the page, `flags` the Translator's {"id", "reason"}.
    A `translations` or a flag that does not fit the page is a `ValueError`, raised before any request.

    The attempts record: {"agent": {"stop_reason", "error", "steps", "answer_parse_error"},
                          "repair": null | {"error", "record", "raw", "parse_error"}}
    The agent and its repair share one budget of `review_budget` tokens, the Reviewer's own.
    """
    ids = [s.index for s in page.segments]
    if set(translations) != set(ids):
        raise ValueError(f"translations must have exactly the region ids {ids}, got {sorted(translations)}")
    if unknown := sorted({f["id"] for f in flags} - set(ids)):
        raise ValueError(f"flags for region ids {unknown} that are not on the page {ids}")

    budget = PageBudget(page.page_id, review_budget)
    agent = run_agent(
        client, budget,
        system_prompt=SYSTEM_PROMPT,
        user_message=build_user_message(page, translations, flags),
        max_tokens=max_tokens, json_output=True, max_steps=max_steps,
    )
    record: dict = {
        "agent": {"stop_reason": agent.stop_reason, "error": agent.error, "steps": agent.steps,
                  "answer_parse_error": None},
        "repair": None,
    }
    if agent.stop_reason in (PROVIDER_ERROR, BUDGET):
        return ReviewResult(None, [], FAILED, record, agent.error)  # nothing more is sent (OUTLINE §3.6)
    if agent.stop_reason != FINAL:  # step_limit or incomplete: there is no answer to repair
        return ReviewResult(None, [], UNUSABLE, record, f"{agent.stop_reason}: {agent.error}")

    try:
        verdict, issues = parse_verdict(agent.text, ids)
        return ReviewResult(verdict, issues, OK, record)
    except ValueError as e:
        answer_error = str(e)
    record["agent"]["answer_parse_error"] = answer_error
    try:
        reply = client.chat(
            repair_messages(agent.messages, answer_error, ids),
            budget=budget, max_tokens=max_tokens, json_output=True,
        )
    except BudgetExceeded as e:
        return ReviewResult(None, [], FAILED, record, f"Stopped: {e}")
    except ProviderError as e:
        record["repair"] = {"error": e.description, "record": asdict(e.record) if e.record is not None else None,
                            "raw": None, "parse_error": None}
        return ReviewResult(None, [], FAILED, record, e.description)
    record["repair"] = {"error": None, "record": asdict(reply.record), "raw": reply.text, "parse_error": None}
    try:
        verdict, issues = parse_verdict(reply.text, ids)
        return ReviewResult(verdict, issues, OK, record)
    except ValueError as e:
        record["repair"]["parse_error"] = str(e)
        return ReviewResult(None, [], UNUSABLE, record,
                            f"unparseable: the answer ({answer_error}) and its repair ({e}) could not be used")


def build_user_message(page: Page, translations: Mapping[int, str], flags: Sequence[Mapping]) -> list[dict]:
    """The one user message: the page image, each region's id, box, Japanese text and translation, and the flags.

    Only these values go in. In particular not the page's reference translations, and nothing of how the Translator
    got to its translations.
    """
    image = baseline.build_messages(page)[1]["content"][0]  # the page image, as baseline sends it
    regions = [{"id": s.index, "box": [s.x, s.y, s.w, s.h], "ja": s.ja, "zh": translations[s.index]}
               for s in page.segments]
    text = "Text regions with their translation, one JSON object per line:\n" + _lines(regions)
    if flags:
        text += "\n\nRegions the translator flagged as uncertain, one JSON object per line:\n" + _lines(
            [{"id": f["id"], "reason": f["reason"]} for f in flags])
    else:
        text += "\n\nThe translator flagged no region."
    return [image, {"type": "text", "text": text}]


def _lines(objects: Sequence[dict]) -> str:
    return "\n".join(json.dumps(o, ensure_ascii=False) for o in objects)


def parse_verdict(text: str, ids: Sequence[int]) -> tuple[str, list[dict]]:
    """(verdict, issues) of a Reviewer's answer. Raises ValueError, saying what is wrong, when it cannot be used.

    Valid: the verdict is `approve` or `revise`; every issue is on a region of the page, at most one per region, with a
    comment; `revise` has at least one issue. `approve` may carry issues as notes. A missing `issues` is an empty list.
    """
    start, end = text.find("{"), text.rfind("}")  # tolerate a ```json fence when the provider has no JSON mode
    if start == -1 or end == -1:
        raise ValueError("no JSON object in the response")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("the answer is not a JSON object")
    verdict = data.get("verdict")
    if verdict not in (APPROVE, REVISE):
        raise ValueError(f'"verdict" must be "{APPROVE}" or "{REVISE}", got {verdict!r}')
    raw = data.get("issues")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError('"issues" must be a list')

    issues: list[dict] = []
    seen: set[int] = set()
    problems: list[str] = []
    unknown: set[int] = set()
    duplicates: set[int] = set()
    for n, item in enumerate(raw):
        if not isinstance(item, dict):
            problems.append(f"issue {n} is not an object")
            continue
        region = item.get("id")
        if not isinstance(region, int) or isinstance(region, bool):
            problems.append(f"issue {n} has no integer region id")
            continue
        if region not in ids:
            unknown.add(region)
        if region in seen:
            duplicates.add(region)
        seen.add(region)
        comment, suggestion = item.get("comment"), item.get("suggestion")
        if not isinstance(comment, str) or not comment.strip():
            problems.append(f"the issue for region {region} has an empty comment")
            continue
        if suggestion is not None and not isinstance(suggestion, str):
            problems.append(f"the suggestion for region {region} must be text or null")
            continue
        issues.append({"id": region, "comment": comment.strip(), "suggestion": (suggestion or "").strip() or None})
    if unknown:
        problems.append(f"issues for region ids not on the page {sorted(unknown)}")
    if duplicates:
        problems.append(f"more than one issue for region ids {sorted(duplicates)}")
    if verdict == REVISE and not raw:
        problems.append(f'the verdict "{REVISE}" needs at least one issue')
    if problems:
        raise ValueError("; ".join(problems))
    return verdict, issues


def repair_messages(history: list[dict], parse_error: str, ids: list[int]) -> list[dict]:
    """The messages of the repair request: the agent's `history` as OpenAI-format dicts, then the repair message."""
    return [*history, {"role": "user", "content": _repair_message(parse_error, ids)}]


def _repair_message(parse_error: str, ids: list[int]) -> str:
    return (f"Your answer could not be used: {parse_error}. Answer again with only the JSON. The region ids of this page "
            f"are {ids}: an issue may only be for one of them, at most one issue per region, each with a comment, and "
            f'the verdict "{REVISE}" needs at least one issue: '
            '{"verdict": "approve" or "revise", "issues": [{"id": 0, "comment": "...", "suggestion": "..." or null}]}')
