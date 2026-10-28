"""The Translator Agent: translates one page with `get_previous_context`, and falls back to `baseline` visibly.

The agent loop (loop.py) runs the model with its one tool and JSON output. The answer is checked with
`parse_translations` (every region id exactly once). A bad answer gets one repair call; when that fails too, or the
agent stopped without an answer (Step limit, a cut-off reply), the page is translated by `baseline` instead and the
attempts record says so (OUTLINE §3.4). After a provider error or a spent budget nothing more is sent: the request may
have been billed, and OUTLINE §3.6 leaves resending to the user.

This module imports no LangChain and no database: the earlier pages are read by the caller before the run starts,
because tool handlers run in LangGraph worker threads.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

from pydantic import BaseModel, Field

from ..datasets.openmantra import Page
from ..llm.client import LLMClient
from ..llm.errors import ProviderError
from ..llm.metering import BudgetExceeded, PageBudget
from ..pipeline import baseline
from .loop import BUDGET, DEFAULT_MAX_STEPS, FINAL, PROVIDER_ERROR, Tool, run_agent

CONFIG_NAME = "translator"
DEFAULT_PAGE_BUDGET = 60_000  # tokens for the agent and its repair; the fallback has its own (baseline's)
MAX_PAGES_BACK = 2

SYSTEM_PROMPT = """You translate Japanese manga dialogue into Simplified Chinese (zh-Hans).
You get one page image and its text regions. Each region has an id, a bounding box [x, y, w, h] in pixels, and the Japanese text.
Use the image for context: who is speaking, their tone, and what is happening. Translate every region into natural spoken Chinese that fits a speech bubble.

You have one tool, get_previous_context. It returns the Japanese text of up to two earlier pages of this book, and their Chinese translation when there is one. Call it when this page alone leaves something open: a sentence that continues from the previous page, who is speaking, how a name or a term was rendered before. Skip it when the page is clear. When you use what it returns, keep names, terms and speech style consistent with it. The same call always returns the same result, so do not repeat one.

When you are done, answer with only JSON in this shape, with one entry per region id of this page:
{"translations": [{"id": 0, "zh": "..."}]}"""

# What parse_translations raises for an answer it cannot use.
PARSE_ERRORS = (ValueError, KeyError, TypeError)


@dataclass(frozen=True)
class PreviousRegion:
    idx: int  # the region id on its page
    ja: str
    zh: str | None  # zh-Hans from the page's latest succeeded Run of this config; None when there is none


@dataclass(frozen=True)
class PreviousPage:
    page_index: int
    regions: tuple[PreviousRegion, ...]
    translated: bool  # whether a succeeded Run of this config exists for the page


class PreviousContextArgs(BaseModel):
    pages_back: int = Field(
        default=1, ge=1, le=MAX_PAGES_BACK,
        description="How many pages back to look: 1 is the previous page, 2 also adds the page before that.",
    )


def previous_context_tool(previous: Sequence[PreviousPage], page_index: int) -> Tool:
    """`get_previous_context` over `previous`, the earlier pages of the book as they were before the run started.

    The handler only returns that data: it reads nothing, so it is safe in a worker thread.
    """

    def get_previous_context(pages_back: int = 1) -> dict:
        pages = [p for p in previous if 0 < page_index - p.page_index <= pages_back]
        if not pages:
            return {"pages": [], "note": "There is no earlier page in this book."}
        return {"pages": [
            {"page_index": p.page_index, "pages_back": page_index - p.page_index, "translated": p.translated,
             "regions": [{"id": r.idx, "ja": r.ja, "zh": r.zh} for r in p.regions]}
            for p in pages
        ]}

    return Tool(
        name="get_previous_context",
        description="Get the Japanese text, and the Chinese translation when there is one, of the pages just before "
                    "this one in the book.",
        args=PreviousContextArgs,
        handler=get_previous_context,
    )


@dataclass
class TranslatorResult:
    translations: dict[int, str] | None  # region id -> zh-Hans; None when the run failed
    attempts: dict  # what `runs.py` stores: see run_translator
    error: str | None = None  # why translations is None

    @property
    def ok(self) -> bool:
        return self.translations is not None

    @property
    def fallback_reason(self) -> str | None:
        fallback = self.attempts.get("fallback")
        return None if fallback is None else fallback["reason"]


def run_translator(
    page: Page,
    client: LLMClient,
    *,
    previous: Sequence[PreviousPage] = (),
    agent_budget: int = DEFAULT_PAGE_BUDGET,
    fallback_budget: int = baseline.DEFAULT_PAGE_BUDGET,
    max_tokens: int = baseline.DEFAULT_MAX_TOKENS,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> TranslatorResult:
    """Translate one page. Never raises for provider errors or bad output: they end up in `error`.

    The attempts record: {"config": "translator",
      "agent": {"stop_reason", "error", "steps", "answer_parse_error"},  # the last: why the final answer did not parse
      "repair": null | {"error", "record", "raw", "parse_error"},        # the one repair call, an attempt
      "fallback": null | {"reason", "attempts"}}                         # `attempts` are run_baseline's
    The agent and its repair share one budget of `agent_budget` tokens; the fallback gets its own.
    """
    ids = [s.index for s in page.segments]
    budget = PageBudget(page.page_id, agent_budget)
    agent = run_agent(
        client, budget,
        system_prompt=SYSTEM_PROMPT,
        user_message=baseline.build_messages(page)[1]["content"],  # the user message baseline sends, unchanged
        tools=[previous_context_tool(previous, page.page_index)],
        max_tokens=max_tokens, json_output=True, max_steps=max_steps,
    )
    record: dict = {
        "config": CONFIG_NAME,
        "agent": {"stop_reason": agent.stop_reason, "error": agent.error, "steps": agent.steps,
                  "answer_parse_error": None},
        "repair": None,
        "fallback": None,
    }
    if agent.stop_reason in (PROVIDER_ERROR, BUDGET):
        return TranslatorResult(None, record, agent.error)  # no fallback: it would send another request (OUTLINE §3.6)

    if agent.stop_reason == FINAL:
        try:
            return TranslatorResult(baseline.parse_translations(agent.text, ids), record)
        except PARSE_ERRORS as e:
            answer_error = str(e)
        record["agent"]["answer_parse_error"] = answer_error
        try:
            reply = client.chat(
                repair_messages(agent.messages, answer_error, ids),
                budget=budget, max_tokens=max_tokens, json_output=True,
            )
        except BudgetExceeded as e:
            return TranslatorResult(None, record, f"Stopped: {e}")
        except ProviderError as e:
            record["repair"] = {"error": e.description, "record": asdict(e.record) if e.record is not None else None,
                                "raw": None, "parse_error": None}
            return TranslatorResult(None, record, e.description)
        record["repair"] = {"error": None, "record": asdict(reply.record), "raw": reply.text, "parse_error": None}
        try:
            return TranslatorResult(baseline.parse_translations(reply.text, ids), record)
        except PARSE_ERRORS as e:
            record["repair"]["parse_error"] = str(e)
            reason = f"unparseable: the answer ({answer_error}) and its repair ({e}) could not be used"
    else:  # step_limit or incomplete
        reason = f"{agent.stop_reason}: {agent.error}"

    fallback = baseline.run_baseline(page, client, PageBudget(page.page_id, fallback_budget), max_tokens=max_tokens)
    record["fallback"] = {"reason": reason, "attempts": fallback.attempts}
    return TranslatorResult(fallback.translations, record, fallback.error)


def repair_messages(history: list[dict], parse_error: str, ids: list[int]) -> list[dict]:
    """The messages of the repair request: the agent's `history` as OpenAI-format dicts, then the repair message.

    The request carries no `tools`, although the history holds tool calls and their results. Whether the provider
    accepts that is what `scripts/m2_tool_smoke.py --repair-check` finds out.
    """
    return [*history, {"role": "user", "content": _repair_message(parse_error, ids)}]


def _repair_message(parse_error: str, ids: list[int]) -> str:
    return (f"Your answer could not be used: {parse_error}. Answer again with only the JSON, with one entry for each "
            f"of the region ids {ids} of this page, each once: " '{"translations": [{"id": 0, "zh": "..."}]}')
