"""The Translator Agent: translates one page with `get_previous_context`, and falls back to `baseline` visibly.

The agent loop (loop.py) runs the model with its one tool and JSON output. The answer is checked with
`parse_translations` (every region id exactly once). A bad answer gets one repair call; when that fails too, or the
agent stopped without an answer (Step limit, a cut-off reply), the page is translated by `baseline` instead and the
attempts record says so (OUTLINE §3.4). After a provider error or a spent budget nothing more is sent: the request may
have been billed, and OUTLINE §3.6 leaves resending to the user.

With the `flagging` switch on, the agent also has `flag_uncertain`, which marks regions for the Reviewer to look at
closely (OUTLINE §3.2). The switch is off for the `translator` config, so its requests stay what they were in M2.

Given a `Revision`, the run is a revision: the Reviewer sent the page back, and the user message also carries the
current translation and the Reviewer's Issues. It has the same tools and the same Repair, but no fallback: a revision
that gives no usable answer is `unusable`, and the translation it was meant to replace stands.

This module imports no LangChain and no database: the earlier pages are read by the caller before the run starts,
because tool handlers run in LangGraph worker threads.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Annotated, Any

from pydantic import BaseModel, Field, StringConstraints, field_validator

from ..datasets.openmantra import Page
from ..llm.client import LLMClient
from ..llm.errors import ProviderError
from ..llm.metering import BudgetExceeded, PageBudget
from ..pipeline import baseline
from .loop import BUDGET, DEFAULT_MAX_STEPS, FINAL, PROVIDER_ERROR, Tool, run_agent

CONFIG_NAME = "translator"
DEFAULT_PAGE_BUDGET = 60_000  # tokens for the agent and its repair; the fallback has its own (baseline's)
MAX_PAGES_BACK = 2
MAX_REASON_CHARS = 200

SYSTEM_PROMPT = """You translate Japanese manga dialogue into Simplified Chinese (zh-Hans).
You get one page image and its text regions. Each region has an id, a bounding box [x, y, w, h] in pixels, and the Japanese text.
Use the image for context: who is speaking, their tone, and what is happening. Translate every region into natural spoken Chinese that fits a speech bubble.

You have one tool, get_previous_context. It returns the Japanese text of up to two earlier pages of this book, and their Chinese translation when there is one. Call it when this page alone leaves something open: a sentence that continues from the previous page, who is speaking, how a name or a term was rendered before. Skip it when the page is clear. When you use what it returns, keep names, terms and speech style consistent with it. The same call always returns the same result, so do not repeat one.

When you are done, answer with only JSON in this shape, with one entry per region id of this page:
{"translations": [{"id": 0, "zh": "..."}]}"""

# The system prompt when `flagging` is on: the same, with flag_uncertain described next to get_previous_context.
FLAG_SYSTEM_PROMPT = """You translate Japanese manga dialogue into Simplified Chinese (zh-Hans).
You get one page image and its text regions. Each region has an id, a bounding box [x, y, w, h] in pixels, and the Japanese text.
Use the image for context: who is speaking, their tone, and what is happening. Translate every region into natural spoken Chinese that fits a speech bubble.

You have two tools.
get_previous_context returns the Japanese text of up to two earlier pages of this book, and their Chinese translation when there is one. Call it when this page alone leaves something open: a sentence that continues from the previous page, who is speaking, how a name or a term was rendered before. Skip it when the page is clear. When you use what it returns, keep names, terms and speech style consistent with it. The same call always returns the same result, so do not repeat one.
flag_uncertain marks one region of this page that you are not sure you translated well: an ambiguous line, a name or term you had to guess, a speaker or tone you could not tell, a pun or an allusion. Give the region id and a short reason in one sentence. A reviewer reads your flags and looks closely at the flagged regions, so flag only the regions that need that, each once, and not the ones you are confident about. A flag does not replace a translation: every region still needs one in your answer.

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


class FlagUncertainArgs(BaseModel):
    id: int = Field(description="The id of the region on this page that you are not sure about.")
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_REASON_CHARS)] = Field(
        description=f"Why, in one short sentence (at most {MAX_REASON_CHARS} characters).",
    )

    @field_validator("id", mode="before")
    @classmethod
    def _not_a_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool):  # pydantic would read true as 1; parse_verdict refuses it too
            raise ValueError("Input should be a valid integer, not a boolean")
        return value


class FlagCollector:
    """The flags one run collects through `flag_uncertain`, for the region ids of one page.

    The calls of one reply run in parallel worker threads, so the list is guarded by a lock. An id that is not on the
    page raises, which the loop passes back to the model as the tool's result; nothing is recorded then.
    """

    def __init__(self, ids: Sequence[int]):
        self._ids = frozenset(ids)
        self._lock = threading.Lock()
        self._flags: list[dict] = []

    def tool(self) -> Tool:
        def flag_uncertain(id: int, reason: str) -> dict:
            if id not in self._ids:
                raise ValueError(f"region {id} is not on this page; the region ids are {sorted(self._ids)}")
            with self._lock:
                self._flags.append({"id": id, "reason": reason})
            return {"flagged": id}

        return Tool(
            name="flag_uncertain",
            description="Mark one region of this page that you are not sure you translated well, with a short reason. "
                        "A reviewer will look closely at the flagged regions.",
            args=FlagUncertainArgs,
            handler=flag_uncertain,
        )

    @property
    def flags(self) -> list[dict]:
        """The flags so far as {"id", "reason"}, in region order (calls that run side by side have no order of their own)."""
        with self._lock:
            return sorted(self._flags, key=lambda f: f["id"])  # stable: one region's flags keep the order they came in


@dataclass(frozen=True)
class Revision:
    """What a revision run is given besides the page: the translation the Reviewer sent back, and its Issues."""

    translations: Mapping[int, str]  # region id -> zh-Hans, for every region of the page
    issues: Sequence[Mapping]  # the Reviewer's {"id", "comment", "suggestion"}


@dataclass
class TranslatorResult:
    translations: dict[int, str] | None  # region id -> zh-Hans; None when the run failed
    attempts: dict  # what `runs.py` stores: see run_translator
    error: str | None = None  # why translations is None
    flags: list[dict] = field(default_factory=list)  # {"id", "reason"}; see run_translator
    unusable: bool = False  # a revision that gave no usable answer; see run_translator. Never true for a first pass

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
    flagging: bool = False,
    revision: Revision | None = None,
) -> TranslatorResult:
    """Translate one page. Never raises for provider errors or bad output: they end up in `error`.

    The attempts record: {"config": "translator",
      "agent": {"stop_reason", "error", "steps", "answer_parse_error"},  # the last: why the final answer did not parse
      "repair": null | {"error", "record", "raw", "parse_error"},        # the one repair call, an attempt
      "fallback": null | {"reason", "attempts"}}                         # `attempts` are run_baseline's
    The agent and its repair share one budget of `agent_budget` tokens; the fallback gets its own.

    With `flagging` on, the agent also has `flag_uncertain` (and `FLAG_SYSTEM_PROMPT` says so), and `flags` holds what
    it flagged: {"id", "reason"}, in region order. Flags count only when the page's answer is the agent's or its repair's;
    after a fallback, or a failed run, `flags` is empty (the agent's steps still show the calls). With it off, which
    is the default, the requests are the M2 `translator`'s.

    With a `revision`, the user message is `baseline`'s followed by one more text part: the current translation and
    the Issues. The record, the tools and the Repair are the same, and so is the budget (`agent_budget`, its own), but
    there is no fallback (`fallback_budget` is not used): when the agent stops with the Step limit or a cut-off reply,
    or the answer and its Repair stay unparseable, the result is `unusable` (no translations, `error` says why).
    A `revision` that does not fit the page is a `ValueError`, raised before any request.
    """
    ids = [s.index for s in page.segments]
    if revision is not None:
        _check_revision(revision, ids)
    budget = PageBudget(page.page_id, agent_budget)
    tools = [previous_context_tool(previous, page.page_index)]
    collector = FlagCollector(ids) if flagging else None
    if collector is not None:
        tools.append(collector.tool())
    agent = run_agent(
        client, budget,
        system_prompt=FLAG_SYSTEM_PROMPT if flagging else SYSTEM_PROMPT,
        user_message=_user_message(page, revision),
        tools=tools,
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
            return TranslatorResult(baseline.parse_translations(agent.text, ids), record,
                                    flags=collector.flags if collector is not None else [])
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
            return TranslatorResult(baseline.parse_translations(reply.text, ids), record,
                                    flags=collector.flags if collector is not None else [])
        except PARSE_ERRORS as e:
            record["repair"]["parse_error"] = str(e)
            reason = f"unparseable: the answer ({answer_error}) and its repair ({e}) could not be used"
    else:  # step_limit or incomplete
        reason = f"{agent.stop_reason}: {agent.error}"

    if revision is not None:  # no fallback: the translation the revision was meant to replace stands
        return TranslatorResult(None, record, reason, unusable=True)
    fallback = baseline.run_baseline(page, client, PageBudget(page.page_id, fallback_budget), max_tokens=max_tokens)
    record["fallback"] = {"reason": reason, "attempts": fallback.attempts}
    return TranslatorResult(fallback.translations, record, fallback.error)


def _user_message(page: Page, revision: Revision | None) -> list[dict]:
    """The user message baseline sends, unchanged; for a revision, one more text part after it."""
    content = baseline.build_messages(page)[1]["content"]
    if revision is None:
        return content
    return [*content, {"type": "text", "text": revision_text(revision)}]


def revision_text(revision: Revision) -> str:
    """The text part of a revision's user message: the current translation, the Issues, and what to do with them.

    Only the Reviewer's Issues (comment and suggestion) and the translation go in, nothing of how either was made.
    """
    current = "\n".join(json.dumps({"id": i, "zh": zh}, ensure_ascii=False)
                        for i, zh in sorted(revision.translations.items()))
    issues = "\n".join(
        json.dumps({"id": i["id"], "comment": i["comment"], "suggestion": i.get("suggestion")}, ensure_ascii=False)
        for i in revision.issues)
    return (
        "A reviewer has read your translation of this page and sent it back for revision.\n\n"
        f"Current translation, one JSON object per line:\n{current}\n\n"
        "The reviewer's issues, one JSON object per line (the comments and suggestions are in Simplified Chinese):\n"
        f"{issues}\n\n"
        "Revise the translation. Fix each region that has an issue, as its comment says. A suggestion is a proposal: "
        "use it, or write something better. Leave the other regions as they are, unless keeping names, terms and "
        "tone consistent needs a change there. Answer with the JSON in the same shape as before, with one entry for "
        "every region id of this page, the changed and the unchanged ones."
    )


def _check_revision(revision: Revision, ids: list[int]) -> None:
    if set(revision.translations) != set(ids):
        raise ValueError(f"the revision's translations must have exactly the region ids {ids}, "
                         f"got {sorted(revision.translations)}")
    if unknown := sorted({i["id"] for i in revision.issues} - set(ids)):
        raise ValueError(f"issues for region ids {unknown} that are not on the page {ids}")


def repair_messages(history: list[dict], parse_error: str, ids: list[int]) -> list[dict]:
    """The messages of the repair request: the agent's `history` as OpenAI-format dicts, then the repair message.

    The request carries no `tools`, although the history holds tool calls and their results. Whether the provider
    accepts that is what `scripts/m2_tool_smoke.py --repair-check` finds out.
    """
    return [*history, {"role": "user", "content": _repair_message(parse_error, ids)}]


def _repair_message(parse_error: str, ids: list[int]) -> str:
    return (f"Your answer could not be used: {parse_error}. Answer again with only the JSON, with one entry for each "
            f"of the region ids {ids} of this page, each once: " '{"translations": [{"id": 0, "zh": "..."}]}')
