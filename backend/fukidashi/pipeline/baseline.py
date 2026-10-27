"""`baseline`: one call per page, no tools, no KB. Image + Japanese regions in, zh-Hans JSON out.

Shared by scripts/m0_first_call.py and the API. Every call goes through LLMClient, so metering and the
per-page token budget always apply.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass, field

from ..datasets.openmantra import Page
from ..llm.client import LLMClient
from ..llm.errors import ProviderError
from ..llm.metering import BudgetExceeded, PageBudget

CONFIG_NAME = "baseline"
DEFAULT_PAGE_BUDGET = 40_000  # tokens per page
DEFAULT_MAX_TOKENS = 16_000  # output cap per call; leaves room for reasoning tokens when thinking is on

SYSTEM_PROMPT = """You translate Japanese manga dialogue into Simplified Chinese (zh-Hans).
You get one page image and its text regions. Each region has an id, a bounding box [x, y, w, h] in pixels, and the Japanese text.
Use the image for context: who is speaking, their tone, and what is happening. Translate every region into natural spoken Chinese that fits a speech bubble.
Return only JSON in this shape, with one entry per region id:
{"translations": [{"id": 0, "zh": "..."}]}"""


def build_messages(page: Page) -> list[dict]:
    # Only the Japanese text and boxes go into the prompt, never the reference translations.
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
    got: dict[int, str] = {}
    duplicates: set[int] = set()
    for t in data["translations"]:
        region = int(t["id"])
        if region in got:
            duplicates.add(region)
        got[region] = str(t["zh"])
    # the ids must be exactly the page's, each once: an unknown id has no segment to land on
    problems = []
    if missing := sorted(set(ids) - set(got)):
        problems.append(f"missing region ids {missing}")
    if unknown := sorted(set(got) - set(ids)):
        problems.append(f"region ids not on the page {unknown}")
    if duplicates:
        problems.append(f"duplicate region ids {sorted(duplicates)}")
    if problems:
        raise ValueError("; ".join(problems))
    return got


@dataclass
class BaselineResult:
    translations: dict[int, str] | None  # region id -> zh-Hans; None when the run failed
    attempts: list[dict] = field(default_factory=list)
    error: str | None = None  # why translations is None

    @property
    def ok(self) -> bool:
        return self.translations is not None


def run_baseline(
    page: Page,
    client: LLMClient,
    budget: PageBudget,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> BaselineResult:
    """Translate one page. Never raises for provider errors or bad output: they end up in `error`."""
    messages = build_messages(page)
    ids = [s.index for s in page.segments]
    result = BaselineResult(translations=None)

    for attempt in (1, 2):  # structured-output failure (including empty content): retry once, then fail visibly
        try:
            reply = client.chat(messages, budget=budget, max_tokens=max_tokens, json_output=True)
        except BudgetExceeded as e:
            result.error = f"Stopped: {e}"
            return result
        except ProviderError as e:
            result.error = e.description
            return result
        result.attempts.append({"attempt": attempt, "finish_reason": reply.finish_reason, "raw": reply.text,
                                "record": asdict(reply.record)})
        try:
            result.translations = parse_translations(reply.text, ids)
            result.error = None  # the first attempt's parse error stays in `attempts`, not on a good Run
            return result
        except (ValueError, KeyError, TypeError) as e:
            result.attempts[-1]["parse_error"] = str(e)
            result.error = f"Could not parse the model response after {attempt} attempt(s): {e}"
    return result
