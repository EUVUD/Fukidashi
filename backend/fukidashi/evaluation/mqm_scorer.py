"""Score a finished eval run with GEMBA-MQM through the judge (OUTLINE 4.4, 4.5, 8).

The judge is a second `LLMClient`, read from the `JUDGE_*` variables (or another prefix, `JUDGE2_*` for the fallback),
with a `Meter` that writes to a ledger of its own (`runs/judge_usage.jsonl`): its calls are the evaluation's cost and
never belong to a Run. The command-line entry point is eval/mqm.py; the logic lives here so that tests can import it.

- **Inputs.** The run's `manifest.json` (it must be finished) and `segments.jsonl`, split on "\\n" only (the harness
  writes U+2028 and U+0085 unescaped). Of a line it reads `book`, `page_index`, `idx`, `run_id`, `ja` and `hyp`, never a
  reference: the judge sees only the Japanese source and the translation (`mqm.build_messages`).
- **Calls.** One per segment whose Japanese text is not empty, in file order, with a `PageBudget` of 8,000 tokens that
  covers both attempts. An answer that is cut off or that the parser cannot read gets one retry (the same request);
  after that the segment is `failed` and has no score. A segment with an empty Japanese text is `skipped`, with no call.
- **Failures.** A `ProviderError` or `BudgetExceeded` leaves here unchanged: the caller writes nothing, and the calls
  already made stay in the judge's ledger.
- **Scores.** The system score is the mean MQM score of the scored segments; it and the breakdown leave the failed
  segments out and say how many there were. Unknown usage stays unknown (`Meter.summary`).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ..llm.client import LLMClient, LLMConfig, LLMNotConfigured
from ..llm.metering import Meter, PageBudget
from . import mqm

if TYPE_CHECKING:
    import httpx

SEGMENT_BUDGET_TOKENS = 8_000  # per segment, both attempts
MAX_TOKENS = 500  # what GEMBA asks for
ATTEMPTS = 2  # an unreadable answer gets one retry
TRANSLATION_PREFIX = "LLM"  # the translation system's variables; the judge must not be its model

_LINE_FIELDS = (("book", str), ("page_index", int), ("idx", int), ("run_id", int), ("ja", str), ("hyp", str))


class Refused(Exception):
    """The run cannot be scored, or the judge cannot be used; the message names the cause, never a variable's value.
    Nothing has been sent and nothing written."""


@dataclass(frozen=True)
class Line:
    book: str
    page_index: int
    idx: int
    run_id: int
    ja: str
    hyp: str

    @property
    def key(self) -> dict:
        return {"book": self.book, "page_index": self.page_index, "idx": self.idx}


@dataclass(frozen=True)
class EvalRun:
    directory: Path
    manifest: dict
    lines: tuple[Line, ...]
    segments_sha256: str  # of the bytes the lines were read from

    @property
    def eval_id(self) -> str:
        return self.manifest["eval_id"]


@dataclass
class SegmentResult:
    line: Line
    status: str  # scored, skipped or failed
    errors: tuple[mqm.MQMError, ...] = ()
    answer: str | None = None  # the answer that was scored, or the last one that could not be read
    attempts: int = 0
    problems: list[str] = field(default_factory=list)  # why each unreadable attempt was unreadable

    @property
    def score(self) -> int | None:
        return mqm.mqm_score(e.severity for e in self.errors) if self.status == "scored" else None

    def to_dict(self) -> dict:
        return {**self.line.key, "run_id": self.line.run_id, "status": self.status,
                "errors": [e.to_dict() for e in self.errors], "score": self.score, "answer": self.answer,
                "attempts": self.attempts, "problems": self.problems}


# --- reading the run ---------------------------------------------------------------------------------------------------

def read_run(run_dir: Path) -> EvalRun:
    """The finished eval run in `run_dir`; raises Refused."""
    if not run_dir.is_dir():
        raise Refused("it is not a directory")
    manifest = _read_manifest(run_dir / "manifest.json")
    path = run_dir / "segments.jsonl"
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise Refused(f"cannot read {path.name}: {e}")
    lines: list[Line] = []
    seen: dict[tuple[int, int, str], int] = {}
    # "\n" only: `splitlines()` also cuts at U+2028, U+2029 and U+0085, which the harness writes unescaped
    for number, raw in enumerate(text.split("\n"), start=1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except ValueError as e:
            raise Refused(f"{path.name} line {number} is not valid JSON: {e}")
        if not isinstance(item, dict):
            raise Refused(f"{path.name} line {number} is not a JSON object")
        for key, kind in _LINE_FIELDS:
            value = item.get(key)
            if not isinstance(value, kind) or isinstance(value, bool):
                raise Refused(f"{path.name} line {number} has no {kind.__name__} `{key}`")
        place = (item["page_index"], item["idx"], item["book"])
        if place in seen:  # a segment judged twice would count twice
            raise Refused(f"{path.name} line {number} repeats page_index {place[0]} idx {place[1]}, which line "
                          f"{seen[place]} has")
        seen[place] = number
        lines.append(Line(item["book"], item["page_index"], item["idx"], item["run_id"], item["ja"], item["hyp"]))
    if not lines:
        raise Refused(f"{path.name} has no segments")
    return EvalRun(run_dir, manifest, tuple(lines), hashlib.sha256(data).hexdigest())


def _read_manifest(path: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise Refused(f"cannot read {path.name}: {e}")
    except ValueError as e:
        raise Refused(f"{path.name} is not valid JSON: {e}")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("eval_id"), str):
        raise Refused(f"{path.name} has no `eval_id`")
    if manifest.get("finished_at") is None:
        raise Refused("the eval is not finished (manifest.json has `finished_at` null)")
    return manifest


# --- the judge -----------------------------------------------------------------------------------------------------------

def bare_model(model: str) -> str:
    """The model without an organisation prefix ("openai/gpt-4o-mini" -> "gpt-4o-mini")."""
    return model.strip().rsplit("/", 1)[-1]


def judge_config(prefix: str, run: EvalRun) -> LLMConfig:
    """The judge's configuration from `<prefix>_*`; raises Refused when it is missing or when its model is the
    translation model (`LLM_MODEL`, or a model the run itself translated with). No message holds a value."""
    if prefix == TRANSLATION_PREFIX:
        raise Refused(f"the judge cannot be the translation system's configuration ({TRANSLATION_PREFIX}_*)")
    try:
        judge = LLMConfig.from_env(prefix)
    except LLMNotConfigured as e:
        raise Refused(f"the judge is not configured: {e}")
    except ValueError:  # float() on a price or a temperature; its message would quote the value
        raise Refused(f"{prefix}_PRICE_INPUT_PER_M, {prefix}_PRICE_OUTPUT_PER_M or {prefix}_TEMPERATURE is set but "
                      "is not a number")
    output_name(judge.model)  # refuses a model that cannot name the output file, before any call
    mine = bare_model(judge.model).lower()
    translation = os.environ.get(f"{TRANSLATION_PREFIX}_MODEL", "")
    if translation.strip() and bare_model(translation).lower() == mine:
        raise Refused(f"the judge's model ({prefix}_MODEL) is the translation model ({TRANSLATION_PREFIX}_MODEL); "
                      "the judge must be another model")
    ran_with = [m for key in ("requested_models", "response_models") for m in run.manifest.get(key) or []
                if isinstance(m, str)]
    if any(bare_model(m).lower() == mine for m in ran_with):
        raise Refused(f"the judge's model ({prefix}_MODEL) is a model this eval run translated with "
                      "(manifest.json requested_models / response_models); the judge must be another model")
    return judge


def make_client(judge: LLMConfig, ledger_path: Path | None, http_client: httpx.Client | None = None) -> LLMClient:
    """The judge's client, with a Meter of its own that writes to `ledger_path`."""
    return LLMClient(judge, Meter(ledger_path=ledger_path), http_client=http_client)


def output_name(model: str) -> str:
    name = bare_model(model)
    if name in ("", ".", "..") or "\\" in name or "\0" in name:
        raise Refused("the judge's model name cannot name the output file")
    return f"mqm-{name}.json"


# --- scoring -------------------------------------------------------------------------------------------------------------

def judge_segment(client: LLMClient, line: Line, eval_id: str) -> SegmentResult:
    """One segment: up to ATTEMPTS calls under one budget. ProviderError and BudgetExceeded leave unchanged."""
    if not line.ja.strip():
        return SegmentResult(line, "skipped")
    budget = PageBudget(f"mqm:{eval_id}:{line.book}:{line.page_index}:{line.idx}", SEGMENT_BUDGET_TOKENS)
    messages = mqm.build_messages(line.ja, line.hyp)
    result = SegmentResult(line, "failed")
    for attempt in range(1, ATTEMPTS + 1):
        reply = client.chat(messages, budget=budget, max_tokens=MAX_TOKENS)
        result.attempts, result.answer = attempt, reply.text
        try:
            if reply.finish_reason != "stop":
                raise mqm.Unreadable(f"the answer was cut off (finish_reason {reply.finish_reason!r})")
            result.errors = mqm.parse_answer(reply.text)
        except mqm.Unreadable as e:
            result.problems.append(f"attempt {attempt}: {e}")
            continue
        result.status = "scored"
        return result
    return result


def score_run(run: EvalRun, client: LLMClient, *, prefix: str) -> dict:
    """The content of mqm-<model>.json: every segment judged with `client`, in file order."""
    started = time.monotonic()
    results = [judge_segment(client, line, run.eval_id) for line in run.lines]
    return build_document(run, client, prefix, results, elapsed_s=time.monotonic() - started)


def build_document(run: EvalRun, client: LLMClient, prefix: str, results: Sequence[SegmentResult], *,
                   elapsed_s: float) -> dict:
    scored = [r for r in results if r.status == "scored"]
    skipped = [r.line.key for r in results if r.status == "skipped"]
    failed = [r.line.key for r in results if r.status == "failed"]
    scores = [r.score for r in scored]
    response_models = sorted({rec.response_model for rec in client.meter.records if rec.response_model})
    config = client.config
    return {
        "eval_id": run.eval_id,
        "book": run.manifest.get("book"),
        "segments_sha256": run.segments_sha256,
        "judge": {
            "prefix": prefix,
            "model": config.model,
            # one string when every response named the same model; a list when they differ; None when none did
            "response_model": (response_models[0] if len(response_models) == 1 else response_models or None),
            "temperature": config.temperature,
            "reasoning_effort": config.reasoning_effort,
            "max_tokens": MAX_TOKENS,
        },
        "gemba": {**mqm.GEMBA, "template_sha256": mqm.template_sha256(), "source_lang": mqm.SOURCE_LANG,
                  "target_lang": mqm.TARGET_LANG},
        "weights": dict(mqm.WEIGHTS),
        "score_rule": mqm.SCORE_RULE,
        "n_segments": len(results),
        "scored": len(scored),
        "skipped": {"count": len(skipped), "segments": skipped},
        "failed": {"count": len(failed), "segments": failed},
        "system_score": sum(scores) / len(scores) if scores else None,
        "breakdown": {"segments": {"scored": len(scored), "failed": len(failed)},
                      **mqm.breakdown([r.errors for r in scored])},
        "segments": [r.to_dict() for r in results],
        "usage": client.meter.summary(),
        "scored_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed_s, 3),
    }


# --- output ----------------------------------------------------------------------------------------------------------------

def write_json(path: Path, data: dict) -> None:
    """Replace `path` in one step: write a temporary file next to it, then rename it over. On failure the temporary
    file is removed and an earlier `path` is left as it was."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def format_summary(document: dict) -> str:
    judge, usage, b = document["judge"], document["usage"], document["breakdown"]
    out = [f"{document['eval_id']}: GEMBA-MQM by {judge['model']} ({judge['prefix']}_*), responses from "
           f"{judge['response_model'] or 'unknown'}"]
    out.append(f"  segments: {document['n_segments']} (scored {document['scored']}, skipped "
               f"{document['skipped']['count']}, failed {document['failed']['count']})")
    score = document["system_score"]
    out.append(f"  system score: {'n/a' if score is None else f'{score:.3f}'} (mean over the scored segments; "
               "0 is no error, -25 the worst)")
    severities = ", ".join(f"{s} {n}" for s, n in b["by_severity"].items())
    top = ", ".join(f"{c} {n}" for c, n in list(b["by_category"].items())[:5]) or "none"
    out.append(f"  errors: {b['errors']} ({severities}); most frequent: {top}")
    cost = usage["cost_usd"]
    cost_text = (f"${cost:.6f}" if cost is not None else
                 f"unknown ({usage['unknown_usage_calls']} call(s) without usage, or no {judge['prefix']}_PRICE_*)")
    out.append(f"  calls: {usage['calls']}, tokens {usage['input_tokens']} in / {usage['output_tokens']} out, "
               f"cost {cost_text}")
    return "\n".join(out)
