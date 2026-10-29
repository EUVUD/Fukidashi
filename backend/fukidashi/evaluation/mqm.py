# SPDX-License-Identifier: CC-BY-SA-4.0
# This file is licensed under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/), not under the
# repository's MIT License. It holds material copied and adapted from GEMBA
# (https://github.com/MicrosoftTranslator/GEMBA, Microsoft Translator: Tom Kocmi, Christian Federmann) at commit
# a7a7eff8e46998447c6cbf09d06affc8f1b99ab4, which is licensed under CC BY-SA 4.0: the GEMBA-MQM prompt template, its
# three examples, its answer parser and its weights. The changes are listed below. See LICENSE.

"""GEMBA-MQM (Kocmi & Federmann 2023): the judge's prompt, the answer parser, the MQM score and the error breakdown.

Adapted from GEMBA, https://github.com/MicrosoftTranslator/GEMBA (Microsoft Translator: Tom Kocmi, Christian
Federmann), at commit a7a7eff8e46998447c6cbf09d06affc8f1b99ab4, file gemba/gemba_mqm_utils.py (unchanged there since
ea90068f9b, "added GEMBA-MQM", 2023-10-25). Paper: "GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4",
WMT 2023, https://aclanthology.org/2023.wmt-1.64.

**Licence.** GEMBA's code and data are released under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).
This module copies GEMBA's system prompt, user template and three few-shot examples verbatim and adapts its answer
parser (`parse_mqm_answer`), its error classes (`parse_error_class`) and its weights, so this module, unlike most of the
repository (MIT; LICENSE lists the files that are not), is licensed under CC BY-SA 4.0. It comes with no warranty.

**Changes from GEMBA:**

- The languages are fixed: "Japanese" source, "Chinese" translation (GEMBA takes both as arguments).
- Only the plain-text answer format that the prompt asks for is parsed. GEMBA's JSON branch (an answer that starts
  with `{"improved translation"`) serves another prompt and is not ported.
- **Unreadable answers.** GEMBA scores every answer: a line that comes before any severity heading is printed ("No
  error level for ...") and dropped, so an answer in another format scores 0, as if it had no error. Here such an
  answer is unreadable (`Unreadable`), and so is an empty one. The scorer retries it once and then marks the segment
  failed (unknown, not 0), where GEMBA would resend it at a higher temperature.
- **Errors.** Each error keeps its severity, its class (GEMBA's `parse_error_class`, read from the text before the span,
  so a span that happens to contain a class's name does not change it) and the span in quotes. GEMBA keeps the
  lower-cased line.

**Unchanged:** the prompt, the examples, the order of the messages, how a line is placed under a severity (an exact
"critical:" / "major:" / "minor:" line opens a section; a line that mentions "no-error" or "no error" is skipped; a line
that mentions "non-translation" is critical wherever it appears), the weights (critical 25, major 5, minor 1), that only
the first five errors count (critical first, then major, then minor), the cap at 25 and the sign: a segment's MQM score
is between -25 and 0, and 0 means no error.

The judge sees only the source segment and the translation: no image, no page context, no reference (OUTLINE 4.4, 4.9).
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

GEMBA = {
    "repository": "https://github.com/MicrosoftTranslator/GEMBA",
    "commit": "a7a7eff8e46998447c6cbf09d06affc8f1b99ab4",
    "file": "gemba/gemba_mqm_utils.py",
    "licence": "CC BY-SA 4.0",
    "licence_url": "https://creativecommons.org/licenses/by-sa/4.0/",
    "paper": "https://aclanthology.org/2023.wmt-1.64",
}

SOURCE_LANG = "Japanese"
TARGET_LANG = "Chinese"

# --- GEMBA's prompt, verbatim (gemba_mqm_utils.py: mqm_fewshot and few_shots) --------------------------------------

SYSTEM_PROMPT = ("You are an annotator for the quality of machine translation. Your task is to identify errors and "
                 "assess the quality of the translation.")

USER_TEMPLATE = """{source_lang} source:
```{source_seg}```
{target_lang} translation:
```{target_seg}```

Based on the source segment and machine translation surrounded with triple backticks, identify error types in the translation and classify them. The categories of errors are: accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling), style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.\nEach error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technically errors, but do not disrupt the flow or hinder comprehension."""  # noqa: E501

FEW_SHOTS = (  # GEMBA's "ende", "encs" and "zhen", in the order TEMPLATE_GEMBA_MQM uses them
    {
        "source_lang": "English",
        "source_seg": "I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.",  # noqa: E501
        "target_lang": "German",
        "target_seg": "Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.",  # noqa: E501
        "answer": """Critical:
no-error
Major:
accuracy/mistranslation - "involvement"
accuracy/omission - "the account holder"
Minor:
fluency/grammar - "wäre"
fluency/register - "dir"
""",
    },
    {
        "source_lang": "English",
        "source_seg": "Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.",  # noqa: E501
        "target_lang": "Czech",
        "target_seg": "Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, přičemž obě partaje se snaží posoudit vyhlídky na úspěch po posledních výměnách v jednáních.",  # noqa: E501
        "answer": """Critical:
no-error
Major:
accuracy/addition - "ve Vídni"
accuracy/omission - "the stop-start"
Minor:
terminology/inappropriate for context - "partaje"
""",
    },
    {
        "source_lang": "Chinese",
        "source_seg": "大众点评乌鲁木齐家居卖场频道为您提供高铁居然之家地址，电话，营业时间等最新商户信息，找装修公司，就上大众点评",  # noqa: E501
        "target_lang": "English",
        "target_seg": "Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.",  # noqa: E501
        "answer": """Critical:
accuracy/addition - "of high-speed rail"
Major:
accuracy/mistranslation - "go to the reviews"
Minor:
style/awkward - "etc.,"
""",
    },
)


def template_messages() -> list[dict]:
    """GEMBA's TEMPLATE_GEMBA_MQM: the system message, the three examples as user / assistant turns, and the user
    template with its placeholders still open."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot in FEW_SHOTS:
        messages.append({"role": "user", "content": USER_TEMPLATE.format(**shot)})
        messages.append({"role": "assistant", "content": shot["answer"]})
    messages.append({"role": "user", "content": USER_TEMPLATE})
    return messages


def template_sha256() -> str:
    """The template's hash: sha256 of `json.dumps(TEMPLATE_GEMBA_MQM, ensure_ascii=False, separators=(",", ":"))` in
    UTF-8. GEMBA's own TEMPLATE_GEMBA_MQM at the pinned commit gives the same value (a test pins it)."""
    blob = json.dumps(template_messages(), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def build_messages(source: str, translation: str) -> list[dict]:
    """One segment's request, as GEMBA's `apply_template` builds it: every turn's content formatted with the data."""
    data = {"source_lang": SOURCE_LANG, "source_seg": source, "target_lang": TARGET_LANG, "target_seg": translation}
    return [{**turn, "content": turn["content"].format(**data)} for turn in template_messages()]


# --- the error typology ---------------------------------------------------------------------------------------------

SEVERITIES = ("critical", "major", "minor")  # GEMBA's order, which is also the order the score counts errors in
WEIGHTS = {"critical": 25, "major": 5, "minor": 1}
COUNTED_ERRORS = 5  # only the first five errors count
SCORE_CAP = 25
SCORE_RULE = ("GEMBA-MQM: the first 5 errors count (critical, then major, then minor), weighted critical 25, major 5, "
              "minor 1; the sum is capped at 25 and negated, so a segment scores between -25 and 0 (no error)")

# The categories the prompt names, written as GEMBA's examples write them ("class/subclass"). The human annotation
# sheet uses exactly these (docs/mqm-annotation.md).
TYPOLOGY = (
    "accuracy/addition", "accuracy/mistranslation", "accuracy/omission", "accuracy/untranslated text",
    "fluency/character encoding", "fluency/grammar", "fluency/inconsistency", "fluency/punctuation",
    "fluency/register", "fluency/spelling",
    "style/awkward",
    "terminology/inappropriate for context", "terminology/inconsistent use",
    "non-translation",
    "other",
)

_HEADINGS = {f"{severity}:": severity for severity in SEVERITIES}


class Unreadable(ValueError):
    """An answer the parser cannot read. The message says why."""


@dataclass(frozen=True)
class MQMError:
    severity: str  # critical, major or minor
    category: str  # GEMBA's class, e.g. "accuracy-mistranslation", "style", "unknown"
    span: str | None  # the quoted part after " - ", without its quotes; None when the line has none
    text: str  # the answer's line, stripped

    def to_dict(self) -> dict:
        return {"severity": self.severity, "category": self.category, "span": self.span, "text": self.text}


def error_class(description: str) -> str:
    """GEMBA's `parse_error_class`: the class of an error from its (lower-cased) description; the last matching
    subclass wins, as there."""
    description = description.lower()
    class_name = "unknown"
    if "accuracy" in description:
        class_name = "accuracy"
        for subclass in ("addition", "mistranslation", "omission", "untranslated text"):
            if subclass in description:
                class_name = f"accuracy-{subclass}"
    elif "fluency" in description:
        class_name = "fluency"
        for subclass in ("character encoding", "grammar", "inconsistency", "punctuation", "register", "spelling"):
            if subclass in description:
                class_name = f"fluency-{subclass}"
    elif "locale convention" in description:
        class_name = "locale convention"
        for subclass in ("currency", "date", "name", "telephone", "time"):
            if subclass in description:
                class_name = f"locale convention-{subclass}"
    elif "style" in description:
        class_name = "style"
    elif "terminology" in description:
        class_name = "terminology"
        for subclass in ("inappropriate", "inconsistent"):
            if subclass in description:
                class_name = f"terminology-{subclass}"
    elif "non-translation" in description:
        class_name = "non-translation"
    elif "other" in description:
        class_name = "other"
    return class_name


def _split(line: str) -> tuple[str, str | None]:
    """(the label before " - ", the span after it without one pair of quotes)."""
    label, separator, rest = line.partition(" - ")
    if not separator:
        return line, None
    span = rest.strip()
    for open_quote, close_quote in (('"', '"'), ("“", "”")):
        if len(span) >= 2 and span.startswith(open_quote) and span.endswith(close_quote):
            span = span[1:-1]
            break
    return label, span


def parse_answer(answer: str) -> tuple[MQMError, ...]:
    """The errors of one answer in GEMBA's text format, in the answer's order. Raises Unreadable for an empty answer
    and for a line GEMBA's parser would drop because no severity heading comes before it."""
    if not answer.strip():
        raise Unreadable("the answer is empty")
    level: str | None = None
    errors: list[MQMError] = []
    for number, raw in enumerate(answer.split("\n"), start=1):
        line = raw.strip()
        lowered = line.lower()
        if "no-error" in lowered or "no error" in lowered or not line:
            continue
        if lowered in _HEADINGS:
            level = _HEADINGS[lowered]
            continue
        if level is None:
            shown = line if len(line) <= 80 else line[:77] + "..."
            raise Unreadable(f"line {number} comes before any severity heading: {shown!r}")
        label, span = _split(line)
        severity = "critical" if "non-translation" in lowered else level
        errors.append(MQMError(severity=severity, category=error_class(label), span=span, text=line))
    return tuple(errors)


def mqm_score(severities: Iterable[str]) -> int:
    """GEMBA's segment score from the errors' severities (rule in SCORE_RULE): 0 for no error, -25 at worst."""
    listed = list(severities)
    unknown = set(listed) - set(SEVERITIES)
    if unknown:
        raise ValueError(f"unknown severities: {sorted(unknown)}")
    total = counted = 0
    for severity in SEVERITIES:
        for _ in range(listed.count(severity)):
            if counted < COUNTED_ERRORS:
                total += WEIGHTS[severity]
                counted += 1
    return -min(total, SCORE_CAP)


def breakdown(errors_per_segment: Sequence[Sequence[MQMError]]) -> dict:
    """Error counts over scored segments: by severity, by category (most frequent first) and by both. Every error the
    parser found counts here, including those past the five that the score counts."""
    by_severity = dict.fromkeys(SEVERITIES, 0)
    by_category: Counter[str] = Counter()
    both: dict[str, dict[str, int]] = {}
    for errors in errors_per_segment:
        for error in errors:
            by_severity[error.severity] += 1
            by_category[error.category] += 1
            both.setdefault(error.category, dict.fromkeys(SEVERITIES, 0))[error.severity] += 1
    order = sorted(by_category, key=lambda c: (-by_category[c], c))
    return {
        "errors": sum(by_severity.values()),
        "segments_with_errors": sum(1 for errors in errors_per_segment if errors),
        "by_severity": by_severity,
        "by_category": {c: by_category[c] for c in order},
        "by_category_and_severity": {c: both[c] for c in order},
    }
