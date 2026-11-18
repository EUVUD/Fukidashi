"""Which boxes a rename changes (OUTLINE §8, "Rename → whole-book sync"; SCOPE.md #6). Pure: no database, no model.

A person renames a KB entry's target term or a profile's Chinese name, and the book's boxes that use it follow. `plan_boxes`
decides which boxes. It is given each box's Japanese text and the translation the page displays (the translation of the
page's latest succeeded Run, of any config), the item's keys, and the old and the new rendering:

- **Keys.** An entry's key is its source term. A profile's keys are its name and each of its aliases.
- **A box changes** when both hold: its Japanese text, normalised (NFKC, trimmed), contains one of the keys (the rule
  `KBSnapshot.snippet_items` uses to decide that an item is relevant to a page), and its displayed translation contains the
  old rendering. The match on the translation is exact, on the string as stored. Every occurrence of the old rendering
  becomes the new one (`str.replace`: the text it puts in is not read again, so 雪 → 小雪 turns 雪と雪 into 小雪と小雪).
- **A box is listed, not changed,** when its Japanese contains a key but its translation does not contain the old rendering,
  or it has no translation (its page has no succeeded Run): a person edits it by hand.
- **A box is left alone** when its translation contains the old rendering but its Japanese contains no key, so a short
  rendering inside a longer word (雪 in 雪花) is not touched unless the source is there too. A box with an empty Japanese text
  is skipped.

`rendering` checks the new rendering the way the KB checks a term's target or a name's rendering (`entities.Name`).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError

from .entities import MAX_NAME_CHARS, Name, normalise


class InvalidRendering(ValueError):
    """A rendering the KB would not store: empty, too long for a name, or not one line of text."""


_RENDERING = TypeAdapter(Name)


def rendering(text: object) -> str:
    """`text` as the KB stores a rendering: NFKC, trimmed, 1 to 60 characters, one line (no newline, tab or other control
    character), which is what a term's target and a character's name have to be. Raises `InvalidRendering`, with the
    reason, for anything else, a value that is not text included. `Name` decides; the reason is worded here, because its own
    messages ("Value should have at least 1 item after validation") are for a payload's field, not for a person's input."""
    try:
        return _RENDERING.validate_python(text)
    except ValidationError as exc:
        raise InvalidRendering(f"invalid new rendering: {_why(text, exc)}") from None


def _why(text: object, exc: ValidationError) -> str:
    if not isinstance(text, str):
        return f"expected text, not {type(text).__name__}"
    length = len(normalise(text))
    if length == 0:
        return "it is empty (or only spaces)"
    if length > MAX_NAME_CHARS:
        return f"it has {length} characters, and a name has at most {MAX_NAME_CHARS}"
    return "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())  # a control character: `Name` says which


@dataclass(frozen=True)
class Box:
    """One box of a page as a rename sees it. `text_zh` is the translation the page displays, or None when it displays none
    (the page has no succeeded Run, or its Run has no translation for the box)."""

    page_index: int
    idx: int
    text_ja: str
    text_zh: str | None


@dataclass(frozen=True)
class BoxChange:
    """A box a rename changes (`after` is the new text) or only lists (`after` is None). `before` is the displayed
    translation: None for a box with none."""

    page_index: int
    idx: int
    text_ja: str
    before: str | None
    after: str | None


@dataclass(frozen=True)
class BoxPlan:
    """What a rename does to the boxes, each list in page order and then by `idx`."""

    changes: tuple[BoxChange, ...]
    listed: tuple[BoxChange, ...]


def plan_boxes(boxes: Iterable[Box], keys: Sequence[str], old: str, new: str) -> BoxPlan:
    """Which of `boxes` a rename from `old` to `new` changes, and which it only lists (see the module's docstring).

    `keys` are used as stored: an empty one matches nothing. An empty `old` is contained in no translation, so no box is
    changed (the boxes that hold a key are listed). Raises ValueError when `new` is `old`: nothing would change, and the
    caller refuses that before it asks."""
    if new == old:
        raise ValueError("a rename needs a new rendering that differs from the old one")
    changes: list[BoxChange] = []
    listed: list[BoxChange] = []
    for box in sorted(boxes, key=lambda b: (b.page_index, b.idx)):
        text = normalise(box.text_ja)
        if not text or not any(key and key in text for key in keys):
            continue
        if old and box.text_zh is not None and old in box.text_zh:
            changes.append(BoxChange(box.page_index, box.idx, box.text_ja, box.text_zh, box.text_zh.replace(old, new)))
        else:
            listed.append(BoxChange(box.page_index, box.idx, box.text_ja, box.text_zh, None))
    return BoxPlan(tuple(changes), tuple(listed))
