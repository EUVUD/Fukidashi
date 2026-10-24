"""Read-only access to the OpenMantra annotation file (field notes: docs/openmantra.md)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data" / "open-mantra-dataset"


@dataclass(frozen=True)
class Segment:
    index: int
    x: int
    y: int
    w: int
    h: int
    ja: str
    en: str
    zh_hant: str  # the dataset's Chinese reference is Traditional Chinese


@dataclass(frozen=True)
class Page:
    book_title: str
    page_index: int  # 1-based, as in annotation.json
    image_path: Path
    segments: list[Segment]

    @property
    def page_id(self) -> str:
        return f"{self.book_title}/{self.page_index:03d}"


def load_pages(root: Path = DEFAULT_ROOT) -> list[Page]:
    data = json.loads((root / "annotation.json").read_text(encoding="utf-8"))
    pages = []
    for book in data:
        for p in book["pages"]:
            segments = [
                Segment(i, t["x"], t["y"], t["w"], t["h"], t["text_ja"], t["text_en"], t["text_zh"])
                for i, t in enumerate(p["text"])
            ]
            pages.append(Page(book["book_title"], p["page_index"], root / p["image_paths"]["ja"], segments))
    return pages


def get_page(book_title: str, page_index: int, root: Path = DEFAULT_ROOT) -> Page:
    for page in load_pages(root):
        if page.book_title == book_title and page.page_index == page_index:
            return page
    raise KeyError(f"no page {book_title}/{page_index} in {root / 'annotation.json'}")
