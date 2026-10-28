"""The dev and test books (OUTLINE 4.1, 8). Split by book: prompts and tools are tuned on dev, test runs once."""

from __future__ import annotations

DEV_BOOKS = ("tojime_no_siora", "boureisougi")
TEST_BOOKS = ("balloon_dream", "tencho_isoro", "rasetugari")

SPLITS = {**{book: "dev" for book in DEV_BOOKS}, **{book: "test" for book in TEST_BOOKS}}


def split_of(book_title: str) -> str | None:
    """"dev" or "test"; None for a book that is in neither list."""
    return SPLITS.get(book_title)
