"""OpenMantra's Chinese references are Traditional; the system outputs Simplified (OUTLINE 4.1, 8)."""

from __future__ import annotations

from functools import lru_cache

from opencc import OpenCC

# t2s converts characters only and leaves the translator's wording alone. tw2sp was rejected: it rewrites it.
OPENCC_CONFIG = "t2s"


@lru_cache(maxsize=1)
def _converter() -> OpenCC:
    return OpenCC(OPENCC_CONFIG)


def to_zh_hans(text_zh_hant: str) -> str:
    return _converter().convert(text_zh_hant)
