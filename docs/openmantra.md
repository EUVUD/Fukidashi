# OpenMantra Data Notes

> Checked 2026-10-22 (M0). Source: [mantra-inc/open-mantra-dataset](https://github.com/mantra-inc/open-mantra-dataset) (depth-1 clone, last commit 2023-03-18), local path `data/open-mantra-dataset/` (gitignored, not part of this repository).
> License: **CC BY-NC 4.0** (`LICENSE.md`). Citation: Hinami et al., *Towards Fully Automated Manga Translation*, AAAI 2021.

## Files

- `annotation.json`: the annotations. The README calls it `annotations.json`; the actual file name wins.
- `images/<book_title>/ja/NNN.jpg`: Japanese originals only, 214 JPEGs; there are no images of the translations.
- Image file name = `page_index − 1`, zero-padded to 3 digits (`page_index` 1 → `000.jpg`).

## Schema (as found)

| Level | Fields | Notes |
|---|---|---|
| book | `book_title`, `pages` | `book_title` is a romanized slug |
| page | `page_index`, `image_paths`, `frame`, `text` | `page_index` starts at 1; `image_paths` has only `ja`; **`frame` is not in the README**: panel bounding boxes, a list of `{x, y, w, h}`, 5.0 per page on average |
| segment (an element of `text`) | `x`, `y`, `w`, `h`, `text_ja`, `text_en`, `text_zh` | One text region (box); every segment has non-empty en and zh |

## Counts

| `book_title` | Pages | Segments | Segments / page | `text_ja` chars | Chars / page | Pages without segments | Image size |
|---|---|---|---|---|---|---|---|
| `tojime_no_siora` | 46 | 334 | 7.3 | 3,523 | 76.6 | 1 | 830×1170 |
| `balloon_dream` | 38 | 314 | 8.3 | 3,316 | 87.3 | 2 | 830×1170 |
| `tencho_isoro` | 40 | 311 | 7.8 | 2,848 | 71.2 | 0 | 824×1170 |
| `boureisougi` | 36 | 274 | 7.6 | 2,921 | 81.1 | 2 | 828×1170 (one each at 830 and 836) |
| `rasetugari` | 54 | 359 | 6.6 | 2,926 | 54.2 | 0 | 830×1170 |
| **Total** | **214** | **1,592** | **7.4** | **15,534** | **72.6** | **5** | |

## Book titles

- `book_title` is a slug, used as the ID in code and docs. Note `tojime_no_siora` (siora, not shiora).
- Only `tojime_no_siora`'s first segment on page 1 is the title (綴じ眼のシオラ); the other books open with dialogue, not a title. Do not invent Japanese titles.

## Findings

1. **`text_zh` is Traditional Chinese.** Counting 16 common Traditional / Simplified character pairs gives 849 Traditional occurrences and 1 Simplified. Simplified output compared directly against these references would lose chrF / BLEU to script differences alone. M0 decision: the system outputs Simplified Chinese, and OpenCC converts the references to Simplified before evaluation (OUTLINE §8).
2. `text_en` casing is inconsistent (for example "bound eye siora", "i don't owe you!").
3. 5 pages have no segments; the eval protocol skips them (`eval/PROTOCOL.md`).
4. Every box lies inside its image; 4 `text_ja` values contain a newline.
5. Whether the order of segments in the JSON is the reading order: **not checked**.
6. **References that do not line up with their segment** (found while building the test glossaries, M5). On some pages two neighbouring segments carry each other's reference, or the references are shifted by one. Checked against the Japanese: `balloon_dream` page 4 (segments 6 and 7) and page 5 (6 and 7), `rasetugari` page 5 (5 and 6) and page 14 (1 and 2) are swapped pairs; `tencho_isoro` page 30 (segments 1 to 3) is shifted. Page numbers are `page_index`. Nobody has checked every page, so more may exist. A reference-based score on such a segment compares the translation with another bubble's text; it adds noise to every config alike, and term accuracy skips a segment whose reference does not use the term.
7. **Typos in `text_ja`.** `rasetugari` writes 璋気 (pages 4, 10 and 29) and 境気 (page 46) where 瘴気 is meant, and 証めるな (page 45, segment 2) where 舐めるな is meant. The system sees the text as written.

## Split

2 dev books / 3 test books, split by book (decided 2026-10-28, OUTLINE §8):

| Split | Books | Pages | Segments |
|---|---|---|---|
| Dev | `tojime_no_siora`, `boureisougi` | 82 | 608 |
| Test | `balloon_dream`, `tencho_isoro`, `rasetugari` | 132 | 984 |

`tojime_no_siora` is a dev book because M0–M2 already translated some of its pages. `boureisougi` is the smallest of the other four, which leaves the most data for the test split. 3 of the 5 pages without segments are in the dev split (`tojime_no_siora` 1, `boureisougi` 2), 2 in the test split (`balloon_dream`).

## Token estimate (for the budget)

Image token counts depend on the provider (Claude, for example, uses about width × height / 750, so about 1.3k tokens for an 830×1170 page); source text and box coordinates add about 0.35k, and the system prompt about 0.6k. By that estimate, `baseline` input is about 2.3k tokens per page.

**Measured in M0 (`deepseek-flash`, 2026-10-24):** an 830×1170 image costs about 591 tokens; the whole `tojime_no_siora/002` request took 922 input and 268 output tokens. Details in `docs/m0.md`; M4 checks again on 3 pages.
