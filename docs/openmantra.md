# OpenMantra Data Notes

> 2026-10-22 核对（M0）。来源：[mantra-inc/open-mantra-dataset](https://github.com/mantra-inc/open-mantra-dataset)（depth-1 clone，最后一次提交 2023-03-18），本地路径 `data/open-mantra-dataset/`（gitignored，不进仓库）。
> License：**CC BY-NC 4.0**（`LICENSE.md`）。Citation：Hinami et al., *Towards Fully Automated Manga Translation*, AAAI 2021。

## 文件

- `annotation.json`：标注。README 里写的是 `annotations.json`，以实际文件名为准。
- `images/<book_title>/ja/NNN.jpg`：只有日文原图，共 214 张 JPEG；没有译文版的图片。
- 图片文件名 = `page_index − 1`，补零到 3 位（`page_index` 1 → `000.jpg`）。

## Schema（实测）

| 层级 | 字段 | 说明 |
|---|---|---|
| book | `book_title`、`pages` | `book_title` 是罗马字 slug |
| page | `page_index`、`image_paths`、`frame`、`text` | `page_index` 从 1 开始；`image_paths` 只有 `ja`；**`frame` 在 README 里没写**：panel bounding boxes，`{x, y, w, h}` 的列表，平均每页 5.0 个 |
| segment（`text` 的元素） | `x`、`y`、`w`、`h`、`text_ja`、`text_en`、`text_zh` | 一个文本区（box）；所有 segment 的 en 和 zh 都不为空 |

## Counts

| `book_title` | Pages | Segments | Segments / page | `text_ja` chars | Chars / page | 0-segment pages | Image size |
|---|---|---|---|---|---|---|---|
| `tojime_no_siora` | 46 | 334 | 7.3 | 3,523 | 76.6 | 1 | 830×1170 |
| `balloon_dream` | 38 | 314 | 8.3 | 3,316 | 87.3 | 2 | 830×1170 |
| `tencho_isoro` | 40 | 311 | 7.8 | 2,848 | 71.2 | 0 | 824×1170 |
| `boureisougi` | 36 | 274 | 7.6 | 2,921 | 81.1 | 2 | 828×1170（830、836 各 1 张） |
| `rasetugari` | 54 | 359 | 6.6 | 2,926 | 54.2 | 0 | 830×1170 |
| **Total** | **214** | **1,592** | **7.4** | **15,534** | **72.6** | **5** | |

## 作品标题写法

- `book_title` 是 slug，代码和文档里统一用它当 ID。注意是 `tojime_no_siora`（siora，不是 shiora）。
- 只有 `tojime_no_siora` 第 1 页的第一个 segment 是标题（綴じ眼のシオラ）；其他作品第 1 页的第一句是正文，不是标题。不要自己补日文标题。

## 发现

1. **`text_zh` 是繁体中文。** 统计 16 对常见繁简字，繁体出现 849 次，简体 1 次。如果输出简体却直接对照参考译文，chrF / BLEU 会被字形差异拉低。M0 的决定：系统输出简体，评测前用 OpenCC 把参考译文转成简体（OUTLINE 第 8 节）。
2. `text_en` 的大小写不统一（例如 "bound eye siora"、"i don't owe you!"）。
3. 有 5 页没有 segment，要在 eval protocol 的 exclusion rules 里写明怎么处理。
4. 所有 box 都在图片范围内；有 4 个 `text_ja` 含换行符。
5. segment 在 JSON 里的顺序是否就是阅读顺序：**未核对**。

## Split

dev 2 部 / test 3 部，按作品划分。具体哪几部在 M3 决定。

## Token 粗估（用于定 budget）

图片的 token 算法因 provider 而异（例如 Claude 约为 宽 × 高 / 750，一页 830×1170 约 1.3k tokens）；原文 + box 坐标约 0.35k；system prompt 和说明约 0.6k。按这个算，`baseline` 每页 input 约 2.3k tokens。

**M0 实测（`deepseek-flash`，2026-10-24）：** 一张 830×1170 的图约 591 tokens；`tojime_no_siora/002` 整个请求 922 input tokens、268 output tokens。细节见 `docs/m0.md`；M4 再用 3 页核对。
