# Evaluation protocol

Written 2026-10-30, before any test page is translated. The git tag `exp-frozen` marks the commit that contains this file; the test runs start from that tag. After the tag, no metric, rule or comparison below is added, dropped or re-ranked because of what a result looks like. The reasons are in `OUTLINE.md` §4 and §8.

## Comparisons and metrics

| Comparison | Primary metric | Secondary metrics |
|---|---|---|
| `full` vs `baseline` | COMET (`Unbabel/wmt22-comet-da`) | chrF, BLEU, term accuracy, term consistency, GEMBA-MQM, cost per page, latency (p50 / p95) |
| `full` vs `full_no_kb` | Term accuracy (the primary score: the gold glossary's non-borderline entries) | Term consistency |

- These are the only two comparisons. `full` vs `baseline` measures the whole system; the Translator's tools and the Reviewer are not ablated, only the KB is.
- Term accuracy and term consistency follow `OUTLINE.md` §4.7, as implemented by `eval/terms.py` and `eval/consistency.py` at the tag.
- **The consistency hypothesis (the KB raises term consistency) was formed after the dev runs,** in which `full` and `full_no_kb` had the same primary term accuracy (61/117). It is why consistency is a secondary metric and not a replacement for term accuracy.

## Data

- **Test split:** `balloon_dream`, `tencho_isoro` and `rasetugari` (132 pages and 984 segments, before pages without text are skipped). The dev books (`tojime_no_siora`, `boureisougi`) are where prompts and tools were tuned; the tuning ended before the tag, and the prompts, tools and eval code are those at the tag.
- **Runs:** each of `baseline`, `full_no_kb` and `full` runs once on each test book, one config and one book at a time, with the same translation model. `full` starts every eval run with a new, empty KB. A run that is stopped is reported as stopped.
- **References:** `text_zh_hant` converted with OpenCC `t2s`. The three test glossaries were built from the converted references before any page of those books was translated.

## Exclusions

- Pages without text are skipped.
- Quality metrics cover the common pages: those scored by every config, per book. The pages left out are listed.
- A failed page is listed in the report and not rerun. An eval run is stopped and reported when more than about 10% of its pages fail or 429s repeat; it is not continued.
- GEMBA-MQM skips segments with an empty Japanese text.
- Term consistency is reported as a result for a run only when the extractor's accuracy on the converted references is at least 90%; otherwise it is reported for reference only.

## GEMBA-MQM

- **Judge:** GPT-4o mini, `gpt-4o-mini-2024-07-18`, with the GEMBA-MQM template at the commit pinned below. Every judge call, scoring and consistency extraction included, runs before OpenAI shuts the model down on 2027-01-14.
- **Rule (`OUTLINE.md` §4.5):** a judge passes when Kendall's τ_b with the annotator's segment-level MQM scores over the 30-item sample is at least 0.30, and it marks a major or critical error on at least 60% of the items where the annotator marked one. If the annotator marked none, τ_b decides alone.
- **Outcome:** the sample has one marked error (a minor one) in 30 items, so τ_b decided. GPT-4o mini scored −0.042. The fallback, GPT-5.1 (`gpt-5.1-2025-11-13`), scored 0.283. Both fail, so **MQM is reported for reference only**, and the report says so.
- **How strong that is:** the annotator is an LLM (Claude), not a person, and with one marked error a fail is weak evidence against a judge, not proof that its errors are wrong.

## Statistics

Computed once, in M8, from the finished test runs:

- A paired bootstrap by page: resample the pages common to the two compared configs, pooled over the three books, with replacement; **10,000 resamples, seed 2027**. Recompute every metric on each resample (corpus chrF and BLEU, mean COMET, pooled term accuracy, consistency) and report the 95% percentile interval of the difference, `full` minus the other config.
- Every quality metric of the two comparisons gets one. Cost and latency are described, without an interval. There are no other tests.
- Each config runs once, so an interval covers which pages were drawn, not how much the model differs from run to run. That spread is measured on the dev books and reported next to the results.

## Pinned inputs

| Input | Value |
|---|---|
| Code | the tag `exp-frozen`; each eval run's manifest records its commit, and a dirty tree or another commit is reported as a deviation |
| Translation model | `deepseek-flash` (DeepSeek-V4.1-Flash-1017) |
| Gold glossaries, sha256 | `balloon_dream` `212379c2eb9deb36431aaaeb7e1ff2679b71459a7a5ca8a4285d51252c62d8ab` |
| | `tencho_isoro` `e2a80fb8516db63971aab7eb8d9c0a271992c0aa3d8c5fdef58318e28f9c4779` |
| | `rasetugari` `60c2e5511a75332b6a8d402d618bb04bee15ad660342e31b64cb0653b1142d41` |
| | `tojime_no_siora` `dd1490237c715580ff1bf9b48e8109de4a095941541f646e7b37ef167d702a15` |
| | `boureisougi` `21b0f695f95293727d151ddd9c497657c42f8ca010d708e0e4b5ef898c5a63cc` |
| Agreement sample's labels, sha256 | `c400077e12c677f31b510bff6cb76b846fdd3d9d053487ab5e216974c51d68c2` |
| Judge | `gpt-4o-mini-2024-07-18` (fallback `gpt-5.1-2025-11-13`, failed the check) |
| GEMBA-MQM template | `MicrosoftTranslator/GEMBA` at `a7a7eff8e46998447c6cbf09d06affc8f1b99ab4`, template sha256 `d3b87df3827195db96ef55f0ebe126be3ed019cb1c1e0bca307b05d745bdb3c6` |
| COMET | `Unbabel/wmt22-comet-da`, revision `2760a223ac957f30acfb18c8aa649b01cf1d75f2`, `unbabel-comet` 2.2.7 |
| References | OpenCC `t2s` |

## Limitations the report states

- Term accuracy compares the system's renderings with the reference translator's; consistency asks whether the system keeps one rendering. Consistency is strict: two renderings count as one only when they are the same after normalisation, so a longer form that contains the usual one (a name with a title, say) counts as a different rendering.
- The MQM labels come from an LLM, with one marked error in 30 items.
- References are single and converted from zh-Hant, so every reference-based score runs conservative (`OUTLINE.md` §4.9).
