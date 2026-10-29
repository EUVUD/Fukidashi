# MQM annotation guide (the GEMBA-MQM agreement sample)

> **License.** This page is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), not under the repository's MIT License. It quotes and adapts the severity definitions and error categories of the GEMBA-MQM prompt from [GEMBA](https://github.com/MicrosoftTranslator/GEMBA) (Microsoft Translator: Tom Kocmi, Christian Federmann) at commit `a7a7eff8e46998447c6cbf09d06affc8f1b99ab4`, which is licensed under CC BY-SA 4.0. The one-line definitions of the categories and the rules for counting errors are written for this project. See [LICENSE](../LICENSE).

This guide is for the author, who annotates the human sample for the GEMBA-MQM agreement check (OUTLINE §4.5) by hand. The judge (GPT-4o mini) annotates the same items later in the same scheme: GEMBA-MQM's error categories and severities. The sample is drawn with `eval/mqm_sample.py draw`. Its directory under `runs/mqm-human/` holds `sheet.csv`, `key.json` and `README.txt`.

## Before you start

- **Blind.** Do not open `key.json`: it says which config translated each item. The 30 items come from `baseline` and `full`, in random order.
- **First.** Annotate before any judge output on these items exists or is looked at.
- **Only the segment.** You see what the judge sees: the Japanese source and the Chinese translation. There is no page image, no earlier or later bubble and no reference translation. Judge each item as it stands. A reading that the segment alone cannot rule out is not an error. For example, a name rendered in another valid transliteration, or a pronoun the context might explain, is not an error.

## Severities

The definitions are GEMBA-MQM's, quoted from its prompt (GEMBA, CC BY-SA 4.0, github.com/MicrosoftTranslator/GEMBA):

| Severity | Definition |
|---|---|
| `critical` | "Critical errors inhibit comprehension of the text." |
| `major` | "Major errors disrupt the flow, but what the text is trying to say is still understandable." |
| `minor` | "Minor errors are technically errors, but do not disrupt the flow or hinder comprehension." |
| `no-error` | The translation has no error. |

## Categories

Write a category exactly as listed. These are the categories that GEMBA-MQM's prompt names, in the `class/subclass` form of its examples.

| Category | One error is |
|---|---|
| `accuracy/addition` | content in the translation that the source does not have |
| `accuracy/mistranslation` | content that says something other than the source |
| `accuracy/omission` | source content that the translation leaves out |
| `accuracy/untranslated text` | part of the source left in Japanese (kana, or kanji that are not Chinese) |
| `fluency/character encoding` | garbled characters or encoding debris |
| `fluency/grammar` | Chinese that is ungrammatical |
| `fluency/inconsistency` | the same thing said two ways within the segment |
| `fluency/punctuation` | wrong, missing or mismatched punctuation |
| `fluency/register` | the wrong level of formality or tone for the speaker |
| `fluency/spelling` | a wrong character, for example a homophone or a Traditional form in Simplified text |
| `style/awkward` | grammatical but unnatural Chinese that a native speaker would not write |
| `terminology/inappropriate for context` | a term that is wrong for what is meant here |
| `terminology/inconsistent use` | one term rendered two ways within the segment |
| `non-translation` | output that is not a translation of the source at all; always `critical`, as GEMBA-MQM's parser treats it |
| `other` | an error that fits none of the above |

## What counts as one error

- **One error per problem.** Mark one error for each contiguous wrong span, under the one category that fits it best. If one span has two problems, mark the more severe one.
- **Repeats.** If the same mistake appears twice in the segment, mark it twice.
- **The span** is the wrong part of the translation. For an omission it is the part of the source that is missing. Keep it short; it is for reading the results later, and the score does not use it.

## Filling in the sheet

- **The first error** of an item goes on the item's own row: fill `severity`, `category` and `span`.
- **Each further error** gets a new row with the same `item` number. Its `ja` and `translation` cells can stay empty.
- **No error.** Write `no-error` in `severity` and leave `category` and `span` empty.
- **`note`** is free text. It is not checked.
- **Format.** Keep the file a UTF-8 CSV with the same columns.

When you are done, run `uv run python eval/mqm_sample.py check runs/mqm-human/<time>-sample/sheet.csv`. It lists anything missing or wrong. Once the sheet is complete, it prints each item's MQM score and the sheet's sha256.

## The score

Each item gets GEMBA-MQM's score: critical 25, major 5, minor 1. Only the first five errors count (critical first, then major, then minor), the sum is capped at 25, and the result is negated. An item scores between −25 and 0, and 0 means no error. The judge's segments are scored the same way, so the two can be compared item by item.
