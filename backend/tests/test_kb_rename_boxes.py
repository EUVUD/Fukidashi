"""Which boxes a rename changes (task 042; OUTLINE §8, "Rename → whole-book sync"): the pure rule of `kb/rename.py`, over plain
values: no database, no model. The writes that follow from it are tested in `test_kb_rename.py` and `test_kb_rename_api.py`."""

from __future__ import annotations

import subprocess
import sys

import pytest

from fukidashi.kb.entities import MAX_NAME_CHARS
from fukidashi.kb.rename import Box, BoxChange, InvalidRendering, plan_boxes, rendering


def box(page: int, idx: int, ja: str, zh: str | None) -> Box:
    return Box(page, idx, ja, zh)


def change(page: int, idx: int, ja: str, before: str | None, after: str | None) -> BoxChange:
    return BoxChange(page, idx, ja, before, after)


# --- a box changes when its Japanese holds a key and its translation holds the old rendering ---


def test_a_key_in_the_japanese_and_the_old_rendering_in_the_translation_change_the_box():
    plan = plan_boxes([box(1, 0, "結界を張る！", "布下结界！")], ("結界",), "结界", "屏障")

    assert plan.changes == (change(1, 0, "結界を張る！", "布下结界！", "布下屏障！"),)
    assert plan.listed == ()


def test_a_key_in_the_japanese_only_is_listed_and_not_changed():
    plan = plan_boxes([box(1, 2, "結界？", "什么？")], ("結界",), "结界", "屏障")

    assert plan.changes == ()
    assert plan.listed == (change(1, 2, "結界？", "什么？", None),)  # `before` is the text it shows, `after` is null


def test_the_old_rendering_without_a_key_is_never_touched():
    boxes = [box(1, 2, "バリアが消えた", "结界消失了")]

    plan = plan_boxes(boxes, ("結界",), "结界", "屏障")

    assert plan.changes == () and plan.listed == ()  # not changed, and not listed either


def test_a_short_rendering_inside_a_longer_word_is_left_alone_unless_the_source_is_there_too():
    """雪 renders the term 雪; 雪花 is another word, whose Japanese (結晶) holds no 雪."""
    boxes = [
        box(1, 0, "雪が降る", "下雪了"),
        box(1, 1, "結晶がきれい", "雪花真美"),
        box(1, 2, "雪の結晶", "雪的雪花"),  # the source is here too: this one is the term's, whatever the word around it
    ]

    plan = plan_boxes(boxes, ("雪",), "雪", "冰雪")

    assert [(c.idx, c.after) for c in plan.changes] == [(0, "下冰雪了"), (2, "冰雪的冰雪花")]
    assert plan.listed == ()
    assert 1 not in {c.idx for c in (*plan.changes, *plan.listed)}  # 雪花 stayed, as the Japanese says it is another word


def test_a_profile_is_matched_through_its_name_and_through_each_alias():
    boxes = [
        box(1, 0, "ユキ、おはよう", "小雪，早上好"),  # the name
        box(1, 1, "ゆきが笑う", "小雪笑了"),  # an alias
        box(1, 2, "雪ちゃん", "小雪酱"),  # another alias
        box(1, 3, "ハルが笑う", "小雪走了"),  # holds the rendering, but no name of the profile
    ]

    plan = plan_boxes(boxes, ("ユキ", "ゆき", "雪ちゃん"), "小雪", "雪乃")

    assert [(c.idx, c.after) for c in plan.changes] == [(0, "雪乃，早上好"), (1, "雪乃笑了"), (2, "雪乃酱")]
    assert plan.listed == ()


def test_a_box_whose_page_has_no_succeeded_run_is_listed_with_no_before():
    plan = plan_boxes([box(4, 0, "結界とユキ", None)], ("結界",), "结界", "屏障")

    assert plan.changes == ()
    assert plan.listed == (change(4, 0, "結界とユキ", None, None),)


def test_every_occurrence_in_one_box_is_replaced():
    plan = plan_boxes([box(2, 0, "結界は結界だ", "结界就是结界，结界")], ("結界",), "结界", "屏障")

    assert plan.changes == (change(2, 0, "結界は結界だ", "结界就是结界，结界", "屏障就是屏障，屏障"),)


def test_a_new_rendering_that_holds_the_old_one_replaces_each_occurrence_once():
    plan = plan_boxes([box(1, 0, "雪と雪", "雪和雪"), box(1, 1, "雪", "雪")], ("雪",), "雪", "小雪")

    assert [c.after for c in plan.changes] == ["小雪和小雪", "小雪"]  # not 小小雪: what is put in is not read again


def test_a_new_rendering_with_the_old_one_around_it_is_not_replaced_twice():
    plan = plan_boxes([box(1, 0, "ab", "aab")], ("ab",), "a", "aa")

    assert plan.changes[0].after == "aaaab"  # each of the two `a` becomes `aa`; the result is not scanned again


def test_the_japanese_is_matched_after_nfkc_and_trimming():
    boxes = [
        box(1, 0, "ＡＢＣ城", "ABC城堡"),  # full-width Latin letters
        box(1, 1, "ﾕｷが笑う", "小雪笑了"),  # half-width katakana: NFKC makes it ユキ
        box(1, 2, "  ユキ  ", "小雪"),  # spaces around the text
    ]

    abc = plan_boxes(boxes, ("ABC",), "ABC", "XYZ")
    yuki = plan_boxes(boxes, ("ユキ",), "小雪", "雪乃")

    assert [c.idx for c in abc.changes] == [0] and abc.changes[0].after == "XYZ城堡"
    assert [c.idx for c in yuki.changes] == [1, 2] and yuki.listed == ()


def test_the_translation_is_matched_exactly_as_stored():
    """Only the Japanese is normalised: a translation that holds the old rendering in another form does not hold it."""
    boxes = [box(1, 0, "ＡＢＣ", "ＡＢＣ城"), box(1, 1, "ＡＢＣ", "abc城"), box(1, 2, "ＡＢＣ", "ABC城")]

    plan = plan_boxes(boxes, ("ABC",), "ＡＢＣ", "XYZ")  # the old rendering, full-width as it is stored

    assert [(c.idx, c.after) for c in plan.changes] == [(0, "XYZ城")]
    assert [c.idx for c in plan.listed] == [1, 2]  # `abc` and `ABC` are not `ＡＢＣ`


# --- order, and what is skipped ---


def test_boxes_come_in_page_order_and_then_by_idx_whatever_the_order_they_were_given_in():
    shuffled = [box(3, 1, "結界", "结界"), box(1, 2, "結界", "结界"), box(3, 0, "結界", "x"), box(1, 0, "結界", "结界"),
                box(2, 5, "結界", None)]

    plan = plan_boxes(shuffled, ("結界",), "结界", "屏障")

    assert [(c.page_index, c.idx) for c in plan.changes] == [(1, 0), (1, 2), (3, 1)]
    assert [(c.page_index, c.idx) for c in plan.listed] == [(2, 5), (3, 0)]


def test_a_box_with_an_empty_japanese_text_is_skipped():
    boxes = [box(1, 0, "", "结界"), box(1, 1, "   ", "结界"), box(1, 2, "　", "结界")]

    plan = plan_boxes(boxes, ("結界",), "结界", "屏障")

    assert plan.changes == () and plan.listed == ()


def test_an_empty_key_matches_nothing():
    """`"" in text` is true of every text: an empty key (a row somebody wrote by hand) must not make every box relevant."""
    plan = plan_boxes([box(1, 0, "結界", "结界"), box(1, 1, "バリア", "结界")], ("", "結界"), "结界", "屏障")

    assert [c.idx for c in plan.changes] == [0] and plan.listed == ()
    assert plan_boxes([box(1, 0, "結界", "结界")], ("",), "结界", "屏障") == plan_boxes([], (), "结界", "屏障")


def test_an_empty_old_rendering_changes_nothing_and_lists_the_boxes_with_a_key():
    """`str.replace("", x)` would put `x` between every two characters: an empty rendering is contained in no translation."""
    plan = plan_boxes([box(1, 0, "結界", "布下结界"), box(1, 1, "バリア", "布下")], ("結界",), "", "屏障")

    assert plan.changes == ()
    assert plan.listed == (change(1, 0, "結界", "布下结界", None),)


def test_no_box_at_all_is_a_plan_with_nothing_in_it():
    plan = plan_boxes([], ("結界",), "结界", "屏障")

    assert (plan.changes, plan.listed) == ((), ())


def test_a_rename_to_the_same_rendering_is_refused_before_it_plans():
    with pytest.raises(ValueError, match="differs"):
        plan_boxes([box(1, 0, "結界", "结界")], ("結界",), "结界", "结界")


def test_the_plan_does_not_read_a_translation_that_the_japanese_does_not_point_to():
    """A translation is only looked at when the Japanese holds a key: with none, `text_zh` may be anything."""
    plan = plan_boxes([box(1, 0, "バリア", "结界结界结界")], ("結界",), "结界", "屏障")

    assert plan.changes == () and plan.listed == ()


# --- the new rendering is validated as the KB's names are ---


@pytest.mark.parametrize("text, stored", [
    ("屏障", "屏障"),
    ("  屏障  ", "屏障"),  # trimmed
    ("　屏障　", "屏障"),  # an ideographic space is trimmed too
    ("ＡＢＣ", "ABC"),  # NFKC
    ("ﾕｷ", "ユキ"),
    ("雪" * MAX_NAME_CHARS, "雪" * MAX_NAME_CHARS),  # the longest a name may be
    ("小雪（正式）", "小雪(正式)"),  # NFKC turns the full-width brackets into ASCII ones, as it does for every stored name
])
def test_a_rendering_is_stored_nfkc_and_trimmed(text, stored):
    assert rendering(text) == stored


@pytest.mark.parametrize("text, why", [
    ("", "at least 1"),
    ("   ", "at least 1"),
    ("　", "at least 1"),
    ("雪" * (MAX_NAME_CHARS + 1), "at most 60"),
    ("a\nb", "one line"),
    ("a\tb", "one line"),
    ("a\rb", "one line"),
    ("a\x00b", "one line"),
    ("a b", "one line"),
    ("a b", "one line"),
    (None, "valid string"),
    (5, "valid string"),
    (["屏障"], "valid string"),
], ids=["empty", "spaces", "ideographic-space", "too-long", "newline", "tab", "carriage-return", "nul", "line-separator",
        "paragraph-separator", "null", "number", "list"])
def test_a_rendering_the_kb_would_not_store_is_refused_with_the_reason(text, why):
    with pytest.raises(InvalidRendering, match=why) as refused:
        rendering(text)

    assert str(refused.value).startswith("invalid new rendering: ")
    assert "Value error" not in str(refused.value)  # pydantic's own prefix is not shown


def test_the_rules_are_the_ones_a_proposal_goes_through():
    """`rendering` is `entities.Name`, so whatever a term or a name may be, a rendering may be, and nothing else."""
    from fukidashi.kb.entities import make_proposal

    for text in ("屏障", "  屏障 ", "ＡＢＣ", "雪" * 60, "", "a\nb", "雪" * 61, "a b"):
        try:
            stored = rendering(text)
        except InvalidRendering:
            with pytest.raises(ValueError):
                make_proposal(1, 1, "term", {"source_term": "結界", "target_term": text}, "translator")
        else:
            assert make_proposal(1, 1, "term", {"source_term": "結界", "target_term": text}, "translator").payload[
                "target_term"] == stored


def test_the_rule_imports_no_database_and_no_model():
    """It is pure: a fresh interpreter that imports it has not loaded the store, sqlite3, the LLM client or LangChain."""
    code = ("import sys, fukidashi.kb.rename; print(sorted(n for n in ('fukidashi.store', 'fukidashi.db', 'sqlite3', "
            "'fukidashi.llm.client', 'openai', 'langchain', 'langchain_core') if n in sys.modules))")

    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert done.stdout.strip() == "[]"
