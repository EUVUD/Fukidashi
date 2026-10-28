"""Every request the agents send over the task-014 scenarios, collected as they went out (a helper for test_kb_seams.py).

The scenarios run the real `LLMClient`, the real openai SDK and LangChain's `create_agent` against a scripted provider
(`mock_llm_client`), with the Knowledge Base off and on: the M2 `translator`, a Revision, the Reviewer, and the
Orchestrator over a first pass, a review, one send-back, two send-backs and a Fallback. Only APIs that `origin/main`
had before task 015 are used, so the same module runs on that tree: the digests in `test_kb_seams.py` were taken there.
"""

from __future__ import annotations

import json

from conftest import mock_llm_client
from test_agent_loop import Provider, answer, ask, call
from test_reviewer import APPROVE, REVISE, issue, verdict_json

from fukidashi.agents import reviewer, translator
from fukidashi.kb import KBSnapshot, ProposalCollector
from fukidashi.kb.entities import CharacterProfile, GlossaryEntry
from fukidashi.orchestrator import run_orchestrator

KB_ID = 7
ZH = json.dumps({"translations": [{"id": 0, "zh": "你好"}, {"id": 1, "zh": "再见"}]}, ensure_ascii=False)
ZH_REVISED = json.dumps({"translations": [{"id": 0, "zh": "你好啊"}, {"id": 1, "zh": "再见"}]}, ensure_ascii=False)
ZH_REVISED_AGAIN = json.dumps({"translations": [{"id": 0, "zh": "你好呀"}, {"id": 1, "zh": "再见"}]}, ensure_ascii=False)


def kb_snapshot() -> KBSnapshot:
    """What is on the `dataset` fixture's page 1 (a term, a pending name with an alias) and what is not."""
    entries = (GlossaryEntry(1, "こんにちは", "你好", "greeting", "approved", "agent", (1,), ()),
               GlossaryEntry(2, "宇宙船", "飞船", "", "approved", "agent", (1,), ()))
    profiles = (CharacterProfile(1, "さようなら", "再见", ("バイバイ",), "口头禅", "pending_human", (1,), ()),
                CharacterProfile(2, "ハル", "小春", (), "", "approved", (1,), ()))
    return KBSnapshot(KB_ID, "book_a", entries, profiles)


def _call(call_id, tool, /, **args):
    return call(call_id, args, name=tool)


def _previous(call_id):
    return _call(call_id, "get_previous_context", pages_back=1)


def _flag(call_id, region, reason):
    return _call(call_id, "flag_uncertain", id=region, reason=reason)


def _first_pass_with_the_kb():
    """One reply that asks for every tool the KB adds, and proposes a term and a character."""
    return ask(
        _previous("g"), _flag("f", 1, "迷い"), _call("t", "lookup_term", source_term="こんにちは"),
        _call("c", "lookup_character", name="バイバイ"),
        _call("p1", "propose_entry", kind="term", payload={"source_term": "封印", "target_term": "封印术",
                                                           "category": "technique"}),
        _call("p2", "propose_entry", kind="character", payload={"name_src": "ユキ", "name_tgt": "小雪",
                                                                "aliases": ["ゆき"], "notes": "说话轻声"}))


def collect(page) -> dict[str, list[dict]]:
    """Scenario name -> the request bodies the provider received, in order."""
    flags = [{"id": 1, "reason": "迷い"}]
    current = {0: "你好", 1: "再见"}
    kb = kb_snapshot()
    collected: dict[str, list[dict]] = {}

    def run(name, *replies, do):
        provider = Provider(*replies)
        do(mock_llm_client(provider))
        collected[name] = provider.bodies

    revision = translator.Revision(current, [issue(0)])
    run("off/translator", ask(_previous("g")), answer(ZH), do=lambda c: translator.run_translator(page, c))
    run("off/translator-flagging-and-repair", ask(_previous("g"), _flag("f", 1, "迷い")), answer("not json"), answer(ZH),
        do=lambda c: translator.run_translator(page, c, flagging=True))
    run("off/revision", answer(ZH_REVISED),
        do=lambda c: translator.run_translator(page, c, flagging=True, revision=revision))
    run("off/review", answer(APPROVE), do=lambda c: reviewer.run_reviewer(page, current, flags, c))
    run("off/review-and-repair", answer("nope"), answer(APPROVE),
        do=lambda c: reviewer.run_reviewer(page, current, flags, c))
    run("on/translator", ask(_previous("g")), answer(ZH),
        do=lambda c: translator.run_translator(page, c, flagging=True, kb=kb, collector=ProposalCollector(KB_ID, 1)))
    run("on/revision", answer(ZH_REVISED), do=lambda c: translator.run_translator(
        page, c, flagging=True, revision=revision, kb=kb, collector=ProposalCollector(KB_ID, 1)))
    run("on/review", ask(_call("t", "lookup_term", source_term="こんにちは")), answer(APPROVE),
        do=lambda c: reviewer.run_reviewer(page, current, flags, c, kb=kb))

    for tag, with_kb in (("off", False), ("on", True)):
        first = _first_pass_with_the_kb() if with_kb else ask(_previous("g"), _flag("f", 1, "迷い"))
        review_lookups = ([ask(_call("rt", "lookup_term", source_term="こんにちは"),
                                _call("rc", "lookup_character", name="ハル"))] if with_kb else [])
        two_send_backs = (first, answer(ZH), answer(REVISE), answer(ZH_REVISED),
                          answer(verdict_json("revise", issue(0, "还是不对", "再改"))), answer(ZH_REVISED_AGAIN))
        for name, replies in {
            "approve": (first, answer(ZH), *review_lookups, answer(APPROVE)),
            "one-send-back": (first, answer(ZH), *review_lookups, answer(REVISE), answer(ZH_REVISED), answer(APPROVE)),
            "two-send-backs": two_send_backs,
            "fallback": (ask(_previous("g")), answer("", finish_reason="length"), answer(ZH), answer(APPROVE)),
        }.items():
            run(f"{tag}/orchestrator-{name}", *replies, do=lambda c: run_orchestrator(page, c, kb=kb if with_kb else None))
    return collected
