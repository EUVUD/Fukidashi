"""The proposal collector and proposal ids: content-derived, validated, kept once, safe across threads."""

from __future__ import annotations

import hashlib
import json
import threading

import pytest

from fukidashi.kb import ProposalCollector
from fukidashi.kb.entities import make_proposal, proposal_id

TERM = {"source_term": "封印", "target_term": "封印"}
NAME = {"name_src": "ユキ", "name_tgt": "小雪"}


def test_the_same_proposal_has_the_same_id_every_time():
    ids = {ProposalCollector(1, 3).add("term", TERM)[0].proposal_id for _ in range(3)}
    ids |= {make_proposal(1, 3, "term", dict(TERM), "someone else").proposal_id}  # who proposes it does not matter

    assert len(ids) == 1
    expected = hashlib.sha256(json.dumps([1, 3, "term", "封印", "封印"], ensure_ascii=False).encode()).hexdigest()[:32]
    assert ids == {expected} == {proposal_id(1, 3, "term", TERM)}


def test_the_id_comes_from_the_normalised_pair_and_not_from_the_other_fields():
    base = ProposalCollector(1, 3).add("term", {"source_term": "ABC", "target_term": "字"})[0]

    spelled = ProposalCollector(1, 3).add("term", {"source_term": " ＡＢＣ　", "target_term": "字 ", "category": "x"})[0]

    assert spelled.proposal_id == base.proposal_id
    assert spelled.payload == {"source_term": "ABC", "target_term": "字", "category": "x"}  # stored normalised
    named = ProposalCollector(1, 3).add("character", {**NAME, "aliases": ["ゆき"], "notes": "a"})[0]
    assert named.proposal_id == ProposalCollector(1, 3).add("character", NAME)[0].proposal_id


def test_a_proposal_made_twice_on_one_page_is_kept_once():
    collector = ProposalCollector(1, 3)

    first, first_is_new = collector.add("term", {**TERM, "category": "magic"})
    again, again_is_new = collector.add("term", {"source_term": " 封印 ", "target_term": "封印", "category": "other"})

    assert (first_is_new, again_is_new) == (True, False)
    assert again is first and again.payload["category"] == "magic"  # the first one stays
    assert collector.proposals == (first,)


def test_the_same_pair_on_another_page_in_another_kb_or_of_another_kind_has_another_id():
    ids = {
        ProposalCollector(1, 3).add("term", TERM)[0].proposal_id,
        ProposalCollector(1, 4).add("term", TERM)[0].proposal_id,  # another page
        ProposalCollector(2, 3).add("term", TERM)[0].proposal_id,  # another KB
        ProposalCollector(1, 3).add("character", {"name_src": "封印", "name_tgt": "封印"})[0].proposal_id,  # another kind
        ProposalCollector(1, 3).add("term", {**TERM, "target_term": "封存"})[0].proposal_id,  # another target
    }

    assert len(ids) == 5


def test_proposals_come_back_in_the_order_they_arrived():
    collector = ProposalCollector(1, 3)
    for source in "ABCDE":
        collector.add("term", {"source_term": source, "target_term": source.lower()})
    collector.add("term", {"source_term": "C", "target_term": "c"})  # again: keeps its place

    assert [p.payload["source_term"] for p in collector.proposals] == list("ABCDE")


@pytest.mark.parametrize("kind, payload, message", [
    ("term", {"source_term": "", "target_term": "x"}, "source_term"),
    ("term", {"source_term": " 　", "target_term": "x"}, "source_term"),
    ("term", {"source_term": "a"}, "target_term: Field required"),
    ("term", {"source_term": "a", "target_term": "b", "surprise": 1}, "surprise"),
    ("term", {"source_term": "a" * 61, "target_term": "b"}, "source_term"),
    ("term", {"source_term": 5, "target_term": "b"}, "source_term: Input should be a valid string"),
    ("term", {"source_term": "a", "target_term": "b", "category": "c" * 31}, "category"),
    ("character", {"name_src": "a"}, "name_tgt: Field required"),
    ("character", {**NAME, "aliases": [str(i) for i in range(9)]}, "aliases"),
    ("character", {**NAME, "notes": "n" * 301}, "notes"),
    ("phrase", TERM, "unknown kind 'phrase'; use one of term, character"),
], ids=lambda v: v if isinstance(v, str) and len(v) < 30 else None)
def test_an_invalid_payload_is_refused_with_a_message_and_nothing_is_kept(kind, payload, message):
    collector = ProposalCollector(1, 3)

    with pytest.raises(ValueError, match=message):
        collector.add(kind, payload)

    assert collector.proposals == ()


def test_a_character_payload_is_cleaned():
    proposal = ProposalCollector(1, 3).add("character", {
        "name_src": "ユキ", "name_tgt": "小雪", "aliases": ["ユキ", " ゆき ", "ゆき", "ﾕｷ"], "notes": " 说话，温柔 ",
    })[0]

    # no alias repeats the name or another alias; notes are trimmed but keep their full-width comma
    assert proposal.payload == {"name_src": "ユキ", "name_tgt": "小雪", "aliases": ["ゆき"], "notes": "说话，温柔"}


class RacingDict(dict):
    """A dict that makes the gap between "is it there?" and "put it there" as wide as it can be: the first read
    returns what was there when it began and then waits until a second thread has read too, so without a lock both
    threads act on the same stale answer. With the lock no second thread can read, and the wait times out."""

    def __init__(self):
        super().__init__()
        self._reads = 0
        self._count = threading.Lock()
        self._second_read = threading.Event()

    def _stale(self, answer):
        with self._count:
            self._reads += 1
            nth = self._reads
        if nth == 1:
            self._second_read.wait(timeout=0.1)
        elif nth == 2:
            self._second_read.set()
        return answer

    def get(self, key, default=None):
        return self._stale(super().get(key, default))

    def __contains__(self, key):
        return self._stale(super().__contains__(key))


def run_side_by_side(count, work):
    barrier = threading.Barrier(count)
    results = [None] * count

    def target(i):
        barrier.wait()  # all the threads start together
        results[i] = work(i)

    threads = [threading.Thread(target=target, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_one_proposal_made_by_many_threads_at_once_is_kept_exactly_once():
    """Fails without the collector's lock: two threads would both find the proposal missing and both keep their own."""
    collector = ProposalCollector(1, 3)
    collector._by_id = RacingDict()

    results = run_side_by_side(16, lambda i: collector.add("term", TERM))

    assert sum(is_new for _, is_new in results) == 1
    assert len({id(proposal) for proposal, _ in results}) == 1  # everybody got the one kept
    assert len(collector.proposals) == 1


def test_many_different_proposals_made_at_once_are_all_kept():
    collector = ProposalCollector(1, 3)

    results = run_side_by_side(40, lambda i: collector.add("term", {"source_term": f"term{i}", "target_term": f"t{i}"}))

    assert all(is_new for _, is_new in results)
    assert sorted(p.payload["source_term"] for p in collector.proposals) == sorted(f"term{i}" for i in range(40))
    assert collector.proposals == tuple({p.proposal_id: p for p in collector.proposals}.values())  # none twice
