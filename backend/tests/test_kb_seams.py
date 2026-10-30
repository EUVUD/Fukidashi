"""The KB seams hardened before the Curator writes (task 015), on a scripted provider and in fresh interpreters: no network.

What must hold: the agents' import graph holds no database, the KB tool code both agents share has one home, the lookup
tools describe `approved` and `pending` as the system prompts do, the payload kept for one proposal id comes from the
first call in the order of the calls and not from the first to arrive, and every request the agents send over the
scenarios of `request_scenarios.py` is pinned to the byte, so that no prompt, tool schema or request layout changes
unnoticed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import request_scenarios
from conftest import mock_llm_client
from kb_support import accept, new_kb, new_run
from test_agent_loop import Provider, answer, ask, call
from test_kb_tools import KB_ID, ZH, ZH_REVISED, imported_names, propose, snapshot, translate_with_kb
from test_reviewer import APPROVE, REVISE
from test_translator import tool_result

from fukidashi import kb as kb_package
from fukidashi import store
from fukidashi.agents import kb_tools, reviewer, translator
from fukidashi.config import ROOT
from fukidashi.datasets.importer import import_openmantra
from fukidashi.datasets.openmantra import get_page
from fukidashi.kb import ProposalCollector
from fukidashi.kb.curator import apply
from fukidashi.kb.entities import make_proposal
from fukidashi.orchestrator import flow, run_orchestrator

PACKAGE = ROOT / "backend" / "fukidashi"


@pytest.fixture
def page(dataset):
    return get_page("book_a", 1, dataset)


# --- AC1: no database in the agents' import graph -----------------------------------------------------------------

DATABASE = ("fukidashi.store", "fukidashi.db", "sqlite3")
NO_DATABASE = ["fukidashi.agents.translator", "fukidashi.agents.reviewer", "fukidashi.agents.kb_tools",
               "fukidashi.orchestrator.flow", "fukidashi.kb", "fukidashi.kb.snapshot", "fukidashi.kb.collector"]
WITH_DATABASE = ["fukidashi.kb.loader", "fukidashi.kb.curator", "fukidashi.store"]  # where the check must see it


def database_loaded_by(module: str) -> list[str]:
    """Which database modules importing `module` loads, in an interpreter of its own (`-I`: nothing but the package's
    installation). A test process cannot say: `store` is loaded in it already."""
    code = f"import {module}, sys; print(repr([n for n in {DATABASE!r} if n in sys.modules]))"
    done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return ast.literal_eval(done.stdout.strip())


@pytest.fixture(scope="module")
def loaded():
    with ThreadPoolExecutor(max_workers=len(NO_DATABASE) + len(WITH_DATABASE)) as pool:  # one interpreter per module
        modules = [*NO_DATABASE, *WITH_DATABASE]
        return dict(zip(modules, pool.map(database_loaded_by, modules)))


@pytest.mark.parametrize("module", NO_DATABASE)
def test_importing_the_agents_the_orchestrator_and_the_kbs_plain_modules_loads_no_database(loaded, module):
    assert loaded[module] == []


@pytest.mark.parametrize("module", WITH_DATABASE)
def test_the_check_does_see_a_database_where_there_is_one(loaded, module):
    assert "fukidashi.store" in loaded[module] and "sqlite3" in loaded[module]


# --- AC2: the module layout ---------------------------------------------------------------------------------------

def imports_translator(source: str) -> bool:
    """Whether a module's source imports `agents/translator.py`, however it is spelled."""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            names = {a.name for a in node.names}
            if (node.module or "").split(".")[-1] == "translator" or (
                    "translator" in names and (node.module or "").split(".")[-1] in ("", "agents")):
                return True
        elif isinstance(node, ast.Import) and any(a.name.split(".")[-1] == "translator" for a in node.names):
            return True
    return False


@pytest.mark.parametrize("source", [
    "from .translator import kb_snippets_part\n", "from fukidashi.agents.translator import lookup_tools\n",
    "from . import translator\n", "from ..agents import reviewer, translator\n", "import fukidashi.agents.translator\n",
], ids=["relative", "absolute", "package-relative", "package", "import"])
def test_the_scan_catches_every_way_of_importing_the_translator(source):
    assert imports_translator(source)


def test_the_scan_leaves_other_imports_alone():
    assert not imports_translator("from .kb_tools import lookup_tools\nfrom .loop import run_agent\nimport json\n")


def test_the_reviewer_imports_nothing_from_the_translator():
    assert not imports_translator((PACKAGE / "agents" / "reviewer.py").read_text(encoding="utf-8"))
    assert imports_translator((PACKAGE / "orchestrator" / "flow.py").read_text(encoding="utf-8"))  # the scan sees


SHARED = ("KB_SNIPPET_CHARS", "kb_snippets_part", "_status", "LookupTermArgs", "LookupCharacterArgs", "lookup_tools")


def defined_at_top_level(path: Path) -> set[str]:
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_the_shared_kb_tool_code_is_defined_once_in_kb_tools():
    homes = {name: sorted(p.name for p in (PACKAGE / "agents").glob("*.py") if name in defined_at_top_level(p))
             for name in SHARED}

    assert homes == {name: ["kb_tools.py"] for name in SHARED}
    assert reviewer.kb_snippets_part is kb_tools.kb_snippets_part and reviewer.lookup_tools is kb_tools.lookup_tools
    assert translator.kb_snippets_part is kb_tools.kb_snippets_part and translator.lookup_tools is kb_tools.lookup_tools


def test_only_the_loader_and_the_curator_import_the_store_in_the_kb_package():
    importers = {p.name for p in (PACKAGE / "kb").glob("*.py") if imported_names(p) & {"store", "sqlite3", "db"}}

    assert importers == {"curator.py", "loader.py"}
    assert "load_snapshot" not in kb_package.__all__ and not hasattr(kb_package, "load_snapshot")


# --- AC3: every request the agents send is pinned to the byte ------------------------------------------------------

LOOKUPS = ("lookup_term", "lookup_character")

# sha256 of the request bodies (json.dumps with sort_keys and ensure_ascii=False) of each scenario in
# request_scenarios.py, taken with the prompts and tool descriptions of the M5 tuning. A change to a prompt, a tool
# schema or the request layout changes them; an intended change re-baselines them in the same change.
DIGESTS = {
    "off/orchestrator-approve": "66545fea4dad5006a1c9d2058d1d932b81ecb05d563ee95e3b638d6f0f01d7c3",
    "off/orchestrator-fallback": "845bbceddafdb3c729dc1af4a1686794c857da5d48b5f50b4f16f3481470da7a",
    "off/orchestrator-one-send-back": "363a353548116ce6bcbc5c3c8438ee531d5ac23d2402a2732a1450c8993d80c4",
    "off/orchestrator-two-send-backs": "5a1a5365097e7031c6e2ad531a310d7df476e61c831db61778b46695f35a4202",
    "off/review": "68a7e328d7761eb7808a33c1c043e12d13e446fec3952d583b55066e42daea09",
    "off/review-and-repair": "95401449b95d2d9f882bd98aede0e83715fc7eae45a9b0e10f16ec66a7785169",
    "off/revision": "2678bd6365450e4b7e9f2c7a35a017d1fb2467804bc6b5f0be6b41e4949d7b06",
    "off/translator": "209fb74cb33062fc9441507bd7c9c2c5faae630e77a6e4a7bcc291871d9eb290",
    "off/translator-flagging-and-repair": "4043d0fc671efa771a28daf9a39dabc2cd0429a51676fa3d8becc9a05955b38b",
    "on/orchestrator-approve": "37061efc1851ad496a49b6aa04698eeeb8ba067b1a150c8b6b5667e70f2fc2e6",
    "on/orchestrator-fallback": "d867d3a10474c22dae551ca1341a5dacdcc14a7f5326f1ba2441ea754efffe93",
    "on/orchestrator-one-send-back": "1d025cd5503dcb7ad987e86418a20d4ee2f505360d6230520242de6fcdf97b09",
    "on/orchestrator-two-send-backs": "86a8d2dca29872422f619d489a81321c1a459719b12fbf69111ddde18be4b449",
    "on/review": "d0025d46f481d0078d41e5cf5f0a1728e0b7bb64b45d82c875d4483a9fb9992f",
    "on/revision": "64944a2faeeb52b9782e9bb641132f8d1942ecce8500160f05a4eadc0eb75cf8",
    "on/translator": "de8f0efe6eece4b52b359eb4177d9b21e5b89df990581d837195725417399b2e",
}


def lookup_functions(bodies: list[dict]) -> list[dict]:
    """The `function` entries of the two lookup tools in every request that has them."""
    return [t["function"] for body in bodies for t in body.get("tools") or [] if t["function"]["name"] in LOOKUPS]


def digest(bodies: list[dict]) -> str:
    return hashlib.sha256(json.dumps(bodies, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def scenarios(tmp_path_factory):
    from conftest import SEGMENTS

    root = tmp_path_factory.mktemp("requests") / "open-mantra-dataset"   # the `dataset` fixture's page 1
    (root / "images" / "book_a" / "ja").mkdir(parents=True)
    (root / "images" / "book_a" / "ja" / "000.jpg").write_bytes(b"\xff\xd8fake-jpeg")
    page = {"page_index": 1, "image_paths": {"ja": "images/book_a/ja/000.jpg"}, "frame": [], "text": SEGMENTS}
    (root / "annotation.json").write_text(json.dumps([{"book_title": "book_a", "pages": [page]}]))
    return request_scenarios.collect(get_page("book_a", 1, root))


def test_every_request_over_the_scenarios_is_the_pinned_one(scenarios):
    assert {name: digest(bodies) for name, bodies in scenarios.items()} == DIGESTS
    lookups = {name: len(lookup_functions(bodies)) for name, bodies in scenarios.items()}
    assert all(count == 0 for name, count in lookups.items() if name.startswith("off/"))   # the KB off: no lookups
    assert all(count > 0 for name, count in lookups.items() if name.startswith("on/"))     # the KB on: they are sent


def test_the_lookup_tools_describe_the_two_states_as_the_prompts_do():
    tools = {t.name: t for t in kb_tools.lookup_tools(snapshot())}

    assert set(tools) == set(LOOKUPS)
    for tool in tools.values():
        assert "approved (in the knowledge base)" in tool.description
        assert "pending (proposed on an earlier page, not yet confirmed by a person)" in tool.description
        assert "whether a person has confirmed" not in tool.description   # the old wording: it made `approved` a person's
    for prompt in (translator.KB_SYSTEM_PROMPT, reviewer.KB_SYSTEM_PROMPT):
        assert "proposed on an earlier page and no person has confirmed it yet" in prompt   # what [pending] means there


def test_the_lookups_answer_with_the_same_two_status_words_as_before():
    kb = snapshot(*request_scenarios.kb_snapshot().entries, *request_scenarios.kb_snapshot().profiles)
    handlers = {t.name: t.handler for t in kb_tools.lookup_tools(kb)}

    assert handlers["lookup_term"](source_term="こんにちは")["status"] == "approved"
    assert handlers["lookup_character"](name="さようなら")["status"] == "pending"


# --- AC4: one payload per proposal id, decided by the order of the calls -------------------------------------------

class ForcedArrival(ProposalCollector):
    """Makes the calls of a reply arrive in the reverse order. `pairs` are (first, second) categories of two calls of one
    reply: the handler of `first` waits until `second` has been added, and `arrivals` lists the categories in the order
    the collector received them.

    Only a reply whose calls run side by side gets past the wait; if they ran one after the other the test fails on the
    timeout instead of hanging.
    """

    def __init__(self, kb_id, page_index, *pairs):
        super().__init__(kb_id, page_index)
        self.waits_for = dict(pairs)
        self.arrived = {category: threading.Event() for pair in pairs for category in pair}
        self.arrivals: list[str] = []

    def add(self, kind, payload, **kwargs):
        category = payload.get("category")
        if category in self.waits_for:
            assert self.arrived[self.waits_for[category]].wait(timeout=5), "the calls of one reply did not run side by side"
        added = super().add(kind, payload, **kwargs)
        self.arrivals.append(category)   # before the event: the one that waits for this one comes after it
        if category in self.arrived:
            self.arrived[category].set()
        return added

    def arrived_before(self, early: str, late: str) -> bool:
        return self.arrivals.index(early) < self.arrivals.index(late)


def term_call(call_id, category, source="封印", target="封印术"):
    return propose(call_id, "term", source_term=source, target_term=target, category=category)


def categories(proposals):
    return [p.payload["category"] for p in proposals]


def test_two_calls_with_one_id_keep_the_first_calls_payload_though_the_second_arrives_first(page):
    provider = Provider(ask(term_call("a", "first"), term_call("b", "second")), answer(ZH))
    collector = ForcedArrival(KB_ID, 1, ("first", "second"))

    result, _ = translate_with_kb(page, provider, snapshot(), collector)

    assert result.ok and collector.arrivals == ["second", "first"]   # the race was forced, and the second one won it
    assert categories(collector.proposals) == ["second"]             # so the collector holds the second call's payload
    assert categories(result.proposals) == ["first"]                 # and the run gives the first call's
    assert result.proposal_ids == [result.proposals[0].proposal_id]
    # the tool's answers may follow arrival: the model only learns that its proposal is recorded
    assert json.loads(tool_result(provider.bodies[1], "b"))["status"] == "new"
    assert json.loads(tool_result(provider.bodies[1], "a"))["status"] == "already_proposed"


def test_the_orchestrator_returns_and_the_curator_stores_the_first_calls_payload(conn, dataset, page, monkeypatch):
    import_openmantra(conn, dataset)
    kb_id, run_id = new_kb(conn), new_run(conn)
    collectors = []

    def forced(kb_id, page_index):
        collectors.append(ForcedArrival(kb_id, page_index, ("first", "second")))
        return collectors[-1]

    monkeypatch.setattr(flow, "ProposalCollector", forced)
    provider = Provider(ask(term_call("a", "first"), term_call("b", "second")), answer(ZH), answer(APPROVE))

    result = run_orchestrator(page, mock_llm_client(provider), kb=snapshot(kb_id=kb_id))

    (collector,) = collectors
    assert result.ok and collector.arrivals == ["second", "first"] and categories(collector.proposals) == ["second"]
    assert categories(result.proposals) == ["first"]
    apply(conn, kb_id, 1, run_id, result.proposals, {p.proposal_id: accept() for p in result.proposals})
    (entry,) = store.kb_entries(conn, kb_id, ("approved",))
    assert (entry["source_term"], entry["category"]) == ("封印", "first")   # what `apply` stored


def test_the_first_pass_beats_the_revision_for_one_id_with_the_arrival_forced_in_both_replies(page, monkeypatch):
    provider = Provider(
        ask(term_call("a", "one"), term_call("b", "two")), answer(ZH), answer(REVISE),
        # the revision: the same id again, and a new id with two calls, each reply forced to arrive reversed
        ask(term_call("c", "three"), term_call("d", "four"),
            term_call("e", "five", source="結界", target="结界"), term_call("f", "six", source="結界", target="结界")),
        answer(ZH_REVISED), answer(APPROVE))
    collectors = []

    def forced(kb_id, page_index):
        collectors.append(ForcedArrival(kb_id, page_index, ("one", "two"), ("three", "four"), ("five", "six")))
        return collectors[-1]

    monkeypatch.setattr(flow, "ProposalCollector", forced)

    result = run_orchestrator(page, mock_llm_client(provider), kb=snapshot())

    (collector,) = collectors
    assert result.ok and result.send_backs == 1
    for early, late in (("two", "one"), ("four", "three"), ("six", "five")):
        assert collector.arrived_before(early, late), (early, late)   # the race was forced in both replies
    assert [(p.payload["source_term"], p.payload["category"]) for p in result.proposals] == [
        ("封印", "one"), ("結界", "five")]   # the first pass's call for 封印, and the revision's first call for 結界


def test_a_proposal_that_only_the_revision_makes_keeps_its_first_call_too(page, monkeypatch):
    provider = Provider(
        answer(ZH), answer(REVISE), ask(term_call("c", "three"), term_call("d", "four")), answer(ZH_REVISED),
        answer(APPROVE))
    collector = ForcedArrival(KB_ID, 1, ("three", "four"))
    monkeypatch.setattr(flow, "ProposalCollector", lambda kb_id, page_index: collector)

    result = run_orchestrator(page, mock_llm_client(provider), kb=snapshot())

    assert result.ok and collector.arrivals == ["four", "three"] and categories(result.proposals) == ["three"]


def fake_call(kind, payload, *, ran=True, error=None, name="propose_entry"):
    """A tool call as the agent loop records it in a step."""
    proposal = make_proposal(KB_ID, 1, kind, payload, "translator") if ran and error is None else None
    return {"name": name, "ran": ran, "error": error, "arguments": json.dumps({"kind": kind, "payload": payload}),
            "result": None if proposal is None else json.dumps({"proposal_id": proposal.proposal_id})}


def term_payload(source, target="译", category=""):
    return {"source_term": source, "target_term": target, "category": category}


def test_step_proposals_keeps_the_first_call_of_each_id_in_the_order_of_the_calls():
    steps = [{"tool_calls": [
        fake_call("term", term_payload("甲", category="first")),
        fake_call("term", term_payload("乙")),
        fake_call("term", term_payload("甲", category="second")),                      # the same id, other details
        fake_call("term", term_payload("丙"), error="Error: ..."),                      # refused by make_proposal
        fake_call("term", term_payload("丁"), ran=False),                               # a duplicate, not run again
        fake_call("term", term_payload("戊"), name="lookup_term"),                      # not a proposal
    ]}, {"tool_calls": [
        fake_call("term", term_payload("己")),
        fake_call("term", term_payload("乙", category="again")),
        fake_call("character", {"name_src": "ユキ", "name_tgt": "小雪", "aliases": [], "notes": ""}),
    ]}]

    made = translator.step_proposals(steps, KB_ID, 1)

    assert [(p.kind, p.payload.get("source_term") or p.payload["name_src"], p.payload.get("category")) for p in made] == [
        ("term", "甲", "first"), ("term", "乙", ""), ("term", "己", ""), ("character", "ユキ", None)]
    assert [p.proposal_id for p in made] == translator.proposal_ids(steps)   # the same calls, the same ids
    assert all((p.kb_id, p.page_index, p.proposer) == (KB_ID, 1, "translator") for p in made)


def test_merge_proposals_keeps_the_first_run_and_the_first_call():
    first = make_proposal(KB_ID, 1, "term", term_payload("甲", category="pass"), "translator")
    again = make_proposal(KB_ID, 1, "term", term_payload("甲", category="revision"), "translator")
    other = make_proposal(KB_ID, 1, "term", term_payload("乙"), "translator")

    assert flow.merge_proposals((first, other), (again,)) == (first, other)
    assert flow.merge_proposals((again,), (first, other)) == (again, other)   # the order of the runs decides
    assert flow.merge_proposals() == () and flow.merge_proposals((), ()) == ()


def test_a_run_without_the_kb_has_no_proposals(page):
    provider = Provider(ask(call("g", {"pages_back": 1}, name="get_previous_context")), answer(ZH))

    result = translator.run_translator(page, mock_llm_client(provider))

    assert result.ok and result.proposals == () and result.proposal_ids == []
