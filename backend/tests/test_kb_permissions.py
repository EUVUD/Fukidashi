"""Least privilege, enforced in code (OUTLINE 1.3): only the Curator's module writes KB entries, profiles and proposals."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from fukidashi import store
from fukidashi.config import ROOT

CURATOR = Path("backend/fukidashi/kb/curator.py")
STORE = Path("backend/fukidashi/store.py")
WRITES_KB_TABLES = re.compile(
    r"(INSERT(?: OR \w+)? INTO|UPDATE|DELETE FROM)\s+(glossary_entries|character_profiles|proposals|\{table\})"
)


def store_writers() -> set[str]:
    """Every function in store.py that writes the entry, profile or proposal tables, directly or through another
    (private helpers included)."""
    sources = {
        name: inspect.getsource(fn) for name, fn in inspect.getmembers(store, inspect.isfunction)
        if fn.__module__ == store.__name__
    }
    writers = {name for name, source in sources.items() if WRITES_KB_TABLES.search(source)}
    while True:
        callers = {name for name, source in sources.items() if any(re.search(rf"\b{w}\(", source) for w in writers)}
        if callers <= writers:
            return writers
        writers |= callers


def uses_of(source: str, names) -> list[tuple[int, str]]:
    """Where `source` mentions one of `names`: a call (`store.f()`, `f()`), a reference (`g = store.f`), an import of
    it, even under another name (`from ..store import f as h`), or its name as a string (`getattr(store, "f")`)."""
    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and node.attr in names:
            hits.append((node.lineno, node.attr))
        elif isinstance(node, ast.Name) and node.id in names:
            hits.append((node.lineno, node.id))
        elif isinstance(node, ast.alias) and node.name in names:
            hits.append((node.lineno if hasattr(node, "lineno") else 0, node.name))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in names:
            hits.append((node.lineno, node.value))
    return hits


def python_files():
    """The package and the scripts that sit beside it."""
    for folder in ("backend/fukidashi", "scripts", "eval"):
        yield from sorted((ROOT / folder).rglob("*.py"))


def test_kb_writes_name_every_public_function_that_writes_the_kb_tables():
    writers = store_writers()

    assert set(store.KB_WRITES) <= writers  # no stale name
    assert writers - set(store.KB_WRITES) == {"_add_evidence"}  # only a private helper of two of them is left out
    assert "create_kb" not in writers  # it makes the empty container; entries are what only the Curator writes


def test_only_the_curator_calls_the_kb_write_functions():
    watched = store_writers()
    violations = {}
    used_by_curator = set()
    for path in python_files():
        relative = path.relative_to(ROOT)
        if relative == STORE:
            continue  # where they are defined
        hits = uses_of(path.read_text(encoding="utf-8"), watched)
        if relative == CURATOR:
            used_by_curator = {name for _, name in hits}
        elif hits:
            violations[str(relative)] = hits

    assert violations == {}, "only kb/curator.py may call the KB write functions in store.py"
    assert used_by_curator == set(store.KB_WRITES)  # the scan sees the Curator, so a clean result is not an empty one


@pytest.mark.parametrize("source", [
    "from fukidashi import store\nstore.insert_entry(conn)\n",
    "from fukidashi.store import insert_proposal\ninsert_proposal(conn)\n",
    "from ..store import set_proposal_status as decide\n",
    "from .. import store\nwrite = store.insert_profile\n",
    "import fukidashi.store\nfukidashi.store.add_entry_evidence(conn)\n",
    "from fukidashi import store\ngetattr(store, 'add_profile_evidence')(conn)\n",
], ids=["call", "imported name", "import as", "reference", "dotted path", "getattr"])
def test_the_scan_catches_every_way_of_reaching_a_kb_write_function(source):
    assert uses_of(source, set(store.KB_WRITES))


def test_the_scan_leaves_the_reads_and_other_code_alone():
    source = (
        "from fukidashi import store\n"
        "store.find_entry(conn, 1, 'a')\nstore.kb_entries(conn, 1, ['approved'])\nstore.create_kb(conn)\n"
        "store.finish_run(conn)\ninsert = 3\n"
    )

    assert uses_of(source, set(store.KB_WRITES)) == []


def test_no_agent_module_imports_the_curator():
    """The agents get a snapshot and a collector (neither can write); `apply` needs a connection, which they never hold."""
    offenders = []
    for path in sorted((ROOT / "backend/fukidashi/agents").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            imported = []
            if isinstance(node, ast.ImportFrom):
                imported = [f"{node.module or ''}.{a.name}" for a in node.names] + [node.module or ""]
            elif isinstance(node, ast.Import):
                imported = [a.name for a in node.names]
            if any(re.search(r"(^|\.)curator($|\.)", name) or name.endswith(".apply") for name in imported):
                offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []
