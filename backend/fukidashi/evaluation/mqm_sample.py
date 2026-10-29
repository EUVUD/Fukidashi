"""The human sample for GEMBA-MQM's agreement check (OUTLINE 4.5): drawn blind, and before any judge output on it.

`draw` reads four finished eval runs (a `baseline` and a `full` run of each dev book), and only their `manifest.json`
and `segments.jsonl`. It picks source segments whose Japanese text is not empty and whose `baseline` and `full`
translations differ, in proportion to the books' segment counts (15 in all: 8 and 7 for the dev books' 334 and 274),
with a seeded `random.Random`. Each segment gives two items, its `baseline` and its `full` translation, and the items are
shuffled. The sheet shows only what the judge sees, the Japanese source and the translation; the key (item to book,
page, idx, config, eval id, run id) is a separate file. `check` validates a filled sheet. The command-line entry point is
eval/mqm_sample.py; the sheet quotes the dataset and model output, so it stays under runs/ (gitignored).

Agreement itself (Kendall's tau_b, recall of major and critical errors) is a later task.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import mqm

SOURCE_SEGMENTS = 15
CONFIGS = ("baseline", "full")
SHEET_COLUMNS = ("item", "ja", "translation", "severity", "category", "span", "note")
NO_ERROR = "no-error"
SEVERITY_LABELS = (*mqm.SEVERITIES, NO_ERROR)
GUIDE = "docs/mqm-annotation.md"

_LINE_FIELDS = (("book", str), ("page_index", int), ("idx", int), ("run_id", int), ("ja", str), ("hyp", str))


class Refused(Exception):
    """The sample cannot be drawn from these runs; nothing has been written."""


@dataclass(frozen=True)
class SampleRun:
    directory: Path
    eval_id: str
    book: str
    config: str
    lines: Mapping[tuple[int, int], dict]  # (page_index, idx) -> the segments.jsonl line
    segments_sha256: str


@dataclass(frozen=True)
class Sample:
    rows: tuple[dict, ...]  # the sheet's rows, item order
    key: dict  # key.json without the sheet's hash


def read_run(directory: Path, config: str) -> SampleRun:
    """A finished eval run of `config`; raises Refused."""
    if not directory.is_dir():
        raise Refused(f"{directory}: not a directory")
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refused(f"{directory}: cannot read manifest.json: {e}")
    if not isinstance(manifest, dict) or not all(isinstance(manifest.get(k), str) for k in ("eval_id", "book")):
        raise Refused(f"{directory}: manifest.json has no `eval_id` and `book`")
    if manifest.get("finished_at") is None:
        raise Refused(f"{directory}: the eval is not finished (manifest.json has `finished_at` null)")
    if manifest.get("config_name") != config:
        raise Refused(f"{directory}: a {manifest.get('config_name')!r} run, not {config!r}")
    path = directory / "segments.jsonl"
    try:
        data = path.read_bytes()
        text = data.decode("utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise Refused(f"{directory}: cannot read segments.jsonl: {e}")
    lines: dict[tuple[int, int], dict] = {}
    for number, raw in enumerate(text.split("\n"), start=1):  # "\n" only, as the other scorers
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except ValueError as e:
            raise Refused(f"{directory}: segments.jsonl line {number} is not valid JSON: {e}")
        if not isinstance(item, dict) or any(not isinstance(item.get(k), t) or isinstance(item.get(k), bool)
                                             for k, t in _LINE_FIELDS):
            raise Refused(f"{directory}: segments.jsonl line {number} lacks one of "
                          + ", ".join(k for k, _ in _LINE_FIELDS))
        if item["book"] != manifest["book"]:
            raise Refused(f"{directory}: segments.jsonl line {number} is of another book than the manifest's")
        place = (item["page_index"], item["idx"])
        if place in lines:
            raise Refused(f"{directory}: segments.jsonl repeats page_index {place[0]} idx {place[1]}")
        lines[place] = item
    if not lines:
        raise Refused(f"{directory}: segments.jsonl has no segments")
    return SampleRun(directory, manifest["eval_id"], manifest["book"], config, lines, hashlib.sha256(data).hexdigest())


def allocate(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """`total` split over the books in proportion to `sizes`, by largest remainder (ties by book name)."""
    whole = sum(sizes.values())
    if whole <= 0:
        raise Refused("the runs have no segments")
    quotas = {book: total * n / whole for book, n in sizes.items()}
    shares = {book: int(q) for book, q in quotas.items()}
    for book in sorted(sizes, key=lambda b: (-(quotas[b] - shares[b]), b))[: total - sum(shares.values())]:
        shares[book] += 1
    return shares


def _by_book(runs: Sequence[SampleRun], config: str) -> dict[str, SampleRun]:
    found: dict[str, SampleRun] = {}
    for run in runs:
        if run.book in found:
            raise Refused(f"two {config} runs of book {run.book!r}: {found[run.book].directory} and {run.directory}")
        found[run.book] = run
    return found


def draw(baseline_dirs: Sequence[Path], full_dirs: Sequence[Path], seed: int,
         total: int = SOURCE_SEGMENTS) -> Sample:
    """Pick `total` source segments and make their 2 * `total` items (rules in the module docstring)."""
    baseline = _by_book([read_run(d, "baseline") for d in baseline_dirs], "baseline")
    full = _by_book([read_run(d, "full") for d in full_dirs], "full")
    if set(baseline) != set(full):
        raise Refused(f"the baseline runs are of {sorted(baseline)} and the full runs of {sorted(full)}: "
                      "give one run of each config per book")
    books = sorted(baseline)
    common: dict[str, list[tuple[int, int]]] = {}
    candidates: dict[str, list[tuple[int, int]]] = {}
    for book in books:
        b, f = baseline[book].lines, full[book].lines
        common[book] = sorted(set(b) & set(f))
        for place in common[book]:
            if b[place]["ja"] != f[place]["ja"]:
                raise Refused(f"the {book} runs disagree on the Japanese text of page_index {place[0]} idx {place[1]}")
        candidates[book] = [p for p in common[book]
                            if b[p]["ja"].strip() and b[p]["hyp"].strip() != f[p]["hyp"].strip()]
    allocation = allocate({book: len(common[book]) for book in books}, total)
    rng = random.Random(seed)
    picked: list[tuple[str, tuple[int, int]]] = []
    for book in books:
        if len(candidates[book]) < allocation[book]:
            raise Refused(f"{book} has {len(candidates[book])} segments to draw from, fewer than {allocation[book]}")
        picked += [(book, place) for place in rng.sample(candidates[book], allocation[book])]
    items = [(book, place, config) for book, place in picked for config in CONFIGS]
    rng.shuffle(items)

    runs = {"baseline": baseline, "full": full}
    rows, keys = [], []
    for number, (book, place, config) in enumerate(items, start=1):
        run = runs[config][book]
        line = run.lines[place]
        rows.append({"item": number, "ja": line["ja"], "translation": line["hyp"], "severity": "", "category": "",
                     "span": "", "note": ""})
        keys.append({"item": number, "book": book, "page_index": place[0], "idx": place[1], "config": config,
                     "eval_id": run.eval_id, "run_id": line["run_id"]})
    key = {
        "about": "The key to sheet.csv. Do not open it before the annotation is finished: the annotation is blind.",
        "seed": seed,
        "source_segments": total,
        "segments": {book: len(common[book]) for book in books},
        "candidates": {book: len(candidates[book]) for book in books},
        "allocation": allocation,
        "runs": [{"config": config, "book": book, "eval_id": runs[config][book].eval_id,
                  "dir": str(runs[config][book].directory), "segments_sha256": runs[config][book].segments_sha256}
                 for config in CONFIGS for book in books],
        "items": keys,
    }
    return Sample(tuple(rows), key)


def sheet_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    """The CSV, UTF-8 with a byte order mark so that spreadsheet programs read the Japanese and Chinese right."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=SHEET_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def readme(key: Mapping) -> str:
    categories = "\n".join(f"  {c}" for c in mqm.TYPOLOGY)
    return f"""GEMBA-MQM human annotation sample (OUTLINE 4.5): {len(key['items'])} items, drawn {key['drawn_at']} with
seed {key['seed']}.

Read the guide first: {GUIDE} in the repository. Annotate before any judge output on these items
exists or is looked at, and do not open key.json: it says which config translated each item.

sheet.csv has one row per item, with the Japanese source (ja) and the translation; nothing else is shown, as for the
judge. Keep it a UTF-8 CSV with the same columns.

- An error: fill severity (minor, major or critical), category (one of the list below) and span (the wrong part of the
  translation; for an omission, the part of the source that is missing) on the item's row.
- Each further error of the same item: add a row with the same item number and fill severity, category and span. Its
  ja and translation cells can stay empty.
- An item with no error: write no-error in severity and leave category and span empty.
- note is free text and is not checked.

Categories:
{categories}

When done, check the sheet (it lists what is missing or wrong, or prints each item's MQM score and the sheet's sha256):
  uv run python eval/mqm_sample.py check <this directory>/sheet.csv
"""


def write_sample(sample: Sample, out_root: Path, now: datetime | None = None) -> Path:
    """Write sheet.csv, key.json and README.txt into a new directory `<out_root>/<UTC time>-sample` (`-2`, `-3`... if
    that exists): an existing directory is never written into. Returns the directory."""
    now = now or datetime.now(timezone.utc)
    out_root.mkdir(parents=True, exist_ok=True)
    base = now.strftime("%Y%m%dT%H%M%SZ") + "-sample"
    candidate, n = base, 1
    while True:
        try:
            (out_root / candidate).mkdir()
            break
        except FileExistsError:
            n += 1
            candidate = f"{base}-{n}"
    out_dir = out_root / candidate
    sheet = sheet_bytes(sample.rows)
    key = {**sample.key, "drawn_at": now.isoformat(timespec="seconds"),
           "sheet_sha256": hashlib.sha256(sheet).hexdigest()}
    (out_dir / "sheet.csv").write_bytes(sheet)
    (out_dir / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.txt").write_text(readme(key), encoding="utf-8")
    return out_dir


# --- check ---------------------------------------------------------------------------------------------------------------

@dataclass
class SheetCheck:
    sha256: str
    items: tuple[int, ...]  # the items the sheet should have, from key.json
    problems: list[str] = field(default_factory=list)
    unlabelled: list[int] = field(default_factory=list)
    errors: dict[int, list[tuple[str, str]]] = field(default_factory=dict)  # item -> (severity, category) per error
    no_error: set[int] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not self.problems and not self.unlabelled

    def score(self, item: int) -> int:
        return mqm.mqm_score(severity for severity, _ in self.errors.get(item, []))


def check_sheet(path: Path) -> SheetCheck:
    """Validate a filled sheet against the items of the key.json beside it. Raises OSError when a file cannot be read."""
    data = path.read_bytes()
    result = SheetCheck(hashlib.sha256(data).hexdigest(), ())
    key_path = path.parent / "key.json"
    try:
        key = json.loads(key_path.read_text(encoding="utf-8"))
        result.items = tuple(sorted(int(entry["item"]) for entry in key["items"]))  # only the numbers are read
    except (OSError, ValueError, KeyError, TypeError) as e:
        result.problems.append(f"cannot read the item numbers from {key_path}: {e}")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        result.problems.append(f"{path.name} is not UTF-8: {e}")
        return result
    reader = csv.DictReader(io.StringIO(text, newline=""))
    missing = [c for c in SHEET_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        result.problems.append(f"{path.name} lacks the column(s) {', '.join(missing)}")
        return result
    expected = set(result.items)
    labelled: set[int] = set()
    for number, row in enumerate(reader, start=2):  # row 1 is the header
        where = f"row {number}"
        try:
            item = int((row["item"] or "").strip())
        except ValueError:
            result.problems.append(f"{where}: item {row['item']!r} is not a number")
            continue
        where = f"row {number} (item {item})"
        if expected and item not in expected:
            result.problems.append(f"{where}: no such item in this sample")
            continue
        severity = (row["severity"] or "").strip().lower()
        category = (row["category"] or "").strip().lower()
        span = (row["span"] or "").strip()
        if not severity:
            if category or span:
                result.problems.append(f"{where}: a category or span without a severity")
            continue
        if severity not in SEVERITY_LABELS:
            result.problems.append(f"{where}: unknown severity {severity!r} (use {', '.join(SEVERITY_LABELS)})")
            continue
        labelled.add(item)
        if severity == NO_ERROR:
            if category or span:
                result.problems.append(f"{where}: no-error with a category or span")
            result.no_error.add(item)
            continue
        if category not in mqm.TYPOLOGY:
            result.problems.append(f"{where}: unknown category {category!r} (see {GUIDE})")
            continue
        if category == "non-translation" and severity != "critical":
            result.problems.append(f"{where}: non-translation is always critical, as in GEMBA-MQM")
            continue
        result.errors.setdefault(item, []).append((severity, category))
    for item in sorted(result.no_error & set(result.errors)):
        result.problems.append(f"item {item} is marked no-error and has an error")
    result.unlabelled = sorted(expected - labelled)
    return result
