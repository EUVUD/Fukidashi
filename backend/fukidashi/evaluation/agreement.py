"""GEMBA-MQM's human agreement check (OUTLINE 4.5): the annotator's labels against the judge's, and the decision.

The inputs are a sample directory written by `mqm_sample.draw` (`sheet.csv`, filled, and `key.json`), the sheet's sha256
as OUTLINE 4.5 recorded it before any judge output on the sample, and the MQM files (eval/mqm.py) of the sample's eval
runs, one per run, all from one judge model. The command-line entry point is eval/mqm_agreement.py; the logic lives here
so that tests can import it. Nothing here calls a model, opens a network connection or reads the database.

- **Scores.** An item's annotator score is the sheet's (`mqm_sample.check_sheet`, with GEMBA's weights); its judge score
  is the score that the MQM file of the item's eval run gives the item's segment (book, page_index, idx).
- **Kendall's tau_b** over the items, ties included (`kendall_tau_b`; the lock has no scipy).
- **Recall.** The unit is the item. An item counts when the annotator marked at least one major or critical error on it;
  it is a hit when the judge marked at least one major or critical error on the same item. Spans are not matched.
- **The decision** (OUTLINE 4.5, decided 2026-10-29). A judge passes when tau_b is at least 0.30 and the recall is at
  least 60%. When the annotator marked no major or critical error, tau_b decides alone. A tau_b that is undefined (every
  item has one score in the annotator's labels or in the judge's) is not at least 0.30, so the judge fails. Both
  thresholds are compared exactly, in integers, so that a value on a threshold is not lost to rounding.
- **Refusals** (`Refused`; nothing is written): the sheet's sha256 is not the expected one; the sheet is not complete and
  valid; an item has no judge score (its segment is failed or skipped, or not in the file); inputs that do not belong
  together: an MQM file of an eval run the key does not name, two files of one eval run, files of different judge
  models, prompts or weights, an eval run of the key without a file, a file that scored another `segments.jsonl` than the
  one the sample was drawn from, or a segment whose run is not the key's.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import mqm, mqm_sample

TAU_B_MIN = Fraction(3, 10)  # 0.30
RECALL_MIN = Fraction(3, 5)  # 60%
SERIOUS = ("critical", "major")  # what "a major or critical error" means
STATUSES = ("scored", "skipped", "failed")  # an MQM file's segment statuses (evaluation/mqm_scorer.py)
RULE = ("OUTLINE 4.5 (decided 2026-10-29): a judge passes when Kendall's tau_b between the annotator's and the judge's "
        "segment-level MQM scores over the sample is at least 0.30 and, on at least 60% of the items where the "
        "annotator marked a major or critical error, the judge marks at least one major or critical error too; when "
        "the annotator marked no major or critical error, tau_b decides alone")
ANNOTATORS = {
    "llm": ("The labels are an LLM annotator's, not a person's (OUTLINE 4.5, 'Recorded'): the check measures the judge's "
            "agreement with another LLM, which is weaker evidence than agreement with a person."),
    "person": "The labels are a person's.",
}


class Refused(Exception):
    """The agreement cannot be computed from these inputs; the message names the cause. Nothing has been written."""


# --- Kendall's tau_b --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TauB:
    """Kendall's tau_b of two score lists, with the pair counts it comes from. `tied_x` and `tied_y` count every pair
    tied in that list, the pairs tied in both included (`tied_both`)."""

    n: int
    pairs: int
    concordant: int
    discordant: int
    tied_x: int
    tied_y: int
    tied_both: int

    @property
    def denominator(self) -> int:
        """(pairs - tied_x) * (pairs - tied_y): tau_b is (concordant - discordant) over its square root."""
        return (self.pairs - self.tied_x) * (self.pairs - self.tied_y)

    @property
    def value(self) -> float | None:
        """tau_b, or None when it is undefined (every item tied in one of the lists, or fewer than 2 items)."""
        if self.denominator <= 0:
            return None
        return (self.concordant - self.discordant) / math.sqrt(self.denominator)

    def at_least(self, threshold: Fraction) -> bool:
        """tau_b >= threshold, decided in integers: (C - D) / sqrt(den) >= t  <=>  C - D >= 0 and (C - D)^2 >= t^2 den
        (for t >= 0). False when tau_b is undefined."""
        if threshold < 0:
            raise ValueError("the threshold must not be negative")
        if self.denominator <= 0:
            return False
        s = self.concordant - self.discordant
        return s >= 0 and s * s * threshold.denominator ** 2 >= threshold.numerator ** 2 * self.denominator

    def to_dict(self) -> dict:
        return {"value": self.value, "n": self.n, "pairs": self.pairs, "concordant": self.concordant,
                "discordant": self.discordant, "tied_annotator": self.tied_x, "tied_judge": self.tied_y,
                "tied_both": self.tied_both}


def _sign(a: float, b: float) -> int:
    return (a > b) - (a < b)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> TauB:
    """Kendall's tau_b of the paired scores x[i], y[i], over all n(n-1)/2 pairs (n is about 30 here, so O(n^2) is
    nothing). A pair is concordant when x and y order it the same way, discordant when they order it oppositely, and
    tied in x (in y) when its two x (y) values are equal."""
    if len(x) != len(y):
        raise ValueError(f"the lists differ in length: {len(x)} and {len(y)}")
    n = len(x)
    concordant = discordant = tied_x = tied_y = tied_both = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = _sign(x[i], x[j]), _sign(y[i], y[j])
            if dx == 0 and dy == 0:
                tied_x, tied_y, tied_both = tied_x + 1, tied_y + 1, tied_both + 1
            elif dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    return TauB(n, n * (n - 1) // 2, concordant, discordant, tied_x, tied_y, tied_both)


# --- the decision -----------------------------------------------------------------------------------------------------

def decide(tau: TauB, serious_items: int, hits: int) -> dict:
    """OUTLINE 4.5's rule. `serious_items` is the number of items with a major or critical error in the annotator's
    labels, `hits` how many of them the judge marked with one too."""
    if not 0 <= hits <= serious_items:
        raise ValueError(f"hits ({hits}) must be between 0 and the serious items ({serious_items})")
    tau_ok = tau.at_least(TAU_B_MIN)
    if serious_items == 0:
        recall_ok = None
        passed = tau_ok
        decided_by = "tau_b alone: the annotator marked no major or critical error"
    else:
        recall_ok = hits * RECALL_MIN.denominator >= RECALL_MIN.numerator * serious_items
        passed = tau_ok and recall_ok
        decided_by = "tau_b and the recall: both must reach their thresholds"
    return {"pass": passed, "tau_b_ok": tau_ok, "recall_ok": recall_ok, "decided_by": decided_by,
            "outcome": ("the judge passes: its MQM breakdown is the report's main error analysis" if passed else
                        "the judge fails: its MQM breakdown is for reference only, unless a later judge passes")}


def is_serious(severities: Sequence[str]) -> bool:
    return any(severity in SERIOUS for severity in severities)


# --- reading the inputs -----------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgeFile:
    """One eval run's MQM file, as far as the check reads it."""

    path: Path
    sha256: str
    eval_id: str
    book: str
    model: str
    document: dict  # without its segments
    segments: dict[tuple[str, int, int], dict]  # (book, page_index, idx) -> the segment


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def read_judge_file(path: Path) -> JudgeFile:
    """An MQM file as eval/mqm.py writes it; raises Refused."""
    try:
        data = path.read_bytes()
        document = json.loads(data.decode("utf-8"))
    except OSError as e:
        raise Refused(f"cannot read {path}: {e}")
    except ValueError as e:  # a UnicodeDecodeError is a ValueError
        raise Refused(f"{path} is not valid JSON: {e}")
    if not isinstance(document, dict):
        raise Refused(f"{path} is not a JSON object")
    judge = document.get("judge")
    if not all(isinstance(document.get(k), str) and document[k] for k in ("eval_id", "book", "segments_sha256")) \
            or not isinstance(judge, dict) or not isinstance(judge.get("model"), str) or not judge["model"] \
            or not isinstance(document.get("segments"), list):
        raise Refused(f"{path} is not shaped as eval/mqm.py writes it: it needs `eval_id`, `book`, `segments_sha256`, "
                      "a judge `model` and `segments`")
    segments: dict[tuple[str, int, int], dict] = {}
    for number, segment in enumerate(document["segments"], start=1):
        where = f"{path}: segment {number}"
        if not isinstance(segment, dict) or not isinstance(segment.get("book"), str) \
                or not all(_is_int(segment.get(k)) for k in ("page_index", "idx", "run_id")):
            raise Refused(f"{where} has no book, page_index, idx and run_id")
        status = segment.get("status")
        if status not in STATUSES:
            raise Refused(f"{where} has an unknown status {status!r}")
        errors = segment.get("errors")
        if not isinstance(errors, list) or not all(
                isinstance(e, dict) and e.get("severity") in mqm.SEVERITIES and isinstance(e.get("category"), str)
                for e in errors):
            raise Refused(f"{where} has errors that are not a list of severities and categories")
        if status == "scored":
            if not _is_int(segment.get("score")) or segment["score"] != mqm.mqm_score(e["severity"] for e in errors):
                raise Refused(f"{where} is scored, but its score is not the MQM score of its errors")
        elif segment.get("score") is not None or errors:
            raise Refused(f"{where} is {status}, but has a score or errors")
        key = (segment["book"], segment["page_index"], segment["idx"])
        if key in segments:
            raise Refused(f"{where} repeats {key[0]}/{key[1]} idx {key[2]}")
        segments[key] = segment
    return JudgeFile(path, hashlib.sha256(data).hexdigest(), document["eval_id"], document["book"], judge["model"],
                     {k: v for k, v in document.items() if k != "segments"}, segments)


def _read_key(path: Path) -> dict:
    try:
        key = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise Refused(f"cannot read {path}: {e}")
    except ValueError as e:
        raise Refused(f"{path} is not valid JSON: {e}")
    try:
        items = [{"item": i["item"], "book": i["book"], "page_index": i["page_index"], "idx": i["idx"],
                  "config": i["config"], "eval_id": i["eval_id"], "run_id": i["run_id"]} for i in key["items"]]
        runs = [{"config": r["config"], "book": r["book"], "eval_id": r["eval_id"],
                 "segments_sha256": r["segments_sha256"]} for r in key["runs"]]
    except (KeyError, TypeError) as e:
        raise Refused(f"{path} is not shaped as eval/mqm_sample.py writes it ({type(e).__name__}: {e})")
    return {"items": items, "runs": runs}


def bare_model(model: str) -> str:
    """The model without an organisation prefix, as eval/mqm.py names its files ("openai/gpt-4o-mini" -> "gpt-4o-mini")."""
    return model.strip().rsplit("/", 1)[-1]


def output_name(model: str) -> str:
    name = bare_model(model)
    if name in ("", ".", "..") or "\\" in name or "\0" in name:
        raise Refused("the judge's model name cannot name the output file")
    return f"agreement-{name}.json"


# --- the check --------------------------------------------------------------------------------------------------------

def _match(key: dict, files: Sequence[JudgeFile]) -> dict[str, JudgeFile]:
    """The MQM files by eval id, checked against the key's runs and one another."""
    by_eval: dict[str, JudgeFile] = {}
    for f in files:
        if f.eval_id in by_eval:
            raise Refused(f"{by_eval[f.eval_id].path} and {f.path} are both MQM files of eval run {f.eval_id}: give one "
                          "file per eval run")
        by_eval[f.eval_id] = f
    first = files[0]
    for f in files[1:]:
        if bare_model(f.model) != bare_model(first.model):
            raise Refused(f"{f.path} is by judge model {f.model!r} and {first.path} by {first.model!r}: the check "
                          "compares one judge with the annotator")
        for field, label in (("template_sha256", "prompt template"), ("commit", "GEMBA commit")):
            if (f.document.get("gemba") or {}).get(field) != (first.document.get("gemba") or {}).get(field):
                raise Refused(f"{f.path} has another {label} than {first.path}")
    for f in files:
        if f.document.get("weights") != mqm.WEIGHTS:
            raise Refused(f"{f.path} was scored with the weights {f.document.get('weights')!r}, not {mqm.WEIGHTS!r}, "
                          "which score the sheet")
    wanted = {r["eval_id"]: r for r in key["runs"]}
    for f in files:
        if f.eval_id not in wanted:
            raise Refused(f"{f.path} is of eval run {f.eval_id}, which is not one of the sample's runs "
                          f"({', '.join(sorted(wanted))})")
        run = wanted[f.eval_id]
        if f.book != run["book"]:
            raise Refused(f"{f.path} is of book {f.book!r}; the sample's run {f.eval_id} is of {run['book']!r}")
        if f.document["segments_sha256"] != run["segments_sha256"]:
            raise Refused(f"{f.path} scored another segments.jsonl than the one the sample was drawn from (its "
                          f"segments_sha256 is not the key's for {f.eval_id})")
    for eval_id, run in wanted.items():
        if eval_id not in by_eval:
            raise Refused(f"no MQM file is given for the sample's run {eval_id} ({run['config']} on {run['book']})")
    return by_eval


def compute(sample_dir: Path, expected_sha256: str, mqm_paths: Sequence[Path], *, annotator: str = "llm",
            now: datetime | None = None) -> dict:
    """The agreement file's content (rules in the module docstring); raises Refused."""
    if annotator not in ANNOTATORS:
        raise ValueError(f"unknown annotator {annotator!r}")
    sheet_path, key_path = sample_dir / "sheet.csv", sample_dir / "key.json"
    if not sheet_path.is_file() or not key_path.is_file():
        raise Refused(f"{sample_dir} has no sheet.csv and key.json")
    try:
        sheet = mqm_sample.check_sheet(sheet_path)
    except OSError as e:
        raise Refused(f"cannot read {sheet_path}: {e}")
    expected = expected_sha256.strip().lower()
    if sheet.sha256 != expected:
        raise Refused(f"the sheet's sha256 is {sheet.sha256}, not the expected {expected}: these are not the labels "
                      "recorded in OUTLINE 4.5")
    if not sheet.ok:
        shown = [*sheet.problems[:5]] + ([f"{len(sheet.unlabelled)} item(s) without a label"] if sheet.unlabelled else [])
        raise Refused(f"{sheet_path} is not complete and valid (eval/mqm_sample.py check): " + "; ".join(shown))
    key = _read_key(key_path)
    if not mqm_paths:
        raise Refused("no MQM file is given")
    by_eval = _match(key, [read_judge_file(p) for p in mqm_paths])

    items, x, y = [], [], []
    for entry in sorted(key["items"], key=lambda i: i["item"]):
        f = by_eval[entry["eval_id"]]
        place = (entry["book"], entry["page_index"], entry["idx"])
        segment = f.segments.get(place)
        where = f"item {entry['item']} ({place[0]}/{place[1]} idx {place[2]}, {entry['config']})"
        if segment is None:
            raise Refused(f"{where} has no judge score: {f.path} has no such segment")
        if segment["run_id"] != entry["run_id"]:
            raise Refused(f"{where}: {f.path} judged run {segment['run_id']}, the key's is run {entry['run_id']}")
        if segment["status"] != "scored":
            raise Refused(f"{where} has no judge score: its segment is {segment['status']} in {f.path}")
        mine = sheet.errors.get(entry["item"], [])
        theirs = [(e["severity"], e["category"]) for e in segment["errors"]]
        items.append({
            **entry,
            "annotator": {"score": sheet.score(entry["item"]), "errors": [list(e) for e in mine],
                          "major_or_critical": is_serious([s for s, _ in mine])},
            "judge": {"score": segment["score"], "errors": [list(e) for e in theirs],
                      "major_or_critical": is_serious([s for s, _ in theirs])},
        })
        x.append(sheet.score(entry["item"]))
        y.append(segment["score"])

    tau = kendall_tau_b(x, y)
    serious = [i for i in items if i["annotator"]["major_or_critical"]]
    hits = sum(i["judge"]["major_or_critical"] for i in serious)
    first = by_eval[key["runs"][0]["eval_id"]]
    response_models = sorted({json.dumps(f.document["judge"].get("response_model")) for f in by_eval.values()})
    return {
        "sample": str(sample_dir),
        "sheet": {"path": str(sheet_path), "sha256": sheet.sha256, "expected_sha256": expected},
        "annotator": {"kind": annotator, "note": ANNOTATORS[annotator]},
        "judge": {"model": first.model, "prefix": first.document["judge"].get("prefix"),
                  "response_model": [json.loads(m) for m in response_models] if len(response_models) > 1
                  else json.loads(response_models[0]),
                  "gemba": {k: (first.document.get("gemba") or {}).get(k) for k in ("commit", "template_sha256")}},
        "inputs": [{"path": str(f.path), "sha256": f.sha256, "eval_id": f.eval_id, "book": f.book,
                    "config": next(r["config"] for r in key["runs"] if r["eval_id"] == f.eval_id),
                    "segments_sha256": f.document["segments_sha256"], "scored_at": f.document.get("scored_at")}
                   for f in (by_eval[r["eval_id"]] for r in key["runs"])],
        "rule": RULE,
        "thresholds": {"tau_b": float(TAU_B_MIN), "recall": float(RECALL_MIN)},
        "n_items": len(items),
        "tau_b": tau.to_dict(),
        "recall": {"items_with_major_or_critical": len(serious), "hits": hits,
                   "value": hits / len(serious) if serious else None},
        "decision": decide(tau, len(serious), hits),
        "items": items,
        "computed_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
    }


# --- output -----------------------------------------------------------------------------------------------------------

def _without_time(document: Mapping) -> dict:
    return {k: v for k, v in document.items() if k != "computed_at"}


def write(document: dict, out: Path) -> bool:
    """Write the agreement file, never over another result: when `out` exists and holds the same result (all but
    `computed_at`) it is left as it is and False is returned; when it holds another one, Refused. Otherwise the file is
    written under a temporary name and renamed, and True is returned."""
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise Refused(f"{out} exists and cannot be read ({e}): move it away to write a new one")
        if isinstance(existing, dict) and _without_time(existing) == json.loads(json.dumps(_without_time(document))):
            return False
        raise Refused(f"{out} exists and holds another result (computed_at {existing.get('computed_at')!r}): it is "
                      "never overwritten; move it away to write a new one")
    temporary = out.with_name(out.name + ".tmp")
    try:
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return True


def format_summary(document: dict) -> str:
    tau, recall, decision = document["tau_b"], document["recall"], document["decision"]
    value = "undefined" if tau["value"] is None else f"{tau['value']:.4f}"
    rate = "n/a" if recall["value"] is None else f"{recall['value']:.1%}"
    lines = [
        f"GEMBA-MQM agreement: {document['judge']['model']} against the annotator ({document['annotator']['kind']}), "
        f"{document['n_items']} items; sheet sha256 {document['sheet']['sha256']}",
        f"  {document['annotator']['note']}",
        f"  tau_b {value} (threshold {document['thresholds']['tau_b']:.2f}): {tau['pairs']} pairs, "
        f"{tau['concordant']} concordant, {tau['discordant']} discordant, {tau['tied_annotator']} tied in the "
        f"annotator's scores, {tau['tied_judge']} in the judge's, {tau['tied_both']} in both",
        f"  recall {rate} (threshold {document['thresholds']['recall']:.0%}): the judge marked a major or critical error "
        f"on {recall['hits']} of the {recall['items_with_major_or_critical']} items where the annotator marked one",
        f"  decision: {'PASS' if decision['pass'] else 'FAIL'} ({decision['decided_by']}); {decision['outcome']}",
    ]
    return "\n".join(lines)
