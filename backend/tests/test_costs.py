"""The cost and latency summary (eval/PROTOCOL.md, "Statistics"): `evaluation/costs.py` and eval/costs.py, on made-up runs.

Nothing here needs data/, the database or a model. The eval runs come from test_report.py's `make_eval`; the ledger is a
file written line by line, priced at $1 per million input tokens and $4 per million output tokens so that every figure
can be worked out by hand. Its first two lines are not JSON and its seventh is a call that belongs to no run, as the
real ledger's early lines and its check call are: a range that does not name them must not read them.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys

import pytest

from fukidashi import config
from fukidashi.evaluation import costs, report
from fukidashi.evaluation.metrics import latency_summary
from test_report import A, B, FixedClock, edit_json, make_eval, make_glossary_dir, page, read_json, snapshot

IN_PRICE, OUT_PRICE = 1e-6, 4e-6  # dollars per token: $1 and $4 per million
MODEL = "deepseek-flash"
EARLY = ["this early line is not even JSON", "{\"page_id\": \"early\"}"]  # lines 1 and 2: outside every range


@pytest.fixture(autouse=True)
def known_code(monkeypatch):
    """The output records the code's commit; the tests do not depend on the checkout they run in."""
    monkeypatch.setattr(costs, "git_state", lambda repo: {"commit": "c0ffee1", "dirty": False})


# --- a ledger and the eval runs that wrote it -------------------------------------------------------------------------

def call(page_id, input_tokens, cached, output, *, model=MODEL):
    """A ledger line of a call with usage, priced at IN_PRICE and OUT_PRICE."""
    return {"page_id": page_id, "requested_model": model, "response_model": model,
            "usage": {"input_tokens": input_tokens, "output_tokens": output, "cached_input_tokens": cached},
            "latency_s": 1.0, "cost_usd": input_tokens * IN_PRICE + output * OUT_PRICE, "error": None}


def no_usage(page_id, error="API error 402: Insufficient Balance"):
    """A ledger line of a call that returned no usage: its tokens and its cost are unknown."""
    return {"page_id": page_id, "requested_model": MODEL, "response_model": None, "usage": None, "latency_s": 1.0,
            "cost_usd": None, "error": error}


def spec(index, calls=(), *, status="scored", latency=1.0, error=None):
    """A page of an eval run: its calls are (input, cached, output) tokens, or None for a call without usage."""
    return {"index": index, "status": status, "calls": list(calls), "latency": latency, "error": error}


class Fixture:
    """Eval runs and the ledger their calls wrote. `lines` are the ledger's lines (dicts, or text for a line that is
    not JSON); `ranges` the lines of each run, by eval id; `dirs` the runs' directories, by (config, book)."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.glossary_dir = make_glossary_dir(tmp_path)
        self.ledger = tmp_path / "usage.jsonl"
        self.lines: list = list(EARLY)
        self.ranges: dict[str, tuple[int, int]] = {}
        self.dirs: dict = {}

    def add_run(self, config_name, book, specs):
        pages, block, failed_latency = [], [], {}
        for s in specs:
            if s["status"] == "skipped":
                pages.append(page(s["index"], "skipped"))
                continue
            page_id = f"{book}/{s['index']:03d}"
            made = [no_usage(page_id) if c is None else call(page_id, *c) for c in s["calls"]]
            block += made
            cost = None if any(l["cost_usd"] is None for l in made) else round(sum(l["cost_usd"] for l in made), 6)
            if s["status"] == "scored":
                pages.append(page(s["index"], calls=len(made), cost_usd=cost, latency_s=s["latency"]))
            else:
                pages.append(page(s["index"], "failed", error=s["error"], cost_usd=cost))
                failed_latency[s["index"]] = s["latency"]
        run_dir = make_eval(self.tmp_path, config_name, book, pages, glossary_dir=self.glossary_dir)
        if failed_latency:  # make_eval writes the same latency for every failed page
            edit_json(run_dir / "scores.json", lambda d: [r.update(latency_s=failed_latency[r["page_index"]])
                                                          for r in d["failed_pages"]])
        self.dirs[config_name, book] = run_dir
        self.ranges[run_dir.name] = (len(self.lines) + 1, len(self.lines) + len(block))
        self.lines += block

    def gap(self):
        """A call that belongs to no run, between two ranges (the real ledger's line 1727)."""
        self.lines.append(call("tojime_no_siora/002", 900, 0, 100))

    def eval_id(self, config_name, book):
        return self.dirs[config_name, book].name

    def write(self):
        self.ledger.write_text("".join((json.dumps(l, ensure_ascii=False) if isinstance(l, dict) else l) + "\n"
                                       for l in self.lines), encoding="utf-8")

    def build(self, **options):
        self.write()
        return costs.build_costs(list(self.dirs.values()), self.glossary_dir, self.ledger, self.ranges, **options)


def make_set(tmp_path):
    """Two books and two configs. `full_no_kb` failed A/3, so the common pages are A: 1, 2 and B: 1, and `baseline`'s A/3
    is outside them. A/9 has no text. Ledger lines 3-5 are baseline A, 6 baseline B, 7 no run's, 8-14 full_no_kb A (its
    failed page is lines 13-14, and 14 returned no usage), 15-16 full_no_kb B."""
    s = Fixture(tmp_path)
    s.add_run("baseline", A, [spec(1, [(1000, 0, 200)], latency=1.0), spec(2, [(2000, 0, 400)], latency=2.0),
                              spec(3, [(3000, 0, 100)], latency=50.0), spec(9, status="skipped")])
    s.add_run("baseline", B, [spec(1, [(1500, 0, 300)], latency=3.0)])
    s.gap()
    s.add_run("full_no_kb", A, [
        spec(1, [(4000, 1000, 500), (5000, 3000, 250)], latency=10.0),
        spec(2, [(6000, 2000, 100), (6200, 4000, 100), (6400, 5000, 200)], latency=20.0),
        spec(3, [(3000, 0, 100), None], status="failed", latency=30.0, error="ProviderError: HTTP 402"),
        spec(9, status="skipped")])
    s.add_run("full_no_kb", B, [spec(1, [(2000, 500, 100), (2500, 1500, 100)], latency=40.0)])
    return s


def line_of(s, config_name, book, nth=0):
    """The ledger line number of the nth call of a run."""
    return s.ranges[s.eval_id(config_name, book)][0] + nth


# --- the split into pages ---------------------------------------------------------------------------------------------

def test_each_runs_range_is_recorded_and_split_into_the_pages_it_attempted(tmp_path):
    s = make_set(tmp_path)

    result = s.build()

    assert s.ranges == {s.eval_id("baseline", A): (3, 5), s.eval_id("baseline", B): (6, 6),
                        s.eval_id("full_no_kb", A): (8, 14), s.eval_id("full_no_kb", B): (15, 16)}
    runs = {(r["config"], r["book"]): r for r in result["runs"]}
    assert [(r["config"], r["book"]) for r in result["runs"]] == [("baseline", A), ("baseline", B), ("full_no_kb", A),
                                                                   ("full_no_kb", B)]
    for key, (first, last) in [(("baseline", A), (3, 5)), (("baseline", B), (6, 6)), (("full_no_kb", A), (8, 14)),
                               (("full_no_kb", B), (15, 16))]:
        assert runs[key]["ledger_range"] == {"first": first, "last": last, "lines": last - first + 1}
        assert runs[key]["calls"] == last - first + 1
        assert runs[key]["directory"] == str(s.dirs[key]) and runs[key]["requested_models"] == [MODEL]
    assert runs["baseline", A]["pages"] == {"attempted": 3, "scored": 3, "failed": 0}
    assert runs["full_no_kb", A]["pages"] == {"attempted": 3, "scored": 2, "failed": 1}  # A/9 was never attempted
    assert result["inputs"]["ledger"] == {"path": str(s.ledger), "lines": 16}  # line 7 is in no run, lines 1-2 neither


def test_a_run_records_the_hash_of_its_lines_so_that_they_can_be_found_again(tmp_path):
    import hashlib

    s = make_set(tmp_path)
    run = {r["eval_id"]: r for r in s.build()["runs"]}[s.eval_id("full_no_kb", A)]

    text = "".join(line + "\n" for line in s.ledger.read_text(encoding="utf-8").split("\n")[7:14])  # sed -n 8,14p

    assert run["lines_sha256"] == hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_the_lines_outside_the_ranges_are_never_read(tmp_path):
    s = make_set(tmp_path)  # line 1 is not JSON and line 7 is another page's call: neither is in a range

    result = s.build()

    assert sum(r["calls"] for r in result["runs"]) == 13 and result["inputs"]["ledger"]["lines"] == 16  # 16 less 3
    assert all(7 not in range(r["ledger_range"]["first"], r["ledger_range"]["last"] + 1) for r in result["runs"])


def test_a_ledger_split_on_newlines_only_keeps_a_line_with_a_unicode_separator_in_one_piece(tmp_path):
    s = make_set(tmp_path)
    i = line_of(s, "baseline", A) - 1
    s.lines[i]["error"] = "a message with a line separator   and a next line \u0085 inside"

    result = s.build()  # `splitlines()` would cut that line in three and shift every line number after it

    assert result["runs"][0]["ledger_range"] == {"first": 3, "last": 5, "lines": 3}
    assert result["runs"][0]["calls"] == 3


def test_a_ledger_that_ends_without_a_newline_or_with_one_gives_the_same_lines(tmp_path):
    s = make_set(tmp_path)
    s.write()
    with_newline = costs.read_ledger(s.ledger)
    s.ledger.write_text(s.ledger.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")

    assert costs.read_ledger(s.ledger) == with_newline and len(with_newline) == 16
    s.ledger.write_text("", encoding="utf-8")
    assert costs.read_ledger(s.ledger) == []


# --- refusals ---------------------------------------------------------------------------------------------------------

def shorten_inside_the_last_page(s):
    first, last = s.ranges[s.eval_id("full_no_kb", B)]  # B/1 made 2 calls; the range keeps one
    s.ranges[s.eval_id("full_no_kb", B)] = (first, last - 1)
    return r"page 1 made 2 calls \(scores.json\), but the ledger has 1 line for 'boureisougi/001'", s.dirs["full_no_kb", B]


def stop_before_the_last_page(s):
    first, last = s.ranges[s.eval_id("baseline", A)]  # the range ends before A/3, so it holds two pages of three
    s.ranges[s.eval_id("baseline", A)] = (first, last - 1)
    return r"form 2 runs of one page_id, but the eval run attempted 3 pages", s.dirs["baseline", A]


def run_into_the_next_run(s):
    first, last = s.ranges[s.eval_id("baseline", B)]  # one line too many: the call of no run, line 7
    s.ranges[s.eval_id("baseline", B)] = (first, last + 1)
    return r"form 2 runs of one page_id, but the eval run attempted 1 page ", s.dirs["baseline", B]


def change_the_page_id_inside_a_page(s):
    s.lines[line_of(s, "full_no_kb", A, 3) - 1]["page_id"] = "tojime_no_siora/999"  # the second call of A/2
    return r"form 5 runs of one page_id, but the eval run attempted 3 pages", s.dirs["full_no_kb", A]


def swap_two_pages(s):
    i = line_of(s, "baseline", A) - 1  # A/1 and A/2 trade places
    s.lines[i], s.lines[i + 1] = s.lines[i + 1], s.lines[i]
    return (r"page 1 is number 1 of the pages the eval run attempted, so its calls should be 'tojime_no_siora/001', "
            r"but the ledger's number 1 \(line 3\) is 'tojime_no_siora/002'"), s.dirs["baseline", A]


def use_another_model(s):
    n = line_of(s, "full_no_kb", B, 1)
    s.lines[n - 1]["requested_model"] = "other-model"
    return rf"ledger line {n} was made with model 'other-model', which the eval run did not ask for", s.dirs["full_no_kb", B]


def overlap_two_ranges(s):
    first, last = s.ranges[s.eval_id("full_no_kb", A)]
    s.ranges[s.eval_id("full_no_kb", A)] = (s.ranges[s.eval_id("baseline", B)][1], last)  # begins on baseline B's last line
    return r"the ledger ranges of .* and .* overlap: a call belongs to one eval run", s.eval_id("baseline", B)


def go_past_the_end(s):
    first, last = s.ranges[s.eval_id("full_no_kb", B)]
    s.ranges[s.eval_id("full_no_kb", B)] = (first, last + 5)
    return r"its ledger range 15-21 goes past the end of .*usage.jsonl, which has 16 lines", s.dirs["full_no_kb", B]


def leave_a_run_without_a_range(s):
    del s.ranges[s.eval_id("baseline", A)]
    return r"it has no ledger range: give --ledger-range", s.dirs["baseline", A]


def give_a_range_to_nothing(s):
    s.ranges["20261029T000000Z-nowhere"] = (1, 1)
    return r"a ledger range is given for 20261029T000000Z-nowhere, which is not one of the eval runs", "20261029T000000Z-nowhere"


def break_a_line(s):
    s.lines[line_of(s, "baseline", A) - 1] = "{not json"
    return rf"usage.jsonl: line {line_of(s, 'baseline', A)} is not valid JSON", s.ledger


def put_an_array_in_the_ledger(s):
    s.lines[line_of(s, "baseline", A) - 1] = "[1, 2]"
    return rf"usage.jsonl: line {line_of(s, 'baseline', A)} is not a JSON object", s.ledger


def drop_the_cached_count(s):
    s.lines[line_of(s, "baseline", A) - 1]["usage"]["cached_input_tokens"] = None
    return r"has usage without a `cached_input_tokens` count: its cached share is unknown", s.ledger


def cache_more_than_the_input(s):
    s.lines[line_of(s, "baseline", A) - 1]["usage"]["cached_input_tokens"] = 1001
    return r"has more cached input tokens than input tokens", s.ledger


def price_a_call_without_usage(s):
    n = line_of(s, "full_no_kb", A, 6)  # the failed page's call that returned nothing
    s.lines[n - 1]["cost_usd"] = 0.5
    return rf"line {n} has a cost but no usage", s.ledger


def misprice_one_call(s):
    i = line_of(s, "full_no_kb", A, 2) - 1
    s.lines[i]["cost_usd"] *= 1.000001  # off by one part in a million
    return rf"usage.jsonl: the costs of the given ranges do not fit one pair of prices.*: line {i + 1} costs .*relatively " \
           r"where 1e-09 is allowed", s.ledger


def read_a_run_that_is_not_finished(s):
    edit_json(s.dirs["baseline", A] / "manifest.json", lambda d: d.update(finished_at=None))
    return r"the eval run is not finished", s.dirs["baseline", A]


def lose_a_failed_pages_row(s):
    edit_json(s.dirs["full_no_kb", A] / "scores.json", lambda d: d.update(failed_pages=[]))
    return r"page 3 is failed in manifest.json but has no row in `failed_pages` of scores.json", s.dirs["full_no_kb", A]


def lose_a_failed_pages_latency(s):
    edit_json(s.dirs["full_no_kb", A] / "scores.json", lambda d: d["failed_pages"][0].update(latency_s=None))
    return r"the row of page 3 in scores.json has a `latency_s` that is not a finite number", s.dirs["full_no_kb", A]


def lose_a_scored_pages_calls(s):
    edit_json(s.dirs["baseline", A] / "scores.json", lambda d: d["pages"][0].pop("calls"))
    return r"the row of page 1 in scores.json has no whole-number `calls`", s.dirs["baseline", A]


def lose_the_requested_models(s):
    edit_json(s.dirs["baseline", B] / "manifest.json", lambda d: d.update(requested_models=[]))
    return r"manifest.json has no `requested_models`", s.dirs["baseline", B]


REFUSALS = [shorten_inside_the_last_page, stop_before_the_last_page, run_into_the_next_run,
            change_the_page_id_inside_a_page, swap_two_pages, use_another_model, overlap_two_ranges, go_past_the_end,
            leave_a_run_without_a_range, give_a_range_to_nothing, break_a_line, put_an_array_in_the_ledger,
            drop_the_cached_count, cache_more_than_the_input, price_a_call_without_usage, misprice_one_call,
            read_a_run_that_is_not_finished, lose_a_failed_pages_row, lose_a_failed_pages_latency,
            lose_a_scored_pages_calls, lose_the_requested_models]


def refusal_case(tmp_path, make_wrong):
    """A fixture made wrong in one way: (the fixture, the text the message must hold, what it must name)."""
    s = make_set(tmp_path)
    expected, names = make_wrong(s)
    s.write()
    return s, expected, names


@pytest.mark.parametrize("make_wrong", REFUSALS, ids=lambda f: f.__name__)
def test_a_range_that_is_not_the_runs_calls_and_ledgers_that_do_not_fit_are_refused_with_the_run_and_the_cause(
        tmp_path, make_wrong):
    s, expected, names = refusal_case(tmp_path, make_wrong)

    with pytest.raises(costs.Refused, match=expected) as raised:
        costs.build_costs(list(s.dirs.values()), s.glossary_dir, s.ledger, s.ranges)

    assert str(names) in str(raised.value)


@pytest.fixture
def cli():
    spec_ = importlib.util.spec_from_file_location("eval_costs_under_test", config.ROOT / "eval" / "costs.py")
    module = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(module)
    return module


def cli_args(s, out_root, *extra):
    args = [*map(str, s.dirs.values())]
    for eval_id, (first, last) in s.ranges.items():
        args += ["--ledger-range", f"{eval_id}={first}-{last}"]
    return [*args, "--ledger", str(s.ledger), "--glossary-dir", str(s.glossary_dir), "--out-root", str(out_root), *extra]


@pytest.mark.parametrize("make_wrong", REFUSALS, ids=lambda f: f.__name__)
def test_the_cli_exits_2_names_the_cause_and_writes_nothing_for_every_refusal(cli, tmp_path, capsys, make_wrong):
    s, expected, names = refusal_case(tmp_path, make_wrong)
    out_root = tmp_path / "out" / "analysis"
    before = snapshot(*s.dirs.values(), s.glossary_dir)
    ledger_before = s.ledger.read_bytes()

    code = cli.main(cli_args(s, out_root))

    err = capsys.readouterr().err
    assert code == 2 and err.startswith("Nothing was written: ") and re.search(expected, err), err
    assert str(names) in err
    assert not out_root.exists()  # no directory, not even an empty one
    assert snapshot(*s.dirs.values(), s.glossary_dir) == before
    assert s.ledger.read_bytes() == ledger_before


def test_the_report_refuses_first_and_the_message_is_its_own(tmp_path):
    s = make_set(tmp_path)
    s.write()
    edit_json(s.dirs["baseline", B] / "manifest.json", lambda d: d.update(finished_at=None))
    with pytest.raises(report.Refused) as expected:
        report.build_report(list(s.dirs.values()), s.glossary_dir)

    with pytest.raises(costs.Refused) as raised:
        costs.build_costs(list(s.dirs.values()), s.glossary_dir, s.ledger, s.ranges)

    assert costs.Refused is report.Refused and str(raised.value) == str(expected.value)


def test_the_report_refuses_two_runs_of_one_config_and_book_before_any_range_is_looked_at(tmp_path):
    s = make_set(tmp_path)
    s.write()

    with pytest.raises(costs.Refused, match="both baseline on tojime_no_siora"):
        costs.build_costs([*s.dirs.values(), s.dirs["baseline", A]], s.glossary_dir, s.ledger, {})


@pytest.mark.parametrize("first, last", [(0, 3), (5, 4), (-2, 3)])
def test_a_range_that_is_empty_or_starts_before_line_1_is_refused_even_when_it_does_not_come_from_the_command_line(
        tmp_path, first, last):
    s = make_set(tmp_path)
    s.ranges[s.eval_id("baseline", A)] = (first, last)  # line 0 would silently be the ledger's last line

    with pytest.raises(costs.Refused, match=rf"its ledger range {first}-{last} is empty or starts before line 1"):
        s.build()


def test_ledger_range_options_are_parsed_and_malformed_ones_refused():
    assert costs.parse_ranges(["20261030T215155Z-baseline=1469-1504", "x=3-3"]) == {
        "20261030T215155Z-baseline": (1469, 1504), "x": (3, 3)}
    assert costs.parse_ranges([]) == {}
    for option, why in [("1469-1504", "is not a ledger range"), ("x=1469", "is not a ledger range"),
                        ("x=a-b", "is not a ledger range"), ("x=0-5", "is empty or starts before line 1"),
                        ("x=9-5", "is empty or starts before line 1"), ("x=-5-9", "is not a ledger range")]:
        with pytest.raises(costs.Refused, match=why):
            costs.parse_ranges([option])
    with pytest.raises(costs.Refused, match=r"x has two ledger ranges \(1-2 and 5-6\)"):
        costs.parse_ranges(["x=1-2", "x=5-6"])


def test_the_cli_exits_2_for_a_malformed_ledger_range_and_writes_nothing(cli, tmp_path, capsys):
    s = make_set(tmp_path)
    s.write()
    out_root = tmp_path / "out"

    code = cli.main([*map(str, s.dirs.values()), "--ledger-range", "not a range", "--ledger", str(s.ledger),
                     "--glossary-dir", str(s.glossary_dir), "--out-root", str(out_root)])

    assert code == 2 and "'not a range' is not a ledger range" in capsys.readouterr().err and not out_root.exists()


def test_a_ledger_that_is_not_there_is_refused_by_the_library_and_the_cli(cli, tmp_path, capsys):
    s = make_set(tmp_path)
    s.write()
    s.ledger = tmp_path / "no-such-ledger.jsonl"
    out_root = tmp_path / "out"

    with pytest.raises(costs.Refused, match="no-such-ledger.jsonl: the ledger cannot be read"):
        costs.build_costs(list(s.dirs.values()), s.glossary_dir, s.ledger, s.ranges)
    code = cli.main(cli_args(s, out_root))

    assert code == 2 and "no-such-ledger.jsonl: the ledger cannot be read" in capsys.readouterr().err
    assert not out_root.exists()


# --- the prices ---------------------------------------------------------------------------------------------------------

def test_the_prices_are_recovered_exactly_from_the_lines_that_have_a_cost(tmp_path):
    s = make_set(tmp_path)

    prices = s.build()["prices"]

    assert (prices["input_usd_per_m_tokens"], prices["output_usd_per_m_tokens"]) == (1.0, 4.0)
    assert prices["input_usd_per_token"] == pytest.approx(IN_PRICE, rel=1e-12)
    assert prices["output_usd_per_token"] == pytest.approx(OUT_PRICE, rel=1e-12)
    assert prices["fitted_lines"] == 12  # the 16 lines, less the 2 early ones, the call of no run and the call without usage
    assert 0 <= prices["max_relative_residual"] < 1e-12 and prices["tolerance"] == 1e-9
    assert prices["requested_models"] == [MODEL]
    assert prices["worst_line"] in {3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16}  # one of the 12 lines that were fitted


def test_a_call_with_a_cost_off_by_less_than_the_tolerance_still_fits_and_one_off_by_more_does_not(tmp_path):
    s = make_set(tmp_path)
    i = line_of(s, "full_no_kb", A, 2) - 1
    s.lines[i]["cost_usd"] *= 1 + 1e-12

    prices = s.build()["prices"]

    assert 1e-13 < prices["max_relative_residual"] < 1e-9
    s.lines[i]["cost_usd"] *= 1 + 1e-6
    with pytest.raises(costs.Refused, match="do not fit one pair of prices"):
        s.build()


def test_a_ledger_whose_lines_were_priced_by_two_pairs_of_prices_is_refused(tmp_path):
    s = make_set(tmp_path)
    for n in range(line_of(s, "baseline", A), line_of(s, "baseline", A) + 3):  # one run at double the prices
        s.lines[n - 1]["cost_usd"] *= 2

    with pytest.raises(costs.Refused, match="do not fit one pair of prices"):
        s.build()


def line(number, input_tokens, output_tokens, cost, cached=0):
    return costs.Line(number, "x/001", MODEL, input_tokens, cached, output_tokens, cost, None)


def test_without_a_line_that_has_a_cost_the_prices_cannot_be_recovered():
    with pytest.raises(costs.Refused, match="no line of the given ledger ranges has a cost"):
        costs.fit_prices([])
    with pytest.raises(costs.Refused, match="no line of the given ledger ranges has a cost"):
        costs.fit_prices([line(1, 100, 10, None), costs.Line(2, "x/001", MODEL, None, None, None, None, "E")])


def test_proportional_token_counts_cannot_tell_one_pair_of_prices_from_another():
    lines = [line(1, 100, 10, 0.001), line(2, 200, 20, 0.002), line(3, 300, 30, 0.003)]

    with pytest.raises(costs.Refused, match="their input and output token counts are proportional"):
        costs.fit_prices(lines)
    with pytest.raises(costs.Refused, match="proportional"):
        costs.fit_prices(lines[:1])  # one line cannot fix two prices either


def test_two_independent_lines_fix_the_two_prices():
    prices = costs.fit_prices([line(1, 1000, 100, 1000 * 3e-7 + 100 * 1.2e-6), line(2, 500, 400, 500 * 3e-7 + 400 * 1.2e-6)])

    assert prices.input_per_token == pytest.approx(3e-7, rel=1e-12) and prices.output_per_token == pytest.approx(1.2e-6, rel=1e-12)
    assert prices.lines == 2 and prices.max_relative_residual < 1e-12


def test_the_early_lines_priced_otherwise_do_not_disturb_the_fit(tmp_path):
    s = make_set(tmp_path)
    s.lines[1] = call("early", 100, 0, 10) | {"cost_usd": 99.0}  # line 2 is priced absurdly, and is in no range

    assert s.build()["prices"]["input_usd_per_m_tokens"] == 1.0


# --- the numbers: both ends of the cost range, worked out by hand ---------------------------------------------------------

def test_the_common_pages_of_baseline_have_the_ledgers_cost_at_both_ends_because_nothing_was_cached(tmp_path):
    result = make_set(tmp_path).build()

    pooled = result["pooled"]["baseline"]["common"]  # A/1, A/2, B/1
    assert result["common_pages"] == {A: [1, 2], B: [1]}
    assert pooled["pages"] == 3 and pooled["calls"] == 3 and pooled["calls_per_page"] == 1.0
    assert pooled["tokens"] == {"input": 4500, "uncached_input": 4500, "cached_input": 0, "output": 900}
    assert pooled["tokens_per_page"] == {"uncached_input": 1500.0, "cached_input": 0.0, "output": 300.0}
    assert pooled["cached_share_of_input"] == 0.0
    # (1000 + 2000 + 1500) x $1/M + (200 + 400 + 300) x $4/M = 0.0045 + 0.0036 = 0.0081
    assert pooled["cost_usd"] == {"upper": 0.0081, "lower": 0.0081, "at_cached_price": None}
    assert pooled["cost_per_page_usd"] == {"upper": 0.0027, "lower": 0.0027, "at_cached_price": None}


def test_the_common_pages_of_a_config_that_cached_have_a_range_from_cached_input_at_zero_to_the_ledgers_cost(tmp_path):
    result = make_set(tmp_path).build()

    pooled = result["pooled"]["full_no_kb"]["common"]  # A/1 (2 calls), A/2 (3), B/1 (2)
    assert pooled["pages"] == 3 and pooled["calls"] == 7 and pooled["calls_per_page"] == 2.333
    # input 9000 + 18600 + 4500, of which cached 4000 + 11000 + 2000; output 750 + 400 + 200
    assert pooled["tokens"] == {"input": 32100, "uncached_input": 15100, "cached_input": 17000, "output": 1350}
    assert pooled["tokens_per_page"] == {"uncached_input": 5033.333, "cached_input": 5666.667, "output": 450.0}
    assert pooled["cached_share_of_input"] == 0.529595  # 17000 / 32100
    # the ledger's cost: A/1 0.006 + 0.006, A/2 0.0064 + 0.0066 + 0.0072, B/1 0.0024 + 0.0029 = 0.0375
    assert pooled["cost_usd"]["upper"] == 0.0375 and pooled["cost_per_page_usd"]["upper"] == 0.0125
    # cached input at $0: 0.0375 less 17000 x $1/M = 0.0205
    assert pooled["cost_usd"]["lower"] == 0.0205 and pooled["cost_per_page_usd"]["lower"] == 0.006833


def test_a_price_for_cached_input_gives_a_value_inside_the_range(tmp_path):
    result = make_set(tmp_path).build(cached_input_price_per_m=0.25)

    pooled = result["pooled"]["full_no_kb"]["common"]
    # 0.0205 + 17000 cached tokens x $0.25/M = 0.0205 + 0.00425
    assert pooled["cost_usd"]["at_cached_price"] == 0.02475 and pooled["cost_per_page_usd"]["at_cached_price"] == 0.00825
    assert pooled["cost_usd"]["lower"] <= pooled["cost_usd"]["at_cached_price"] <= pooled["cost_usd"]["upper"]
    assert result["pooled"]["baseline"]["common"]["cost_usd"]["at_cached_price"] == 0.0081  # nothing was cached
    assert result["inputs"]["cached_input_price_per_m"] == 0.25


def test_a_cached_price_of_zero_gives_the_lower_end_and_the_input_price_the_upper_end(tmp_path):
    s = make_set(tmp_path)

    low = s.build(cached_input_price_per_m=0)["pooled"]["full_no_kb"]["common"]["cost_usd"]
    high = s.build(cached_input_price_per_m=1.0)["pooled"]["full_no_kb"]["common"]["cost_usd"]

    assert low["at_cached_price"] == low["lower"] == 0.0205 and high["at_cached_price"] == high["upper"] == 0.0375


@pytest.mark.parametrize("price, why", [(-0.1, "is not a non-negative number"), (float("nan"), "is not a non-negative number"),
                                        (float("inf"), "is not a non-negative number"),
                                        (1.5, "is above the input price recovered from the ledger")])
def test_a_price_for_cached_input_must_lie_between_zero_and_the_input_price(tmp_path, price, why):
    s = make_set(tmp_path)

    with pytest.raises(costs.Refused, match=why):
        s.build(cached_input_price_per_m=price)


def test_without_a_cached_price_the_value_at_that_price_is_not_there_at_all(tmp_path):
    result = make_set(tmp_path).build()

    assert result["inputs"]["cached_input_price_per_m"] is None
    assert all(scope["cost_usd"]["at_cached_price"] is None and scope["cost_per_page_usd"]["at_cached_price"] is None
               for by_which in result["pooled"].values() for scope in by_which.values())
    assert result["method"]["cost_usd"]["at_cached_price"] is None


def test_each_book_has_its_own_figures_and_the_pooled_ones_are_over_all_the_pages(tmp_path):
    result = make_set(tmp_path).build()

    a, b = result["by_book"][A]["full_no_kb"]["common"], result["by_book"][B]["full_no_kb"]["common"]
    # A: pages 1 and 2, 5 calls, 0.012 + 0.0202 = 0.0322, of which 4000 + 11000 cached tokens = $0.015 at the input price
    assert (a["pages"], a["calls"], a["tokens"]["cached_input"]) == (2, 5, 15000)
    assert (a["cost_usd"]["upper"], a["cost_usd"]["lower"]) == (0.0322, 0.0172)
    # B: page 1, 2 calls, 0.0024 + 0.0029 = 0.0053, of which 500 + 1500 cached tokens = $0.002
    assert (b["pages"], b["calls"], b["tokens"]["cached_input"]) == (1, 2, 2000)
    assert (b["cost_usd"]["upper"], b["cost_usd"]["lower"]) == (0.0053, 0.0033)
    assert list(result["by_book"]) == result["books"] == [A, B]


# --- common pages against all attempted pages, with a failed page ---------------------------------------------------------

def test_the_attempted_pages_add_the_pages_outside_the_common_pages_and_the_failed_ones(tmp_path):
    result = make_set(tmp_path).build()

    base = result["pooled"]["baseline"]["attempted"]  # A/1, A/2, A/3 (off the common pages), B/1
    assert (base["pages"], base["calls"], base["cost_usd"]["upper"], base["cost_per_page_usd"]["upper"]) == (
        4, 4, 0.0115, 0.002875)
    assert base["tokens"] == {"input": 7500, "uncached_input": 7500, "cached_input": 0, "output": 1000}
    assert result["pooled"]["baseline"]["common"]["pages"] == 3  # the common pages are a part of the attempted ones


def test_a_failed_page_counts_its_calls_in_the_attempted_pages_and_leaves_what_it_does_not_know_unknown(tmp_path):
    result = make_set(tmp_path).build()

    pooled = result["pooled"]["full_no_kb"]["attempted"]  # A/1, A/2, A/3 failed (2 calls, 1 without usage), B/1
    assert pooled["pages"] == 4 and pooled["calls"] == 9 and pooled["calls_per_page"] == 2.25
    assert pooled["lines_without_usage"] == [14] and pooled["lines_without_cost"] == [14]
    # the tokens and the cost of a scope with a call that has none are unknown, never a partial sum
    assert pooled["tokens"] is None and pooled["tokens_per_page"] is None and pooled["cached_share_of_input"] is None
    assert pooled["cost_usd"] == {"upper": None, "lower": None, "at_cached_price": None}
    assert pooled["cost_per_page_usd"] == {"upper": None, "lower": None, "at_cached_price": None}
    by_book = result["by_book"]
    assert by_book[A]["full_no_kb"]["attempted"]["tokens"] is None  # the book of the failed page
    assert by_book[B]["full_no_kb"]["attempted"]["tokens"] is not None  # the other book has all of its usage
    assert by_book[B]["full_no_kb"]["attempted"]["lines_without_usage"] == []
    assert result["pooled"]["full_no_kb"]["common"]["tokens"] is not None  # the common pages leave the failed page out


def test_the_failed_pages_are_listed_with_their_page_calls_and_error(tmp_path):
    result = make_set(tmp_path).build()

    assert result["failed_pages"] == [{
        "eval_id": "20261029T000000Z-full_no_kb-tojime_no_siora", "config": "full_no_kb", "book": A, "page_index": 3,
        "page_id": "tojime_no_siora/003", "calls": 2, "ledger_lines": [13, 14], "latency_s": 30.0,
        "lines_without_usage": [14], "lines_without_cost": [14], "error": "ProviderError: HTTP 402"}]


def test_a_line_without_cost_is_listed_and_never_counted_as_zero(tmp_path):
    result = make_set(tmp_path).build()

    assert result["lines_without_cost"] == [{
        "line": 14, "eval_id": "20261029T000000Z-full_no_kb-tojime_no_siora", "config": "full_no_kb", "book": A,
        "page_index": 3, "page_status": "failed", "has_usage": False, "error": "API error 402: Insufficient Balance"}]
    run = {r["eval_id"]: r for r in result["runs"]}["20261029T000000Z-full_no_kb-tojime_no_siora"]
    assert run["calls"] == 7 and run["lines_without_usage"] == [14] and run["lines_without_cost"] == [14]
    # the run's own totals are the sums over the lines that have them, named so: 6 lines have usage, 6 have a cost
    assert run["tokens_of_lines_with_usage"] == {"input": 30600, "uncached_input": 15600, "cached_input": 15000,
                                                  "output": 1250}
    assert run["cost_usd_of_lines_with_cost"] == 0.0356  # 0.012 + 0.0202 + the failed page's first call, 0.0034
    assert all(r["lines_without_cost"] == [] for r in result["runs"] if r["config"] == "baseline")


def test_a_call_with_usage_and_no_price_has_tokens_but_no_cost_and_stays_out_of_the_fit(tmp_path):
    s = make_set(tmp_path)
    n = line_of(s, "baseline", A, 1)
    s.lines[n - 1]["cost_usd"] = None  # the price was not set when A/2 was translated

    result = s.build()

    assert result["lines_without_cost"][0]["line"] == n and result["lines_without_cost"][0]["has_usage"] is True
    baseline = result["pooled"]["baseline"]["common"]
    assert baseline["tokens"] is not None and baseline["lines_without_usage"] == [] and baseline["lines_without_cost"] == [n]
    assert baseline["cost_usd"] == {"upper": None, "lower": None, "at_cached_price": None}
    assert result["prices"]["fitted_lines"] == 11 and result["prices"]["input_usd_per_m_tokens"] == 1.0


def test_a_scope_with_no_page_has_no_per_page_figure_not_a_zero(tmp_path):
    s = Fixture(tmp_path)
    s.add_run("baseline", A, [spec(1, [(1000, 0, 200)], latency=1.0)])
    s.add_run("full_no_kb", A, [spec(1, [(500, 0, 50)], status="failed", latency=2.0, error="E")])

    result = s.build()  # baseline scored A/1 and full_no_kb failed it, so no page is common

    common = result["pooled"]["baseline"]["common"]
    assert result["common_pages"] == {A: []} and common["pages"] == 0 and common["calls"] == 0
    assert common["calls_per_page"] is None
    assert common["tokens_per_page"] == {"uncached_input": None, "cached_input": None, "output": None}
    assert common["cost_usd"]["upper"] == 0.0 and common["cost_per_page_usd"]["upper"] is None
    assert common["cached_share_of_input"] is None
    assert common["latency_s"] == {"n": 0, "p50_s": None, "p95_s": None, "mean_s": None}
    assert result["pooled"]["baseline"]["attempted"]["pages"] == 1
    assert result["pooled"]["full_no_kb"]["attempted"]["cost_usd"]["upper"] == pytest.approx(0.0007)
    assert "n/a" in costs.render(result)


# --- latency ---------------------------------------------------------------------------------------------------------

def test_latency_is_the_summary_the_report_uses_of_the_pages_latencies_with_the_mean(tmp_path):
    result = make_set(tmp_path).build()

    common = result["pooled"]["baseline"]["common"]["latency_s"]  # A/1, A/2, B/1: 1, 2, 3
    assert common == {**latency_summary([1.0, 2.0, 3.0]), "mean_s": 2.0}
    assert (common["p50_s"], common["p95_s"], common["mean_s"]) == (2.0, 2.9, 2.0)
    attempted = result["pooled"]["baseline"]["attempted"]["latency_s"]  # and A/3: 50
    assert attempted == {**latency_summary([1.0, 2.0, 3.0, 50.0]), "mean_s": 14.0}
    assert (attempted["p50_s"], attempted["p95_s"]) == (2.5, 42.95)


def test_a_failed_pages_latency_counts_in_the_attempted_pages_only(tmp_path):
    result = make_set(tmp_path).build()

    assert result["pooled"]["full_no_kb"]["common"]["latency_s"] == {**latency_summary([10.0, 20.0, 40.0]),
                                                                       "mean_s": 23.333}
    assert result["pooled"]["full_no_kb"]["attempted"]["latency_s"] == {
        **latency_summary([10.0, 20.0, 30.0, 40.0]), "mean_s": 25.0}  # A/3 failed after 30 s
    assert result["by_book"][A]["full_no_kb"]["attempted"]["latency_s"]["n"] == 3


def test_the_figures_agree_with_the_report_on_the_common_pages(tmp_path):
    s = make_set(tmp_path)
    s.write()
    pooled = report.build_report(list(s.dirs.values()), s.glossary_dir)
    result = costs.build_costs(list(s.dirs.values()), s.glossary_dir, s.ledger, s.ranges)

    assert result["common_pages"] == pooled["common_pages"] and result["configs"] == pooled["configs"]
    assert result["books"] == pooled["books"]
    for config_name in pooled["configs"]:
        cells = [(result["pooled"][config_name]["common"], pooled["pooled"][config_name])]
        cells += [(result["by_book"][book][config_name]["common"], pooled["by_book"][book][config_name])
                  for book in pooled["books"]]
        for mine, theirs in cells:
            assert mine["pages"] == theirs["pages"]
            assert mine["latency_s"]["n"] == theirs["latency"]["n"]
            assert (mine["latency_s"]["p50_s"], mine["latency_s"]["p95_s"]) == (theirs["latency"]["p50_s"],
                                                                                 theirs["latency"]["p95_s"])
            assert abs(mine["cost_usd"]["upper"] - theirs["cost"]["total_usd"]) <= 0.00001
            assert abs(mine["cost_per_page_usd"]["upper"] - theirs["cost"]["per_page_usd"]) <= 0.00001


def test_the_reports_warnings_are_carried_over(tmp_path):
    s = make_set(tmp_path)
    s.write()
    edit_json(s.dirs["baseline", B] / "manifest.json", lambda d: d.update(git_dirty=True))

    result = costs.build_costs(list(s.dirs.values()), s.glossary_dir, s.ledger, s.ranges)

    assert result["warnings"] and "dirty working tree" in result["warnings"][0]
    assert result["warnings"] == report.build_report(list(s.dirs.values()), s.glossary_dir)["warnings"]


def test_the_document_holds_only_what_json_holds_and_records_the_code_it_was_made_with(tmp_path):
    result = make_set(tmp_path).build()

    assert result == json.loads(json.dumps(result, allow_nan=False))
    assert result["code"] == {"git_commit": "c0ffee1", "git_dirty": False}
    assert list(result) == ["generated_at", "code", "inputs", "prices", "method", "configs", "books", "common_pages",
                            "runs", "failed_pages", "lines_without_cost", "pooled", "by_book", "warnings"]
    assert set(result["pooled"]["baseline"]) == {"common", "attempted"} == set(costs.SCOPES)


def test_the_order_of_the_directories_and_of_the_ranges_changes_nothing(tmp_path):
    s = make_set(tmp_path)
    s.write()
    straight = costs.build_costs(list(s.dirs.values()), s.glossary_dir, s.ledger, s.ranges)

    shuffled = costs.build_costs(list(reversed(list(s.dirs.values()))), s.glossary_dir, s.ledger,
                                 dict(reversed(list(s.ranges.items()))))

    assert shuffled == straight | {"generated_at": shuffled["generated_at"]}


# --- the files: reading and writing ---------------------------------------------------------------------------------

def test_the_ledger_and_the_runs_are_only_read(tmp_path):
    s = make_set(tmp_path)
    s.write()
    before = snapshot(*s.dirs.values(), s.glossary_dir) | {"ledger": s.ledger.read_bytes()}

    costs.build_costs(list(s.dirs.values()), s.glossary_dir, s.ledger, s.ranges)

    assert snapshot(*s.dirs.values(), s.glossary_dir) | {"ledger": s.ledger.read_bytes()} == before


def test_the_cli_writes_costs_json_into_a_new_directory_and_prints_the_table(cli, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(costs, "datetime", FixedClock)
    s = make_set(tmp_path)
    s.write()
    out_root = tmp_path / "out" / "analysis"  # does not exist yet

    code = cli.main(cli_args(s, out_root, "--cached-input-price-per-m", "0.25"))

    assert code == 0
    (out_dir,) = list(out_root.iterdir())
    assert out_dir.name == "20261029T120000Z-costs" and sorted(p.name for p in out_dir.iterdir()) == ["costs.json"]
    written = read_json(out_dir / "costs.json")
    again = costs.build_costs(list(s.dirs.values()), s.glossary_dir, s.ledger, s.ranges, cached_input_price_per_m=0.25)
    assert written == json.loads(json.dumps(again | {"generated_at": written["generated_at"]}))
    out = capsys.readouterr().out
    assert out.endswith(f"Wrote {out_dir / 'costs.json'}\n")
    assert "cost per page, lower (cached input $0)" in out and "cost per page, cached at $0.25/M" in out
    assert "warnings: none" in out


def test_a_second_run_in_the_same_second_gets_a_directory_of_its_own_and_never_writes_into_an_existing_one(tmp_path,
                                                                                                          monkeypatch):
    monkeypatch.setattr(costs, "datetime", FixedClock)
    document = make_set(tmp_path).build()
    out_root = tmp_path / "analysis"
    first = costs.write_costs(document, out_root)
    (first / "costs.json").write_text("first", encoding="utf-8")
    (out_root / "20261029T120000Z-costs-2").mkdir()  # taken, too

    second = costs.write_costs(document, out_root)

    assert (first.name, second.name) == ("20261029T120000Z-costs", "20261029T120000Z-costs-3")
    assert (first / "costs.json").read_text(encoding="utf-8") == "first"
    assert sorted(p.name for p in second.iterdir()) == ["costs.json"]


def test_a_failing_write_leaves_no_directory_and_no_temporary_file(tmp_path, monkeypatch):
    document = make_set(tmp_path).build()
    monkeypatch.setattr(costs.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk on fire")))

    with pytest.raises(OSError, match="disk on fire"):
        costs.write_costs(document, tmp_path / "analysis")

    assert list((tmp_path / "analysis").iterdir()) == []


def test_the_cli_exits_1_when_the_file_cannot_be_written(cli, tmp_path, capsys):
    s = make_set(tmp_path)
    s.write()
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("a file", encoding="utf-8")

    code = cli.main(cli_args(s, blocker / "analysis"))

    assert code == 1 and "Nothing was written: could not write the costs" in capsys.readouterr().err


def test_the_default_paths_are_the_projects(cli):
    assert cli.GLOSSARY_DIR == config.ROOT / "eval" / "gold_glossary"
    assert cli.LEDGER == config.LEDGER_PATH == config.ROOT / "runs" / "usage.jsonl"
    assert cli.ANALYSIS_DIR == config.ROOT / "runs" / "analysis"


def test_the_script_runs_as_a_program(tmp_path):
    s = make_set(tmp_path)
    s.write()
    script = str(config.ROOT / "eval" / "costs.py")
    args = [sys.executable, script, *cli_args(s, tmp_path / "r")]

    done = subprocess.run(args, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "baseline" in done.stdout and (next((tmp_path / "r").iterdir()) / "costs.json").is_file()

    refused = subprocess.run([sys.executable, script, str(tmp_path / "nowhere"), "--ledger-range", "x=1-2"],
                             capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, timeout=120)
    assert refused.returncode == 2 and "Nothing was written" in refused.stderr


def test_the_tool_reads_files_only():
    names = set()
    for path in (config.ROOT / "backend" / "fukidashi" / "evaluation" / "costs.py", config.ROOT / "eval" / "costs.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names.update(part for alias in node.names for part in alias.name.split("."))
            elif isinstance(node, ast.ImportFrom):
                names.update((node.module or "").split("."))
                names.update(alias.name for alias in node.names)
    assert not names & {"sqlite3", "store", "db", "runs", "llm", "openai", "httpx", "requests", "socket", "subprocess"}


def test_the_tool_never_opens_a_file_for_writing_except_its_own_output():
    source = (config.ROOT / "backend" / "fukidashi" / "evaluation" / "costs.py").read_text(encoding="utf-8")
    writes = [node.lineno for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr in {"write_text", "write_bytes", "open"}]

    assert len(writes) == 1  # costs.json.tmp, in write_costs: the ledger and the eval runs are only read


# --- the printed table ------------------------------------------------------------------------------------------------

def rows(text, label):
    """The cells of every row of the table that starts with `label`, one list for each block it appears in."""
    return [re.split(r"\s{2,}", line.strip())[1:] for line in text.splitlines() if line.startswith(label)]


def test_the_table_has_a_column_for_each_config_and_every_figure_under_its_own_label(tmp_path):
    text = costs.render(make_set(tmp_path).build())

    # blocks, in order: the common pages (pooled, A, B), then the attempted pages (pooled, A, B)
    assert [r[0] for r in rows(text, "pages ")] == ["3", "2", "1", "4", "3", "1"]
    assert rows(text, "pages ")[0] == ["3", "3"]  # baseline, full_no_kb
    assert rows(text, "calls per page")[0] == ["1.00", "2.33"]
    assert rows(text, "tokens per page: uncached input")[0] == ["1500.0", "5033.3"]
    assert rows(text, "tokens per page: cached input")[0] == ["0.0", "5666.7"]
    assert rows(text, "tokens per page: output")[0] == ["300.0", "450.0"]
    assert rows(text, "cached share of input")[0] == ["0.0%", "53.0%"]
    assert rows(text, "cost per page, lower")[0] == ["$0.002700", "$0.006833"]
    assert rows(text, "cost per page, upper")[0] == ["$0.002700", "$0.012500"]
    assert rows(text, "latency p50 / p95 (s)")[0] == ["2.0 / 2.9", "20.0 / 38.0"]
    assert rows(text, "latency mean (s)")[0] == ["2.0", "23.3"]


def test_an_unknown_figure_is_n_a_with_the_number_of_calls_that_make_it_so(tmp_path):
    text = costs.render(make_set(tmp_path).build())

    pooled_attempted = 3  # the fourth block: all attempted pages, pooled
    assert rows(text, "tokens per page: uncached input")[pooled_attempted] == ["1875.0", "n/a (1 unknown)"]
    assert rows(text, "cached share of input")[pooled_attempted] == ["0.0%", "n/a (1 unknown)"]
    assert rows(text, "cost per page, upper")[pooled_attempted] == ["$0.002875", "n/a (1 unknown)"]
    assert rows(text, "calls per page")[pooled_attempted] == ["1.00", "2.25"]  # a call is a call, whatever it returned


def test_every_row_is_as_wide_as_the_others_and_no_cell_touches_the_one_before_it(tmp_path):
    text = costs.render(make_set(tmp_path).build())

    widths = {len(l) for l in text.splitlines() if l.startswith(("pages ", "calls per page", "cost per page,", "tokens per"))}
    assert len(widths) == 1
    assert "  n/a (1 unknown)" in text


def test_the_table_names_the_runs_the_failed_pages_and_the_lines_without_a_cost(tmp_path):
    text = costs.render(make_set(tmp_path).build())

    assert "prices recovered from 12 lines: input $1 and output $4 per million tokens" in text
    assert "20261029T000000Z-baseline-tojime_no_siora  baseline on tojime_no_siora: lines 3-5, 3 calls, $0.008800" in text
    assert ("full_no_kb on tojime_no_siora: lines 8-14, 7 calls, $0.035600 (+ 1 without a cost)") in text
    assert ("  full_no_kb: tojime_no_siora/3: 2 calls (ledger lines 13-14, 1 without usage): ProviderError: HTTP 402"
            in text)
    assert "ledger lines without a cost (never counted as 0): 14 (full_no_kb tojime_no_siora/3)" in text
    assert text.rstrip().endswith("warnings: none")


def test_a_single_call_page_is_one_call_not_one_calls(tmp_path):
    s = Fixture(tmp_path)
    s.add_run("baseline", A, [spec(1, [(1000, 0, 200)])])
    s.add_run("full_no_kb", A, [spec(1, [(500, 0, 50)], status="failed", latency=2.0, error="E" * 300)])

    text = costs.render(s.build())

    assert "tojime_no_siora/1: 1 call (ledger line 4)" in text and "1 calls" not in text
    assert "baseline on tojime_no_siora: line 3, 1 call, $0.001800" in text
    assert "E" * 99 + "…" in text and "E" * 101 not in text  # the error is cut in the table, whole in costs.json
    assert s.build()["failed_pages"][0]["error"] == "E" * 300


def test_without_a_failed_page_or_a_missing_cost_the_table_says_none(tmp_path):
    s = Fixture(tmp_path)
    s.add_run("baseline", A, [spec(1, [(1000, 0, 200)]), spec(2, [(2000, 0, 300)])])

    text = costs.render(s.build())

    assert "failed pages: none" in text and "ledger lines without a cost: none" in text
