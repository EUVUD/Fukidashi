"""`eval/charts.py`, the self-contained chart script. Nothing here draws: `matplotlib` is not installed in the project's
environment, so the script's `draw` hook gets a fake that writes placeholder files. Rendering is checked by the official run,
not by the suite."""

import ast
import copy
import hashlib
import importlib.util
import json
import re
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fukidashi import config

SCRIPT = config.ROOT / "eval" / "charts.py"

spec = importlib.util.spec_from_file_location("charts_script_under_test", SCRIPT)
charts_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(charts_script)

CONFIGS = ["baseline", "full_no_kb", "full"]
NOW = datetime(2026, 11, 4, 12, 30, 5, tzinfo=timezone.utc)
DIRECTORY = "20261104T123005Z-charts"
EVAL_IDS = [f"2026110{i}T00000{b}Z-{c}" for c in CONFIGS for i, b in enumerate("123", start=1)]

# The official full-vs-baseline and full-vs-full_no_kb numbers (point estimate, low, high), in the files' own units.
OFFICIAL = {
    "full vs baseline": {
        "comet": (-0.0027804839686661476, -0.0068311630115811955, 0.0014200391837191266),
        "chrf": (-1.2061874142039812, -2.331980260566402, -0.11919744416885242),
        "bleu": (-2.1313960638085767, -3.4981110360321455, -0.7945970867731954),
        "term_accuracy": (0.02051282051282055, -0.027027027027027084, 0.06763285024154586),
        "term_consistency": (0.04489619816252988, 0.008048166533839416, 0.06740351635093202),
        "mqm": (0.32696390658174046, -0.12085774083633267, 0.7827645992440024),
    },
    "full vs full_no_kb": {
        "term_accuracy": (0.0, -0.051136363636363646, 0.04999999999999993),
        "term_consistency": (0.02604166666666663, -0.011173768294293026, 0.06877073996397992),
    },
}


def breakdown(cells):
    """An MQM breakdown as the report holds it, from {category: (critical, major, minor)}."""
    by_both = {c: dict(zip(("critical", "major", "minor"), n)) for c, n in cells.items()}
    return {"errors": sum(sum(n) for n in cells.values()), "segments_with_errors": 4,
            "by_severity": {s: sum(n[i] for n in cells.values()) for i, s in enumerate(("critical", "major", "minor"))},
            "by_category": {c: sum(n) for c, n in cells.items()}, "by_category_and_severity": by_both}


CELLS = {
    "baseline": {"accuracy-mistranslation": (2, 3, 1), "style": (0, 0, 4)},
    "full_no_kb": {"accuracy-mistranslation": (1, 3, 2), "style": (0, 1, 3), "unknown": (0, 0, 2)},
    "full": {"accuracy-mistranslation": (1, 2, 2), "style": (0, 0, 3), "other": (1, 0, 0)},
}


def run_numbers(config_index, scale):
    """Slightly different numbers per config, so that a value read from the wrong config shows."""
    return {"comet": 0.84 + 0.001 * config_index, "accuracy": 0.8 + 0.01 * config_index * scale,
            "occurrences": 10 * scale}


def make_report():
    report = {"runs": [{"eval_id": e} for e in EVAL_IDS], "books": ["book_a", "book_b"],
              "common_pages": {"book_a": [1, 2, 3], "book_b": [1, 2]}, "pooled": {}, "by_book": {"book_a": {}, "book_b": {}},
              "mqm": {"use": "for reference only", "why": "the judge failed the agreement check (OUTLINE 4.5)"}}
    for i, c in enumerate(CONFIGS):
        for scope, scale, pages in ((report["pooled"], 5, 5), (report["by_book"]["book_a"], 3, 3),
                                    (report["by_book"]["book_b"], 2, 2)):
            n = run_numbers(i, scale)
            scope[c] = {"pages": pages, "comet": {"mean": n["comet"] + 0.01 * scale},
                        "terms": {"primary": {"accuracy": n["accuracy"], "occurrences": n["occurrences"]}}}
        report["pooled"][c]["mqm"] = {"scored": 9, "segments": 10, "breakdown": breakdown(CELLS[c])}
    return report


def make_bootstrap():
    comparisons = []
    for name, other, n_pages in (("full vs baseline", "baseline", 7), ("full vs full_no_kb", "full_no_kb", 5)):
        metrics = {}
        for key, (point, low, high) in OFFICIAL[name].items():
            use = "for reference only" if key == "mqm" else "result"
            why = "neither GEMBA-MQM judge passed the agreement check (OUTLINE 4.5)" if key == "mqm" else None
            metrics[key] = {"point_estimate": point, "interval": {"low": low, "high": high},
                            "label": {"use": use, "why": why}}
        comparisons.append({"name": name, "config": "full", "other": other, "n_pages": n_pages, "metrics": metrics})
    return {"method": {"resamples": 10000}, "inputs": {"runs": [{"eval_id": e} for e in EVAL_IDS]},
            "comparisons": comparisons}


def make_costs():
    costs = {"runs": [{"eval_id": e} for e in EVAL_IDS], "common_pages": {"book_a": [1, 2, 3], "book_b": [1, 2]},
             "pooled": {}}
    for i, c in enumerate(CONFIGS):
        costs["pooled"][c] = {"common": {
            "pages": 5, "cost_per_page_usd": {"lower": 0.001 * (i + 1), "upper": 0.002 * (i + 1)},
            "latency_s": {"n": 5, "p50_s": 3.0 * (i + 1), "p95_s": 9.0 * (i + 1), "mean_s": 4.0 * (i + 1)}}}
    return costs


class FakeDraw:
    """Stands in for `render`: records what it is asked to draw, writes a placeholder for each file, names a version."""

    def __init__(self, version="9.9.9", fail=None):
        self.version, self.fail, self.calls = version, fail, []

    def __call__(self, charts, out_dir):
        self.calls.append(copy.deepcopy(charts))
        for name in charts:
            (out_dir / f"{name}.svg").write_text(f"<svg>{name}</svg>", encoding="utf-8")
            (out_dir / f"{name}.png").write_bytes(b"png:" + name.encode())
        if self.fail:
            raise self.fail
        return self.version


def write_inputs(directory, report=None, bootstrap=None, costs=None):
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, data, fallback in (("report", report, make_report), ("bootstrap", bootstrap, make_bootstrap),
                                 ("costs", costs, make_costs)):
        paths[name] = directory / f"{name}.json"
        paths[name].write_text(json.dumps(fallback() if data is None else data), encoding="utf-8")
    return paths


def run(tmp_path, *, report=None, bootstrap=None, costs=None, draw=None, out="out", directory="in", now=NOW):
    paths = write_inputs(tmp_path / directory, report, bootstrap, costs)
    draw = draw or FakeDraw()
    code = charts_script.main(["--report", str(paths["report"]), "--bootstrap", str(paths["bootstrap"]),
                               "--costs", str(paths["costs"]), "--out-root", str(tmp_path / out)],
                              draw=draw, now=lambda: now)
    return code, draw, tmp_path / out


def output(out_root):
    (directory,) = [p for p in out_root.iterdir()]
    return directory, json.loads((directory / "charts.json").read_text(encoding="utf-8"))


def sources():
    data = {"report": make_report(), "bootstrap": make_bootstrap(), "costs": make_costs()}
    return {name: charts_script.Source(name, f"{name}.json", d) for name, d in data.items()}, data


def build(**changes):
    srcs, data = sources()
    for name, change in changes.items():
        change(data[name])
    return charts_script.build_charts(*[charts_script.Source(n, f"{n}.json", data[n]) for n in ("report", "bootstrap", "costs")])


def resolve(data, path):
    node = data
    for token in path.split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        node = node[int(token)] if isinstance(node, list) else node[token]
    return node


def refused(tmp_path, capsys, *, expect, **inputs):
    """Run on inputs that must be refused: exit 2, the cause on stderr, nothing drawn and nothing written."""
    code, draw, out_root = run(tmp_path, **inputs)
    err = capsys.readouterr().err
    assert code == 2 and draw.calls == []
    assert err.startswith("Nothing was written: ") and expect in err, err
    assert not out_root.exists()
    return err


def test_the_script_is_self_contained():
    """Its environment is its inline metadata (matplotlib at one exact version), its top-level imports are the standard
    library, and `matplotlib` is imported in one function only; neither `pyproject.toml` nor `uv.lock` mentions it."""
    source = SCRIPT.read_text(encoding="utf-8")
    block = re.search(r"^# /// script\n((?:#.*\n)+?)# ///$", source, re.MULTILINE)
    assert block is not None
    metadata = tomllib.loads("\n".join(line.removeprefix("# ").removeprefix("#") for line in block.group(1).splitlines()))
    (dependency,) = metadata["dependencies"]
    assert re.fullmatch(r"matplotlib==\d+\.\d+\.\d+", dependency), dependency

    tree = ast.parse(source)

    def imported(node):
        return [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]

    top_level = {name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                 for name in imported(node)}
    assert top_level <= sys.stdlib_module_names, top_level - sys.stdlib_module_names

    everywhere = {name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                  for name in imported(node)}
    assert "fukidashi" not in everywhere and "dotenv" not in everywhere
    (renderer,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "render"]
    inside = {name.split(".")[0] for node in ast.walk(renderer) if isinstance(node, (ast.Import, ast.ImportFrom))
              for name in imported(node)}
    assert "matplotlib" in inside
    assert everywhere - inside <= sys.stdlib_module_names  # nothing outside `render` imports a third-party module

    for name in ("pyproject.toml", "uv.lock"):
        assert "matplotlib" not in (config.ROOT / name).read_text(encoding="utf-8")


def test_everything_but_drawing_runs_without_matplotlib(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)  # `import matplotlib` raises ImportError

    code, draw, out_root = run(tmp_path)

    assert code == 0 and len(draw.calls) == 1 and output(out_root)[0].name == DIRECTORY


def test_every_number_equals_the_value_at_its_key_path_after_the_stated_conversion():
    """The numbers come from the right file, key and config, with the fractions converted to percentage points."""
    built = build()
    _, data = sources()
    checked = 0
    for name, chart in built["charts"].items():
        assert chart["numbers"], name
        for n in chart["numbers"]:
            raw = resolve(data[n["source"]], n["path"])
            assert n["value"] == (raw * 100 if n["conversion"] == "x100" else raw), (name, n)
            assert n["conversion"] in (None, "x100")
            checked += 1
    assert checked > 100
    by_label = {n["label"]: n for chart in built["charts"].values() for n in chart["numbers"]}
    term = by_label["full vs baseline: Term accuracy point estimate"]
    assert (term["path"], term["conversion"], term["unit"]) == (
        "/comparisons/0/metrics/term_accuracy/point_estimate", "x100", "pp")
    assert term["value"] == 0.02051282051282055 * 100
    assert by_label["full vs full_no_kb: Term consistency interval high"]["path"] == (
        "/comparisons/1/metrics/term_consistency/interval/high")
    assert by_label["full: pooled COMET"]["source"] == "report"
    assert by_label["Pooled: full term accuracy"]["path"] == "/pooled/full/terms/primary/accuracy"
    assert by_label["Pooled: full term accuracy"]["unit"] == "percent"
    assert by_label["book_b: baseline COMET"]["path"] == "/by_book/book_b/baseline/comet/mean"
    assert by_label["full: cost per page, cached input at $0"]["path"] == "/pooled/full/common/cost_per_page_usd/lower"
    assert by_label["full_no_kb: latency p95"]["path"] == "/pooled/full_no_kb/common/latency_s/p95_s"
    assert by_label["full_no_kb: latency p95"]["value"] == 18.0
    assert by_label["bootstrap resamples"]["value"] == 10000


def test_the_charts_draw_the_numbers_of_the_result_and_nothing_else():
    """`render` gets what `build_charts` built, so a drawn number is a listed number."""
    built = build()
    intervals = built["charts"]["intervals"]
    comparison = intervals["comparisons"][0]
    assert (comparison["name"], comparison["n_pages"]) == ("full vs baseline", 7)
    assert [p["metric"] for p in comparison["panels"]] == ["comet", "chrf", "bleu", "term_accuracy", "term_consistency",
                                                            "mqm"]
    panel = {p["metric"]: p for p in comparison["panels"]}["term_consistency"]
    assert (panel["point"], panel["low"], panel["high"]) == tuple(x * 100 for x in OFFICIAL["full vs baseline"]["term_consistency"])
    assert panel["spread"] == 7.0 and panel["unit"] == "pp"
    assert [p["metric"] for p in intervals["comparisons"][1]["panels"]] == ["term_accuracy", "term_consistency"]

    rows = built["charts"]["by_book"]["rows"]
    assert [(r["key"], r["pages"], r["occurrences"]) for r in rows] == [("book_a", 3, 30), ("book_b", 2, 20), ("pooled", 5, 50)]
    assert rows[0]["term_accuracy"]["baseline"] == pytest.approx(0.8 * 100) and rows[2]["comet"]["full"] == pytest.approx(0.892)
    assert built["charts"]["by_book"]["pages"] == 5

    cost = {p["config"]: p for p in built["charts"]["cost_vs_comet"]["points"]}
    assert (cost["baseline"]["lower"], cost["baseline"]["upper"]) == (0.001, 0.002)
    assert (cost["full"]["lower"], cost["full"]["upper"]) == (0.003, 0.006) and cost["full"]["comet"] == pytest.approx(0.892)

    latency = {p["stat"]: p["values"] for p in built["charts"]["latency"]["panels"]}
    assert latency == {"p50_s": {"baseline": 3.0, "full_no_kb": 6.0, "full": 9.0},
                       "p95_s": {"baseline": 9.0, "full_no_kb": 18.0, "full": 27.0},
                       "mean_s": {"baseline": 4.0, "full_no_kb": 8.0, "full": 12.0}}
    assert built["charts"]["latency"]["pages"] == built["charts"]["cost_vs_comet"]["pages"] == 5


EXPECTED_READINGS = {
    ("full vs baseline", "comet"): "interval includes 0",
    ("full vs baseline", "chrf"): "lower",
    ("full vs baseline", "bleu"): "lower",
    ("full vs baseline", "term_accuracy"): "interval includes 0",
    ("full vs baseline", "term_consistency"): "not separated from run-to-run variation",
    ("full vs baseline", "mqm"): "interval includes 0",
    ("full vs full_no_kb", "term_accuracy"): "interval includes 0",
    ("full vs full_no_kb", "term_consistency"): "interval includes 0",
}


def test_the_readings_on_the_official_numbers_are_the_ones_the_outline_gives(tmp_path, capsys):
    built = build()

    assert {(r["comparison"], r["metric"]): r["reading"] for r in built["readings"]} == EXPECTED_READINGS
    assert {r["metric"]: r["use"] for r in built["readings"]} == {
        "comet": "result", "chrf": "result", "bleu": "result", "term_accuracy": "result", "term_consistency": "result",
        "mqm": "for reference only"}
    code, _, out_root = run(tmp_path)
    printed = capsys.readouterr().out
    assert code == 0
    for (comparison, metric), reading in EXPECTED_READINGS.items():
        label = charts_script.METRICS[metric].label
        (line,) = [line for line in printed.splitlines() if line.startswith(comparison) and f" {label} " in line]
        assert reading in line
    assert "[for reference only]" in [line for line in printed.splitlines() if " MQM " in line][0]
    assert f"charts -> {out_root / DIRECTORY}" in printed
    assert json.loads((out_root / DIRECTORY / "charts.json").read_text())["readings"] == built["readings"]


@pytest.mark.parametrize("point, low, high, spread, reading", [
    (-1.0, -3.0, 2.0, 0.5, "interval includes 0"),                   # 0 inside, whatever the size
    (-1.0, -3.0, 0.0, 0.5, "interval includes 0"),                   # the interval touches 0 from below
    (1.0, 0.0, 3.0, 0.5, "interval includes 0"),                     # and from above
    (0.2, 0.1, 0.3, 0.5, "not separated from run-to-run variation"),  # excludes 0, smaller than the spread
    (-0.2, -0.3, -0.1, 0.5, "not separated from run-to-run variation"),
    (0.5, 0.1, 0.9, 0.5, "higher"),                                  # a difference equal to the spread is 'at least'
    (-0.5, -0.9, -0.1, 0.5, "lower"),
    (2.0, 0.1, 3.0, 0.5, "higher"),
    (-2.0, -3.0, -0.1, 0.5, "lower"),
])
def test_the_three_readings(point, low, high, spread, reading):
    assert charts_script.read_interval(point, low, high, spread) == reading


def test_a_fraction_converted_to_percentage_points_does_not_decide_a_tie():
    """0.29 x 100 is 28.999999999999996, which is below 29.0 by arithmetic and equal to it by the figures."""
    assert 0.29 * 100 < 29.0
    assert charts_script.read_interval(0.29 * 100, 0.1 * 100, 0.5 * 100, 29.0) == "higher"
    assert charts_script.read_interval(-0.29 * 100, -0.5 * 100, -0.1 * 100, 29.0) == "lower"
    assert 0.017 * 100 > 1.7  # and the other way round
    assert charts_script.read_interval(0.017 * 100, 0.01 * 100, 0.03 * 100, 1.7) == "higher"
    assert charts_script.read_interval(0.0169 * 100, 0.01 * 100, 0.03 * 100, 1.7) == "not separated from run-to-run variation"


def test_the_readings_follow_the_files_through_build_charts():
    """A difference equal to the spread, and an interval touching 0, read from the bootstrap file itself."""
    def change(bootstrap):
        metrics = bootstrap["comparisons"][0]["metrics"]
        metrics["term_accuracy"].update(point_estimate=0.017, interval={"low": 0.01, "high": 0.03})  # 1.7 pp = the spread
        metrics["term_consistency"].update(point_estimate=0.05, interval={"low": 0.0, "high": 0.07})  # touches 0
        metrics["chrf"].update(point_estimate=0.55, interval={"low": 0.1, "high": 1.0})
        metrics["bleu"].update(point_estimate=-0.8, interval={"low": -1.0, "high": -0.2})

    readings = {(r["comparison"], r["metric"]): r["reading"] for r in build(bootstrap=change)["readings"]}

    assert readings[("full vs baseline", "term_accuracy")] == "higher"
    assert readings[("full vs baseline", "term_consistency")] == "interval includes 0"
    assert readings[("full vs baseline", "chrf")] == "higher"
    assert readings[("full vs baseline", "bleu")] == "not separated from run-to-run variation"


def test_the_spreads_are_the_ones_the_outline_row_states():
    """The spreads are in no output file, so the script holds them as constants that cite OUTLINE §8."""
    text = (config.ROOT / "OUTLINE.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| Reading the intervals next to the run-to-run spread"))
    found = re.search(r"COMET ([\d.]+), chrF ([\d.]+), BLEU ([\d.]+), primary term accuracy ([\d.]+) pp, term consistency "
                      r"([\d.]+) pp, MQM ([\d.]+)\)", row)
    assert found is not None
    metrics = charts_script.METRICS
    assert [metrics[k].spread for k in ("comet", "chrf", "bleu", "term_accuracy", "term_consistency", "mqm")] == [
        float(v) for v in found.groups()]
    assert metrics["term_accuracy"].scale == metrics["term_consistency"].scale == 100  # their spreads are in pp
    constants = build()["constants"]
    assert {c["metric"]: c["value"] for c in constants} == {k: m.spread for k, m in metrics.items()}
    assert all("OUTLINE §8" in c["cites"] for c in constants)


def test_a_label_that_is_for_reference_only_is_read_from_the_file_not_assumed():
    def change(bootstrap):
        bootstrap["comparisons"][1]["metrics"]["term_consistency"]["label"] = {"use": "for reference only", "why": "x"}
        bootstrap["comparisons"][0]["metrics"]["mqm"]["label"] = {"use": "result", "why": None}

    uses = {(r["comparison"], r["metric"]): r["use"] for r in build(bootstrap=change)["readings"]}

    assert uses[("full vs full_no_kb", "term_consistency")] == "for reference only"
    assert uses[("full vs baseline", "mqm")] == "result"
    assert uses[("full vs baseline", "term_consistency")] == "result"


def test_the_mqm_chart_shows_every_category_of_the_report_and_takes_its_use_from_the_report():
    chart = build()["charts"]["mqm_breakdown"]

    assert chart["use"] == "for reference only" and chart["why"].startswith("the judge failed")
    assert sorted(chart["categories"]) == sorted({c for cells in CELLS.values() for c in cells})
    assert chart["categories"][0] == "accuracy-mistranslation" and "other" in chart["categories"]  # no hidden bucket
    # a category the judge named for one config only is absent from the others, not a 0
    assert "unknown" not in chart["configs"]["baseline"]["categories"]
    assert chart["configs"]["full_no_kb"]["categories"]["unknown"]["total"] == 2
    assert chart["configs"]["full"]["categories"]["accuracy-mistranslation"] == {
        "total": 5, "by_severity": {"critical": 1, "major": 2, "minor": 2}}
    assert chart["configs"]["baseline"]["by_severity"] == {"critical": 2, "major": 3, "minor": 5}
    assert chart["configs"]["baseline"]["errors"] == 10

    def other_use(report):
        report["mqm"]["use"] = "result"

    assert build(report=other_use)["charts"]["mqm_breakdown"]["use"] == "result"


def test_a_null_number_is_left_out_and_listed_under_missing_never_drawn_as_0(tmp_path):
    report, bootstrap, costs = make_report(), make_bootstrap(), make_costs()
    costs["pooled"]["full"]["common"]["cost_per_page_usd"]["lower"] = None
    costs["pooled"]["baseline"]["common"]["latency_s"]["p95_s"] = None
    report["pooled"]["full_no_kb"]["comet"]["mean"] = None
    report["by_book"]["book_a"]["full"]["terms"]["primary"]["accuracy"] = None
    report["pooled"]["baseline"]["mqm"]["breakdown"]["segments_with_errors"] = None
    bootstrap["comparisons"][0]["metrics"]["comet"]["interval"]["high"] = None

    code, draw, out_root = run(tmp_path, report=report, bootstrap=bootstrap, costs=costs)

    assert code == 0
    (charts,) = draw.calls
    cost = {p["config"]: p for p in charts["cost_vs_comet"]["points"]}
    assert cost["full"]["lower"] is None and cost["full"]["upper"] == 0.006
    assert cost["full_no_kb"]["comet"] is None
    latency = {p["stat"]: p["values"] for p in charts["latency"]["panels"]}
    assert latency["p95_s"]["baseline"] is None and latency["p50_s"]["baseline"] == 3.0
    rows = {r["key"]: r for r in charts["by_book"]["rows"]}
    assert rows["book_a"]["term_accuracy"]["full"] is None and rows["book_a"]["term_accuracy"]["baseline"] is not None
    assert rows["pooled"]["comet"]["full_no_kb"] is None
    panel = {p["metric"]: p for p in charts["intervals"]["comparisons"][0]["panels"]}["comet"]
    assert panel["high"] is None and panel["reading"] is None
    assert charts["mqm_breakdown"]["configs"]["baseline"]["segments_with_errors"] is None

    _, record = output(out_root)
    assert {(m["chart"], m["path"]) for m in record["missing"]} == {
        ("cost_vs_comet", "/pooled/full/common/cost_per_page_usd/lower"),
        ("latency", "/pooled/baseline/common/latency_s/p95_s"),
        ("cost_vs_comet", "/pooled/full_no_kb/comet/mean"),
        ("by_book", "/pooled/full_no_kb/comet/mean"),
        ("by_book", "/by_book/book_a/full/terms/primary/accuracy"),
        ("mqm_breakdown", "/pooled/baseline/mqm/breakdown/segments_with_errors"),
        ("intervals", "/comparisons/0/metrics/comet/interval/high"),
    }
    assert all(set(m) == {"chart", "label", "source", "path"} for m in record["missing"])
    for name, chart in record["charts"].items():  # a missing number is not among the numbers either
        listed = {n["path"] for n in chart["numbers"]}
        assert not listed & {m["path"] for m in record["missing"] if m["chart"] == name}
    assert [r["reading"] for r in record["readings"] if (r["comparison"], r["metric"]) == ("full vs baseline", "comet")] == [None]


@pytest.mark.parametrize("which", ["report", "bootstrap", "costs"])
def test_a_missing_input_is_refused_and_named(tmp_path, capsys, which):
    paths = write_inputs(tmp_path / "in")
    paths[which].unlink()
    draw = FakeDraw()

    code = charts_script.main(["--report", str(paths["report"]), "--bootstrap", str(paths["bootstrap"]),
                               "--costs", str(paths["costs"]), "--out-root", str(tmp_path / "out")], draw=draw)

    err = capsys.readouterr().err
    assert code == 2 and draw.calls == [] and f"{which}.json" in err and "missing" in err
    assert err.startswith("Nothing was written: ") and not (tmp_path / "out").exists()


@pytest.mark.parametrize("which", ["report", "bootstrap", "costs"])
@pytest.mark.parametrize("text, cause", [("{not json", "not valid JSON"), ("[1, 2]", "not a JSON object"),
                                          ('{"a": 1}', "is missing")])
def test_an_input_that_is_not_the_expected_json_is_refused_and_named(tmp_path, capsys, which, text, cause):
    paths = write_inputs(tmp_path / "in")
    paths[which].write_text(text, encoding="utf-8")
    draw = FakeDraw()

    code = charts_script.main(["--report", str(paths["report"]), "--bootstrap", str(paths["bootstrap"]),
                               "--costs", str(paths["costs"]), "--out-root", str(tmp_path / "out")], draw=draw)

    err = capsys.readouterr().err
    assert code == 2 and draw.calls == [] and f"{which}.json" in err and cause in err
    assert not (tmp_path / "out").exists()


def test_inputs_that_name_other_eval_runs_are_refused(tmp_path, capsys):
    costs = make_costs()
    costs["runs"][0]["eval_id"] = "20261101T000000Z-baseline"  # one eval id changed

    err = refused(tmp_path, capsys, costs=costs, expect="do not name the same eval runs")

    assert "costs.json lacks" in err and EVAL_IDS[0] in err
    assert "report.json lacks" in err and "20261101T000000Z-baseline" in err


@pytest.mark.parametrize("where", ["report", "bootstrap"])
def test_an_eval_run_missing_from_one_file_is_refused(tmp_path, capsys, where):
    data = {"report": make_report(), "bootstrap": make_bootstrap()}
    runs = data["report"]["runs"] if where == "report" else data["bootstrap"]["inputs"]["runs"]
    del runs[3]

    err = refused(tmp_path, capsys, **data, expect="do not name the same eval runs")

    assert f"{where}.json lacks [{EVAL_IDS[3]!r}]" in err


def test_an_eval_run_named_twice_is_refused(tmp_path, capsys):
    report = make_report()
    report["runs"].append(dict(report["runs"][0]))

    refused(tmp_path, capsys, report=report, expect="report.json names an eval run more than once")


def test_common_pages_that_differ_are_refused(tmp_path, capsys):
    costs = make_costs()
    costs["common_pages"]["book_a"] = [1, 2]

    refused(tmp_path, capsys, costs=costs, expect="do not name the same common pages")


@pytest.mark.parametrize("name", ["full vs baseline", "full vs full_no_kb"])
def test_a_missing_comparison_is_refused(tmp_path, capsys, name):
    bootstrap = make_bootstrap()
    bootstrap["comparisons"] = [c for c in bootstrap["comparisons"] if c["name"] != name]

    refused(tmp_path, capsys, bootstrap=bootstrap, expect=f"the comparison {name!r} is absent")


def test_a_comparison_that_is_not_full_minus_the_other_config_is_refused(tmp_path, capsys):
    bootstrap = make_bootstrap()
    bootstrap["comparisons"][1]["other"] = "baseline"

    refused(tmp_path, capsys, bootstrap=bootstrap, expect="is not full minus full_no_kb")


@pytest.mark.parametrize("comparison, metric", [(0, m) for m in OFFICIAL["full vs baseline"]]
                         + [(1, m) for m in OFFICIAL["full vs full_no_kb"]])
def test_a_missing_metric_is_refused(tmp_path, capsys, comparison, metric):
    bootstrap = make_bootstrap()
    name = bootstrap["comparisons"][comparison]["name"]
    del bootstrap["comparisons"][comparison]["metrics"][metric]

    refused(tmp_path, capsys, bootstrap=bootstrap, expect=f"the comparison {name!r} has no metric {metric!r}")


def test_a_breakdown_whose_categories_do_not_sum_to_its_errors_is_refused(tmp_path, capsys):
    report = make_report()
    report["pooled"]["full_no_kb"]["mqm"]["breakdown"]["by_category"]["style"] += 1
    report["pooled"]["full_no_kb"]["mqm"]["breakdown"]["by_category_and_severity"]["style"]["minor"] += 1

    err = refused(tmp_path, capsys, report=report, expect="report.json: /pooled/full_no_kb/mqm/breakdown/by_category sums to")

    assert "not to /pooled/full_no_kb/mqm/breakdown/errors" in err


def test_a_category_whose_severities_do_not_add_up_is_refused(tmp_path, capsys):
    report = make_report()
    report["pooled"]["baseline"]["mqm"]["breakdown"]["by_category_and_severity"]["style"]["major"] += 1

    refused(tmp_path, capsys, report=report,
            expect="/pooled/baseline/mqm/breakdown/by_category_and_severity/style does not sum to")


def test_categories_that_the_two_breakdown_tables_do_not_share_are_refused(tmp_path, capsys):
    report = make_report()
    del report["pooled"]["baseline"]["mqm"]["breakdown"]["by_category_and_severity"]["style"]

    refused(tmp_path, capsys, report=report, expect="does not list the same categories")


def test_configs_that_cover_other_pages_than_each_other_are_refused(tmp_path, capsys):
    report = make_report()
    report["by_book"]["book_a"]["full"]["pages"] = 2

    refused(tmp_path, capsys, report=report, expect="book_a: pages differs between the configs")


@pytest.mark.parametrize("which, mutate, expect", [
    ("report", lambda r: r["pooled"]["full"].pop("comet"), "report.json: /pooled/full/comet/mean is missing"),
    ("report", lambda r: r["mqm"].pop("use"), "report.json: /mqm/use is missing"),
    ("report", lambda r: r.pop("books"), "report.json: /books is missing"),
    ("costs", lambda c: c["pooled"]["baseline"].pop("common"), "costs.json: /pooled/baseline/common/pages is missing"),
    ("costs", lambda c: c["pooled"]["baseline"]["common"].pop("latency_s"),
     "costs.json: /pooled/baseline/common/latency_s/p50_s is missing"),
    ("costs", lambda c: c["pooled"]["full"]["common"]["cost_per_page_usd"].pop("upper"),
     "costs.json: /pooled/full/common/cost_per_page_usd/upper is missing"),
    ("bootstrap", lambda b: b["method"].pop("resamples"), "bootstrap.json: /method/resamples is missing"),
])
def test_a_key_the_charts_need_that_is_absent_is_refused_and_named(tmp_path, capsys, which, mutate, expect):
    data = {"report": make_report(), "bootstrap": make_bootstrap(), "costs": make_costs()}
    mutate(data[which])

    refused(tmp_path, capsys, **data, expect=expect)


@pytest.mark.parametrize("bad", ["0.84", True, float("nan")])
def test_a_value_that_is_not_a_number_is_refused(tmp_path, capsys, bad):
    report = make_report()
    report["pooled"]["baseline"]["comet"]["mean"] = bad

    refused(tmp_path, capsys, report=report, expect="/pooled/baseline/comet/mean is not a number")


def test_a_use_that_is_neither_result_nor_for_reference_only_is_refused(tmp_path, capsys):
    bootstrap = make_bootstrap()
    bootstrap["comparisons"][0]["metrics"]["comet"]["label"]["use"] = "maybe"

    refused(tmp_path, capsys, bootstrap=bootstrap, expect="/comparisons/0/metrics/comet/label/use is 'maybe'")


def test_charts_json_lists_the_inputs_the_script_the_readings_and_no_timestamp(tmp_path):
    paths = write_inputs(tmp_path / "in")
    draw = FakeDraw(version="3.11.2")

    code = charts_script.main(["--report", str(paths["report"]), "--bootstrap", str(paths["bootstrap"]),
                               "--costs", str(paths["costs"]), "--out-root", str(tmp_path / "out")],
                              draw=draw, now=lambda: NOW)

    assert code == 0
    directory, record = output(tmp_path / "out")
    assert directory.name == DIRECTORY
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        ["charts.json", *(f"{n}.{e}" for n in ("intervals", "by_book", "mqm_breakdown", "cost_vs_comet", "latency")
                          for e in ("svg", "png"))])
    assert set(record) == {"path_syntax", "inputs", "eval_runs", "script", "readings", "constants", "charts", "missing"}
    assert record["inputs"] == {name: {"file": f"in/{name}.json", "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest()}
                                for name in ("report", "bootstrap", "costs")}
    assert record["script"] == {"file": "eval/charts.py", "sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
                                "matplotlib": "3.11.2"}
    assert record["eval_runs"] == sorted(EVAL_IDS)
    assert list(record["charts"]) == ["intervals", "by_book", "mqm_breakdown", "cost_vs_comet", "latency"]
    assert record["charts"]["latency"]["files"] == ["latency.svg", "latency.png"]
    assert record["missing"] == []
    text = (directory / "charts.json").read_text(encoding="utf-8")
    assert not re.search(r"\d{4}-\d\d-\d\d|\d{8}T\d{6}Z-charts|generated|scored_at|_at\"", text)
    assert str(tmp_path) not in text  # no path of this machine
    assert [c["name"] for c in draw.calls[0]["intervals"]["comparisons"]] == ["full vs baseline", "full vs full_no_kb"]
    assert list(draw.calls[0]) == ["intervals", "by_book", "mqm_breakdown", "cost_vs_comet", "latency"]


def test_the_same_inputs_give_the_same_charts_json_whatever_the_out_root_and_the_time(tmp_path):
    first, _, out_a = run(tmp_path, out="a", now=NOW)
    second, _, out_b = run(tmp_path, out="b", now=datetime(2026, 12, 1, 8, 0, 0, tzinfo=timezone.utc))

    assert first == second == 0
    (dir_a, dir_b) = (output(out_a)[0], output(out_b)[0])
    assert dir_a.name != dir_b.name
    assert (dir_a / "charts.json").read_bytes() == (dir_b / "charts.json").read_bytes()


def test_the_directory_is_written_in_one_step_and_a_failing_draw_leaves_nothing(tmp_path, capsys):
    draw = FakeDraw(fail=RuntimeError("font not found"))

    code, _, out_root = run(tmp_path, draw=draw)

    err = capsys.readouterr().err
    assert code == 1 and "Nothing was written" in err and "font not found" in err
    assert list(out_root.iterdir()) == []  # not even the half-written directory


def test_a_failing_write_of_charts_json_leaves_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(charts_script, "write_json", lambda *a: (_ for _ in ()).throw(OSError("disk full")))

    code, draw, out_root = run(tmp_path)

    assert code == 1 and "disk full" in capsys.readouterr().err
    assert len(draw.calls) == 1 and list(out_root.iterdir()) == []


def test_a_directory_that_exists_already_is_not_overwritten(tmp_path, capsys):
    code, _, out_root = run(tmp_path)
    assert code == 0
    before = sorted(p.name for p in (out_root / DIRECTORY).iterdir())
    (out_root / DIRECTORY / "charts.json").write_text("{}", encoding="utf-8")

    code, draw, _ = run(tmp_path)  # the same second

    assert code == 1 and "exists already" in capsys.readouterr().err and draw.calls == []
    assert [p.name for p in out_root.iterdir()] == [DIRECTORY]
    assert sorted(p.name for p in (out_root / DIRECTORY).iterdir()) == before
    assert (out_root / DIRECTORY / "charts.json").read_text(encoding="utf-8") == "{}"


def test_the_default_out_root_is_runs_analysis():
    args = charts_script.parse_args(["--report", "a", "--bootstrap", "b", "--costs", "c"])

    assert args.out_root == Path("runs/analysis")


def test_the_inputs_are_read_never_rewritten(tmp_path):
    paths = write_inputs(tmp_path / "in")
    before = {n: p.read_bytes() for n, p in paths.items()}

    code = charts_script.main(["--report", str(paths["report"]), "--bootstrap", str(paths["bootstrap"]),
                               "--costs", str(paths["costs"]), "--out-root", str(tmp_path / "out")], draw=FakeDraw())

    assert code == 0 and {n: p.read_bytes() for n, p in paths.items()} == before
    assert sorted(p.name for p in (tmp_path / "in").iterdir()) == ["bootstrap.json", "costs.json", "report.json"]


def test_signed_numbers_use_the_minus_sign_on_the_charts_and_a_hyphen_in_the_terminal():
    assert charts_script.signed(-0.0028, 4) == "−0.0028"
    assert charts_script.signed(-0.0028, 4, minus="-") == "-0.0028"
    assert charts_script.signed(2.1, 1) == "+2.1"
    assert charts_script.signed(0.0, 1) == "0.0"
    assert charts_script.pointer("pooled", "full", 0, "a/b", "c~d") == "/pooled/full/0/a~1b/c~0d"
