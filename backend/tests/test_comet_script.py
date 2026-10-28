"""`eval/comet.py`, the self-contained COMET scorer. Nothing here loads a checkpoint: `comet` and `torch` are not installed in
the project's environment, so the script's `load` hook gets a fake scorer."""

import ast
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from fukidashi import config

SCRIPT = config.ROOT / "eval" / "comet.py"

spec = importlib.util.spec_from_file_location("comet_script_under_test", SCRIPT)
comet_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comet_script)

MODEL = "Unbabel/wmt22-comet-da"
OUTPUT = "comet-wmt22-comet-da.json"


def segment(idx, *, hyp=None, **overrides):
    return {
        "book": "book_a", "page_index": 2, "idx": idx, "run_id": 7,
        "ja": f"日本語{idx}", "hyp": f"译文{idx}" if hyp is None else hyp,
        "ref_zh_hans": f"参考{idx}", "ref_zh_hant": f"參考-hant-{idx}",
        **overrides,
    }


def make_run(tmp_path, segments=None, *, finished_at="2026-10-28T15:49:03+00:00", name="20261028T000000Z-full"):
    """An eval run's directory as the harness leaves it: manifest.json and segments.jsonl."""
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"eval_id": name, "finished_at": finished_at}), encoding="utf-8")
    segments = [segment(0), segment(1)] if segments is None else segments
    (run_dir / "segments.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in segments),
                                            encoding="utf-8")
    return run_dir


class FakeScorer:
    """Stands in for the COMET model: records what it is asked, answers one score per triple."""

    def __init__(self, scores=None, *, device="cpu"):
        self.scores, self.device = scores, device
        self.loaded, self.triples, self.batch_sizes = [], [], []

    def load(self, model, gpus):
        self.loaded.append((model, gpus))

        def predict(triples, batch_size):
            self.triples.append(triples)
            self.batch_sizes.append(batch_size)
            scores = [0.5 + 0.1 * i for i in range(len(triples))] if self.scores is None else self.scores
            return scores, sum(scores) / len(scores)

        return comet_script.Scorer(predict, self.device, {"unbabel-comet": "2.2.7", "torch": "9.9.9"})


def run(run_dir, *args, scorer=None):
    scorer = scorer or FakeScorer()
    return comet_script.main([str(run_dir), *args], load=scorer.load), scorer


def listing(directory):
    return sorted(p.name for p in directory.iterdir())


def test_the_script_is_self_contained():
    """AC2: its environment is its inline metadata, its top-level imports are the standard library, and `comet` and
    `torch` are touched in one function only."""
    source = SCRIPT.read_text(encoding="utf-8")
    block = re.search(r"^# /// script\n((?:#.*\n)+?)# ///$", source, re.MULTILINE)
    assert block is not None
    metadata = tomllib.loads("\n".join(line.removeprefix("# ").removeprefix("#") for line in block.group(1).splitlines()))
    assert metadata["requires-python"] == ">=3.12,<3.14" and "unbabel-comet==2.2.7" in metadata["dependencies"]

    tree = ast.parse(source)

    def imported(node):
        return [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]

    top_level = {name.split(".")[0] for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
                 for name in imported(node)}
    assert top_level <= sys.stdlib_module_names, top_level - sys.stdlib_module_names

    everywhere = {name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                  for name in imported(node)}
    assert "fukidashi" not in everywhere and "dotenv" not in everywhere
    (loader,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "load_scorer"]
    inside = {name.split(".")[0] for node in ast.walk(loader) if isinstance(node, (ast.Import, ast.ImportFrom))
              for name in imported(node)}
    assert {"comet", "torch"} <= inside
    assert everywhere - inside <= sys.stdlib_module_names  # nothing outside `load_scorer` imports a third-party module


def test_the_fake_predictor_gets_the_japanese_the_hypothesis_and_the_zh_hans_reference_in_file_order(tmp_path):
    """AC3"""
    run_dir = make_run(tmp_path, [segment(4, ref_zh_hant="繁體不是參考"), segment(1, ref_zh_hant="繁體2")])

    code, scorer = run(run_dir)

    assert code == 0
    assert scorer.triples == [[{"src": "日本語4", "mt": "译文4", "ref": "参考4"},
                               {"src": "日本語1", "mt": "译文1", "ref": "参考1"}]]  # file order, not idx order
    assert "繁體" not in json.dumps(scorer.triples, ensure_ascii=False)
    assert scorer.loaded == [(MODEL, 0)] and scorer.batch_sizes == [8]


def test_the_output_file_holds_every_field_and_the_scores_in_file_order(tmp_path):
    """AC4"""
    run_dir = make_run(tmp_path, [segment(3), segment(1, page_index=4, run_id=9)])
    scorer = FakeScorer([0.25, 0.75], device="cpu")

    code, _ = run(run_dir, scorer=scorer)

    assert code == 0 and listing(run_dir) == [OUTPUT, "manifest.json", "segments.jsonl"]
    out = json.loads((run_dir / OUTPUT).read_text(encoding="utf-8"))
    assert set(out) == {"eval_id", "model", "segments_sha256", "n_segments", "system_score", "segments", "versions",
                        "device", "gpus", "batch_size", "elapsed_s", "scored_at"}
    assert (out["eval_id"], out["model"], out["n_segments"]) == ("20261028T000000Z-full", MODEL, 2)
    assert out["segments_sha256"] == hashlib.sha256((run_dir / "segments.jsonl").read_bytes()).hexdigest()
    assert out["system_score"] == 0.5
    assert out["segments"] == [{"book": "book_a", "page_index": 2, "idx": 3, "run_id": 7, "score": 0.25},
                               {"book": "book_a", "page_index": 4, "idx": 1, "run_id": 9, "score": 0.75}]
    assert out["versions"] == {"unbabel-comet": "2.2.7", "torch": "9.9.9", "python": sys.version.split()[0]}
    assert (out["device"], out["gpus"], out["batch_size"]) == ("cpu", 0, 8)
    assert out["elapsed_s"] >= 0 and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\+00:00", out["scored_at"])


def test_the_gpu_options_reach_the_scorer_and_the_file_names_the_device(tmp_path):
    scorer = FakeScorer(device="NVIDIA GeForce RTX 4080")

    code, _ = run(make_run(tmp_path), "--model", "Unbabel/XCOMET-XL", "--gpus", "1", "--batch-size", "16", scorer=scorer)

    assert code == 0
    assert scorer.loaded == [("Unbabel/XCOMET-XL", 1)] and scorer.batch_sizes == [16]
    out = json.loads((tmp_path / "20261028T000000Z-full" / "comet-XCOMET-XL.json").read_text(encoding="utf-8"))
    assert (out["model"], out["device"], out["gpus"], out["batch_size"]) == ("Unbabel/XCOMET-XL", "NVIDIA GeForce RTX 4080", 1, 16)


def test_scoring_again_replaces_the_file(tmp_path):
    run_dir = make_run(tmp_path)
    run(run_dir, scorer=FakeScorer([0.1, 0.2]))

    code, _ = run(run_dir, scorer=FakeScorer([0.3, 0.4]))

    assert code == 0 and listing(run_dir) == [OUTPUT, "manifest.json", "segments.jsonl"]
    assert json.loads((run_dir / OUTPUT).read_text(encoding="utf-8"))["segments"][0]["score"] == 0.3


@pytest.mark.parametrize("failing", ["rename", "write"])
def test_a_failing_write_keeps_the_earlier_file_and_leaves_no_temporary_one(tmp_path, monkeypatch, capsys, failing):
    """AC4: the file is replaced in one step."""
    run_dir = make_run(tmp_path)
    earlier = '{"earlier": true}\n'
    (run_dir / OUTPUT).write_text(earlier, encoding="utf-8")
    if failing == "rename":
        monkeypatch.setattr(comet_script.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk on fire")))
    else:
        def broken_write(self, *a, **k):
            self.touch()  # a partly written temporary file
            raise OSError("disk full")
        monkeypatch.setattr(comet_script.Path, "write_text", broken_write)

    code, _ = run(run_dir)

    assert code == 1 and "Nothing was written" in capsys.readouterr().err
    monkeypatch.undo()
    assert (run_dir / OUTPUT).read_text(encoding="utf-8") == earlier
    assert listing(run_dir) == [OUTPUT, "manifest.json", "segments.jsonl"]


def test_a_predictor_that_returns_the_wrong_number_of_scores_writes_nothing(tmp_path, capsys):
    run_dir = make_run(tmp_path)

    code, _ = run(run_dir, scorer=FakeScorer([0.5]))

    assert code == 1 and "1 scores for 2 segments" in capsys.readouterr().err
    assert listing(run_dir) == ["manifest.json", "segments.jsonl"]


def refused(tmp_path, capsys, run_dir, *args):
    """Run on a directory that must be refused: exit 2, the cause on stderr, no model loaded, nothing written."""
    before = listing(run_dir.parent) if run_dir.exists() else None
    inside = listing(run_dir) if run_dir.exists() else None
    code, scorer = run(run_dir, *args)
    err = capsys.readouterr().err
    assert code == 2 and scorer.loaded == [] and scorer.triples == []
    assert err.startswith("Nothing was scored or written: ")
    assert (listing(run_dir.parent) if run_dir.exists() else None) == before
    assert (listing(run_dir) if run_dir.exists() else None) == inside
    return err


def test_a_missing_directory_is_refused(tmp_path, capsys):
    """AC5"""
    err = refused(tmp_path, capsys, tmp_path / "does-not-exist")

    assert "does-not-exist is not a directory" in err


def test_an_unfinished_eval_run_is_refused(tmp_path, capsys):
    """AC5: since #37 the manifest of an eval that has not reached its last page says `finished_at: null`."""
    err = refused(tmp_path, capsys, make_run(tmp_path, finished_at=None))

    assert "not finished" in err and "finished_at" in err and "20261028T000000Z-full" in err


def test_a_manifest_without_finished_at_is_refused_too(tmp_path, capsys):
    run_dir = make_run(tmp_path)
    (run_dir / "manifest.json").write_text(json.dumps({"eval_id": "x"}), encoding="utf-8")

    assert "not finished" in refused(tmp_path, capsys, run_dir)


def test_an_empty_segments_file_is_refused(tmp_path, capsys):
    """AC5"""
    run_dir = make_run(tmp_path, [])

    assert "segments.jsonl has no lines" in refused(tmp_path, capsys, run_dir)


def test_a_line_without_hyp_is_refused_and_the_message_names_the_line_and_the_field(tmp_path, capsys):
    """AC5"""
    broken = segment(1)
    del broken["hyp"]

    err = refused(tmp_path, capsys, make_run(tmp_path, [segment(0), broken]))

    assert "line 2" in err and "'hyp'" in err


@pytest.mark.parametrize("field", ["ja", "ref_zh_hans", "book", "page_index", "idx", "run_id"])
def test_every_field_the_scores_need_is_required(tmp_path, capsys, field):
    broken = segment(0)
    del broken[field]

    assert f"'{field}'" in refused(tmp_path, capsys, make_run(tmp_path, [broken]))


def test_a_reference_in_traditional_characters_is_not_needed(tmp_path):
    """`ref_zh_hant` is carried by the harness for reading only; the scorer never looks at it."""
    lean = segment(0)
    del lean["ref_zh_hant"]

    code, scorer = run(make_run(tmp_path, [lean]))

    assert code == 0 and scorer.triples[0][0]["ref"] == "参考0"


@pytest.mark.parametrize("text, cause", [("not json\n", "not valid JSON"), ("[1, 2]\n", "not a JSON object"),
                                          ('{"ja": 1}\n', "'book'")])
def test_a_malformed_line_is_refused(tmp_path, capsys, text, cause):
    run_dir = make_run(tmp_path)
    (run_dir / "segments.jsonl").write_text(text, encoding="utf-8")

    assert cause in refused(tmp_path, capsys, run_dir)


def test_a_text_field_that_is_not_a_string_is_refused(tmp_path, capsys):
    assert "'hyp' is not a string" in refused(tmp_path, capsys, make_run(tmp_path, [segment(0, hyp=None) | {"hyp": 5}]))


def test_a_missing_manifest_or_segments_file_is_refused(tmp_path, capsys):
    run_dir = make_run(tmp_path)
    (run_dir / "segments.jsonl").unlink()

    assert "segments.jsonl is missing" in refused(tmp_path, capsys, run_dir)


@pytest.mark.parametrize("args", [["--gpus", "-1"], ["--batch-size", "0"], ["--model", "Unbabel/"]])
def test_bad_arguments_are_refused_before_anything_is_read(tmp_path, capsys, args):
    scorer = FakeScorer()

    with pytest.raises(SystemExit) as stopped:
        comet_script.main([str(make_run(tmp_path)), *args], load=scorer.load)

    assert stopped.value.code == 2 and scorer.loaded == []


def run_script(*args):
    """The script as a process of its own, in this (the project's) interpreter: no `comet`, no `torch`, no `fukidashi` path
    tricks beyond what the interpreter already has."""
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, timeout=60)


def test_the_script_runs_as_a_program_without_the_comet_environment(tmp_path):
    """The model is imported only once the run has been checked, so options and refusals need no torch."""
    helped = run_script("--help")
    assert helped.returncode == 0
    assert all(option in helped.stdout for option in ("--model", "--gpus", "--batch-size", "Unbabel/wmt22-comet-da"))

    missing = run_script(str(tmp_path / "does-not-exist"))
    assert (missing.returncode, missing.stdout) == (2, "")
    assert "Nothing was scored or written" in missing.stderr and "ModuleNotFoundError" not in missing.stderr

    unfinished = run_script(str(make_run(tmp_path, finished_at=None)))
    assert unfinished.returncode == 2 and "not finished" in unfinished.stderr


def fake_comet_site(tmp_path):
    """A directory that stands in for the script's environment: a `comet` and a `torch` that do no work, and the
    distribution metadata `importlib.metadata` reads the version from. The fake model writes what it was asked to the
    file named by FAKE_COMET_LOG."""
    site = tmp_path / "fake-site"
    (site / "comet").mkdir(parents=True)
    (site / "comet" / "__init__.py").write_text(
        "import json, os, types\n"
        "def download_model(name):\n"
        "    return 'checkpoint-of-' + name\n"
        "class Model:\n"
        "    def predict(self, data, batch_size, gpus, **kwargs):\n"
        "        with open(os.environ['FAKE_COMET_LOG'], 'w') as f:\n"
        "            json.dump({'batch_size': batch_size, 'gpus': gpus, 'kwargs': kwargs, 'data': data}, f)\n"
        "        scores = [0.25 * (i + 1) for i in range(len(data))]\n"
        "        return types.SimpleNamespace(scores=scores, system_score=sum(scores) / len(scores))\n"
        "def load_from_checkpoint(path):\n"
        "    assert path == 'checkpoint-of-Unbabel/wmt22-comet-da', path\n"
        "    return Model()\n", encoding="utf-8")
    (site / "torch").mkdir()
    (site / "torch" / "__init__.py").write_text(
        "import os, types\n__version__ = '9.9.9'\n"
        "cuda = types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda i: 'Fake GPU')\n"
        "backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: os.environ.get('FAKE_MPS') == '1'))\n",
        encoding="utf-8")
    (site / "unbabel_comet-2.2.7.dist-info").mkdir()
    (site / "unbabel_comet-2.2.7.dist-info" / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: unbabel-comet\nVersion: 2.2.7\n", encoding="utf-8")
    return site


@pytest.mark.parametrize("args, mps, device, asked", [
    ([], "0", "cpu", {"batch_size": 8, "gpus": 0, "kwargs": {}}),
    # COMET asks for the "fork" start method when MPS exists, and torch refuses that with no worker processes
    ([], "1", "cpu", {"batch_size": 8, "gpus": 0, "kwargs": {"num_workers": 1}}),
    (["--gpus", "1", "--batch-size", "4"], "1", "Fake GPU", {"batch_size": 4, "gpus": 1, "kwargs": {}}),
])
def test_the_real_loader_finds_the_comet_package_and_not_the_script_that_is_called_comet(tmp_path, args, mps, device, asked):
    """`uv run eval/comet.py` puts eval/ first on `sys.path`, where the script is itself `comet`. With a stand-in package
    in the environment, the script as a process must reach the package: this is the wiring of `load_scorer`."""
    run_dir = make_run(tmp_path, [segment(0), segment(1)])
    log = tmp_path / "asked.json"
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(fake_comet_site(tmp_path)), "FAKE_COMET_LOG": str(log),
           "FAKE_MPS": mps}

    done = subprocess.run([sys.executable, str(SCRIPT), str(run_dir), *args], capture_output=True, text=True, env=env,
                          timeout=60)

    assert done.returncode == 0, done.stderr
    got = json.loads(log.read_text(encoding="utf-8"))
    assert got.pop("data") == [{"src": "日本語0", "mt": "译文0", "ref": "参考0"}, {"src": "日本語1", "mt": "译文1", "ref": "参考1"}]
    assert got == asked
    out = json.loads((run_dir / OUTPUT).read_text(encoding="utf-8"))
    assert [s["score"] for s in out["segments"]] == [0.25, 0.5] and out["system_score"] == 0.375
    assert out["versions"] == {"unbabel-comet": "2.2.7", "torch": "9.9.9", "python": sys.version.split()[0]}
    assert (out["device"], out["gpus"], out["batch_size"]) == (device, asked["gpus"], asked["batch_size"])


def test_the_script_directory_is_taken_off_the_path_and_the_rest_stays(monkeypatch):
    monkeypatch.setattr(sys, "path", [str(SCRIPT.parent), "/somewhere/else", str(SCRIPT.parent) + "/../eval"])

    comet_script.keep_script_directory_off_the_path()

    assert sys.path == ["/somewhere/else"]
