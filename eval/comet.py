# /// script
# requires-python = ">=3.12,<3.14"  # unbabel-comet 2.2.7 does not import on 3.14 (functools._HashedSeq is gone)
# dependencies = [
#     "unbabel-comet==2.2.7",
#     "setuptools<82",  # unbabel-comet pins torchmetrics 0.10.3, which imports pkg_resources; setuptools 82 removed it
# ]
# ///
"""Score a finished eval run with COMET (OUTLINE §4.4).

Usage:
  uv run eval/comet.py runs/eval/<id>
  uv run eval/comet.py runs/eval/<id> --model Unbabel/XCOMET-XL --gpus 1 --batch-size 16

This script is the one entry point in eval/ whose logic is not in `fukidashi.evaluation`. It has an environment of its
own (the inline metadata above; `uv run` builds it and leaves the project's alone) and imports nothing from `fukidashi`,
for three reasons: torch would otherwise enter every `uv sync`, the CI and the lock; COMET's pins would have to be
reconciled with OpenTelemetry's and LangChain's; and the script has to run on another machine (the Windows GPU, for
xCOMET) with nothing but itself and a `segments.jsonl`.

It reads `manifest.json` and `segments.jsonl` of an eval run and scores `src = ja`, `mt = hyp`, `ref = ref_zh_hans` of
every line, in file order. It makes no model call to a provider, does not translate, and does not touch the database or
the eval run's own files. It writes `comet-<model name without the organisation>.json` into the run's directory, replaced
in one step (a temporary file, then a rename). An eval run whose manifest has `finished_at: null` is refused, because
its pages are not all there.

The file holds `eval_id`, `model`, `segments_sha256` (of `segments.jsonl` as read), `n_segments`, `system_score` (COMET's
system score, the mean of the segment scores), `segments` (`book`, `page_index`, `idx`, `run_id` and `score`, in the
file's order), `versions`, `device`, `gpus`, `batch_size`, `elapsed_s` (the scoring itself, not the model's download or
loading) and `scored_at`.

The first run with a model downloads its checkpoint from Hugging Face into the Hugging Face cache (`Unbabel/wmt22-comet-da`:
Apache-2.0, about 2.3 GB). `Unbabel/XCOMET-XL` is gated: its terms are accepted on Hugging Face, by the person running it.

Exit codes: 0 the scores were written; 2 nothing was written because the run cannot be scored (the directory is missing,
the manifest is unfinished, `segments.jsonl` has no lines, or a line lacks a field; the message says which) or the
arguments are wrong; 1 scoring or writing failed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

DEFAULT_MODEL = "Unbabel/wmt22-comet-da"
KEY_FIELDS = ("book", "page_index", "idx", "run_id")  # copied to the output; they say which segment a score belongs to
TEXT_FIELDS = ("ja", "hyp", "ref_zh_hans")  # the triple; `ref_zh_hant` is never read


class RunRejected(Exception):
    """The eval run cannot be scored; the message says why. Nothing has been written."""


class Scorer(NamedTuple):
    """What `main` needs from COMET: its predict, and the facts the output records about where it ran."""

    predict: Callable[[list[dict], int], tuple[list[float], float]]  # (triples, batch size) -> (scores, system score)
    device: str  # "cpu", or the GPU's name
    versions: dict[str, str]  # {"unbabel-comet": ..., "torch": ...}


def keep_script_directory_off_the_path() -> None:
    """`uv run eval/comet.py` puts eval/ first on `sys.path`, where this file is `comet`: `import comet` would find the
    script itself instead of the package."""
    here = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != here]


def load_scorer(model_name: str, gpus: int) -> Scorer:
    """Download (when not cached) and load the checkpoint. The only place that touches `comet` and `torch`."""
    keep_script_directory_off_the_path()
    from importlib.metadata import version

    import torch
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(model_name))

    # With MPS present (any Apple Silicon Mac), COMET asks the DataLoader for the "fork" start method, and its default of
    # no worker processes on the CPU (`2 * gpus`) makes torch refuse that: one worker is enough to get past it.
    workers = {"num_workers": 1} if gpus == 0 and torch.backends.mps.is_available() else {}

    def predict(triples: list[dict], batch_size: int) -> tuple[list[float], float]:
        prediction = model.predict(triples, batch_size=batch_size, gpus=gpus, **workers)
        return list(prediction.scores), prediction.system_score

    if gpus == 0:
        device = "cpu"
    else:
        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "gpu (name unknown)"
    return Scorer(predict, device, {"unbabel-comet": version("unbabel-comet"), "torch": torch.__version__})


def load_run(run_dir: Path) -> tuple[dict, list[dict], str]:
    """The manifest, the segments in file order and the SHA-256 of `segments.jsonl`. Raises `RunRejected`."""
    if not run_dir.is_dir():
        raise RunRejected(f"{run_dir} is not a directory")
    manifest_path, segments_path = run_dir / "manifest.json", run_dir / "segments.jsonl"
    for path in (manifest_path, segments_path):
        if not path.is_file():
            raise RunRejected(f"{path} is missing")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise RunRejected(f"{manifest_path} is not valid JSON: {e}")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("eval_id"), str):
        raise RunRejected(f"{manifest_path} has no eval_id")
    if manifest.get("finished_at") is None:
        raise RunRejected(f"the eval run {manifest['eval_id']} is not finished (finished_at is null in its manifest): "
                          "its pages are not all there")

    raw = segments_path.read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as e:
        raise RunRejected(f"{segments_path} is not valid UTF-8: {e}")
    segments = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            segment = json.loads(line)
        except ValueError as e:
            raise RunRejected(f"{segments_path} line {number} is not valid JSON: {e}")
        if not isinstance(segment, dict):
            raise RunRejected(f"{segments_path} line {number} is not a JSON object")
        for field in (*KEY_FIELDS, *TEXT_FIELDS):
            if field not in segment:
                raise RunRejected(f"{segments_path} line {number} has no {field!r}")
        for field in TEXT_FIELDS:
            if not isinstance(segment[field], str):
                raise RunRejected(f"{segments_path} line {number}: {field!r} is not a string")
        segments.append(segment)
    if not segments:
        raise RunRejected(f"{segments_path} has no lines")
    return manifest, segments, hashlib.sha256(raw).hexdigest()


def build_triples(segments: list[dict]) -> list[dict]:
    """COMET's input, in the segments' order: the Japanese, the system's translation and the zh-Hans reference."""
    return [{"src": s["ja"], "mt": s["hyp"], "ref": s["ref_zh_hans"]} for s in segments]


def output_name(model: str) -> str:
    """`Unbabel/wmt22-comet-da` -> `comet-wmt22-comet-da.json`."""
    return f"comet-{model.rsplit('/', 1)[-1]}.json"


def write_json(path: Path, data: dict) -> None:
    """Replace `path` in one step: write a temporary file next to it, then rename it over, so a reader sees the old file
    or the new one, never half of either. A failure leaves the old file and no temporary one."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Score a finished eval run's segments with COMET and write the scores "
                                             "next to them.")
    ap.add_argument("run_dir", type=Path, help="an eval run's directory, e.g. runs/eval/<id>")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"a COMET checkpoint on Hugging Face (default: {DEFAULT_MODEL}; --model Unbabel/XCOMET-XL "
                         "is gated: accept its terms on Hugging Face first)")
    ap.add_argument("--gpus", type=int, default=0, help="GPUs to use; 0 runs on the CPU (default: 0)")
    ap.add_argument("--batch-size", type=int, default=8, help="segments per batch (default: 8)")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"[\w.-]+", args.model.rsplit("/", 1)[-1]):
        ap.error(f"--model {args.model!r} does not end in a checkpoint name like Unbabel/wmt22-comet-da")
    if args.gpus < 0:
        ap.error("--gpus must be 0 or more")
    if args.batch_size < 1:
        ap.error("--batch-size must be 1 or more")
    return args


def main(argv: list[str] | None = None, *, load: Callable[[str, int], Scorer] = load_scorer) -> int:
    """`load` builds the scorer after the run has been checked; tests pass one that needs no torch."""
    args = parse_args(argv)
    try:
        manifest, segments, segments_sha256 = load_run(args.run_dir)
    except RunRejected as e:
        print(f"Nothing was scored or written: {e}", file=sys.stderr)
        return 2

    scorer = load(args.model, args.gpus)
    started = time.monotonic()
    scores, system_score = scorer.predict(build_triples(segments), args.batch_size)
    elapsed = time.monotonic() - started
    if len(scores) != len(segments) or not all(math.isfinite(s) for s in [*scores, system_score]):
        print(f"Nothing was written: COMET returned {len(scores)} scores for {len(segments)} segments, "
              "or one that is not a finite number.", file=sys.stderr)
        return 1

    result = {
        "eval_id": manifest["eval_id"],
        "model": args.model,
        "segments_sha256": segments_sha256,
        "n_segments": len(segments),
        "system_score": system_score,
        "segments": [{**{field: s[field] for field in KEY_FIELDS}, "score": score}
                     for s, score in zip(segments, scores)],
        "versions": {**scorer.versions, "python": platform.python_version()},
        "device": scorer.device,
        "gpus": args.gpus,
        "batch_size": args.batch_size,
        "elapsed_s": round(elapsed, 3),
        "scored_at": _now(),
    }
    path = args.run_dir / output_name(args.model)
    try:
        write_json(path, result)
    except OSError as e:
        print(f"Nothing was written: could not replace {path}: {e}", file=sys.stderr)
        return 1
    print(f"{manifest['eval_id']}: {args.model} system score {system_score:.4f} over {len(segments)} segments "
          f"({scorer.device}, {elapsed:.1f}s) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
