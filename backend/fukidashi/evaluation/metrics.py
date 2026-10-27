"""Corpus metrics (sacreBLEU), cost and latency summaries. Pure functions: no DB, no model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sacrebleu.metrics import BLEU, CHRF


@dataclass(frozen=True)
class CorpusScores:
    chrf: float | None  # None when there is nothing to score; sacreBLEU gives no signature then either
    bleu: float | None
    chrf_signature: str | None
    bleu_signature: str | None


def score_corpus(hypotheses: Sequence[str], references: Sequence[str]) -> CorpusScores:
    """Corpus-level chrF and BLEU, one reference per segment. BLEU tokenizes Chinese with `zh`."""
    if len(hypotheses) != len(references):
        raise ValueError(f"{len(hypotheses)} hypotheses but {len(references)} references")
    chrf, bleu = CHRF(), BLEU(tokenize="zh")
    if not hypotheses:
        return CorpusScores(None, None, None, None)
    # a signature exists only after the metric has seen references, so take it after scoring
    chrf_score = chrf.corpus_score(list(hypotheses), [list(references)]).score
    bleu_score = bleu.corpus_score(list(hypotheses), [list(references)]).score
    return CorpusScores(chrf_score, bleu_score, str(chrf.get_signature()), str(bleu.get_signature()))


def percentile(values: Sequence[float], q: float) -> float | None:
    """The q-th percentile (0-100) by linear interpolation between ranks; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def latency_summary(seconds: Sequence[float]) -> dict:
    p50, p95 = percentile(seconds, 50), percentile(seconds, 95)
    return {"n": len(seconds), "p50_s": None if p50 is None else round(p50, 3),
            "p95_s": None if p95 is None else round(p95, 3)}


def run_cost(cost_usd: float | None, unknown_usage_calls: int) -> float | None:
    """A Run's cost, or None when it cannot be trusted: a call without usage makes any sum partial."""
    return None if cost_usd is None or unknown_usage_calls else cost_usd


def cost_summary(runs: Sequence[tuple[float | None, int]]) -> dict:
    """Total and per-page cost over Runs given as (cost_usd, unknown_usage_calls).

    The total is None when any Run's cost is unknown, never a partial sum. No Runs cost 0.0 and have no
    per-page cost.
    """
    costs = [run_cost(cost, unknown) for cost, unknown in runs]
    unknown_cost_runs = sum(c is None for c in costs)
    total = None if unknown_cost_runs else round(sum(costs, 0.0), 6)
    return {
        "total_usd": total,
        "per_page_usd": None if total is None or not costs else round(total / len(costs), 6),
        "unknown_cost_runs": unknown_cost_runs,
        "unknown_usage_calls": sum(unknown for _, unknown in runs),
    }
