"""Per-call usage records and per-page token budgets.

Unknown usage is never recorded as zero: a call whose provider returns no usage
is counted separately, and the page budget is flagged as unverified.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int | None = None

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class CallRecord:
    page_id: str
    requested_model: str
    response_model: str | None
    usage: Usage | None  # None means the provider returned no usage: unknown, not zero
    latency_s: float
    cost_usd: float | None


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class PageBudget:
    page_id: str
    limit_tokens: int
    used_tokens: int = 0
    unknown_calls: int = 0

    @property
    def verified(self) -> bool:
        return self.unknown_calls == 0

    def check(self) -> None:
        if self.used_tokens >= self.limit_tokens:
            raise BudgetExceeded(
                f"page {self.page_id}: used {self.used_tokens} of {self.limit_tokens} tokens"
            )

    def charge(self, usage: Usage | None) -> None:
        if usage is None:
            self.unknown_calls += 1
        else:
            self.used_tokens += usage.total


@dataclass
class Meter:
    ledger_path: Path | None = None
    records: list[CallRecord] = field(default_factory=list)

    def record(self, rec: CallRecord) -> None:
        self.records.append(rec)
        if self.ledger_path is not None:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        known = [r for r in self.records if r.usage is not None]
        costs = [r.cost_usd for r in known if r.cost_usd is not None]
        # None when there are no calls, a call has no usage, or a call with usage has no price:
        # a partial sum would understate spend
        complete = bool(self.records) and len(known) == len(self.records) and len(costs) == len(known)
        return {
            "calls": len(self.records),
            "unknown_usage_calls": len(self.records) - len(known),
            "input_tokens": sum(r.usage.input_tokens for r in known),
            "output_tokens": sum(r.usage.output_tokens for r in known),
            "cost_usd": round(sum(costs), 6) if complete else None,
        }
