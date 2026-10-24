import pytest
from openai.types import CompletionUsage

from app.llm.client import usage_from_api
from app.llm.metering import BudgetExceeded, CallRecord, Meter, PageBudget, Usage


def record(usage, cost=None):
    return CallRecord("book/001", "m", "m", usage, 0.1, cost)


def test_missing_usage_is_unknown_not_zero():
    assert usage_from_api(None) is None
    budget = PageBudget("book/001", limit_tokens=100)
    budget.charge(None)
    assert budget.used_tokens == 0
    assert budget.unknown_calls == 1
    assert not budget.verified


def test_usage_from_api_reads_tokens():
    u = usage_from_api(CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
    assert u == Usage(input_tokens=10, output_tokens=5, cached_input_tokens=None)


def test_summary_keeps_unknown_calls_separate():
    meter = Meter()
    meter.record(record(Usage(10, 5), cost=0.01))
    meter.record(record(None))
    s = meter.summary()
    assert s["calls"] == 2
    assert s["unknown_usage_calls"] == 1
    assert (s["input_tokens"], s["output_tokens"]) == (10, 5)
    assert s["cost_usd"] == 0.01


def test_cost_is_none_when_a_known_call_has_no_price():
    meter = Meter()
    meter.record(record(Usage(10, 5), cost=0.01))
    meter.record(record(Usage(10, 5), cost=None))
    assert meter.summary()["cost_usd"] is None


def test_budget_blocks_the_next_call_once_spent():
    budget = PageBudget("book/001", limit_tokens=100)
    budget.check()
    budget.charge(Usage(80, 30))
    with pytest.raises(BudgetExceeded):
        budget.check()
